import ast
import json
import logging
import math
import os
import random
import sys
import braceexpand
from dataclasses import dataclass
from multiprocessing import Value
from functools import partial
from typing import Dict, Any
import pickle

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import (
    Dataset,
    DataLoader,
    SubsetRandomSampler,
    IterableDataset,
    get_worker_info,
)
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import (
    base_plus_ext,
    url_opener,
    tar_file_expander,
    valid_sample,
)
import collections

from torchvision import transforms


from .abo import get_zero_shot_classification_data


try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


class CsvDataset(Dataset):
    def __init__(
        self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None
    ):
        logging.debug(f"Loading csv data from {input_filename}.")
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug("Done loading data.")

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split("::")
        assert len(weights) == len(
            urllist
        ), f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, "sizes.json")
    len_filename = os.path.join(dir_path, "__len__")
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, "r"))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, "r").read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset

        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype("int")
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = "txt" in sample
    has_image = (
        "png" in sample or "jpg" in sample or "jpeg" in sample or "webp" in sample
    )
    return has_caption and has_image



def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


def group_by_keys_nothrow(
    data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None
):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        if "fname" not in filesample:
            continue
        fname, value = filesample["fname"], filesample["data"]
        
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if (
            current_sample is None
            or prefix != current_sample["__key__"]
            or suffix in current_sample
        ):
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
        self,
        bufsize=1000,
        initial=100,
        seed=0,
        epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(
                self.weights
            ), f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(
                    url=self.rng.choices(self.urls, weights=self.weights, k=1)[0]
                )
                
def load_image(path):
    im = Image.open(path)
    return im.convert('RGB')


def load_key_to_caption(generated_captions_path):
    image_paths = os.listdir(generated_captions_path)
    image_ids = [os.path.splitext(image_path)[0] for image_path in image_paths]
    key_to_caption = {}
    for image_id in image_ids:
        with open(os.path.join(generated_captions_path, f"{image_id}.txt"), "r") as f:
            caption = f.read().strip()
        key_to_caption[image_id] = caption
    return key_to_caption

def load_key_to_image(path):
    image_paths = os.listdir(path)
    image_ids = [os.path.splitext(image_path)[0] for image_path in image_paths]
    key_to_image = {}
    for image_id in image_ids:
        image_path = os.path.join(path, f"{image_id}.jpg")
        image = load_image(image_path)
        key_to_image[image_id] = image
    return key_to_image


def load_key_to_clip_prediction(clip_prediction_path):
    with open(clip_prediction_path, "rb") as f:
        key_to_clip_prediction = pickle.load(f)
    return key_to_clip_prediction
 
 
def captions_from_clip_predicted_classes(sample, args, key_to_clip_prediction, 
                                         key_to_caption=None, 
                                         key_to_common_caption=None,
                                         class_to_image_ids=None):
    """
    Load and create captions from clip prediction, per-image, and per-image-group caption
    :param sample: data dict of webdataset loader
    :param args: program args
    :param key_to_clip_prediction: dict of image_id to clip prediction
    :param key_to_caption: dict of image_id to per-image caption
    :param key_to_common_caption: dict of image_id to per-image-group caption
    :param class_to_image_ids: dict of class to image_ids
    :return:
    """
    image_id = sample["__key__"]
    clip_prediction = key_to_clip_prediction[image_id]
    k = args.class_per_image
    
    classnames = clip_prediction["class_names"]
    if key_to_caption is not None:
        random_id = np.random.choice(len(key_to_caption))
        generated_captions = key_to_caption[random_id][image_id].split("\n")
        
    if key_to_common_caption is not None:
        image_id = np.random.choice(class_to_image_ids[classnames[0]])
        common_captions = key_to_common_caption[0][image_id].split("\n")
        sample['common_text'] = [f"{common_captions[choice]}. a photo of a {classnames[choice]}" for choice in range(k)]
    else:
        sample['common_text'] = [f"a photo of a {classnames[choice]}" for choice in range(k)]
    
    

    if args.text_type == "concat":
        sample['text'] = [f"{generated_captions[choice]}. a photo of a {classnames[choice]}" for choice in range(k)]
    elif args.text_type == "label":
        sample['text'] = [f"a photo of a {classnames[choice]}." for choice in range(k)]
    elif args.text_type == "gen":
        sample['text'] = [f"{generated_captions[choice]}." for choice in range(k)]
    else:
        raise ValueError(f"Invalid text_type: {args.text_type}")
    
    sample['label_text'] = [f"a photo of a {classnames[0]}."]
    sample['per_image_text'] = [f"{generated_captions[0]}"]
    sample['per_image_group_text'] = [f"{common_captions[0]}"]
    sample['common_text'] = sample['per_image_group_text']

    
    sample['text_raw'] = sample['per_image_text']

    sample['zeroshot_classnames'] = [classnames[choice] for choice in range(k)]
    sample['image_id'] = image_id
  
    return sample
    
    

def randomly_use_alternative_caption(
    sample: Dict[str, Any],
    key_to_caption: Dict[int, str] = None,
    args = None,
    key_to_caption_hard_mining=None,
    probability_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Replace the caption (e.g. AltText) randomly with an alternative caption (e.g. generated by MLLM).

    :param sample: Webdataset sample, a dict with keys txt, image and metadata
    :param key_to_caption: A dictionary that maps from running Webdataset index to alternative captions.
    :param probability_threshold: Float between 0 and 1, probability of replacing AltText with alternative caption
    :return: An updated sample with randomly replaced caption.
    """
    
    running_idx = sample["__key__"]  # running index over webdataset
    captions = sample["text"].split("\n")
    select_caption_id = random.randint(0, len(captions) - 1)
    sample['text'] = captions[select_caption_id]
    # import pdb;pdb.set_trace()
    if args.train_with_gt_text:
        sample["text"] = sample["metadata"]["class_name"].lower()
        
    
    
    # if random.random() < probability_threshold:
    #     sample["text"] = key_to_caption[running_idx]
    
    if args.similar_images_path is not None and args.hard_mining_captions_path is not None:
        running_idx = sample["__key__"]
        similar_images_json_path = os.path.join(args.similar_images_path, f"{running_idx}.json")
        with open(similar_images_json_path, "r") as f:
            similar_images_json = json.load(f)
            
        # the first one is the reference image to double check, thus we take the second one
        similar_image_id = similar_images_json['similar_images'][11]
        similar_image_path = os.path.join(args.preprocess_path, "train", f"{similar_image_id}.jpg")
        similar_image = load_image(similar_image_path)
        
        hard_mining_caption_1 = key_to_caption_hard_mining[running_idx]
        hard_mining_caption_2 = key_to_caption_hard_mining[similar_image_id]
        # hard_mining_caption_path_1 = os.path.join(args.hard_mining_captions_path, f"{running_idx}.txt")
        # with open(hard_mining_caption_path_1, "r") as f:
        #     hard_mining_caption_1 = f.read().strip()
        
            
        # hard_mining_caption_path_2 = os.path.join(args.hard_mining_captions_path, f"{similar_image_id}.txt")
        # with open(hard_mining_caption_path_2, "r") as f:
        #     hard_mining_caption_2 = f.read().strip()
        
        sample["hard_mining_caption_1"] = hard_mining_caption_1.replace('"', '').replace("\n", " ")
        sample["hard_mining_caption_2"] = hard_mining_caption_2.replace('"', '').replace("\n", " ")

        sample["similar_image"] = similar_image
        sample['image_id'] = running_idx
    return sample


def get_wds_dataset(
    args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None
):
    
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, "dataset_resampled", False) and is_train

    num_shards = None
    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)

            if not num_samples:
                raise RuntimeError(
                    "Currently, the number of dataset samples must be specified for the training dataset. "
                    "Please specify it via `--train-num-samples` if no dataset length info is present."
                )
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0

    shared_epoch = SharedEpoch(
        epoch=epoch
    )  # create a shared epoch store to sync epoch to dataloader worker proc

    if is_train and args.train_data_upsampling_factors is not None:
        assert (
            resampled
        ), "--train_data_upsampling_factors is only supported when sampling with replacement (with --dataset-resampled)."

    if resampled:
        pipeline = [
            ResampledShards2(
                input_shards,
                weights=args.train_data_upsampling_factors,
                deterministic=True,
                epoch=shared_epoch,
            )
        ]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend(
                [
                    detshuffle2(
                        bufsize=_SHARD_SHUFFLE_SIZE,
                        initial=_SHARD_SHUFFLE_INITIAL,
                        seed=args.seed,
                        epoch=shared_epoch,
                    ),
                    wds.split_by_node,
                    wds.split_by_worker,
                ]
            )
        pipeline.extend(
            [
                # at this point, we have an iterator over the shards assigned to each worker at each node
                tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
                wds.shuffle(
                    bufsize=_SAMPLE_SHUFFLE_SIZE,
                    initial=_SAMPLE_SHUFFLE_INITIAL,
                ),
                # wds.slice(int(num_samples * 0.6)),
            ]
        )
        # if args.subsample_ratio < 1.0:
        #     pipeline.extend(
        #         wds.slice(int(num_samples * args.subsample_ratio))
        #     )
    else:
        pipeline.extend(
            [
                wds.split_by_worker,
                # at this point, we have an iterator over the shards assigned to each worker
                wds.tarfile_to_samples(handler=log_and_continue),
            ]
        )
    
    pipeline.extend(
        [
            wds.select(filter_no_caption_or_no_image),
            # wds.select(lambda sample: random.random() < args.subsample_ratio),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.rename(image="jpg;png;jpeg;webp", 
                       distill_image="jpg;png;jpeg;webp", 
                       text="txt", metadata="json"),
        ]
    )


    if is_train:
        if args.clip_prediction_path is not None:
            # Load mapping image_id to clip prediction
            key_to_clip_prediction = load_key_to_clip_prediction(args.clip_prediction_path)
            key_to_caption = None
  
            # Load mapping of image_id to per-image captions
            key_to_caption = []
            if args.generated_captions_path:
                for caption_path in args.generated_captions_path:
                    key_to_caption.append(load_key_to_caption(caption_path))
                    
            # Load mapping of image_id to per-image-group captions
            if args.generated_common_captions_path:
                key_to_common_caption = []
                for caption_path in args.generated_common_captions_path:
                    key_to_common_caption.append(load_key_to_caption(caption_path))
            else:
                key_to_common_caption = None
            
            # Create mapping of class to image_ids belong to each class
            class_to_image_ids = collections.defaultdict(list)
            image_ids = key_to_clip_prediction.keys()
            for image_id in image_ids:
                class_name = key_to_clip_prediction[image_id]["class_names"][0]
                class_to_image_ids[class_name].append(image_id)
       
       
            text_augment = partial(
                captions_from_clip_predicted_classes,
                key_to_clip_prediction=key_to_clip_prediction,
                key_to_caption=key_to_caption,
                key_to_common_caption=key_to_common_caption,
                class_to_image_ids=class_to_image_ids,
                args=args, 
                
            )
            
     
            pipeline.extend(
                [   
                    wds.map(text_augment),
                    wds.map_dict(image=preprocess_img, 
                                 distill_image=preprocess_img,
                                 text_raw=lambda texts: texts,     
                                 text=lambda texts: torch.stack([tokenizer(text)[0] for text in texts]),
                                #  common_text=lambda texts: torch.stack([tokenizer(text)[0] for text in texts]),
                                 common_text=lambda texts: texts,
                                 label_text=lambda texts: torch.stack([tokenizer(text)[0] for text in texts]),
                                 per_image_text=lambda texts: torch.stack([tokenizer(text)[0] for text in texts]),
                                 per_image_group_text=lambda texts: torch.stack([tokenizer(text)[0] for text in texts]),
                                 ),
                    wds.to_tuple("image", "distill_image", "text", "common_text", "text_raw", 
                                 "label_text", "per_image_text", "per_image_group_text",
                                 "metadata", "zeroshot_classnames"),
                    wds.batched(args.batch_size, partial=not is_train),
                ]
            )
        elif args.hard_mining_captions_path is not None:
            # NOTE: NOT USE
            key_to_caption_hard_mining = load_key_to_caption(args.hard_mining_captions_path)
            
            text_augment = partial(
                randomly_use_alternative_caption, 
                key_to_caption_hard_mining=key_to_caption_hard_mining,
                # key_to_caption=key_to_caption, 
                args=args, 
            )
            pipeline.extend(
                [
                    wds.map(text_augment),
                    wds.map_dict(image=preprocess_img, 
                                text=lambda text: tokenizer(text)[0],
                                hard_mining_caption_1=lambda text: tokenizer(text)[0],
                                hard_mining_caption_2=lambda text: tokenizer(text)[0],
                                similar_image=preprocess_img),
                    wds.to_tuple("image", "text", "metadata", "similar_image", "hard_mining_caption_1", "hard_mining_caption_2"),
                    wds.batched(args.batch_size, partial=not is_train),
                ]
            )
        else:
            
            text_augment = partial(
                randomly_use_alternative_caption, 
                args=args, 
            )
            pipeline.extend(
                [
                    wds.map(text_augment),
                    wds.map_dict(image=preprocess_img, 
                                text=lambda text: tokenizer(text)[0],
                                ),
                    wds.to_tuple("image", "text", "metadata"),
                    wds.batched(args.batch_size, partial=not is_train),
                ]
            )
    else:
        pipeline.extend([
            wds.map_dict(image=preprocess_img, 
                         text=lambda text: tokenizer(text)[0]),
            wds.to_tuple("image", "text", "metadata"),
            wds.batched(args.batch_size, partial=not is_train),
        ])
      
            

    

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert (
                num_shards >= args.workers * args.world_size
            ), "number of shards must be >= total workers"
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(
            num_batches / num_workers
        )  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
  
        dataset = dataset.with_epoch(
            num_worker_batches
        )  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)
    
 
    # To debug
    # if args.debug:
    for item in dataset:
        break
    
    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer,
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None
    
    

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        # drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


class SyntheticDataset(Dataset):

    def __init__(
        self,
        transform=None,
        image_size=(224, 224),
        caption="Dummy caption",
        dataset_size=100,
        tokenizer=None,
    ):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new("RGB", image_size)
        self.dataset_size = dataset_size

        self.preprocess_txt = lambda text: tokenizer(text)[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        return image, self.preprocess_txt(self.caption)


def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn,
        image_size=image_size,
        dataset_size=args.train_num_samples,
        tokenizer=tokenizer,
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "auto":
        ext = data_path.split(".")[-1]
        if ext in ["csv", "tsv"]:
            return get_csv_dataset
        elif ext in ["tar"]:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}."
            )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data or args.dataset_type == "synthetic":
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer
        )

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer
        )

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    # if args.abo_zeroshot_eval:
    if args.zeroshot_eval_data is not None:
        data[f"{args.zeroshot_eval_data}-val-zero-shot-classification"] = get_zero_shot_classification_data(
            args, f"{args.zeroshot_eval_data}_val_zeroshot_classification"
        )
        data[f"{args.zeroshot_eval_data}-train-zero-shot-classification"] = get_zero_shot_classification_data(
            args, f"{args.zeroshot_eval_data}_train_zeroshot_classification", 
        )
 

    return data

 
def get_classnames(args):
    class_names = get_zero_shot_classification_data(
        args, f"{args.zeroshot_eval_data}_val_zeroshot_classification"
    ).class_names

    return class_names


def get_classnames_and_templates(args):
    dataset = get_zero_shot_classification_data(
        args, f"{args.zeroshot_eval_data}_val_zeroshot_classification"
    )
    return dataset.class_names, dataset.templates