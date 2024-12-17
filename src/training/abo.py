from training.distributed import is_master
import os
import logging
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision.datasets import ImageFolder
from argparse import Namespace
from typing import List, Optional
from .datautils import AugMixAugmenter
import PIL
import yaml
import json
import random

import torch
from torch.utils.data import Dataset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToTensor,
)
from torchvision import transforms

BICUBIC = InterpolationMode.BICUBIC


class EnsureRGB:
    def __call__(self, image: PIL.Image):
        if image.mode != "RGB":
            return image.convert("RGB")
        else:
            return image


TRANSFORM_LIBRARY = {
    "default_32p": Compose([EnsureRGB(), Resize((224, 224)), ToTensor()]),
    "default_clip": Compose(
        [
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            EnsureRGB(),
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    ),
    "ffcv_clip": Compose(
        [
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            EnsureRGB(),
        ]
    ),
}


def get_transform(transform: str):
    if transform is None:
        return None
    elif callable(transform):
        return transform
    elif isinstance(transform, str):
        assert transform in TRANSFORM_LIBRARY
        return TRANSFORM_LIBRARY[transform]
    else:
        raise NotImplementedError



class BaseDataset(Dataset):
    """Base class to unify the dataset creation, parsing and provision across multiple datasets."""

    def __init__(
        self,
        preprocess_path: str,
        transform=None,
        train: bool = False,
        rank: int = 0,
        ngpus_per_node: int = 8,
        id_to_class_path = "id_to_product_type.json",
        class_to_id_path = "product_type_to_id.json",
        class_name_field = "product_type",
        **kwargs,
    ):
        self.preprocess_path = preprocess_path
        self.train = train
        self.rank = rank
        self.world_size = ngpus_per_node
        self.transform = get_transform(transform)

        
        if self.train:
            self.split_path = os.path.join(preprocess_path, "webdataset",  "train")
        else:
            self.split_path = os.path.join(preprocess_path, "webdataset", "val")
        all_files = os.listdir(self.split_path)
        self.unique_image_ids = list(
            set(os.path.splitext(file)[0] for file in all_files)
        )

        id_to_class_path = os.path.join(
            preprocess_path, id_to_class_path
        )
        
        class_to_id_path = os.path.join(
            preprocess_path, class_to_id_path
        )
        with open(id_to_class_path, "r") as f:
            self.id_to_class = json.load(f)
        with open(class_to_id_path, "r") as f:
            self.class_to_id = json.load(f)
        
        max_id = max([int(k) for k in self.id_to_class.keys()])
        self.class_names = [""] * (max_id + 1)
  
        for i in self.id_to_class:
            self.class_names[int(i)] = self.id_to_class[i]

        self.class_name_field = class_name_field
        self.templates = [
            lambda c: f"a photo of a {c}.",
        ]
    
    def __getitem__(self, index):
        image_id = self.unique_image_ids[index]
        img_path = os.path.join(self.split_path, image_id + ".jpg")
        image = Image.open(img_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        

        if self.transform is not None:
            image = self.transform(image)
        
        return image_id, image, self.get_annotation(image_id)
    

    def get_annotation(self, image_id):
        json_file_path = os.path.join(self.split_path, image_id + ".json")
        with open(json_file_path, "r") as f:
            json_data = json.load(f)
        class_name = json_data[self.class_name_field]
        return int(self.class_to_id[class_name])
    
    def __len__(self):
        return len(self.unique_image_ids)


class ABODataset(BaseDataset):
    """Base class to unify the dataset creation, parsing and provision across multiple datasets."""

    def __init__(
        self,
        preprocess_path: str,
        # filename: str,
        transform=None,
        train: bool = False,
        rank: int = 0,
        ngpus_per_node: int = 8,
        **kwargs,
    ):
        id_to_class_path = "id_to_product_type.json"
        class_to_id_path = "product_type_to_id.json"
        super().__init__(preprocess_path, transform, 
                         train, rank, ngpus_per_node, 
                         id_to_class_path, class_to_id_path,
                         class_name_field="product_type")
        
    
class INatDataset(BaseDataset):
    """Base class to unify the dataset creation, parsing and provision across multiple datasets."""

    def __init__(
        self,
        preprocess_path: str = "/home/ubuntu/data/uned_preprocess/webdataset/inat",
        # filename: str,
        transform=None,
        train: bool = False,
        rank: int = 0,
        ngpus_per_node: int = 8,
        **kwargs,
    ):
        id_to_class_path = "id_to_class.json"
        class_to_id_path = "class_to_id.json"
        super().__init__(preprocess_path, transform, 
                         train, rank, ngpus_per_node, 
                         id_to_class_path, class_to_id_path,
                         class_name_field="class_name")
        
        
    
class Caltech101Dataset(BaseDataset):
    """Base class to unify the dataset creation, parsing and provision across multiple datasets."""

    def __init__(
        self,
        preprocess_path: str = "$LATTECLIP_DATA_DIR/caltech101_preprocess",
        # filename: str,
        transform=None,
        train: bool = False,
        rank: int = 0,
        ngpus_per_node: int = 8,
        **kwargs,
    ):
        id_to_class_path = "id_to_class.json"
        class_to_id_path = "class_to_id.json"
        super().__init__(preprocess_path, transform, 
                         train, rank, ngpus_per_node, 
                         id_to_class_path, class_to_id_path,
                         class_name_field="class_name")
        
        
class Flower102Dataset(BaseDataset):
    """Base class to unify the dataset creation, parsing and provision across multiple datasets."""
    def __init__(
        self,
        preprocess_path: str = "$LATTECLIP_DATA_DIR/flower102_preprocess",
        # filename: str,
        transform=None,
        train: bool = False,
        rank: int = 0,
        ngpus_per_node: int = 8,
        **kwargs,
    ):
        id_to_class_path = "id_to_class.json"
        class_to_id_path = "class_to_id.json"
        super().__init__(preprocess_path, transform, 
                         train, rank, ngpus_per_node, 
                         id_to_class_path, class_to_id_path,
                         class_name_field="class_name")
        self.templates = [
            lambda c: f"a photo of a {c}, a type of flower.",
        ]
        
class OxfordPetsDataset(BaseDataset):
    """Base class to unify the dataset creation, parsing and provision across multiple datasets."""
    def __init__(
        self,
        preprocess_path: str = "$LATTECLIP_DATA_DIR/oxford_pets_preprocess",
        # filename: str,
        transform=None,
        train: bool = False,
        rank: int = 0,
        ngpus_per_node: int = 8,
        **kwargs,
    ):
        id_to_class_path = "id_to_class.json"
        class_to_id_path = "class_to_id.json"
        super().__init__(preprocess_path, transform, 
                         train, rank, ngpus_per_node, 
                         id_to_class_path, class_to_id_path,
                         class_name_field="class_name")
        # self.templates = [
        #     lambda c: f"a photo of a {c}, a type of pet.",
        # ]
        
        
class EurosatDataset(BaseDataset):
    """Base class to unify the dataset creation, parsing and provision across multiple datasets."""
    def __init__(
        self,
        preprocess_path: str = "$LATTECLIP_DATA_DIR/eurosat_preprocess",
        # filename: str,
        transform=None,
        train: bool = False,
        rank: int = 0,
        ngpus_per_node: int = 8,
        **kwargs,
    ):
        id_to_class_path = "id_to_class.json"
        class_to_id_path = "class_to_id.json"
        super().__init__(preprocess_path, transform, 
                         train, rank, ngpus_per_node, 
                         id_to_class_path, class_to_id_path,
                         class_name_field="class_name")
        # self.templates = [
        #     lambda c: f"a centered satellite photo of {c}",
        # ]
        
        
class FGVCAircraftDataset(BaseDataset):
    """Base class to unify the dataset creation, parsing and provision across multiple datasets."""
    def __init__(
        self,
        preprocess_path: str = "$LATTECLIP_DATA_DIR/fgvc_aircraft_preprocess",
        # filename: str,
        transform=None,
        train: bool = False,
        rank: int = 0,
        ngpus_per_node: int = 8,
        **kwargs,
    ):
        id_to_class_path = "id_to_class.json"
        class_to_id_path = "class_to_id.json"
        super().__init__(preprocess_path, transform, 
                         train, rank, ngpus_per_node, 
                         id_to_class_path, class_to_id_path,
                         class_name_field="class_name")
        self.templates = [
            lambda c: f"a photo of a {c}, a type of aircraft.",
        ]
        
        
class StandfordCarsDataset(BaseDataset):
    """Base class to unify the dataset creation, parsing and provision across multiple datasets."""
    def __init__(
        self,
        preprocess_path: str = "$LATTECLIP_DATA_DIR/stanford_cars_preprocess",
        # filename: str,
        transform=None,
        train: bool = False,
        rank: int = 0,
        ngpus_per_node: int = 8,
        **kwargs,
    ):
        id_to_class_path = "id_to_class.json"
        class_to_id_path = "class_to_id.json"
        super().__init__(preprocess_path, transform, 
                         train, rank, ngpus_per_node, 
                         id_to_class_path, class_to_id_path,
                         class_name_field="class_name")
        
  
class DtdDataset(BaseDataset):
    """Base class to unify the dataset creation, parsing and provision across multiple datasets."""
    def __init__(
        self,
        preprocess_path: str = "$LATTECLIP_DATA_DIR/dtd_preprocess",
        # filename: str,
        transform=None,
        train: bool = False,
        rank: int = 0,
        ngpus_per_node: int = 8,
        **kwargs,
    ):
        id_to_class_path = "id_to_class.json"
        class_to_id_path = "class_to_id.json"
        super().__init__(preprocess_path, transform, 
                         train, rank, ngpus_per_node, 
                         id_to_class_path, class_to_id_path,
                         class_name_field="class_name")
        self.templates = [
            lambda c: f"{c} texture.",
        ]



class Sun397Dataset(BaseDataset):
    """Base class to unify the dataset creation, parsing and provision across multiple datasets."""
    def __init__(
        self,
        preprocess_path: str = "$LATTECLIP_DATA_DIR/sun397_preprocess",
        # filename: str,
        transform=None,
        train: bool = False,
        rank: int = 0,
        ngpus_per_node: int = 8,
        **kwargs,
    ):
        id_to_class_path = "id_to_class.json"
        class_to_id_path = "class_to_id.json"
        super().__init__(preprocess_path, transform, 
                         train, rank, ngpus_per_node, 
                         id_to_class_path, class_to_id_path,
                         class_name_field="class_name")
        
        
class UCF101Dataset(BaseDataset):
    """Food-101 dataset."""

    def __init__(
        self,
        preprocess_path: str = "$LATTECLIP_DATA_DIR/ucf101_preprocess",
        # filename: str,
        transform=None,
        train: bool = False,
        rank: int = 0,
        ngpus_per_node: int = 8,
        **kwargs,
    ):
        id_to_class_path = "id_to_class.json"
        class_to_id_path = "class_to_id.json"
        super().__init__(preprocess_path, transform, 
                         train, rank, ngpus_per_node, 
                         id_to_class_path, class_to_id_path,
                         class_name_field="class_name")
        self.templates = [
            lambda c: f"a photo of a person doing {c}",
        ]
        
class Food101Dataset(BaseDataset):
    """Food-101 dataset."""

    def __init__(
        self,
        preprocess_path: str = "$LATTECLIP_DATA_DIR/food101_preprocess",
        # filename: str,
        transform=None,
        train: bool = False,
        rank: int = 0,
        ngpus_per_node: int = 8,
        **kwargs,
    ):
        id_to_class_path = "id_to_class.json"
        class_to_id_path = "class_to_id.json"
        super().__init__(preprocess_path, transform, 
                         train, rank, ngpus_per_node, 
                         id_to_class_path, class_to_id_path,
                         class_name_field="class_name")
        self.templates = [
            lambda c: f"a photo of a {c}, a type of food.",
        ]
        
    
        
class iFood2019Dataset(BaseDataset):
    """Food-101 dataset."""

    def __init__(
        self,
        preprocess_path: str = "/home/ubuntu/data/ifood2019_preprocess/webdataset/original",
        # filename: str,
        transform=None,
        train: bool = False,
        rank: int = 0,
        ngpus_per_node: int = 8,
        **kwargs,
    ):
        id_to_class_path = "id_to_class.json"
        class_to_id_path = "class_to_id.json"
        super().__init__(preprocess_path, transform, 
                         train, rank, ngpus_per_node, 
                         id_to_class_path, class_to_id_path,
                         class_name_field="class_name")
        # self.templates = [
        #     lambda c: f"a photo of a {c}, a type of food.",
        # ]


def get_dataset(
    data_loading_kwargs: Optional[dict],
    data_specific_kwargs: Optional[dict],
) -> Dataset:
    """Template for adding more datasets for downstream tasks"""

    if data_loading_kwargs["dataset_name"] == "inat_zero_shot":
        return INatDataset(
            **data_specific_kwargs, data_loading_kwargs=data_loading_kwargs
        )
        
    if data_loading_kwargs["dataset_name"] == "flower102_zero_shot":
        return Flower102Dataset(
            **data_specific_kwargs, data_loading_kwargs=data_loading_kwargs
        )
    
    if data_loading_kwargs["dataset_name"] == "eurosat_zero_shot":
        return EurosatDataset(
            **data_specific_kwargs, data_loading_kwargs=data_loading_kwargs
        )
    if data_loading_kwargs["dataset_name"] == "oxford_pets_zero_shot":
        return OxfordPetsDataset(
            **data_specific_kwargs, data_loading_kwargs=data_loading_kwargs
        )

    if data_loading_kwargs["dataset_name"] == "fgvc_aircraft_zero_shot":
        return FGVCAircraftDataset(
            **data_specific_kwargs, data_loading_kwargs=data_loading_kwargs
        )
    if data_loading_kwargs["dataset_name"] == "stanford_cars_zero_shot":
        return StandfordCarsDataset(
            **data_specific_kwargs, data_loading_kwargs=data_loading_kwargs
        )
    if data_loading_kwargs["dataset_name"] == "caltech101_zero_shot":
        return Caltech101Dataset(
            **data_specific_kwargs, data_loading_kwargs=data_loading_kwargs
        )
    if data_loading_kwargs["dataset_name"] == "food101_zero_shot":
        return Food101Dataset(
            **data_specific_kwargs, data_loading_kwargs=data_loading_kwargs
        )
    if data_loading_kwargs["dataset_name"] == "ucf101_zero_shot":
        return UCF101Dataset(
            **data_specific_kwargs, data_loading_kwargs=data_loading_kwargs
        )
    if data_loading_kwargs["dataset_name"] == "dtd_zero_shot":
        return DtdDataset(
            **data_specific_kwargs, data_loading_kwargs=data_loading_kwargs
        )
    if data_loading_kwargs["dataset_name"] == "ifood2019_zero_shot":
        return iFood2019Dataset(
            **data_specific_kwargs, data_loading_kwargs=data_loading_kwargs
        )
    if data_loading_kwargs["dataset_name"] == "sun397_zero_shot":
        return Sun397Dataset(
            **data_specific_kwargs, data_loading_kwargs=data_loading_kwargs
        )
        
    if data_loading_kwargs["dataset_name"] == "ABO_zero_shot":
        return ABODataset(
            **data_specific_kwargs, data_loading_kwargs=data_loading_kwargs
        )
    else:
        raise ValueError(f"Can't parse dataset: {data_loading_kwargs['dataset_name']}")


def get_loader(
    batch_size: int,
    num_workers: int = 1,
    distributed: bool = False,
    rank: int = 0,
    ngpus_per_node: int = None,
    is_downstream: bool = False,
    dataset_loading_kwargs: Optional[dict] = None,
    dataset_specific_kwargs: Optional[dict] = None,
    loader_kwargs: Optional[dict] = None,
    **kwargs,
):
    dataset_loading_kwargs["rank"] = rank
    dataset_loading_kwargs["ngpus_per_node"] = ngpus_per_node
    dataset = get_dataset(
        data_loading_kwargs=dict(dataset_loading_kwargs),
        data_specific_kwargs=dataset_specific_kwargs,
    )
    if distributed:
        sampler = DistributedSampler(dataset, rank=rank, shuffle=True)
    else:
        sampler = None
    
    for item in dataset:
        break
    
    return (
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            num_workers=num_workers,
            pin_memory=False,
            sampler=sampler,
            # prefetch_factor=2,
            drop_last=False,
            # persistent_workers=True,
            **(loader_kwargs or {}),
        ),
        sampler,
        dataset.class_names if dataset.class_names else None,
        dataset.templates
    )


def _load_template(path: str):
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def get_abo_zero_shot_classification_data(args):
    config = _load_template(args.eval_config_path)
    amazon_eval_data = {}

    task_config = config["tasks"]["zeroshot_classification"]
    dataloader = get_loader(**task_config, batch_size=args.batch_size)

    # assert dataloader[2], "Make sure that dataset has classnames available for zero-shot labels embeddings."

    amazon_eval_data = Namespace(
        **{
            "dataloader": dataloader[0],
            "class_names": [c.lower().replace("_", " ") for c in dataloader[2]],
        }
    )
    return amazon_eval_data

def get_caltech101_zero_shot_classification_data(args):
    config = _load_template(args.eval_config_path)
    
    amazon_eval_data = {}

    task_config = config["tasks"]["caltech101_zeroshot_classification"]
    dataloader = get_loader(**task_config, batch_size=args.batch_size)

    # assert dataloader[2], "Make sure that dataset has classnames available for zero-shot labels embeddings."

    amazon_eval_data = Namespace(
        **{
            "dataloader": dataloader[0],
            "class_names": [c.lower().replace("_", " ") for c in dataloader[2]],
        }
    )
    return amazon_eval_data

def get_zero_shot_classification_data(args, task_name):
    config = _load_template(args.eval_config_path)
    
    amazon_eval_data = {}

    if args.tta:
        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                        std=[0.26862954, 0.26130258, 0.27577711])
        
        base_transform = transforms.Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
        ])
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        data_transform = AugMixAugmenter(base_transform, preprocess, n_views=63, 
                                            augmix=True, hard_aug=False)
        batchsize = 1
    
        task_config = config["tasks"][task_name]
        task_config["dataset_specific_kwargs"]["transform"] = data_transform
        dataloader = get_loader(**task_config, batch_size=batchsize)
    else:
        task_config = config["tasks"][task_name]
        dataloader = get_loader(**task_config, batch_size=args.batch_size)

    # assert dataloader[2], "Make sure that dataset has classnames available for zero-shot labels embeddings."
    
    amazon_eval_data = Namespace(
        **{
            "dataloader": dataloader[0],
            "class_names": [c.lower().replace("_", " ") for c in dataloader[2]],
            "templates": dataloader[3]
        }
    )
    return amazon_eval_data
