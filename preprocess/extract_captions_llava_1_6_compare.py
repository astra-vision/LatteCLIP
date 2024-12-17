"""
This script is inspired from https://github.com/haotian-liu/LLaVA/blob/main/llava/serve/cli.py
"""

import pickle
import argparse
import time
from pathlib import Path
from typing import List
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from functools import partial
import torch
import webdataset as wds
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from tqdm import tqdm
import os
import collections
from PIL import Image
import numpy as np
import math
from open_clip import create_model_and_transforms
import torch.nn.functional as F


LATTECLIP_DATA_DIR = os.getenv('LATTECLIP_DATA_DIR')
if LATTECLIP_DATA_DIR is None:
    raise EnvironmentError("Environment variable 'LATTECLIP_DATA_DIR' is not set.")


def concatenate_images_vertical(images, dist_images=20):
    """
    This function takes a list of PIL images and concatenates them into a new image vertically.
    The new image is resized to fit in the output_size.
    param images: list of PIL images
    param dist_images: distance between images
    """
    # calc max width from imgs
    width = max(img.width for img in images)
    # calc total height of imgs + dist between them
    total_height = sum(img.height for img in images) + dist_images * (len(images) - 1)

    # create new img with calculated dimensions, black bg
    new_img = Image.new('RGB', (width, total_height), (0, 0, 0))

    # init var to track current height pos
    current_height = 0
    for img in images:
        # paste img in new_img at current height
        new_img.paste(img, (0, current_height))
        # update current height for next img
        current_height += img.height + dist_images

    return new_img


def concatenate_images_horizontal(images, dist_images=20):
    """
    This function takes a list of PIL images and concatenates them into a new image horizontally.
    param images: list of PIL images
    param dist_images: distance between images
    """
    # calc total width of imgs + dist between them
    total_width = sum(img.width for img in images) + dist_images * (len(images) - 1)
    # calc max height from imgs
    height = max(img.height for img in images)

    # create new img with calculated dimensions, black bg
    new_img = Image.new('RGB', (total_width, height), (0, 0, 0))

    # init var to track current width pos
    current_width = 0
    for img in images:
        # paste img in new_img at current width
        new_img.paste(img, (current_width, 0))
        # update current width for next img
        current_width += img.width + dist_images

    return new_img


def concatenate_images_grid(images, dist_images, output_size):
    """
    This function takes a list of PIL images and concatenates them into a new image.
    The new image is resized to fit in the output_size.
    param images: list of PIL images
    param dist_images: distance between images
    param output_size: (width, height) of output image
    """
    num_images = len(images)
    # calc grid size based on amount of input imgs
    grid_size = max(2, math.ceil(math.sqrt(num_images)))

    cell_width = (output_size[0] - dist_images * (grid_size - 1)) // grid_size
    cell_height = (output_size[1] - dist_images * (grid_size - 1)) // grid_size

    # create new img with output_size, black bg
    new_img = Image.new('RGB', output_size, (0, 0, 0))

    for index, img in enumerate(images):
        # calc img aspect ratio
        img_ratio = img.width / img.height
        # calc target aspect ratio per cell
        target_ratio = cell_width / cell_height

        # resize img to fit in cell
        if img_ratio > target_ratio:
            new_width = cell_width
            new_height = int(cell_width / img_ratio)
        else:
            new_width = int(cell_height * img_ratio)
            new_height = cell_height

        # resize img using lanczos filter
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)

        row = index // grid_size
        col = index % grid_size

        # calc x, y offsets for img positioning
        x_offset = col * (cell_width + dist_images) + (cell_width - new_width) // 2
        y_offset = row * (cell_height + dist_images) + (cell_height - new_height) // 2

        # paste resized img in calc pos
        new_img.paste(resized_img, (x_offset, y_offset))

    return new_img


def load_other_images(sample, args, key_to_clip_prediction, images_per_class):
    """
    This function loads 4 other images from the same class and concatenates them into a new image.
    It also adds the new image to the sample with key other_jpg.
    The new image is saved to the /compare_img_diff directory.
    param sample: the current sample
    param args: the command line arguments
    param key_to_clip_prediction: a dictionary mapping image ids to clip predictions
    param images_per_class: a dictionary mapping class names to lists of image ids
    """
    n_images = args.n_images
    
    image_id = sample["__key__"]
    class_name = key_to_clip_prediction[image_id]['class_names'][0]
   
    replace = False
    if len(images_per_class[class_name]) < n_images:
        replace=True
    other_image_ids = np.random.choice(images_per_class[class_name], n_images, replace=replace)

    
    if "flower102" in args.prompt_type:
        preprocess_path = f"{LATTECLIP_DATA_DIR}/flower102_preprocess"
    elif "food101" in args.prompt_type:
        preprocess_path = f"{LATTECLIP_DATA_DIR}/food101_preprocess"
    elif "eurosat" in args.prompt_type:
        preprocess_path = f"{LATTECLIP_DATA_DIR}/eurosat_preprocess"
    elif "sun397" in args.prompt_type:
        preprocess_path = f"{LATTECLIP_DATA_DIR}/sun397_preprocess"
    elif "dtd" in args.prompt_type:
        preprocess_path = f"{LATTECLIP_DATA_DIR}/dtd_preprocess"
    else:
        preprocess_path = f"{LATTECLIP_DATA_DIR}/{args.dataset_name}_preprocess"
        
    image_paths = [f"{preprocess_path}/webdataset/train/{other_image_id}.jpg" for other_image_id in other_image_ids]
    other_imgs = [Image.open(image_path) for image_path in image_paths]

    
    new_img = concatenate_images_grid(other_imgs, dist_images=20, output_size=(672, 672))
    save_dir = f"{preprocess_path}/group_{args.n_images}/"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{image_id}.jpg"
    new_img.save(save_path)
    sample['other_jpg'] = new_img
    return sample
    

def ask_llava(image, prompt, model, conv_mode, tokenizer):
    prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], prompt)
    prompt = conv.get_prompt()
    
    input_ids = (
        tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        .unsqueeze(0)
        .to(model.device)
    )

    t0 = time.time()
    output_ids = model.generate(
        input_ids,
        image_sizes=[image.shape[-2:]],
        images=image,
        do_sample=True if args.temperature > 0 else False,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
    )
    forward_time = time.time() - t0
    print(f"{forward_time} sec")

    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return caption


def extract_captions_of_one_tarfile(
    text_prompt, tar_path, model, image_processor, tokenizer, output_dir, key_to_clip_prediction, conv_mode
):
    """
    Extract captions of one tarfile using the following text_prompts
    TEXT_PROMPTs = {
        "captions": "USER: <image>\nDescribe the image concisely, less than 20 words. ASSISTANT:",
        "MaxCaptions": "USER: <image>\nDescribe the image concisely, less than 20 words, pointing out what makes it different to other images. ASSISTANT:",
        "DomainVisualCaptions": "USER: <image>\n Concisely describe, in under 20 words, the visual features that distinguish the product in the image, highlighting its unique visual characteristic. ASSISTANT:"
    }
    :param text_prompt: type of prompts (captions/MaxCaptions/DomainVisualCaptions)
    :param tar_path: path to the tar file
    :param model: LLAVA model
    :param image_processor: preprocess image function go with LLAVA
    :param tokenizer: LLAVA tokenizer
    :param output_dir: output directory that the captions will be written
    """
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    images_per_class = {}
    for k in key_to_clip_prediction:
        class_name = key_to_clip_prediction[k]['class_names'][0]
        if class_name in images_per_class:
            images_per_class[class_name].append(k)
        else:
            images_per_class[class_name] = [k]
            
    augment = partial(
                load_other_images, 
                key_to_clip_prediction=key_to_clip_prediction,
                images_per_class=images_per_class,
                args=args, 
            )
    
    # Preprocess text prompt only once as it is always the same
    print(f"Initialize dataloader for {tar_path}")
    dataset = (
        wds.WebDataset(str(tar_path), empty_check=False)
        .decode("pilrgb")
        .map(augment)
        .rename(other_image="other_jpg", image="jpg", text="txt", metadata="json")
        .map_dict(
            other_image=lambda image: process_images([image], image_processor, model.config),
            image=lambda image: process_images([image], image_processor, model.config),
            text=lambda text: text,  # do nothing
            metadata=lambda metadata: metadata,
        )
        .to_tuple("image", "other_image", "text", "metadata")
    )
    for item in dataset:
        break
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=8, batch_size=1)
    
    with torch.inference_mode():
        for idx, (images, other_images, text, metadata) in enumerate(dataloader):
            image_id = metadata["image_id"][0]
            output_path = os.path.join(output_dir, f"{image_id}.txt")
            if os.path.exists(output_path):
                print(f"File {output_path} already exists. Skipping...")
                continue
            
            class_names = key_to_clip_prediction[image_id]['class_names'][:1] # This is just legacy of previous exp, I always use top-1 prediction
            out = []
            
            other_image = other_images[0][0].to(model.device, dtype=torch.float16)
            for i, class_name in enumerate(class_names):
                
                prompt2 = text_prompt[0].format(class_name) # Legacy of experiment with multi-turn conversation
                caption2  = ask_llava(other_image, prompt2, model, conv_mode, tokenizer)
                out.append(caption2.replace("\n", " "))

            line = "\n".join(out)
            # Save captions
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"{line}")
            print(caption2)
            print(f"Saved to {output_path}")


def run_single_worker(tar_paths: List[Path], args, output_dir, text_prompt, key_to_clip_prediction):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        args.load_8bit,
        args.load_4bit,
        device=args.device,
    )
    
    
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
        
    
    

    # Extract image embeddings
    for tar_path in tqdm(tar_paths, desc="Extract image captions"):
        extract_captions_of_one_tarfile(
            text_prompt, tar_path, model, image_processor, tokenizer, output_dir, key_to_clip_prediction, conv_mode
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    # parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-13b")
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.6-mistral-7b")
    # parser.add_argument("--model-path", type=str, default="liuhaotian/liuhaotian/llava-v1.6-34b")
    parser.add_argument("--clip-prediction-path", type=str)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument(
        "--prompt-type",
        "-pt",
        type=str,
        default="veclip",
        help="switch between different type of prompts",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=77)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true", default=True)
    parser.add_argument("--debug", action="store_true")
    
    # Added by Max
    parser.add_argument("--split", default="train")
    parser.add_argument("--data-dir", "-dd", default="$LATTECLIP_DATA_DIR/abo_preprocess")
    parser.add_argument("--output-dir", "-od", default="$LATTECLIP_DATA_DIR/abo_preprocess")
    parser.add_argument("--process-id", "-p", type=int)
    parser.add_argument("--world-size", "-w", type=int)
    parser.add_argument("--n-images", type=int, default=4)
    args = parser.parse_args()

    TEXT_PROMPTs = {
        "flower102_describe_common_v3": [
            "Describe the common visual attributes of the flowers in all the photos concisely, less than 20 words."
        ],
        "eurosat_describe_common_v3": [
            "Describe the common visual attributes of the land use in all the satellite images concisely, less than 20 words."
        ],
        "food101_describe_common_v3": ["Describe the common visual attributes of the foods in all the photos concisely, less than 20 words."],
        "sun397_describe_common_v3": ["Describe the common visual attributes of the scenes in all the photos concisely, less than 20 words."],
        "dtd_describe_common_v3": ["Describe the common visual attributes of the textures in all the photos concisely, less than 20 words."],

        "aircraft_describe_common_v3": ["Describe the common visual attributes of the aircrafts in all the photos concisely, less than 20 words."],
        "car_describe_common_v3": ["Describe the common visual attributes of the cars in all the photos concisely, less than 20 words."],
        "pets_describe_common_v3": ["Describe the common visual attributes of the pets in all the photos concisely, less than 20 words."],
        "ufc_describe_common_v3": ["Describe the common visual attributes of the person's actions in all the photos concisely, less than 20 words."],
        "caltech_describe_common_v3": ["Describe the common visual attributes of the objects in all the photos concisely, less than 20 words."],
    }

    
    data_dir = os.path.join(args.data_dir, f"{args.split}_tar")
    tar_paths = sorted(Path(data_dir).glob("*.tar"))
    model_name = get_model_name_from_path(args.model_path)
    
    with open(args.clip_prediction_path, "rb") as f:
        key_to_clip_prediction = pickle.load(f)


    print(f"Still need to extract {len(tar_paths)}/{len(tar_paths)} tar files")

    split = tar_paths[args.process_id :: args.world_size]
    print(f"Process {args.process_id}, work on {split[:5]}")
    
    if args.load_8bit:
        bit_text = "8bit"
    elif args.load_4bit:
        bit_text = "4bit"
    else:
        raise ValueError("Must specify --load-8bit or --load-4bit")
    
    for prompt_type in [args.prompt_type]:
        output_dir = (
            Path(args.output_dir) / f"{args.split}_{args.n_images}images_{prompt_type}_{args.max_new_tokens}_{model_name}_{bit_text}"
        )
        text_prompt = TEXT_PROMPTs[prompt_type]
        print(f"Text prompt: {text_prompt}")
        run_single_worker(split, args, output_dir, text_prompt, key_to_clip_prediction)
