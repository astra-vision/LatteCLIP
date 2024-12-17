"""
This script is inspired from https://github.com/haotian-liu/LLaVA/blob/main/llava/serve/cli.py
"""

import pickle
import argparse
import time
from pathlib import Path
from typing import List
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

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
from transformers import TextStreamer



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

    
    # Preprocess text prompt only once as it is always the same
    print(f"Initialize dataloader for {tar_path}")
    dataset = (
        wds.WebDataset(str(tar_path), empty_check=False)
        .decode("pilrgb")
        .rename(image="jpg", text="txt", metadata="json")
        .map_dict(
            image=lambda image: process_images([image], image_processor, model.config),
            text=lambda text: text,  # do nothing
            metadata=lambda metadata: metadata,
        )
        .to_tuple("image", "text", "metadata")
    )
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=3, batch_size=1)
    
    # Extract image embeddings
    all_generated_captions = []
    forward_times = []

    with torch.inference_mode():
        for idx, (images, text, metadata) in enumerate(dataloader):
            image_id = metadata["image_id"][0]
            output_path = os.path.join(output_dir, f"{image_id}.txt")
            if os.path.exists(output_path):
                print(f"File {output_path} already exists. Skipping...")
                continue
            
            class_names = key_to_clip_prediction[image_id]['class_names'][:1]
            out = []
            image = images[0][0].to(model.device, dtype=torch.float16)
            for i, class_name in enumerate(class_names):
                
                class_prompt = text_prompt.format(class_name, class_name)
                class_prompt = DEFAULT_IMAGE_TOKEN + '\n' + class_prompt
                conv = conv_templates[conv_mode].copy()
                conv.append_message(conv.roles[0], class_prompt)
                prompt = conv.get_prompt()
                
                input_ids = (
                    tokenizer_image_token(
                        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                    )
                    .unsqueeze(0)
                    .to(model.device)
                )
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]

                t0 = time.time()
                output_ids = model.generate(
                    input_ids,
                    image_sizes=[image.shape[-2:]],
                    images=image,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    # streamer=streamer,
                    use_cache=True,
                )
                forward_time = time.time() - t0
                forward_times.append(forward_time)
                print(f"Avg: {sum(forward_times) / len(forward_times):.2f} sec")

                caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
                out.append(caption.replace("\n", " "))

            line = "\n".join(out)
            # Save captions
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"{line}")
            print(line)
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
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.6-mistral-7b")
    parser.add_argument("--clip-prediction-path", type=str)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument(
        "--prompt-type",
        "-pt",
        type=str,
        default="veclip",
        help="switch between different type of prompts",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=77)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true", default=True)
    parser.add_argument("--debug", action="store_true")
    
    # Added by Max
    parser.add_argument("--split", default="train")
    parser.add_argument("--data-dir", "-dd", default="/home/ubuntu/data/abo_preprocess")
    parser.add_argument("--output-dir", "-od", default="/home/ubuntu/data/abo_preprocess")
    parser.add_argument("--process-id", "-p", type=int)
    parser.add_argument("--world-size", "-w", type=int)
    args = parser.parse_args()

    TEXT_PROMPTs = {
        "classname": 'Describe the {} flower in the photo concisely, less than 20 words.',
        "classname_food101": 'Describe the {} food in the photo concisely, less than 20 words.',
        "classname_eurosat": 'Describe the land use in the satellite image concisely, less than 20 words.',
        "classname_scene": 'Describe the scene in the photo concisely, less than 20 words.',
        "classname_dtd": 'Describe the texture in the photo concisely, less than 20 words.',
        "classname_aircraft": 'Describe the aircraft in the photo concisely, less than 20 words.',
        "classname_pets": 'Describe the pet in the photo concisely, less than 20 words.',
        "classname_car": 'Describe the car in the photo concisely, less than 20 words.',
        "classname_ufc": 'Describe the action of the person in the photo concisely, less than 20 words.',
        "classname_caltech": 'Describe the object in the photo concisely, less than 20 words.',

    }

    
    

    data_dir = os.path.join(args.data_dir, f"{args.split}_tar")
    tar_paths = sorted(Path(data_dir).glob("*.tar"))
    # Check if output already exists
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
            Path(args.output_dir) / f"{args.split}_{prompt_type}_{args.max_new_tokens}_{model_name}_{bit_text}"
        )
        text_prompt = TEXT_PROMPTs[prompt_type]
        print(f"Text prompt: {text_prompt}")
        run_single_worker(split, args, output_dir, text_prompt, key_to_clip_prediction)
