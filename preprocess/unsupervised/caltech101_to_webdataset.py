from tqdm import tqdm
import os
import pandas as pd
import json
import shutil
import click
import glob
from sklearn.model_selection import train_test_split


@click.command()
@click.option("--exp_name", default="")
@click.option("--dataset_dir", default="$LATTECLIP_DATA_DIR/caltech-101")
@click.option("--preprocess_dir", default="$LATTECLIP_DATA_DIR/caltech101_preprocess")
@click.option(
    "--text_dirs", '-td',
    multiple=True,
    default=[],
)
def main(exp_name, dataset_dir, text_dirs, preprocess_dir):
    """
    Convert caltech-101 dataset to WebDataset format using the image ID - product title mapping. 
    Args:
        dataset_dir (str): The directory path of the ABO dataset.
        preprocess_dir (str): The directory path to save the dataset splits and image2product mapping.
    Returns:
        None. Saves the filtered dataset and image2product mapping to the preprocess_dir.
        Also creates a directory for the filtered dataset and copies the filtered images to the directory.
        Also creates a text file for each image with the product title in the WebDataset format.
    """
    os.makedirs(preprocess_dir, exist_ok=True)
    image_dir = os.path.join(dataset_dir, "101_ObjectCategories")
    id_to_class = {}
    class_to_id = {}
    data_split_path = os.path.join(dataset_dir, "split_zhou_Caltech101.json")
    with open(data_split_path, "r") as f:
        data_split = json.load(f)
    data_train = data_split['train'] + data_split['val']
    data_test = data_split['test']
    data_all = data_train + data_test
    for image_path, class_id, class_name in data_all:
        id_to_class[class_id] = class_name
        class_to_id[class_name] = class_id
   
    
    save_path = os.path.join(preprocess_dir, "id_to_class.json")
    with open(save_path, "w") as f:
        json.dump(id_to_class, f)
    save_path = os.path.join(preprocess_dir, "class_to_id.json")
    with open(save_path, "w") as f:
        json.dump(class_to_id, f)
    
   
    
    
    for split, data in zip(["train", "val"], [data_train, data_test]):
        split_dir = os.path.join(preprocess_dir, "webdataset", f"{split}{exp_name}")
        cnt = 0
        os.makedirs(split_dir, exist_ok=True)
        for image_path, _, class_name in tqdm(data):
      
            
            image_name = os.path.basename(image_path).split(".")[0]
            image_id = "{}_{}".format(class_name, image_name)
            
            # Write txt file
            folder_name = image_path.split("/")[0]
            # print(folder_name, class_name)
            text = ""
            if split == "train":
                for i, text_dir in enumerate(text_dirs):
                    text_path = os.path.join(text_dir, f"{folder_name}_{image_name}.txt")
                    with open(text_path, "r") as f:
                        text += f.read().replace("\n", "")
                    if i != (len(text_dirs) - 1):
                        text += "\n"
                    
            text_path = os.path.join(split_dir, f"{image_id}.txt")
            with open(text_path, "w") as f2:
                f2.write(text)

            # Write json file
            json_path = os.path.join(split_dir, f"{image_id}.json")
        
            with open(json_path, "w") as f2:
                json.dump({
                    "image_id": image_id, 
                    "class_name": class_name
                }, f2)

            # Copy image
            src_image_path = os.path.join(dataset_dir, "101_ObjectCategories", image_path)
            dst_image_path = os.path.join(split_dir, f"{image_id}.jpg")
            shutil.copy(src_image_path, dst_image_path)
            
            cnt += 1
        print(f"Total {cnt} images in {split} split.")




if __name__ == "__main__":
    main()
