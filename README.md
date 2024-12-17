<div align='center'>

# LatteCLIP: Unsupervised CLIP Fine-Tuning via LMM-Synthetic Texts

WACV 2025

[Anh-Quan Cao](https://anhquancao.github.io)<sup>1</sup>&nbsp;&nbsp;&nbsp;
[Maximilian Jaritz](https://scholar.google.co.uk/citations?user=yt2IsdAAAAAJ&hl=de)<sup>2</sup>&nbsp;&nbsp;&nbsp;
[Matthieu Guillaumin](https://scholar.google.com/citations?user=jFdZ8s4AAAAJ&hl=en)<sup>2</sup>&nbsp;&nbsp;&nbsp;
[Raoul de Charette](https://team.inria.fr/rits/membres/raoul-de-charette/)<sup>1</sup>&nbsp;&nbsp;&nbsp;
[Loris Bazzani](https://lorisbaz.github.io/)<sup>2</sup>&nbsp;&nbsp;&nbsp;

<div>
<sup>1</sup> Inria
<sup>2</sup> Amazon
</div>

<br/>

[![arXiv](https://img.shields.io/badge/arXiv-2410.08211-darkred)](https://arxiv.org/abs/2410.08211) 

</div>

If you find this work or code useful, please cite our [paper](https://arxiv.org/abs/2410.08211) and [give this repo a star](https://github.com/astra-vision/LatteCLIP/stargazers):
```
@InProceedings{cao2024latteclip,
      title={LatteCLIP: Unsupervised CLIP Fine-Tuning via LMM-Synthetic Texts}, 
      author={Anh-Quan Cao and Maximilian Jaritz and Matthieu Guillaumin and Raoul de Charette and Loris Bazzani},
      year={2024},
      booktitle = {arXiv}
}
```


# News
- 17/12/2024: code is released.
- 14/10/2024: code will be available soon.

# Table of Contents

1. [Installation](#installation)
   - [Install OpenCLIP's Dependencies](#1-install-openclips-dependencies)
   - [Install LLaVA](#2-install-llava)
2. [Data Preparation](#data-preparation)
3. [Generate Descriptions](#generate-descriptions)
   - [Generate Image Descriptions](#1-generate-image-descriptions)
   - [Generate Group Descriptions](#2-generate-group-descriptions)
4. [Training](#training)
5. [Acknowledgement](#acknowledgement)


# Installation

Follow these steps to install the necessary dependencies:

## 1. Install OpenCLIP's Dependencies
Create a new conda environment and install the dependencies:
```bash
conda create -n latteclip python=3.10
conda activate latteclip
```

Navigate to the `latteclip` directory and run the following command:
```bash
make install
make install-training
```

## 2. Install LLaVA
Follow the official instructions [here](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#install).
```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
pip install -e .
```

---

# Data Preparation

## 1. Create the Data Directory
Create a folder to store the data and set the path in the bash variable `$LATTECLIP_DATA_DIR`:
```bash
mkdir -p /path/to/data
export LATTECLIP_DATA_DIR=/path/to/data
```

## 2. Download the Data
Download the data from [this link](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) and extract all files into the `$LATTECLIP_DATA_DIR`.

## 3. Run the Preprocess Script
Navigate to the `latteclip` directory and run the preprocess script to create the webdataset, tarfiles, and extract the clip features:
```bash
cd latteclip
bash scripts/preprocess/preprocess.sh
```

---

# Generate Descriptions

## 1. Generate Image Descriptions

To generate image descriptions, follow these steps:

### Example with `dtd` Dataset
Run the following command:
```bash
bash scripts/unsupervised/extract_captions_llava_multiprocess.sh $MACHINE_ID $NUM_MACHINE classname_dtd dtd $NUM_PROCESSES_PER_GPU $NUM_GPUS
```

### If You Have Multiple Machines
Assume you have **2 machines**, **1 GPU per machine**, and **5 generation processes** per Tesla V100 32g GPU:

**Machine 0**:
```bash
bash scripts/unsupervised/extract_captions_llava_multiprocess.sh 0 2 classname_dtd dtd 5 1
```

**Machine 1**:
```bash
bash scripts/unsupervised/extract_captions_llava_multiprocess.sh 1 2 classname_dtd dtd 5 1
```

### Generate Image Descriptions for Other Datasets
Use the following commands:

```bash
bash scripts/unsupervised/extract_captions_llava_multiprocess.sh 0 1 classname_dtd dtd 5 1
bash scripts/unsupervised/extract_captions_llava_multiprocess.sh 0 1 classname_eurosat eurosat 5 1
bash scripts/unsupervised/extract_captions_llava_multiprocess.sh 0 1 classname_scene sun397 5 1
bash scripts/unsupervised/extract_captions_llava_multiprocess.sh 0 1 classname_flower flower102 5 1
bash scripts/unsupervised/extract_captions_llava_multiprocess.sh 0 1 classname_food101 food101 5 1
bash scripts/unsupervised/extract_captions_llava_multiprocess.sh 0 1 classname_pets oxford_pets 5 1
bash scripts/unsupervised/extract_captions_llava_multiprocess.sh 0 1 classname_car stanford_cars 5 1
bash scripts/unsupervised/extract_captions_llava_multiprocess.sh 0 1 classname_ufc ucf101 5 1
bash scripts/unsupervised/extract_captions_llava_multiprocess.sh 0 1 classname_caltech caltech101 5 1
```

## 2. Generate Group Descriptions

The process is similar to generating image descriptions. Use the following commands:

```bash
bash scripts/unsupervised/extract_captions_llava_compare.sh 0 1 dtd_describe_common_v3 dtd 5 1
bash scripts/unsupervised/extract_captions_llava_compare.sh 0 1 eurosat_describe_common_v3 eurosat 5 1
bash scripts/unsupervised/extract_captions_llava_compare.sh 0 1 sun397_describe_common_v3 sun397 5 1
bash scripts/unsupervised/extract_captions_llava_compare.sh 0 1 flower102_describe_common_v3 flower102 5 1
bash scripts/unsupervised/extract_captions_llava_compare.sh 0 1 food101_describe_common_v3 food101 5 1
bash scripts/unsupervised/extract_captions_llava_compare.sh 0 1 pets_describe_common_v3 oxford_pets 5 1
bash scripts/unsupervised/extract_captions_llava_compare.sh 0 1 car_describe_common_v3 stanford_cars 5 1
bash scripts/unsupervised/extract_captions_llava_compare.sh 0 1 ufc_describe_common_v3 ucf101 5 1
bash scripts/unsupervised/extract_captions_llava_compare.sh 0 1 caltech_describe_common_v3 caltech101 5 1
```

---

# Training

To train the model on `dtd`, run:
```bash
bash scripts/unsupervised/dtd/dtd_fine_tune_multiclass.sh $lr $class_per_image $device $port $seed $exp_name
```
- `$lr`: Learning rate  
- `$class_per_image`: Number of classes per image (always set to 1)  
- `$device`: Device ID  
- `$port`: Port for the job  (Not used)
- `$seed`: Random seed  
- `$exp_name`: Experiment name  

### Example
To train with **learning rate 1e-7**, on **device 0**, with **port 25680**, random seed **3**, and experiment name `exp_dtd`:
```bash
bash scripts/unsupervised/dtd_fine_tune_multiclass.sh 1e-7 1 0 25680 1 exp_dtd
```

### Train on Other Datasets
```bash
bash scripts/unsupervised/eurosat_fine_tune_multiclass.sh 1e-7 1 0 25666 1 exp_eurosat
bash scripts/unsupervised/caltech101_fine_tune_multiclass.sh 1e-7 1 0 25665 1 exp_caltech101
bash scripts/unsupervised/fgvc_aircraft/fgvc_aircraft_fine_tune_multiclass.sh 1e-7 1 0 25667 1 exp_fgvc_aircraft
bash scripts/unsupervised/flower102_fine_tune_multiclass.sh 1e-7 1 0 25668 1 exp_flower102
bash scripts/unsupervised/food101_fine_tune_multiclass.sh 1e-7 1 0 25669 1 exp_food101
bash scripts/unsupervised/oxford_pets_fine_tune_multiclass.sh 1e-7 1 0 25670 1 exp_oxford_pets
bash scripts/unsupervised/stanford_cars/stanford_cars_fine_tune_multiclass.sh 1e-7 1 0 25671 1 exp_stanford_cars
bash scripts/unsupervised/sun397_fine_tune_multiclass.sh 1e-7 1 0 25672 1 exp_sun397
bash scripts/unsupervised/ucf101_fine_tune_multiclass.sh 1e-7 1 0 25673 1 exp_ucf101
```

> **Note**: Logs will be stored in the `logs` folder.

---

# Acknowledgement

This repository is built upon [OpenCLIP](https://github.com/mlfoundations/open_clip?tab=readme-ov-file) and [LLaVA](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file).

