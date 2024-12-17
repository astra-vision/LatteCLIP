python preprocess/unsupervised/flower102_to_webdataset.py --dataset_dir $LATTECLIP_DATA_DIR/oxford_flowers --preprocess_dir $LATTECLIP_DATA_DIR/flower102_preprocess


python preprocess/create_tarfiles.py --input_directory $LATTECLIP_DATA_DIR/flower102_preprocess/webdataset/train \
                                    --output_directory $LATTECLIP_DATA_DIR/flower102_preprocess/webdataset/train_tar \
                                    --items_per_tar 100

python preprocess/create_tarfiles.py --input_directory $LATTECLIP_DATA_DIR/flower102_preprocess/webdataset/val \
                                    --output_directory $LATTECLIP_DATA_DIR/flower102_preprocess/webdataset/val_tar \
                                    --items_per_tar 100


python -m training.main \
        --zeroshot-eval-data flower102 \
        --extract-features-path $LATTECLIP_DATA_DIR/flower102_preprocess \
        --extract-features-split train \
        --model ViT-B-32 \
        --pretrained laion2b_s34b_b79k  \
        --batch-size 512 



python -m training.main \
        --zeroshot-eval-data flower102 \
        --extract-features-path $LATTECLIP_DATA_DIR/flower102_preprocess \
        --extract-features-split val \
        --model ViT-B-32 \
        --pretrained laion2b_s34b_b79k  \
        --batch-size 512 
