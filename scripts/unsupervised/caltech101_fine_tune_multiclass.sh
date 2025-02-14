echo "lr: $1"
echo "class per image: $2"
echo "device: $3"
echo "port: $4"
echo "seed: $5"
export PREFIX=$6
echo "prefix: $PREFIX"

export CUDA_VISIBLE_DEVICES=$3

python  -m training.main \
--train-data "${LATTECLIP_DATA_DIR}/caltech101_preprocess/webdataset/train_tar/{00000..00057}.tar" \
--clip-prediction-path "${LATTECLIP_DATA_DIR}/caltech101_preprocess/clip_features_train.pkl" \
--generated-captions-path "${LATTECLIP_DATA_DIR}/caltech101_preprocess/generated_captions/train_classname_caltech_77_llava-v1.6-mistral-7b_4bit" \
--generated-common-captions-path "${LATTECLIP_DATA_DIR}/caltech101_preprocess/generated_captions/train_caltech_describe_common_v3_77_llava-v1.6-mistral-7b_4bit" \
--zeroshot-eval-data caltech101 \
--train-num-samples 5777 \
--dataset-type webdataset \
--epochs 50 \
--batch-size 512 \
--precision amp \
--local-loss \
--gather-with-grad \
--grad-checkpointing \
--ddp-static-graph \
--workers 3 \
--lr $1 \
--logs /lustre/fsn1/projects/rech/kvd/uyl37fq/logs/amz/caltech101 \
--report-to tensorboard \
--resume "latest" \
--zeroshot-frequency 1 \
--model ViT-B-32 \
--warmup 10 \
--class-per-image ${2} \
--alpha 0.01 \
--beta 0.01 \
--gamma 0.0 \
--text-type concat \
--seed $5 \
--name "${PREFIX}_seed${5}_${2}_lr${1}" \
--pretrained laion2b_s34b_b79k \
--distill-model ViT-B-32 \
--distill-pretrained laion2b_s34b_b79k \
--wd 1e-7 
