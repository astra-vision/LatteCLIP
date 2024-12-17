echo "lr: $1"
echo "class per image: $2"
echo "device: $3"
echo "port: $4"
echo "seed: $5"
export PREFIX=$6
echo "prefix: $PREFIX"

export CUDA_VISIBLE_DEVICES=$3

python  -m training.main \
--train-data "${LATTECLIP_DATA_DIR}/sun397_preprocess/webdataset/train_tar/{00000..00198}.tar" \
--clip-prediction-path "${LATTECLIP_DATA_DIR}/sun397_preprocess/clip_features_train.pkl" \
--generated-captions-path "${LATTECLIP_DATA_DIR}/sun397_preprocess/generated_captions/train_classname_scene_77_llava-v1.6-mistral-7b_4bit" \
--generated-common-captions-path "${LATTECLIP_DATA_DIR}/sun397_preprocess/generated_captions/train_sun397_describe_common_v3_77_llava-v1.6-mistral-7b_4bit" \
--zeroshot-eval-data sun397 \
--train-num-samples 19850 \
--dataset-type webdataset \
--epochs 15 \
--batch-size 512 \
--precision amp \
--local-loss \
--gather-with-grad \
--grad-checkpointing \
--ddp-static-graph \
--workers 3 \
--lr $1 \
--logs /lustre/fsn1/projects/rech/kvd/uyl37fq/logs/amz/sun397 \
--report-to tensorboard \
--resume "latest" \
--zeroshot-frequency 1 \
--model ViT-B-32 \
--warmup 40 \
--class-per-image ${2} \
--alpha 0.01 \
--beta 0.01 \
--gamma 0.0 \
--text-type concat \
--seed $5 \
--name "${PREFIX}_seed${5}_${2}_lr${1}" \
--pretrained laion2b_s34b_b79k \
--distill-model ViT-B-32 \
--distill-pretrained laion2b_s34b_b79k
