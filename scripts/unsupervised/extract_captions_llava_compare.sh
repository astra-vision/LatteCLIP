MACHINE_ID=$1
NUM_MACHINES=$2
echo "MACHINE_ID: ${MACHINE_ID}"
echo "NUM_MACHINES: ${NUM_MACHINES}"
NUM_GPUS=$6
NUM_PROCESSES_PER_GPU=$5
END_IDX_PROCESS_ID=$((NUM_PROCESSES_PER_GPU-1))
END_IDX_GPU=$((NUM_GPUS-1))
WORLD_SIZE_PER_MACHINE=$((NUM_PROCESSES_PER_GPU*NUM_GPUS))
WORLD_SIZE=$((WORLD_SIZE_PER_MACHINE*NUM_MACHINES))


PROMPT_TYPE=$3
MAX_NEW_TOKENS=77
echo "PROMPT_TYPE: ${PROMPT_TYPE}"
DATASET_NAME=$4
N_IMAGES=4

DATA_DIR=$LATTECLIP_DATA_DIR/${DATASET_NAME}_preprocess/webdataset
OUTPUT_DIR=$LATTECLIP_DATA_DIR/${DATASET_NAME}_preprocess/generated_captions
CLIP_PREDICTION_PATH=$LATTECLIP_DATA_DIR/${DATASET_NAME}_preprocess/clip_features_train.pkl


for gpu_id in $( seq 0 ${END_IDX_GPU})
do
  for process_id in $( seq 0 ${END_IDX_PROCESS_ID})
  do
    current_process_id=$((gpu_id*NUM_PROCESSES_PER_GPU+process_id+WORLD_SIZE_PER_MACHINE*MACHINE_ID))
    echo ${current_process_id}
    CUDA_VISIBLE_DEVICES=${gpu_id} python preprocess/extract_captions_llava_1_6_compare.py -p ${current_process_id} \
      -w ${WORLD_SIZE} -od ${OUTPUT_DIR} --dataset-name ${DATASET_NAME} \
      --split train \
      --clip-prediction-path ${CLIP_PREDICTION_PATH}  --n-images ${N_IMAGES} \
      -dd ${DATA_DIR} -pt ${PROMPT_TYPE} --max-new-tokens ${MAX_NEW_TOKENS} &
  done
done

wait