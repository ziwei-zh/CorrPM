#!/bin/bash
uname -a
date

CS_PATH='/data/zzw/segment/data/lip/images_labels'
POSE_ANNO='/data/zzw/segment/data/lip/TrainVal_pose_annotations/LIP_SP_TRAIN_annotations.json'
LR=1e-3
WD=5e-4
BS=4
GPU_IDS=4,5
RESTORE_FROM='/data/zzw/segment/data/pre_trained_models/resnet101-imagenet.pth'
INPUT_SIZE='384,384'
SNAPSHOT_DIR='/data/zzw/segment/snapshots/test_corrpm'
DATA_LIST='_id.txt'
DATASET='train'
NUM_CLASSES=20
NPOINTS=16
EPOCHS=180
START_EPOCH=0

if [[ ! -e ${SNAPSHOT_DIR} ]]; then
    mkdir -p  ${SNAPSHOT_DIR}
fi

python3 train.py --data-dir ${CS_PATH} \
       --pose-anno-file ${POSE_ANNO} \
       --random-mirror\
       --random-scale\
       --restore-from ${RESTORE_FROM}\
       --gpu ${GPU_IDS}\
       --learning-rate ${LR} \
       --weight-decay ${WD}\
       --batch-size ${BS} \
       --input-size ${INPUT_SIZE}\
       --snapshot-dir ${SNAPSHOT_DIR}\
       --dataset ${DATASET}\
       --dataset-list ${DATA_LIST} \
       --num-classes ${NUM_CLASSES} \
       --num-points ${NPOINTS} \
       --epochs ${EPOCHS} \
       --start-epoch ${START_EPOCH}
