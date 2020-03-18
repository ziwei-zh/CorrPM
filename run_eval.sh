#!/bin/bash

CS_PATH='/data/zzw/segment/data/lip/images_labels'
POSE_ANNO='/data/zzw/segment/data/lip/TrainVal_pose_annotations/LIP_SP_VAL_annotations.json'
BS=12
GPU_IDS='6,7'
INPUT_SIZE='384,384'
SNAPSHOT_FROM='./pth/LIP_best.pth'
DATASET='val'
DATANAME='lip'
NUM_CLASSES=20
NPOINTS=16
OUTPUT_DIR='./output_seg/'

python3 evaluate.py --data-dir ${CS_PATH} \
       --pose-anno-file ${POSE_ANNO} \
       --gpu ${GPU_IDS} \
       --batch-size ${BS} \
       --input-size ${INPUT_SIZE}\
       --restore-from ${SNAPSHOT_FROM}\
       --dataset ${DATASET} \
       --data-name ${DATANAME} \
       --num-classes ${NUM_CLASSES} \
       --num-points ${NPOINTS} \
       --save-dir ${OUTPUT_DIR}
