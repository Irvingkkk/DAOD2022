#!/bin/bash
set -e

name=${2}
echo $name
net=$(echo $name | sed 's/models\/\([^\/:]*\).*/\1/g')
echo $net
dataset=$(echo $name | sed 's/.*target\_\(.*\)\_session.*/\1/g')
echo $dataset
for i in $(seq 1 3)
do    
    model_name=${name%epoch*}epoch_${i}_step${name#*step}
    echo $model_name
    CUDA_VISIBLE_DEVICES=${1} python aut_test.py --dataset $dataset --net $net --load_name $model_name 2>&1 | tee -a test_log/aut/$(basename $model_name).log
done


#  CUDA_VISIBLE_DEVICES=1 python aut_test.py --dataset foggy_cityscape --net vgg16 --load_name /space0/zhaofz/2022/faster-rcnn.pytorch/models/vgg16/cityscape/warmup-s+tlikeGPA_SWAug_True_mutual-+tlikeGPA_SWAug_True_bs_1_unsup_loss_w_1_adv_loss_w_0.1_threshold_0.8_aut_lr_0.001_warmup_epoch_1_mutual_epoch_5_ema_weight_0.9996_nolrdc_5/teacher_target_foggy_cityscape_session_1_epoch_5_step_10000_mAP31.8.pth