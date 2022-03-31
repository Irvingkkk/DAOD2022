hints
--unsup_loss_w
--adv_loss_w
--th
--aut_lr
--warmup_epoch
--mutual_epoch
--ema_weight

# CUDA_VISIBLE_DEVICES=3 python aut_train.py --dataset cityscape --dataset_t foggy_cityscape --net vgg16 --unsup_loss_w 1 --adv_loss_w 0 --aut_lr 0.01 --warmup_epoch 1 --mutual_epoch 5 --epochs 1  --use_tfb --unsup True --bs 1 

CUDA_VISIBLE_DEVICES=2 python aut_train.py \
--dataset cityscape --dataset_t foggy_cityscape --net vgg16 \
--unsup_loss_w 1 --adv_loss_w 0.1 --aut_lr 0.001  \
--warmup_epoch 1 --mutual_epoch 5  \
--epochs 6  --use_tfb --unsup True --bs 1 \

### check GPA 
CUDA_VISIBLE_DEVICES=2 python aut_train_stplus.py \
--dataset cityscape --dataset_t foggy_cityscape --net vgg16 \
--unsup_loss_w 1 --adv_loss_w 0.1 --aut_lr 0.001  \
--warmup_epoch 1 --mutual_epoch 5  \
--epochs 6  --use_tfb --unsup True --bs 1 \
--warmup_use_GPA --warmup_use_targetlike  --mutual_use_GPA --mutual_use_targetlike --warmup_use_SWAug --mutual_use_SWAug \
--reliable_path '/space0/zhaofz/2022/faster-rcnn.pytorch/data/foggyCityscape/VOC2007/ImageSets/Main'

CUDA_VISIBLE_DEVICES=0 python aut_train_stplus.py \
--dataset cityscape --dataset_t foggy_cityscape --net vgg16 \
--unsup_loss_w 1 --adv_loss_w 0.1 --aut_lr 0.001 \
--warmup_epoch 1 --mutual_epoch 5 \
--epochs 6  --use_tfb --unsup True --bs 1 --r True \
--student_load_name models/vgg16/cityscape/SWAug_True-warmup-s+tlike+sGPA_mutual-+tlike+sGPA_bs_1_unsup_loss_w_1_adv_loss_w_0.1_threshold_0.8_aut_lr_0.001_warmup_epoch_1_mutual_epoch_5_ema_weight_0.9996_nolrdc_5/teacher_target_foggy_cityscape_session_1_epoch_1_step_10000_mAP32.8.pth \
--teacher_load_name models/vgg16/cityscape/SWAug_True-warmup-s+tlike+sGPA_mutual-+tlike+sGPA_bs_1_unsup_loss_w_1_adv_loss_w_0.1_threshold_0.8_aut_lr_0.001_warmup_epoch_1_mutual_epoch_5_ema_weight_0.9996_nolrdc_5/teacher_target_foggy_cityscape_session_1_epoch_1_step_10000_mAP32.8.pth \
--mutual_use_SWAug  --warmup_use_SWAug \
--warmup_use_GPA --warmup_use_targetlike --mutual_use_GPA --mutual_use_targetlike \

CUDA_VISIBLE_DEVICES=1 python aut_train.py \
--dataset cityscape --dataset_t foggy_cityscape --net vgg16 \
--unsup_loss_w 1 --adv_loss_w 0.1 --aut_lr 0.001 \
--warmup_epoch 1 --mutual_epoch 5 \
--epochs 6  --use_tfb --unsup True --bs 1 --r True \
--student_load_name models/vgg16/cityscape/batchaize_1_unsup_loss_w_1_adv_loss_w_0.1_threshold_0.8_aut_lr_0.001_warmup_epoch_1_mutual_epoch_5_ema_weight_0.9996/teacher_target_foggy_cityscape_session_1_epoch_1_step_10000_mAP20.22.pth \
--teacher_load_name models/vgg16/cityscape/batchaize_1_unsup_loss_w_1_adv_loss_w_0.1_threshold_0.8_aut_lr_0.001_warmup_epoch_1_mutual_epoch_5_ema_weight_0.9996/teacher_target_foggy_cityscape_session_1_epoch_1_step_10000_mAP20.22.pth \

CUDA_VISIBLE_DEVICES=0,1 python aut_train.py \
--dataset cityscape --dataset_t foggy_cityscape --net vgg16 \
--unsup_loss_w 1 --adv_loss_w 0 --aut_lr 0.04 \
--warmup_epoch 1 --mutual_epoch 5 \
--epochs 6  --use_tfb --unsup True --bs 2 --mGPUs\


 python aut_train.py \
--dataset cityscape --dataset_t foggy_cityscape --net vgg16 \
--unsup_loss_w 1 --adv_loss_w 0.1 --aut_lr 0.04 \
--warmup_epoch 1 --mutual_epoch 5  \
--epochs 6  --use_tfb --unsup True --bs 16 --mGPUs \