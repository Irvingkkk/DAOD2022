### check adaptive thresh 
CUDA_VISIBLE_DEVICES=3 python train_adaptive_thresh.py \
--dataset cityscape --dataset_t foggy_cityscape --net vgg16 \
--unsup_loss_w 1 --adv_loss_w 1 --aut_lr 0.001  --ema_weight 0.9996 \
--warmup_epoch 1 --mutual_epoch 7  --lr_decay_step 9 \
--epochs 8  --use_tfb --unsup True --bs 2 \
--warmup_use_GPA --warmup_use_targetlike  --mutual_use_GPA --mutual_use_targetlike --warmup_use_SWAug --mutual_use_SWAug \
--reliable_path '/space0/zhaofz/2022/DAOD/data/foggyCityscape/VOC2007/ImageSets/Main' \
--r True \
--student_load_name models/vgg16/cityscape/foggy-0.02only-ICRCCR-ST++warmup-s+tlike+sGPA_SWAug_True_mutual-+tlike+sGPA_SWAug_True_bs_2_unsup_loss_w_1_adv_loss_w_1.0_EF_False_threshold_0.8_aut_lr_0.001_warmup_epoch_1_mutual_epoch_7_ema_weight_0.9996_lrdc_epoch8/student_target_foggy_cityscape_session_1_epoch_1_step_10000.pth \
--teacher_load_name models/vgg16/cityscape/foggy-0.02only-ICRCCR-ST++warmup-s+tlike+sGPA_SWAug_True_mutual-+tlike+sGPA_SWAug_True_bs_2_unsup_loss_w_1_adv_loss_w_1.0_EF_False_threshold_0.8_aut_lr_0.001_warmup_epoch_1_mutual_epoch_7_ema_weight_0.9996_lrdc_epoch8/student_target_foggy_cityscape_session_1_epoch_1_step_10000.pth \

