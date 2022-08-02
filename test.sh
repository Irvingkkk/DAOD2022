


CUDA_VISIBLE_DEVICES=0 python test.py \
--dataset foggy_cityscape --net vgg16 \
--load_name models/vgg16/cityscape/rerun-mladv-tinycutout-st++warmup-s+tlike+sGPA_SWAug_True_mutual-+tlike+sGPA_SWAug_True_bs_2_unsup_loss_w_1_adv_loss_w_1.0_EF_False_threshold_0.8_aut_lr_0.001_warmup_epoch_1_mutual_epoch_7_ema_weight_0.9996_lrdc_epoch5/teacher_target_foggy_cityscape_session_1_epoch_5_step_7000.pth