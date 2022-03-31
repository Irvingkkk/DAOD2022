# coding:utf-8
# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from asyncio import FastChildWatcher

import cv2
import os
import numpy as np
import pprint
import pdb
import time
from numpy.core.fromnumeric import shape
from tqdm import tqdm

# from torchvision.transforms import autoaugment
import _init_paths

import torch
from torch.autograd import Variable
import torch.nn as nn
import random


from model.utils.config import cfg, cfg_from_file, cfg_from_list
from model.utils.net_utils import (
    adjust_learning_rate,
    save_checkpoint,
    get_dataloader,
    get_dataloader_aut,
    setup_seed,
    clip_gradient,
    EFocalLoss
)
from model.ema.optim_weight_ema import WeightEMA
from model.utils.parser_func import parse_args, set_dataset_args
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv

from prettytimer import PrettyTimer

from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# loader使用torchvision中自带的transforms函数
loader = transforms.Compose([
    transforms.ToTensor()])  

unloader = transforms.ToPILImage()

import imageio
imsave = imageio.imsave

def get_cfg():
    args = parse_args()

    print("Called with args:")
    print(args)
    args = set_dataset_args(args)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print("Using config:")
    pprint.pprint(cfg)
    # np.random.seed(cfg.RNG_SEED)
    setup_seed(cfg.RNG_SEED)
    return args

def putBox(img,boxes):
    num_classes=9
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(num_classes)]
    for bbox in boxes:
        bbox = tuple(int(np.round(x)) for x in bbox[:5])
        category_id=int(bbox[4])
        
        # cv2.rectangle(img, (x,y), (x+w,y+h), (B,G,R), Thickness)
        if category_id==0:
            continue
        cv2.rectangle(img, bbox[0:2], bbox[2:4], colors[category_id], 2)
        text = str(category_id)
        Font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, text, (x,y), Font, Size, (B,G,R), Thickness)
        # cv2.putText(img, text, (bbox[0],bbox[1]-10), Font, 1.5, colors[category_id], 2)
        cv2.putText(img, text, (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, colors[category_id], thickness=1)
    return img

def checkdataloader(data_iter):
    for step in range(1,11):
        data_s=next(data_iter_s)

        img_weak=data_s[0].squeeze(0)
        img_strong=data_s[1].squeeze(0)
        # print('img_strong_get size:',img_strong.shape)
        # print('img_weak_get size:',img_weak.shape)
        name=data_s[-1][0].split('/')[-1]  
        print(name) 
        # print('img_weak.size',img_weak.size())

        img_weak=img_weak.permute(1,2,0).contiguous()
        img_strong=img_strong.permute(1,2,0).contiguous()
        img_weak=img_weak.numpy()  # in BGR order
        img_strong=img_strong.numpy()

        gt_boxes=data_s[3].squeeze(0).numpy()      
        # print(gt_boxes)
        img_weak=putBox(img=img_weak,boxes=gt_boxes)
        img_strong=putBox(img=img_strong,boxes=gt_boxes)
        imsave('aug_examples/visgpa_weak_{}_{}'.format(step,name),img_weak[:, :, ::-1])
        imsave('aug_examples/visgpa_strong_{}_{}'.format(step,name),img_strong[:, :, ::-1])
        # img_w = cv2.cvtColor(img_weak, cv2.COLOR_BGR2RGB)
        # cv2.normalize(img_weak, img_weak, 0, 255, cv2.NORM_MINMAX)
        # cv2.normalize(img_strong, img_strong, 0, 255, cv2.NORM_MINMAX)
        # cv2.imwrite('no_aug/cv2_data_withgt_weak_{}'.format(name),img_weak)
        # cv2.imwrite('no_aug/cv2_data_withgt_strong_{}'.format(name),img_strong)
    quit()



if __name__ == "__main__":
    args = get_cfg()

    output_dir = f"{args.save_dir}/{args.net}/{args.dataset}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.dataset_t == "water":
        args.aug = False

    if args.dataset_t == "foggy_cityscape":
        # initilize the network here.
        from model.aut_faster_rcnn.aut_vgg16_adv import vgg16
        # from model.umt_faster_rcnn_truncate.umt_resnet import resnet
    else:
        from model.umt_faster_rcnn.umt_vgg16 import vgg16
        # from model.umt_faster_rcnn.umt_resnet import resnet

    exp_setup="warmup-s{}{}_SWAug_{}_mutual-{}{}_SWAug_{}_bs_{}_unsup_loss_w_{}_adv_loss_w_{}_threshold_{}_aut_lr_{}_warmup_epoch_{}_mutual_epoch_{}_ema_weight_{}_nolrdc_{}".format(
            "+tlike" if args.warmup_use_targetlike == True else '',
            "+tlikeGPA" if args.warmup_use_GPA == True else '',
            args.warmup_use_SWAug ,
            "+tlike" if args.mutual_use_targetlike == True else '',
            "+tlikeGPA" if args.mutual_use_GPA == True else '',
            args.mutual_use_SWAug ,
            args.batch_size,
            args.unsup_loss_weight,
            args.adv_loss_weight,
            args.threshold, 
            args.aut_lr,
            args.warmup_epoch,
            args.mutual_epoch,
            args.ema_weight,
            args.lr_decay_step,
        )

    model_save_path = os.path.join(
        output_dir,
        exp_setup
        # 'lr_0.01_warmup_10kstep'
        
    )
    print("Model will be saved to: ")
    print(model_save_path)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    print("cfg.TRAIN.USE_FLIPPED",cfg.TRAIN.USE_FLIPPED)
    cfg.TRAIN.USE_FLIPPED = False   # do not apply this flip
    cfg.USE_GPU_NMS = args.cuda


    #######################################################
    ############## dataloader for warmup ##################
    #######################################################
    s_imdb_warm, s_train_size_warm, s_dataloader_warm = get_dataloader_aut(args.imdb_name, args, sequential=False, augment=args.warmup_use_SWAug)
    
    if args.warmup_use_targetlike == True:
        s_imdb_target_like, s_train_size_target_like , s_dataloader_target_like = get_dataloader_aut(args.imdb_name_fake_target,args,sequential=False,augment=args.warmup_use_SWAug)
    
    if args.warmup_use_GPA == True:
        if args.dataset_t=="foggy_cityscape":
            tgt_listpath="/space0/zhaofz/2022/faster-rcnn.pytorch/data/foggyCityscape/VOC2007/ImageSets/Main/trainval.txt"
            tgt_imgpath="/space0/zhaofz/2022/faster-rcnn.pytorch/data/foggyCityscape/VOC2007/JPEGImages"
        with open(tgt_listpath) as handle:
            tgt_content=handle.readlines()
        tgt_files=[]
        for fname in tgt_content:
            name = fname.strip()
            tgt_files.append(os.path.join(tgt_imgpath, "%s.jpg" % (name)))
        
        ### do gpa on source image
        # s_imdb_sourceGPA, s_train_size_sourceGPA , s_dataloader_sourceGPA = get_dataloader_aut(args.imdb_name, args, sequential=False, augment=args.warmup_use_SWAug, use_GPA=True, tgt_files=tgt_files)
        ### do gpa on tlike image
        s_imdb_sourceGPA, s_train_size_sourceGPA , s_dataloader_sourceGPA = get_dataloader_aut(args.imdb_name_fake_target, args, sequential=False, augment=args.warmup_use_SWAug, use_GPA=True, tgt_files=tgt_files)        
        
        # data_iter_s = iter(s_dataloader_sourceGPA)
        # checkdataloader(data_iter_s)
        # data_s=next(data_iter_s)

    # data_iter_s = iter(s_dataloader_target_like)
    # data_s=next(data_iter_s)
    # checkdataloader(data_iter_s)  #if you want to see what the data and gt looks like use this

    #######################################################
    ########## dataloader for mutual learning  ############
    #######################################################
    # source train set only use strong aug img
    ## use source-GPA as source img
    if args.mutual_use_GPA == True:
        if args.dataset_t=="foggy_cityscape":
            tgt_listpath="/space0/zhaofz/2022/faster-rcnn.pytorch/data/foggyCityscape/VOC2007/ImageSets/Main/trainval.txt"
            tgt_imgpath="/space0/zhaofz/2022/faster-rcnn.pytorch/data/foggyCityscape/VOC2007/JPEGImages"
        with open(tgt_listpath) as handle:
            tgt_content=handle.readlines()
        tgt_files=[]
        for fname in tgt_content:
            name = fname.strip()
            tgt_files.append(os.path.join(tgt_imgpath, "%s.jpg" % (name)))

        # s_imdb_mutual_GPA, s_train_size_mutual_GPA, s_dataloader_mutual_GPA = get_dataloader_aut(args.imdb_name, args, sequential=False, augment=args.mutual_use_SWAug, use_GPA=True, tgt_files=tgt_files)
        s_imdb_mutual_GPA, s_train_size_mutual_GPA, s_dataloader_mutual_GPA = get_dataloader_aut(args.imdb_name_fake_target, args, sequential=False, augment=args.mutual_use_SWAug, use_GPA=True, tgt_files=tgt_files)

    if args.mutual_use_targetlike == True:
        s_imdb_mutual_target_like, s_train_size_mutual_target_like , s_dataloader_mutual_target_like = get_dataloader_aut(args.imdb_name_fake_target,args,sequential=False,augment=args.mutual_use_SWAug)
    # else:
    s_imdb, s_train_size, s_dataloader = get_dataloader_aut(args.imdb_name, args, sequential=False, augment=args.mutual_use_SWAug)
    
   
    
    ###### data_s[0]:weak_img N*C*H*W | data_s[1]:strong img N*C*H*W
    #      data_s[2]:im_info  N*3     | data_s[3]:gt_boxes_padding N*30*5 
    #      data_s[4]:num_boxes N*1    | data_s[5]: blobs['path'] N tuple

     
    
    # target train set use weak + strong aug 
    t_imdb, t_train_size, t_dataloader = get_dataloader_aut(args.imdb_name_target, args, sequential=False, augment=args.mutual_use_SWAug)
    # data_iter_s = iter(t_dataloader)
    # data_s=next(data_iter_s)
    # checkdataloader(data_iter_s)  #if you want to see what the data and gt looks like use this
    

    print("{:d} source roidb entries".format(s_train_size))
    print("{:d} target roidb entries".format(t_train_size))

    # initilize the tensor holder here.
    im_data_warm = torch.FloatTensor(1)
    im_data_warm_targetlike = torch.FloatTensor(1)
    im_data_warm_sGPA = torch.FloatTensor(1)
    im_data_weak = torch.FloatTensor(1)
    im_data_strong = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    # ship to cuda
    if args.cuda:
        im_data_warm = im_data_warm.cuda()
        im_data_warm_targetlike = im_data_warm_targetlike.cuda()
        im_data_warm_sGPA = im_data_warm_sGPA.cuda()
        im_data_weak = im_data_weak.cuda()
        im_data_strong = im_data_strong.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data_warm = Variable(im_data_warm)
    im_data_warm_targetlike = Variable(im_data_warm_targetlike)
    im_data_warm_sGPA = Variable(im_data_warm_sGPA)
    im_data_weak = Variable(im_data_weak)
    im_data_strong = Variable(im_data_strong)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if args.cuda:
        cfg.CUDA = True

    imdb = s_imdb

    if args.net == "vgg16":
        student_fasterRCNN = vgg16(
            imdb.classes,
            pretrained=True,
            class_agnostic=args.class_agnostic,
        )
        teacher_fasterRCNN = vgg16(
            imdb.classes,
            pretrained=True,
            class_agnostic=args.class_agnostic,
        )
    elif args.net == "res101":
        student_fasterRCNN = resnet(
            imdb.classes,
            101,
            pretrained=True,
            class_agnostic=args.class_agnostic,
            conf=args.conf,
        )
        teacher_fasterRCNN = resnet(
            imdb.classes,
            101,
            pretrained=True,
            class_agnostic=args.class_agnostic,
            conf=args.conf,
        )
    elif args.net == "res50":
        student_fasterRCNN = resnet(
            imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic
        )
        teacher_fasterRCNN = resnet(
            imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic
        )
    else:
        print("network is not defined")
        pdb.set_trace()

    student_fasterRCNN.create_architecture()
    teacher_fasterRCNN.create_architecture()

    lr = cfg.TRAIN.LEARNING_RATE
    # lr = args.lr    #0.001 for the first 50k iterations  
                    #0.0001 for the next 30k iterations
    lr = args.aut_lr
    print('lr:',lr)

    if args.resume:
        student_checkpoint = torch.load(args.student_load_name)
        args.session = student_checkpoint["session"]
        args.start_epoch = 2 #student_checkpoint["epoch"]
        student_fasterRCNN.load_state_dict(student_checkpoint["model"])
        # student_optimizer.load_state_dict(student_checkpoint["optimizer"])
        # lr = student_optimizer.param_groups[0]["lr"]
        if "pooling_mode" in student_checkpoint.keys():
            cfg.POOLING_MODE = student_checkpoint["pooling_mode"]
        print("loaded checkpoint %s" % (args.student_load_name))

        teacher_checkpoint = torch.load(args.teacher_load_name)
        teacher_fasterRCNN.load_state_dict(teacher_checkpoint["model"])
        if "pooling_mode" in teacher_checkpoint.keys():
            cfg.POOLING_MODE = teacher_checkpoint["pooling_mode"]
        print("loaded checkpoint %s" % (args.teacher_load_name))

    student_detection_params = []
    params = []
    for key, value in dict(student_fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if "bias" in key:
                params += [
                    {
                        "params": [value],
                        "lr": lr * (cfg.TRAIN.DOUBLE_BIAS + 1),
                        "weight_decay": cfg.TRAIN.BIAS_DECAY
                        and cfg.TRAIN.WEIGHT_DECAY
                        or 0,
                    }
                ]
            else:
                params += [
                    {
                        "params": [value],
                        "lr": lr,
                        "weight_decay": cfg.TRAIN.WEIGHT_DECAY,
                    }
                ]
            student_detection_params += [value]
    
    # print("student_detection_params",student_detection_params[0])
    # print('params:',params[0])
    

    teacher_detection_params = []
    for key, value in dict(teacher_fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            teacher_detection_params += [value]
            value.requires_grad = False

    if args.optimizer == "adam":
        lr = lr * 0.1
        student_optimizer = torch.optim.Adam(params)
    elif args.optimizer == "sgd":
        student_optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)
    teacher_optimizer = WeightEMA(teacher_detection_params, student_detection_params, alpha=args.ema_weight)
    mutual_flag=False

    if args.cuda:
        student_fasterRCNN.cuda()
        teacher_fasterRCNN.cuda()

    if args.mGPUs:
        student_fasterRCNN = nn.DataParallel(student_fasterRCNN)
        teacher_fasterRCNN = nn.DataParallel(teacher_fasterRCNN)
    # iters_per_epoch = int(10000 / args.batch_size)
    iters_per_epoch = 10000 

    FL = EFocalLoss(class_num=2, gamma=3)

    if args.use_tfboard:
        from tensorboardX import SummaryWriter

        logger = SummaryWriter("logs-"+exp_setup)

    count_iter = 0
    # conf_gamma = args.conf_gamma
    # pretrained_epoch = args.pretrained_epoch
    warmup_epoch=args.warmup_epoch

    
    timer = PrettyTimer()
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        data_iter_warm = iter(s_dataloader_warm)
        if args.warmup_use_targetlike == True:
            data_iter_warm_targetlike = iter(s_dataloader_target_like)
        if args.warmup_use_GPA == True:
            data_iter_warm_sGPA = iter(s_dataloader_sourceGPA)

        data_iter_s = iter(s_dataloader)
        data_iter_t = iter(t_dataloader)
        if args.mutual_use_GPA ==True:
            data_iter_s_mutual_GPA = iter(s_dataloader_mutual_GPA)
        if args.mutual_use_targetlike ==True:
            data_iter_s_mutual_targetlike = iter(s_dataloader_mutual_target_like)

        # torch.cuda.empty_cache()
        # setting to train mode
        student_fasterRCNN.train()
        teacher_fasterRCNN.train()
        loss_temp = 0

        start = time.time()
        epoch_start = time.time()
        # if epoch % (args.lr_decay_step + 1) == 0:
        #     adjust_learning_rate(student_optimizer, args.lr_decay_gamma)
        #     lr *= args.lr_decay_gamma

        for step in range(1, iters_per_epoch + 1):
            print('step:',step)
            timer.start("iter")
        
            # eta = 1.0
            count_iter += 1
            if epoch <= warmup_epoch:
                print('warm up')
                try:
                    data_warm = next(data_iter_warm)
                except:
                    data_iter_warm = iter(s_dataloader_warm)
                    data_warm = next(data_iter_warm)
                if args.warmup_use_targetlike == True:
                    try:
                        data_warm_targetlike = next(data_iter_warm_targetlike)
                    except:
                        data_iter_warm_targetlike = iter(s_dataloader_target_like)
                        data_warm_targetlike = next(data_iter_warm_targetlike)
                if args.warmup_use_GPA:
                    try:
                        data_warm_sGPA = next(data_iter_warm_sGPA)
                    except:
                        data_iter_warm_sGPA = iter(s_dataloader_sourceGPA)
                        data_warm_sGPA = next(data_iter_warm_sGPA)


                with torch.no_grad():
                    im_data_warm.resize_(data_warm[0].size()).copy_(data_warm[0])
                    im_info.resize_(data_warm[2].size()).copy_(data_warm[2])
                    gt_boxes.resize_(data_warm[3].size()).copy_(data_warm[3])
                    num_boxes.resize_(data_warm[4].size()).copy_(data_warm[4])
                
                student_fasterRCNN.zero_grad()
                (
                rois_warm, 
                cls_prob_warm,
                bbox_pred_warm,
                rpn_loss_cls_warm,
                rpn_loss_box_warm,
                RCNN_loss_cls_warm,
                RCNN_loss_bbox_warm,
                rois_label_warm,
                _
                ) = student_fasterRCNN(im_data_warm, im_info, gt_boxes, num_boxes)

                loss = (
                    rpn_loss_cls_warm.mean()
                    + rpn_loss_box_warm.mean()
                    + RCNN_loss_cls_warm.mean()
                    + RCNN_loss_bbox_warm.mean()
                    )
                if args.warmup_use_targetlike == True:
                    with torch.no_grad():
                        im_data_warm_targetlike.resize_(data_warm_targetlike[0].size()).copy_(data_warm_targetlike[0])
                        im_info.resize_(data_warm_targetlike[2].size()).copy_(data_warm_targetlike[2])
                        gt_boxes.resize_(data_warm_targetlike[3].size()).copy_(data_warm_targetlike[3])
                        num_boxes.resize_(data_warm_targetlike[4].size()).copy_(data_warm_targetlike[4])

                    # student_fasterRCNN.zero_grad()
                    (
                    rois_warm_targetlike, 
                    cls_prob_warm_targetlike,
                    bbox_pred_warm_targetlike,
                    rpn_loss_cls_warm_targetlike,
                    rpn_loss_box_warm_targetlike,
                    RCNN_loss_cls_warm_targetlike,
                    RCNN_loss_bbox_warm_targetlike,
                    rois_label_warm_targetlike,
                    _
                    ) = student_fasterRCNN(im_data_warm_targetlike, im_info, gt_boxes, num_boxes)

                    loss += ( # +=
                        rpn_loss_cls_warm_targetlike.mean()
                        + rpn_loss_box_warm_targetlike.mean()
                        + RCNN_loss_cls_warm_targetlike.mean()
                        + RCNN_loss_bbox_warm_targetlike.mean()
                        )

                    ### if use GPA ####
                if args.warmup_use_GPA == True:
                    with torch.no_grad():
                        im_data_warm_sGPA.resize_(data_warm_sGPA[0].size()).copy_(data_warm_sGPA[0])
                        im_info.resize_(data_warm_sGPA[2].size()).copy_(data_warm_sGPA[2])
                        gt_boxes.resize_(data_warm_sGPA[3].size()).copy_(data_warm_sGPA[3])
                        num_boxes.resize_(data_warm_sGPA[4].size()).copy_(data_warm_sGPA[4])

                    # student_fasterRCNN.zero_grad()
                    (
                    rois_warm_sGPA, 
                    cls_prob_warm_sGPA,
                    bbox_pred_warm_sGPA,
                    rpn_loss_cls_warm_sGPA,
                    rpn_loss_box_warm_sGPA,
                    RCNN_loss_cls_warm_sGPA,
                    RCNN_loss_bbox_warm_sGPA,
                    rois_label_warm_sGPA,
                    _
                    ) = student_fasterRCNN(im_data_warm_sGPA, im_info, gt_boxes, num_boxes)

                    loss += ( # +=
                        rpn_loss_cls_warm_sGPA.mean()
                        + rpn_loss_box_warm_sGPA.mean()
                        + RCNN_loss_cls_warm_sGPA.mean()
                        + RCNN_loss_bbox_warm_sGPA.mean()
                    )
                

            elif epoch > warmup_epoch and args.enable_unsup==True:
                loss=0
                print('mutual learning')
                # torch.cuda.empty_cache()
                mutual_flag=True
                # if step==1 and epoch==warmup_epoch+1:
                    # del data_iter_warm, data_warm, im_data_warm, rois_warm, cls_prob_warm, bbox_pred_warm 
                # try:
                #     data_s = next(data_iter_s) # next(data_iter_warm_targetlike) # 
                # except:
                #     data_iter_s = iter(s_dataloader) #iter(s_dataloader_target_like) # 
                #     data_s = next(data_iter_s)

                
                

                #### use source-GPA image as source image ####
                if args.mutual_use_GPA == True:
                    try:
                        data_s_mutual_GPA = next(data_iter_s_mutual_GPA) # 
                    except:
                        data_iter_s_mutual_GPA = iter(s_dataloader_mutual_GPA) # 
                        data_s_mutual_GPA = next(data_iter_s_mutual_GPA)

                #### use target-like image as source image ####
                if args.mutual_use_targetlike == True:
                    try:
                        data_s_mutual_targetlike = next(data_iter_s_mutual_targetlike) # 
                    except:
                        data_iter_s_mutual_targetlike= iter(s_dataloader_mutual_target_like) # 
                        data_s_mutual_targetlike = next(data_iter_s_mutual_targetlike)
                try:
                    data_t = next(data_iter_t)
                except:
                    data_iter_t = iter(t_dataloader)
                    data_t = next(data_iter_t)

                # assert (data_s[0].size() == data_t[0].size()), "The size should be same between source_weak and target_weak"
                # assert (data_s[1].size() == data_t[1].size()), "The size should be same between source_strong and target_strong"

                ########## do student supervised training ###############
                # put source data into variable
                student_fasterRCNN.zero_grad()
                if args.mutual_use_targetlike== True:

                    with torch.no_grad():
                        im_data_strong.resize_(data_s_mutual_targetlike[1].size()).copy_(data_s_mutual_targetlike[1])
                        im_info.resize_(data_s_mutual_targetlike[2].size()).copy_(data_s_mutual_targetlike[2])
                        gt_boxes.resize_(data_s_mutual_targetlike[3].size()).copy_(data_s_mutual_targetlike[3])
                        num_boxes.resize_(data_s_mutual_targetlike[4].size()).copy_(data_s_mutual_targetlike[4])

                    
                    (
                        rois_s_s, # s_s:strong source
                        cls_prob_s_s,
                        bbox_pred_s_s,
                        rpn_loss_cls_s_s,
                        rpn_loss_box_s_s,
                        RCNN_loss_cls_s_s,
                        RCNN_loss_bbox_s_s,
                        rois_label_s_s,
                        domain_predict_source
                    ) = student_fasterRCNN(im_data_strong, im_info, gt_boxes, num_boxes)
                    loss += (
                        rpn_loss_cls_s_s.mean()
                        + rpn_loss_box_s_s.mean()
                        + RCNN_loss_cls_s_s.mean()
                        + RCNN_loss_bbox_s_s.mean()
                    )

                    domain_s = Variable(torch.zeros(domain_predict_source.size(0)).long().cuda()) # domain_predict_source.size(0): batch_size
                    # print('domain_predict_source.size(0)',domain_predict_source.size(0))
                    # print("domain_s:",domain_s)

                    ### adv loss for source sample ####\
                    #                         [1, 2]             1
                    advloss_s = 0.5 * FL(domain_predict_source, domain_s)
                # print('domain_predict_source.shape',domain_predict_source.shape)
                # print('advloss_source',advloss_s.item())

                if args.mutual_use_GPA== True:

                    with torch.no_grad():
                        im_data_strong.resize_(data_s_mutual_GPA[1].size()).copy_(data_s_mutual_GPA[1])
                        im_info.resize_(data_s_mutual_GPA[2].size()).copy_(data_s_mutual_GPA[2])
                        gt_boxes.resize_(data_s_mutual_GPA[3].size()).copy_(data_s_mutual_GPA[3])
                        num_boxes.resize_(data_s_mutual_GPA[4].size()).copy_(data_s_mutual_GPA[4])

                    # student_fasterRCNN.zero_grad()
                    (
                        rois_s_s, # s_s:strong source
                        cls_prob_s_s,
                        bbox_pred_s_s,
                        rpn_loss_cls_s_s,
                        rpn_loss_box_s_s,
                        RCNN_loss_cls_s_s,
                        RCNN_loss_bbox_s_s,
                        rois_label_s_s,
                        domain_predict_source
                    ) = student_fasterRCNN(im_data_strong, im_info, gt_boxes, num_boxes)
                    loss += (
                        rpn_loss_cls_s_s.mean()
                        + rpn_loss_box_s_s.mean()
                        + RCNN_loss_cls_s_s.mean()
                        + RCNN_loss_bbox_s_s.mean()
                    )
                    domain_s = Variable(torch.zeros(domain_predict_source.size(0)).long().cuda()) # domain_predict_source.size(0): batch_size
                    # print('domain_predict_source.size(0)',domain_predict_source.size(0))
                    # print("domain_s:",domain_s)

                    ### adv loss for source sample ####\
                    #                         [1, 2]             1
                    advloss_s = 0.5 * FL(domain_predict_source, domain_s)

                print('sup-loss',loss.item())


                ##### do unsupervised training #########            
                teacher_fasterRCNN.eval()

                # put target data into variable(weak aug)
                with torch.no_grad():
                    im_data_weak.resize_(data_t[0].size()).copy_(data_t[0])
                    im_info.resize_(data_t[2].size()).copy_(data_t[2])
                    # gt is empty
                    gt_boxes.resize_(1, 1, 5).zero_()
                    num_boxes.resize_(1).zero_()
                # put weak_aug target img to teacher to get pseudo label
                (
                    rois_w_t,
                    cls_prob_w_t,
                    bbox_pred_w_t,
                    rpn_loss_cls_w_t,
                    rpn_loss_box__w_t,
                    RCNN_loss_cls_w_t,
                    RCNN_loss_bbox_w_t,
                    rois_label_w_t,
                    _
                ) = teacher_fasterRCNN(im_data_weak, im_info, gt_boxes, num_boxes, test=True)

                scores = cls_prob_w_t.data
                boxes = rois_w_t.data[:, :, 1:5]

                if cfg.TEST.BBOX_REG:
                    # Apply bounding-box regression deltas
                    box_deltas = bbox_pred_w_t.data
                    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                        # Optionally normalize targets by a precomputed mean and stdev
                        if args.class_agnostic:
                            box_deltas = (
                                box_deltas.view(-1, 4)
                                * torch.FloatTensor(
                                    cfg.TRAIN.BBOX_NORMALIZE_STDS
                                ).cuda()
                                + torch.FloatTensor(
                                    cfg.TRAIN.BBOX_NORMALIZE_MEANS
                                ).cuda()
                            )
                            box_deltas = box_deltas.view(args.batch_size, -1, 4)
                        else:
                            box_deltas = (
                                box_deltas.view(-1, 4)
                                * torch.FloatTensor(
                                    cfg.TRAIN.BBOX_NORMALIZE_STDS
                                ).cuda()
                                + torch.FloatTensor(
                                    cfg.TRAIN.BBOX_NORMALIZE_MEANS
                                ).cuda()
                            )
                            box_deltas = box_deltas.view(args.batch_size, -1, 4 * len(imdb.classes))

                    pred_boxes = bbox_transform_inv(boxes, box_deltas, args.batch_size)
                    pred_boxes = clip_boxes(pred_boxes, im_info.data, args.batch_size)
                else:
                    # Simply repeat the boxes, once for each class
                    pred_boxes = np.tile(boxes, (1, scores.shape[1]))

                # for batchsize>1 do not squeeze()
                # scores = scores.squeeze()  #[2, 300, 9]
                # pred_boxes = pred_boxes.squeeze() #[2, 300, 36]

                # print('scores.shape',scores.shape)
                # print('pred_boxes.shape|||',pred_boxes.shape)
              
                gt_boxes_target = [[] for i in range(args.batch_size)]
                pre_thresh = 0.0
                thresh = args.threshold #threshold for pseudo label default 0.8
                empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
                for idx in range(args.batch_size):
                    for j in range(1, len(imdb.classes)):
                        # print('i',idx)
                        # print('j',j)
                        inds = torch.nonzero(scores[idx, :, j] > pre_thresh).view(-1)
                        
                        # if there is det
                        if inds.numel() > 0:
                            cls_scores = scores[idx, :, j][inds]
                            _, order = torch.sort(cls_scores, 0, True)
                            if args.class_agnostic:
                                cls_boxes = pred_boxes[idx, inds, :]
                            else:
                                cls_boxes = pred_boxes[idx, inds][:, j * 4 : (j + 1) * 4]

                            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                            
                            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                            cls_dets = cls_dets[order]
                            # keep = nms(cls_dets, cfg.TEST.NMS)
                            keep = nms(cls_boxes[order, :], cls_scores[order],cfg.TEST.NMS)
                            cls_dets = cls_dets[keep.view(-1).long()]
                            # all_boxes[j][i] = cls_dets.cpu().numpy()
                            cls_dets_numpy = cls_dets.cpu().numpy()
                            # print('cls_dets_numpy.shape',cls_dets_numpy.shape)
                            
                            for i in range(np.minimum(10, cls_dets_numpy.shape[0])):   #Q:limit the number within 10??  A:per class pseudo_box is limited within 10
                                bbox = tuple(
                                    int(np.round(x)) for x in cls_dets_numpy[i, :4]
                                )
                                score = cls_dets_numpy[i, -1]
                                if score > thresh:                                    # select proposals with confidence_score > 0.8 as pseudo label for distillation loss
                                    gt_boxes_target[idx].append(list(bbox[0:4]) + [j])
                # print('real_gt:',data_t[3])
                # print('image:',data_t[-1])
                # print('gt_boxes_target:',gt_boxes_target)
                pseudo_bbox_num=[len(gt_boxes_target[q]) for q in range(args.batch_size)] #len(gt_boxes_target) # put to tfb to check if pseudo box per image is less than 10
                mean_pseudo_bbox_num = np.mean(pseudo_bbox_num)
                print('mean pseudo_bbox_num', mean_pseudo_bbox_num)
                # print('real gt:',data_t[3])
                # print('pseudo_bbox',gt_boxes_target)
                
                gt_boxes_padding = torch.FloatTensor(args.batch_size,cfg.MAX_NUM_GT_BOXES, 5).zero_()
                for idx in range(args.batch_size):
                    if len(gt_boxes_target[idx]) != 0:
                        gt_boxes_numpy = torch.FloatTensor(gt_boxes_target[idx])
                        num_boxes_cpu = torch.LongTensor(
                            [min(gt_boxes_numpy.size(0), cfg.MAX_NUM_GT_BOXES)]
                        )
                        gt_boxes_padding[idx, :num_boxes_cpu, :] = gt_boxes_numpy[:num_boxes_cpu]
                    else:
                        num_boxes_cpu = torch.LongTensor([0])
                

                # teacher_fasterRCNN.train()
                # put target data into variable(strong_aug)
                # put strong_aug img to student to get prediction
                with torch.no_grad():
                    im_data_strong.resize_(data_t[1].size()).copy_(data_t[1])
                    im_info.resize_(data_t[2].size()).copy_(data_t[2])
                    # gt_boxes_padding = torch.unsqueeze(gt_boxes_padding, 0)
                    # print('gt_boxes_padding.shape',gt_boxes_padding.shape)
                    
                    gt_boxes.resize_(gt_boxes_padding.size()).copy_(gt_boxes_padding)
                    num_boxes.resize_(num_boxes_cpu.size()).copy_(num_boxes_cpu)

                (
                    rois_s_t, # s_t: strong aug target
                    cls_prob_s_t,
                    bbox_pred_s_t,
                    rpn_loss_cls_s_t,
                    rpn_loss_box_s_t,
                    RCNN_loss_cls_s_t,
                    RCNN_loss_bbox_s_t,
                    rois_label_s_t,
                    domain_predict_target
                ) = student_fasterRCNN(im_data_strong, im_info, gt_boxes, num_boxes)

                rpn_loss_box_s_t = rpn_loss_box_s_t * 0
                RCNN_loss_bbox_s_t = RCNN_loss_bbox_s_t * 0

                domain_t = Variable(torch.ones(domain_predict_target.size(0)).long().cuda())
                ### adv loss for target sample ####
                advloss_t = 0.5 * FL(domain_predict_target, domain_t)
                # print('domain_predict_target',domain_predict_target)
                # print('advloss_target',advloss_t.item())


                loss += args.unsup_loss_weight * (
                    rpn_loss_cls_s_t.mean()
                    + rpn_loss_box_s_t.mean()
                    + RCNN_loss_cls_s_t.mean()
                    + RCNN_loss_bbox_s_t.mean()
                )

                adv_loss=advloss_s+advloss_t
                print('adv_loss:',adv_loss.item())
                # print((epoch - 1) * iters_per_epoch + step)

                loss += adv_loss * args.adv_loss_weight

                if args.use_tfboard:
                    info = {
                        "mean pseudo_bbox_num": mean_pseudo_bbox_num if epoch > warmup_epoch else 0,
                        "adv_loss": adv_loss if epoch > warmup_epoch else 0,
                        "adv_loss_source": advloss_s.item() if epoch > warmup_epoch else 0,
                        "adv_loss_target": advloss_t.item() if epoch > warmup_epoch else 0,
                    }
                    logger.add_scalars(
                        "logs_s_{}/losses".format(args.session),
                        info,
                        (epoch - 1) * iters_per_epoch + step,
                    )


            print('total loss:',loss.item())
            loss_temp += loss.item()
            student_optimizer.zero_grad()
            loss.backward()
            if args.net == "vgg16":
                clip_gradient(student_fasterRCNN, 10.)
            student_optimizer.step()
            teacher_fasterRCNN.zero_grad()
            teacher_optimizer.step(mutual=mutual_flag)
            timer.end("iter")
            estimate_time = timer.eta(
                "iter", count_iter, args.max_epochs * iters_per_epoch
            )
            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= args.disp_interval

                if args.mGPUs:
                    suploss_rpn_cls = rpn_loss_cls_s_s.mean().item()
                    suploss_rpn_box = rpn_loss_box_s_s.mean().item()
                    suploss_rcnn_cls = RCNN_loss_cls_s_s.mean().item()
                    suploss_rcnn_box = RCNN_loss_bbox_s_s.mean().item()
                    fg_cnt_source = torch.sum(rois_label_s_s.data.ne(0))
                    bg_cnt_source = rois_label_s_s.data.numel() - fg_cnt_source
                    if args.enable_unsup==True and epoch > warmup_epoch:
                        unsuploss_rpn_cls_target = rpn_loss_cls_s_t.mean().item()
                        unsuploss_rpn_box_target = rpn_loss_box_s_t.mean().item()
                        unsuploss_rcnn_cls_target = RCNN_loss_cls_s_t.mean().item()
                        unsuploss_rcnn_box_target = RCNN_loss_bbox_s_t.mean().item()
                        fg_cnt_target = torch.sum(rois_label_s_t.data.ne(0))
                        bg_cnt_target = rois_label_s_t.data.numel() - fg_cnt_target


                    # dloss_s_fake = dloss_s_fake.mean().item()
                    # dloss_t_fake = dloss_t_fake.mean().item()
                    # dloss_s_p_fake = dloss_s_p_fake.mean().item()
                    # dloss_t_p_fake = dloss_t_p_fake.mean().item()
                else:
                    if epoch <= warmup_epoch:
                        suploss_rpn_cls_warm = rpn_loss_cls_warm.item()
                        suploss_rpn_box_warm = rpn_loss_box_warm.item()
                        suploss_rcnn_cls_warm = RCNN_loss_cls_warm.item()
                        suploss_rcnn_box_warm = RCNN_loss_bbox_warm.item()
                        fg_cnt_source_warm = torch.sum(rois_label_warm.data.ne(0))
                        bg_cnt_source_warm = rois_label_warm.data.numel() - fg_cnt_source_warm
                        ## target like ####
                        if args.warmup_use_targetlike == True:
                            suploss_rpn_cls_warm_targetlike = rpn_loss_cls_warm_targetlike.item()
                            suploss_rpn_box_warm_targetlike = rpn_loss_box_warm_targetlike.item()
                            suploss_rcnn_cls_warm_targetlike = RCNN_loss_cls_warm_targetlike.item()
                            suploss_rcnn_box_warm_targetlike = RCNN_loss_bbox_warm_targetlike.item()
                            fg_cnt_source_warm_targetlike = torch.sum(rois_label_warm_targetlike.data.ne(0))
                            bg_cnt_source_warm_targetlike = rois_label_warm_targetlike.data.numel() - fg_cnt_source_warm_targetlike
                        ### GPA ### 
                        if args.warmup_use_GPA==True:
                            suploss_rpn_cls_warm_sGPA = rpn_loss_cls_warm_sGPA.item()
                            suploss_rpn_box_warm_sGPA = rpn_loss_box_warm_sGPA.item()
                            suploss_rcnn_cls_warm_sGPA = RCNN_loss_cls_warm_sGPA.item()
                            suploss_rcnn_box_warm_sGPA = RCNN_loss_bbox_warm_sGPA.item()
                            fg_cnt_source_warm_sGPA = torch.sum(rois_label_warm_sGPA.data.ne(0))
                            bg_cnt_source_warm_sGPA = rois_label_warm_sGPA.data.numel() - fg_cnt_source_warm_sGPA

                    if  epoch > warmup_epoch:
                        suploss_rpn_cls = rpn_loss_cls_s_s.item()
                        suploss_rpn_box = rpn_loss_box_s_s.item()
                        suploss_rcnn_cls = RCNN_loss_cls_s_s.item()
                        suploss_rcnn_box = RCNN_loss_bbox_s_s.item()
                        fg_cnt_source = torch.sum(rois_label_s_s.data.ne(0))
                        bg_cnt_source = rois_label_s_s.data.numel() - fg_cnt_source


                    if args.enable_unsup==True and epoch > warmup_epoch:
                        unsuploss_rpn_cls_target = rpn_loss_cls_s_t.item()
                        unsuploss_rpn_box_target = rpn_loss_box_s_t.item()
                        unsuploss_rcnn_cls_target = RCNN_loss_cls_s_t.item()
                        unsuploss_rcnn_box_target = RCNN_loss_bbox_s_t.item()
                        fg_cnt_target = torch.sum(rois_label_s_t.data.ne(0))
                        bg_cnt_target = rois_label_s_t.data.numel() - fg_cnt_target


                print(
                    "[session %d][epoch %2d][iter %4d/%4d] lr: %.2e, loss: %.4f, eta: %s"
                    % (
                        args.session,
                        epoch,
                        step,
                        iters_per_epoch,
                        lr,
                        loss_temp,
                        estimate_time,
                    )
                )
                if epoch <= warmup_epoch:
                    print(
                        "\t\t\tfg_source_warm/bg_source_warm=(%d/%d), time cost: %f" % (fg_cnt_source_warm, bg_cnt_source_warm, end - start)
                    )
                    print(
                        "\t\t\tsuploss_rpn_cls_warm: %.4f, suploss_rpn_box_warm: %.4f, suploss_rcnn_cls_warm: %.4f, suploss_rcnn_box_warm %.4f"
                        % (suploss_rpn_cls_warm, suploss_rpn_box_warm, suploss_rcnn_cls_warm, suploss_rcnn_box_warm)
                    )
                    if args.warmup_use_targetlike == True:
                        ### target like #####
                        print(
                            "\t\t\tfg_source_warm_targetlike/bg_source_warm_targetlike=(%d/%d), time cost: %f" % (fg_cnt_source_warm_targetlike, bg_cnt_source_warm_targetlike, end - start)
                        )
                        print(
                            "\t\t\tsuploss_rpn_cls_warm_targetlike: %.4f, suploss_rpn_box_warm_targetlike: %.4f, suploss_rcnn_cls_warm_targetlike: %.4f, suploss_rcnn_box_warm_targetlike %.4f"
                            % (suploss_rpn_cls_warm_targetlike, suploss_rpn_box_warm_targetlike, suploss_rcnn_cls_warm_targetlike, suploss_rcnn_box_warm_targetlike)
                        )
                    ### GPA ###
                    if args.warmup_use_GPA:
                        print(
                        "\t\t\tfg_source_warm_sGPA/bg_source_warm_sGPA=(%d/%d), time cost: %f" % (fg_cnt_source_warm_sGPA, bg_cnt_source_warm_sGPA, end - start)
                        )
                        print(
                            "\t\t\tsuploss_rpn_cls_warm_sGPA: %.4f, suploss_rpn_box_warm_sGPA: %.4f, suploss_rcnn_cls_warm_sGPA: %.4f, suploss_rcnn_box_warm_sGPA %.4f"
                            % (suploss_rpn_cls_warm_sGPA, suploss_rpn_box_warm_sGPA, suploss_rcnn_cls_warm_sGPA, suploss_rcnn_box_warm_sGPA)
                        )
                if epoch > warmup_epoch:
                    print(
                        "\t\t\tfg_source/bg_source=(%d/%d), time cost: %f" % (fg_cnt_source, bg_cnt_source, end - start)
                    )
                    print(
                        "\t\t\tsuploss_rpn_cls: %.4f, suploss_rpn_box: %.4f, suploss_rcnn_cls: %.4f, suploss_rcnn_box %.4f"
                        % (suploss_rpn_cls, suploss_rpn_box, suploss_rcnn_cls, suploss_rcnn_box)
                    )

                if args.enable_unsup==True and epoch > warmup_epoch:
                    print("\t\t\tfg_target/bg_target=(%d/%d)" % (fg_cnt_target, bg_cnt_target))
                    print(
                        "\t\t\tunsuploss_rpn_cls_target: %.4f, unsuploss_rpn_box_target: %.4f, unsuploss_rcnn_cls_target: %.4f, unsuploss_rcnn_box_target %.4f"
                        % (
                            unsuploss_rpn_cls_target,
                            unsuploss_rpn_box_target,
                            unsuploss_rcnn_cls_target,
                            unsuploss_rcnn_box_target,
                        )
                    )

            

                if args.use_tfboard:
                    info = {
                        "loss": loss_temp,
                        "suploss_rpn_cls_warm": suploss_rpn_cls_warm if epoch <= warmup_epoch else 0,
                        "suploss_rpn_box_warm": suploss_rpn_box_warm if epoch <= warmup_epoch else 0,
                        "suploss_rcnn_cls_warm": suploss_rcnn_cls_warm if epoch <= warmup_epoch else 0,
                        "suploss_rcnn_box_warm": suploss_rcnn_box_warm if epoch <= warmup_epoch else 0,
                        "suploss_rpn_cls_warm_targetlike": suploss_rpn_cls_warm_targetlike if epoch <= warmup_epoch and args.warmup_use_targetlike==True else 0,
                        "suploss_rpn_box_warm_targetlike": suploss_rpn_box_warm_targetlike if epoch <= warmup_epoch and args.warmup_use_targetlike==True else 0,
                        "suploss_rcnn_cls_warm_targetlike": suploss_rcnn_cls_warm_targetlike if epoch <= warmup_epoch and args.warmup_use_targetlike==True else 0,
                        "suploss_rcnn_box_warm_targetlike": suploss_rcnn_box_warm_targetlike if epoch <= warmup_epoch and args.warmup_use_targetlike==True else 0,
                        ### GPA ###
                        
                        "suploss_rpn_cls_warm_sGPA": suploss_rpn_cls_warm_sGPA if epoch <= warmup_epoch and args.warmup_use_GPA == True else 0,
                        "suploss_rpn_box_warm_sGPA": suploss_rpn_box_warm_sGPA if epoch <= warmup_epoch and args.warmup_use_GPA == True else 0,
                        "suploss_rcnn_cls_warm_sGPA": suploss_rcnn_cls_warm_sGPA if epoch <= warmup_epoch and args.warmup_use_GPA == True else 0,
                        "suploss_rcnn_box_warm_sGPA": suploss_rcnn_box_warm_sGPA if epoch <= warmup_epoch and args.warmup_use_GPA == True else 0,
                        
                        "suploss_rpn_cls": suploss_rpn_cls if epoch > warmup_epoch else 0,
                        "suploss_rpn_box": suploss_rpn_box if epoch > warmup_epoch else 0,
                        "suploss_rcnn_cls": suploss_rcnn_cls if epoch > warmup_epoch else 0,
                        "suploss_rcnn_box": suploss_rcnn_box if epoch > warmup_epoch else 0,
                        "unsuploss_rpn_cls_target": unsuploss_rpn_cls_target if epoch > warmup_epoch else 0,
                        "unsuploss_rpn_box_target": unsuploss_rpn_box_target if epoch > warmup_epoch else 0,
                        "unsuploss_rcnn_cls_target": unsuploss_rcnn_cls_target if epoch > warmup_epoch else 0,
                        "unsuploss_rcnn_box_target": unsuploss_rcnn_box_target if epoch > warmup_epoch else 0,
                        # "pseudo_bbox_num": pseudo_bbox_num if epoch > warmup_epoch else 0,
                    }
                    logger.add_scalars(
                        "logs_s_{}/losses".format(args.session),
                        info,
                        (epoch - 1) * iters_per_epoch + step,
                    )

                loss_temp = 0

                start = time.time()

        student_save_name = os.path.join(
            model_save_path,
            "student_target_{}_session_{}_epoch_{}_step_{}.pth".format(
                args.dataset_t,
                args.session,
                epoch,
                step,
            ),
        )
        save_checkpoint(
            {
                "session": args.session,
                "epoch": epoch + 1,
                "model": student_fasterRCNN.module.state_dict()
                if args.mGPUs
                else student_fasterRCNN.state_dict(),
                "optimizer": student_optimizer.state_dict(),
                "pooling_mode": cfg.POOLING_MODE,
                "class_agnostic": args.class_agnostic,
            },
            student_save_name,
        )
        print("save student model: {}".format(student_save_name))

        teacher_save_name = os.path.join(
            model_save_path,
            "teacher_target_{}_session_{}_epoch_{}_step_{}.pth".format(
                args.dataset_t,
                args.session,
                epoch,
                step,
            ),
        )
        save_checkpoint(
            {
                "session": args.session,
                "epoch": epoch + 1,
                "model": teacher_fasterRCNN.module.state_dict()
                if args.mGPUs
                else teacher_fasterRCNN.state_dict(),
                "pooling_mode": cfg.POOLING_MODE,
                "class_agnostic": args.class_agnostic,
            },
            teacher_save_name,
        )
        print("save teacher model: {}".format(teacher_save_name))
        epoch_end = time.time()
        print("epoch cost time: {} min".format((epoch_end - epoch_start) / 60.0))

        # cmd = (
        #     f"python test_net_global_local.py --dataset {args.dataset_t} --net {args.net}"
        #     f" --load_name {student_save_name}"
        # )
        # print("cmd: ", cmd)
        # cmd = [i.strip() for i in cmd.split(" ") if len(i.strip()) > 0]
        # try:
        #     proc = subprocess.Popen(cmd)
        #     proc.wait()
        # except (KeyboardInterrupt, SystemExit):
        #     pass

        # cmd = (
        #     f"python test_net_global_local.py --dataset {args.dataset_t} --net {args.net}"
        #     f" --load_name {teacher_save_name}"
        # )
        # print("cmd: ", cmd)
        # cmd = [i.strip() for i in cmd.split(" ") if len(i.strip()) > 0]
        # try:
        #     proc = subprocess.Popen(cmd)
        #     proc.wait()
        # except (KeyboardInterrupt, SystemExit):
        #     pass

    if args.use_tfboard:
        logger.close()
