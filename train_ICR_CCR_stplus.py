# coding:utf-8
# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ast import arg
from asyncio import FastChildWatcher
from tkinter import image_names
try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3

import math
import cv2
import os
import numpy as np
import pprint
import pdb
import time
from numpy.core.fromnumeric import shape
from copy import deepcopy
from tqdm import tqdm
from lib.roi_data_layer.roidb import combined_roidb
from lib.roi_data_layer.roibatchLoader import roibatchLoader
# from torchvision.transforms import autoaugment
import _init_paths

import torch
from torch.autograd import Variable
import torch.nn as nn
import random
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import SubElement
from xml.etree.ElementTree import ElementTree
import xml.etree.ElementTree as ET

from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list
from lib.model.utils.net_utils import (
    FocalLoss,
    adjust_learning_rate,
    save_checkpoint,
    get_dataloader,
    get_dataloader_aut,
    setup_seed,
    clip_gradient,
    EFocalLoss
)
from lib.model.ema.optim_weight_ema import WeightEMA
from lib.model.utils.parser_func import parse_args, set_dataset_args
from lib.model.rpn.bbox_transform import clip_boxes
# from lib.model.nms.nms_wrapper import nms
from lib.model.roi_layers import nms
from lib.model.rpn.bbox_transform import bbox_transform_inv

from prettytimer import PrettyTimer

from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn.functional as F
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
    for step in range(1,80):
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
        if not os.path.exists("aug_examples"):
            os.mkdir("aug_examples")
        imsave('aug_examples/weak_{}_{}'.format(step,name),img_weak[:, :, ::-1])
        imsave('aug_examples/strong_{}_{}'.format(step,name),img_strong[:, :, ::-1])
        # img_w = cv2.cvtColor(img_weak, cv2.COLOR_BGR2RGB)
        # cv2.normalize(img_weak, img_weak, 0, 255, cv2.NORM_MINMAX)
        # cv2.normalize(img_strong, img_strong, 0, 255, cv2.NORM_MINMAX)
        # cv2.imwrite('no_aug/cv2_data_withgt_weak_{}'.format(name),img_weak)
        # cv2.imwrite('no_aug/cv2_data_withgt_strong_{}'.format(name),img_strong)
    quit()

def create_xml(ann_path):
    root = Element('annotation')

    folder=SubElement(root,'folder')
    folder.text='VOC2007'

    filename=SubElement(root,'filename')
    basename=os.path.basename(ann_path)
    filename.text=basename.replace('.xml','.jpg')

    segmented=SubElement(root,'segmented')
    segmented.text='0'

    size=SubElement(root,'size')
    width=SubElement(size,'width')
    width.text='2048'
    height=SubElement(size,'height')
    height.text='1024'
    depth=SubElement(size,'depth')
    depth.text='3'

    tree = ElementTree(root)
    tree.write(ann_path,encoding='utf-8')

def add_node(xml_path,classname,bbox):
    tree= ElementTree()
    tree.parse(xml_path)

    root = tree.getroot()
    object=Element('object')

    bndbox=Element('bndbox')
    xmin=Element('xmin')
    xmin.text=str(bbox[0])
    ymin=Element('ymin')
    ymin.text=str(bbox[1])
    xmax=Element('xmax')
    xmax.text=str(bbox[2])
    ymax=Element('ymax')
    ymax.text=str(bbox[3])

    bndbox.append(xmin)
    bndbox.append(ymin)
    bndbox.append(xmax)
    bndbox.append(ymax)
    
    object.append(bndbox)
    name=Element('name')
    name.text=classname
    object.append(name)

    difficult=Element('difficult')
    difficult.text='0'
    object.append(difficult)

    truncated=Element('truncated')
    truncated.text='0'
    object.append(truncated)

    root.append(object)
    tree.write(xml_path,encoding='utf-8')

def prettyXml(element, indent, newline, level = 0): 
    # 判断element是否有子元素
    if element:
        # 如果element的text没有内容      
        if element.text == None or element.text.isspace():     
            element.text = newline + indent * (level + 1)      
        else:    
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)    
    # 此处两行如果把注释去掉，Element的text也会另起一行 
    #else:     
        #element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level    
    temp = list(element) # 将elemnt转成list    
    for subelement in temp:    
        # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
        if temp.index(subelement) < (len(temp) - 1):     
            subelement.tail = newline + indent * (level + 1)    
        else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个    
            subelement.tail = newline + indent * level   
        # 对子元素进行递归操作 
        prettyXml(subelement, indent, newline, level = level + 1) 

def write_annotation_xml(imgname,model_k_predpath,model_k_annotation_path):
    # img name: fake_b_munster_000089_000019_leftImg8bit
    ann_name=imgname+'.xml'
    ann_name=os.path.join(model_k_annotation_path,ann_name)
    if not os.path.exists(ann_name):
        create_xml(ann_name)
    
    files=os.listdir(model_k_predpath)
    for file in files:
        class_name=file.split('_')[-1].split('.')[0]
        with open(os.path.join(model_k_predpath,file)) as f:
            lines = f.readlines()
        splitlines = [x.strip().split(' ') for x in lines]
        for line in splitlines:
            BB = np.array([round(float(z)) for z in line[2:]])
            add_node(ann_name,class_name,BB)

    ## beautify ###
    tree = ET.parse(ann_name)
    root=tree.getroot()
    prettyXml(root,'\t','\n')
    tree.write(ann_name,encoding='utf-8')
    
def select_reliable(imdbval_name,models,args,epoch):
    if not os.path.exists(args.reliable_path):
        os.mkdir(args.reliable_path)
    
    for i in range(len(models)):
        models[i].eval()
    
    
    imdb, roidb, ratio_list, ratio_index = combined_roidb(imdbval_name, False)
    imdb.competition_mode(on=True)
    
    print("{:d} roidb entries (selective phase)".format(len(roidb)))
    dataset = roibatchLoader(
        roidb,
        ratio_list,
        ratio_index,
        1,
        imdb.num_classes,
        training=False,
        normalize=False,
        path_return=True,
        )           
    bs=1
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)
    
    tbar=tqdm(dataloader)
    id_to_reliability = []
    output_dir_root='/space0/zhaofz/2022/DAOD/data/foggyCityscape/VOC2007/ST'
    model_k_predpath=output_dir_root+'/fakegt_predictions'
    model_k_annotation_path=output_dir_root+'/Annotations'
    if not os.path.exists(model_k_annotation_path):
                os.mkdir(model_k_annotation_path)
    os.system("rm {}/*".format(model_k_annotation_path))
    print("delete formal fake gt")

    with torch.no_grad():
        for data in tbar:
            print(data[-1])
            preds=[]
            for idx, model in enumerate(models):
                if idx==len(models)-1:
                    thresh=0.8
                else:
                    thresh=0.0
                all_boxes=get_predictions(data,model,imdb,thresh)
                preds.append(all_boxes)
            print("get prediction from K models")
            imgpath=data[-2] #['/space0/zhaofz/2022/DAOD/data/foggyCityscape/VOC2007/JPEGImages/target_munster_000089_000019_leftImg8bit_foggy_beta_0.02.jpg']
            imgname=imgpath[0].split('/')[-1]
            imgname=imgname.replace('.jpg','')
            # imgname=imgname.replace('target','fake_b')
            print("imgname:",imgname)  # img name: fake_b_munster_000089_000019_leftImg8bit

            with open("/space0/zhaofz/2022/DAOD/data/foggyCityscape/VOC2007/ST/test.txt",'w') as f:
                f.write(imgname)

            fake_gt = preds[-1]
            
            imdb._write_voc_results_file_st(fake_gt,output_dir=output_dir_root,imgname=imgname,pred_type='fakegt_predictions')  #ST/fakegt_predictions
            
            
            write_annotation_xml(imgname,model_k_predpath,model_k_annotation_path)

            mAP=[]
            for i in range(len(preds) - 1):
                print("Evaluating detections for K Models")
                map=imdb.evaluate_detections_st(preds[i], imgname=imgname, output_dir=output_dir_root,ST=True)
                mAP.append(map)
            print("imgname {} mAP:".format(imgname),mAP)
            reliability = sum(mAP) / len(mAP)
            id_to_reliability.append((imgname, reliability))
    
    id_to_reliability.sort(key=lambda elem: elem[1], reverse=True)
    reliable_id_path = '/space0/zhaofz/2022/DAOD/data/foggyCityscape/VOC2007/ImageSets/Main'
    os.rename(reliable_id_path+'/reliable.txt',reliable_id_path+'/reliable_epoch{}.txt'.format(epoch))
    os.rename(reliable_id_path+'/unreliable.txt',reliable_id_path+'/unreliable_epoch{}.txt'.format(epoch))
    with open(os.path.join(reliable_id_path, 'reliable.txt'), 'w') as f:
        for elem in id_to_reliability[:len(id_to_reliability) // 2]:
            f.write(elem[0] + ' ' +str(elem[1]) + '\n')
    with open(os.path.join(reliable_id_path, 'unreliable.txt'), 'w') as f:
        for elem in id_to_reliability[len(id_to_reliability) // 2:]:
            f.write(elem[0] + ' ' + str(elem[1]) + '\n')

def get_predictions(data,model,imdb,thresh):
    model.eval()
    num_images = 1 #len(imdb.image_index)
    print("num_images:",num_images)
    all_boxes = [[[] for _ in xrange(num_images)] for _ in xrange(imdb.num_classes)]
    bs=1
    # thresh = 0.8

    img = data[0].cuda()
    im_info = data[1].cuda()
    gt_boxes = data[2].cuda() 
    num_boxes = data[3].cuda()
    im_cls_lb = data[5].cuda()
    (
    rois,
    cls_prob,
    bbox_pred,
    rpn_loss_cls,
    rpn_loss_box,
    RCNN_loss_cls,
    RCNN_loss_bbox,
    rois_label,
    _,_,_,_
    ) = model(img, im_info, im_cls_lb, gt_boxes, num_boxes)

    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))

    if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            a=box_deltas.view(-1, 4)
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if args.class_agnostic:
                    box_deltas = (
                        box_deltas.view(-1, 4)
                        * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda()
                        + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    )
                    box_deltas = box_deltas.view(bs, -1, 4)
                else:
                    box_deltas = (
                        box_deltas.view(-1, 4)
                        * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda()
                        + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    )
                    box_deltas = box_deltas.view(bs, -1, 4 * len(imdb.classes))

            # print('box_deltas.size()',box_deltas.size())
            # print('boxes.size()',boxes.size())
            
            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, bs)
        # print('pred_boxes.size',pred_boxes.size())
        
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))
    pred_boxes /= data[1][0][2].item()  # height width im_scale(0.58594 for city)
    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()
    for j in xrange(1, imdb.num_classes):
        inds = torch.nonzero(scores[:, j] > thresh).view(-1)
        # if there is det
        if inds.numel() > 0:
            cls_scores = scores[:, j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            if args.class_agnostic:
                cls_boxes = pred_boxes[inds, :]
            else:
                cls_boxes = pred_boxes[inds][:, j * 4 : (j + 1) * 4]
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            # keep = nms(cls_dets, cfg.TEST.NMS)
            keep = nms(cls_boxes[order, :], cls_scores[order],cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            
            # imdb.image_path_at(i)
            all_boxes[j][0] = cls_dets.cpu().numpy()
        else:
            all_boxes[j][0] = empty_array
    # Limit to max_per_image detections *over all classes*
    max_per_image = 100
    if max_per_image > 0:
        image_scores = np.hstack(
            [all_boxes[j][0][:, -1] for j in xrange(1, imdb.num_classes)]
        )
        if len(image_scores) > max_per_image:
            image_thresh = np.sort(image_scores)[-max_per_image]
            for j in xrange(1, imdb.num_classes):
                keep = np.where(all_boxes[j][0][:, -1] >= image_thresh)[0]
                all_boxes[j][0] = all_boxes[j][0][keep, :]
    return all_boxes

def save_checkpoints(args,cfg,epoch,step,model_save_path):
    ### save checkpoint for student and teacher model at the end of each epoch ####
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

if __name__ == "__main__":
    with torch.autograd.set_detect_anomaly(True):
        args = get_cfg()

        output_dir = f"{args.save_dir}/{args.net}/{args.dataset}"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if args.dataset_t == "water":
            args.aug = False

        if args.dataset_t == "foggy_cityscape":
            # initilize the network here.
            from lib.model.ICRCCR_faster_rcnn.vgg16 import vgg16
            # from lib.model.umt_faster_rcnn_truncate.umt_resnet import resnet
        else:
            from lib.model.umt_faster_rcnn.umt_vgg16 import vgg16
            # from lib.model.umt_faster_rcnn.umt_resnet import resnet

        exp_setup="insRight-ICRCCR-tinycutout-st++warmup-s{}{}_SWAug_{}_mutual-{}{}_SWAug_{}_bs_{}_unsup_loss_w_{}_adv_loss_w_{}_EF_{}_threshold_{}_aut_lr_{}_warmup_epoch_{}_mutual_epoch_{}_ema_weight_{}_lrdc_epoch{}".format(
                "+tlike" if args.warmup_use_targetlike == True else '',
                "+sGPA" if args.warmup_use_GPA == True else '',
                args.warmup_use_SWAug,
                "+tlike" if args.mutual_use_targetlike == True else '',
                "+sGPA" if args.mutual_use_GPA == True else '',
                args.mutual_use_SWAug,
                args.batch_size,
                args.unsup_loss_weight,
                args.adv_loss_weight,
                args.ef,
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
            s_imdb_target_like, s_train_size_target_like , s_dataloader_target_like = get_dataloader_aut(args.imdb_name_fake_target, args, sequential=False, augment=args.warmup_use_SWAug)

        if args.warmup_use_GPA == True:
            if args.dataset_t=="foggy_cityscape":
                tgt_listpath="/space0/zhaofz/2022/DAOD/data/foggyCityscape/VOC2007/ImageSets/Main/trainval.txt"
                tgt_imgpath="/space0/zhaofz/2022/DAOD/data/foggyCityscape/VOC2007/JPEGImages"
            with open(tgt_listpath) as handle:
                tgt_content=handle.readlines()
            tgt_files=[]
            for fname in tgt_content:
                name = fname.strip()
                tgt_files.append(os.path.join(tgt_imgpath, "%s.jpg" % (name)))

            ### do gpa on source image
            s_imdb_sourceGPA, s_train_size_sourceGPA , s_dataloader_sourceGPA = get_dataloader_aut(args.imdb_name, args, sequential=False, augment=args.warmup_use_SWAug, use_GPA=True, tgt_files=tgt_files)
            ### do gpa on tlike image
            # s_imdb_sourceGPA, s_train_size_sourceGPA , s_dataloader_sourceGPA = get_dataloader_aut(args.imdb_name_fake_target, args, sequential=False, augment=args.warmup_use_SWAug, use_GPA=True, tgt_files=tgt_files)        
            # data_iter_s = iter(s_dataloader_sourceGPA)
            # checkdataloader(data_iter_s)
            # data_s=next(data_iter_s)

        # data_iter_s = iter(s_dataloader_target_like)
        # data_s=next(data_iter_s)
        # checkdataloader(data_iter_s)  #if you want to see what the data and gt looks like use this

        #######################################################
        ########## dataloader for mutual learning  ############
        #######################################################

        ## use source-GPA as source img
        if args.mutual_use_GPA == True:
            if args.dataset_t=="foggy_cityscape":
                tgt_listpath="/space0/zhaofz/2022/DAOD/data/foggyCityscape/VOC2007/ImageSets/Main/trainval.txt"
                tgt_imgpath="/space0/zhaofz/2022/DAOD/data/foggyCityscape/VOC2007/JPEGImages"
            with open(tgt_listpath) as handle:
                tgt_content=handle.readlines()
            tgt_files=[]
            for fname in tgt_content:
                name = fname.strip()
                tgt_files.append(os.path.join(tgt_imgpath, "%s.jpg" % (name)))

            s_imdb_mutual_GPA, s_train_size_mutual_GPA, s_dataloader_mutual_GPA = get_dataloader_aut(args.imdb_name, args, sequential=False, augment=args.mutual_use_SWAug, use_GPA=True, tgt_files=tgt_files)
            # s_imdb_mutual_GPA, s_train_size_mutual_GPA, s_dataloader_mutual_GPA = get_dataloader_aut(args.imdb_name_fake_target, args, sequential=False, augment=args.mutual_use_SWAug, use_GPA=True, tgt_files=tgt_files)

        if args.mutual_use_targetlike == True:
            s_imdb_mutual_target_like, s_train_size_mutual_target_like , s_dataloader_mutual_target_like = get_dataloader_aut(args.imdb_name_fake_target,args,sequential=False,augment=args.mutual_use_SWAug)
        # else:
        s_imdb, s_train_size, s_dataloader = get_dataloader_aut(args.imdb_name, args, sequential=False, augment=args.mutual_use_SWAug)

    

        ###### data_s[0]:weak_img N*C*H*W | data_s[1]:strong img N*C*H*W
        #      data_s[2]:im_info  N*3     | data_s[3]:gt_boxes_padding N*30*5 
        #      data_s[4]:num_boxes N*1    | data_s[5]: blobs['path'] N tuple



        # target train set use weak + strong aug 
        # t_imdb, t_train_size, t_dataloader = get_dataloader_aut(args.imdb_name_target, args, sequential=False, augment=args.mutual_use_SWAug)
        # data_iter_s = iter(t_dataloader)
        # data_s=next(data_iter_s)
        # checkdataloader(data_iter_s)  #if you want to see what the data and gt looks like use this


        # print("{:d} source roidb entries".format(s_train_size))
        # print("{:d} target roidb entries".format(t_train_size))

        # initilize the tensor holder here.
        im_data_warm = torch.FloatTensor(1)
        im_data_warm_targetlike = torch.FloatTensor(1)
        im_data_warm_sGPA = torch.FloatTensor(1)
        im_data_weak = torch.FloatTensor(1)
        im_data_strong = torch.FloatTensor(1)
        im_info = torch.FloatTensor(1)
        num_boxes = torch.LongTensor(1)
        gt_boxes = torch.FloatTensor(1)
        im_cls_lb = torch.FloatTensor(1)
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
            im_cls_lb = im_cls_lb.cuda()

        # make variable
        im_data_warm = Variable(im_data_warm)
        im_data_warm_targetlike = Variable(im_data_warm_targetlike)
        im_data_warm_sGPA = Variable(im_data_warm_sGPA)
        im_data_weak = Variable(im_data_weak)
        im_data_strong = Variable(im_data_strong)
        im_info = Variable(im_info)
        num_boxes = Variable(num_boxes)
        gt_boxes = Variable(gt_boxes)
        im_cls_lb = Variable(im_cls_lb)

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
            args.start_epoch = student_checkpoint["epoch"]
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
        if args.ef:
            FL = EFocalLoss(class_num=2, gamma=3)
        else:
            FL=FocalLoss(class_num=2, gamma=3)

        if args.use_tfboard:
            from tensorboardX import SummaryWriter

            logger = SummaryWriter("logs-"+exp_setup)

        count_iter = 0
        # conf_gamma = args.conf_gamma
        # pretrained_epoch = args.pretrained_epoch
        warmup_epoch=args.warmup_epoch


        timer = PrettyTimer()
        for epoch in range(args.start_epoch, args.max_epochs + 1):
            student_fasterRCNN.train()
            teacher_fasterRCNN.train()
            loss_temp = 0

            start = time.time()
            epoch_start = time.time()
            if epoch % (args.lr_decay_step + 1) == 0:
                adjust_learning_rate(student_optimizer, args.lr_decay_gamma)
                lr *= args.lr_decay_gamma
            if epoch == warmup_epoch:
                warmup_checkpoints=[]   # store models trained in the warm up phase

                data_iter_warm = iter(s_dataloader_warm)
                if args.warmup_use_targetlike == True:
                    data_iter_warm_targetlike = iter(s_dataloader_target_like)
                if args.warmup_use_GPA == True:
                    data_iter_warm_sGPA = iter(s_dataloader_sourceGPA)

                for step in range(1, iters_per_epoch + 1):
                    count_iter += 1
                    print('warm up')
                    print('step',step)
                    timer.start("iter")
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
                        im_cls_lb.resize_(data_warm[6].size()).copy_(data_warm[6])
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
                    category_loss_cls_s,_,_,_
                    ) = student_fasterRCNN(im_data_warm, im_info, im_cls_lb, gt_boxes, num_boxes)

                    loss = (
                        category_loss_cls_s.mean() +
                        rpn_loss_cls_warm.mean()
                        + rpn_loss_box_warm.mean()
                        + RCNN_loss_cls_warm.mean()
                        + RCNN_loss_bbox_warm.mean()
                        )
                    print('warmup source suploss:',loss.item())
                    loss_temp = loss_temp + loss.item()
                    student_optimizer.zero_grad()
                    loss.backward()
                    if args.net == "vgg16":
                        clip_gradient(student_fasterRCNN, 10.)
                    student_optimizer.step()
                    teacher_fasterRCNN.zero_grad()
                    teacher_optimizer.step(mutual=mutual_flag)

                    if args.warmup_use_targetlike == True:
                        with torch.no_grad():
                            im_data_warm_targetlike.resize_(data_warm_targetlike[0].size()).copy_(data_warm_targetlike[0])
                            im_info.resize_(data_warm_targetlike[2].size()).copy_(data_warm_targetlike[2])
                            gt_boxes.resize_(data_warm_targetlike[3].size()).copy_(data_warm_targetlike[3])
                            num_boxes.resize_(data_warm_targetlike[4].size()).copy_(data_warm_targetlike[4])
                            im_cls_lb.resize_(data_warm_targetlike[6].size()).copy_(data_warm_targetlike[6])
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
                        category_loss_cls_tlike,_,_,_
                        ) = student_fasterRCNN(im_data_warm_targetlike, im_info, im_cls_lb, gt_boxes, num_boxes)

                        loss = ( # +=
                            category_loss_cls_tlike.mean() +
                             rpn_loss_cls_warm_targetlike.mean()
                            + rpn_loss_box_warm_targetlike.mean()
                            + RCNN_loss_cls_warm_targetlike.mean()
                            + RCNN_loss_bbox_warm_targetlike.mean()
                            )
                        print('warm up tlike suploss:',loss.item())
                        loss_temp = loss_temp + loss.item()
                        student_optimizer.zero_grad()
                        loss.backward()
                        if args.net == "vgg16":
                            clip_gradient(student_fasterRCNN, 10.)
                        student_optimizer.step()
                        teacher_fasterRCNN.zero_grad()
                        teacher_optimizer.step(mutual=mutual_flag)

                        ### if use GPA ####
                    if args.warmup_use_GPA == True:
                        with torch.no_grad():
                            im_data_warm_sGPA.resize_(data_warm_sGPA[0].size()).copy_(data_warm_sGPA[0])
                            im_info.resize_(data_warm_sGPA[2].size()).copy_(data_warm_sGPA[2])
                            gt_boxes.resize_(data_warm_sGPA[3].size()).copy_(data_warm_sGPA[3])
                            num_boxes.resize_(data_warm_sGPA[4].size()).copy_(data_warm_sGPA[4])
                            im_cls_lb.resize_(data_warm_sGPA[6].size()).copy_(data_warm_sGPA[6])
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
                        category_loss_cls_GPA,_,_,_
                        ) = student_fasterRCNN(im_data_warm_sGPA, im_info, im_cls_lb, gt_boxes, num_boxes)

                        loss =  ( # +=
                            category_loss_cls_GPA.mean() +
                            rpn_loss_cls_warm_sGPA.mean()
                            + rpn_loss_box_warm_sGPA.mean()
                            + RCNN_loss_cls_warm_sGPA.mean()
                            + RCNN_loss_bbox_warm_sGPA.mean()
                        )
                    # print("category_loss_cls_s:",category_loss_cls_s)
                    # print("category_loss_cls_tlike:",category_loss_cls_tlike)
                    # print("category_loss_cls_GPA:",category_loss_cls_GPA)
                    # print("total category_loss:",category_loss_cls_GPA+category_loss_cls_s+category_loss_cls_tlike)
                    print('warmup GPA suploss:',loss.item())
                    loss_temp = loss_temp + loss.item()
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

                    if epoch == warmup_epoch and step in [iters_per_epoch // 3, iters_per_epoch *2  // 3, iters_per_epoch]:
                        warmup_checkpoints.append(deepcopy(teacher_fasterRCNN))

                    if step == iters_per_epoch:
                        # imdb_target, _, target_dataloader = get_dataloader_aut(args.imdb_name_target, args, sequential=False, augment=args.mutual_use_SWAug)
                        imdbval_name="foggy_cityscape_trainval"
                        select_reliable(imdbval_name,warmup_checkpoints,args,epoch)
                    if step % 1000 ==0:
                        save_checkpoints(args,cfg,epoch,step,model_save_path)
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

                        else:
                            if epoch <= warmup_epoch:
                                suploss_rpn_cls_warm = rpn_loss_cls_warm.item()
                                suploss_rpn_box_warm = rpn_loss_box_warm.item()
                                suploss_rcnn_cls_warm = RCNN_loss_cls_warm.item()
                                suploss_rcnn_box_warm = RCNN_loss_bbox_warm.item()
                                suploss_classify_s = category_loss_cls_s.item()
                                fg_cnt_source_warm = torch.sum(rois_label_warm.data.ne(0))
                                bg_cnt_source_warm = rois_label_warm.data.numel() - fg_cnt_source_warm
                                ## target like ####
                                if args.warmup_use_targetlike == True:
                                    suploss_rpn_cls_warm_targetlike = rpn_loss_cls_warm_targetlike.item()
                                    suploss_rpn_box_warm_targetlike = rpn_loss_box_warm_targetlike.item()
                                    suploss_rcnn_cls_warm_targetlike = RCNN_loss_cls_warm_targetlike.item()
                                    suploss_rcnn_box_warm_targetlike = RCNN_loss_bbox_warm_targetlike.item()
                                    suploss_classify_tlike = category_loss_cls_tlike.item()
                                    fg_cnt_source_warm_targetlike = torch.sum(rois_label_warm_targetlike.data.ne(0))
                                    bg_cnt_source_warm_targetlike = rois_label_warm_targetlike.data.numel() - fg_cnt_source_warm_targetlike
                                ### GPA ### 
                                if args.warmup_use_GPA==True:
                                    suploss_rpn_cls_warm_sGPA = rpn_loss_cls_warm_sGPA.item()
                                    suploss_rpn_box_warm_sGPA = rpn_loss_box_warm_sGPA.item()
                                    suploss_rcnn_cls_warm_sGPA = RCNN_loss_cls_warm_sGPA.item()
                                    suploss_rcnn_box_warm_sGPA = RCNN_loss_bbox_warm_sGPA.item()
                                    suploss_classify_GPA = category_loss_cls_GPA.item()
                                    fg_cnt_source_warm_sGPA = torch.sum(rois_label_warm_sGPA.data.ne(0))
                                    bg_cnt_source_warm_sGPA = rois_label_warm_sGPA.data.numel() - fg_cnt_source_warm_sGPA



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
                                "\t\t\tsuploss_rpn_cls_warm: %.4f, suploss_rpn_box_warm: %.4f, suploss_rcnn_cls_warm: %.4f, suploss_rcnn_box_warm %.4f suploss_classify_s %.4f"
                                % (suploss_rpn_cls_warm, suploss_rpn_box_warm, suploss_rcnn_cls_warm, suploss_rcnn_box_warm, suploss_classify_s)
                            )
                            if args.warmup_use_targetlike == True:
                                ### target like #####
                                print(
                                    "\t\t\tfg_source_warm_targetlike/bg_source_warm_targetlike=(%d/%d), time cost: %f" % (fg_cnt_source_warm_targetlike, bg_cnt_source_warm_targetlike, end - start)
                                )
                                print(
                                    "\t\t\tsuploss_rpn_cls_warm_targetlike: %.4f, suploss_rpn_box_warm_targetlike: %.4f, suploss_rcnn_cls_warm_targetlike: %.4f, suploss_rcnn_box_warm_targetlike %.4f suploss_classify_tlike %.4f"
                                    % (suploss_rpn_cls_warm_targetlike, suploss_rpn_box_warm_targetlike, suploss_rcnn_cls_warm_targetlike, suploss_rcnn_box_warm_targetlike, suploss_classify_tlike)
                                )
                            ### GPA ###
                            if args.warmup_use_GPA:
                                print(
                                "\t\t\tfg_source_warm_sGPA/bg_source_warm_sGPA=(%d/%d), time cost: %f" % (fg_cnt_source_warm_sGPA, bg_cnt_source_warm_sGPA, end - start)
                                )
                                print(
                                    "\t\t\tsuploss_rpn_cls_warm_sGPA: %.4f, suploss_rpn_box_warm_sGPA: %.4f, suploss_rcnn_cls_warm_sGPA: %.4f, suploss_rcnn_box_warm_sGPA %.4f suploss_classify_GPA %.4f"
                                    % (suploss_rpn_cls_warm_sGPA, suploss_rpn_box_warm_sGPA, suploss_rcnn_cls_warm_sGPA, suploss_rcnn_box_warm_sGPA,suploss_classify_GPA)
                                )


                        if args.use_tfboard:
                            info = {
                                "loss": loss_temp,
                                ### origin source ###
                                "suploss_rpn_cls_warm": suploss_rpn_cls_warm if epoch <= warmup_epoch else 0,
                                "suploss_rpn_box_warm": suploss_rpn_box_warm if epoch <= warmup_epoch else 0,
                                "suploss_rcnn_cls_warm": suploss_rcnn_cls_warm if epoch <= warmup_epoch else 0,
                                "suploss_rcnn_box_warm": suploss_rcnn_box_warm if epoch <= warmup_epoch else 0,
                                "suploss_classify_s": suploss_classify_s if epoch <= warmup_epoch and args.warmup_use_targetlike==True else 0,
                                ### target like source ###
                                "suploss_rpn_cls_warm_targetlike": suploss_rpn_cls_warm_targetlike if epoch <= warmup_epoch and args.warmup_use_targetlike==True else 0,
                                "suploss_rpn_box_warm_targetlike": suploss_rpn_box_warm_targetlike if epoch <= warmup_epoch and args.warmup_use_targetlike==True else 0,
                                "suploss_rcnn_cls_warm_targetlike": suploss_rcnn_cls_warm_targetlike if epoch <= warmup_epoch and args.warmup_use_targetlike==True else 0,
                                "suploss_rcnn_box_warm_targetlike": suploss_rcnn_box_warm_targetlike if epoch <= warmup_epoch and args.warmup_use_targetlike==True else 0,
                                "suploss_classify_tlike": suploss_classify_tlike if epoch <= warmup_epoch and args.warmup_use_targetlike==True else 0,
                                ### GPA source###
                                "suploss_rpn_cls_warm_sGPA": suploss_rpn_cls_warm_sGPA if epoch <= warmup_epoch and args.warmup_use_GPA == True else 0,
                                "suploss_rpn_box_warm_sGPA": suploss_rpn_box_warm_sGPA if epoch <= warmup_epoch and args.warmup_use_GPA == True else 0,
                                "suploss_rcnn_cls_warm_sGPA": suploss_rcnn_cls_warm_sGPA if epoch <= warmup_epoch and args.warmup_use_GPA == True else 0,
                                "suploss_rcnn_box_warm_sGPA": suploss_rcnn_box_warm_sGPA if epoch <= warmup_epoch and args.warmup_use_GPA == True else 0,
                                "suploss_classify_GPA": suploss_classify_GPA if epoch <= warmup_epoch and args.warmup_use_targetlike==True else 0,

                                # "suploss_rpn_cls": suploss_rpn_cls if epoch > warmup_epoch else 0,
                                # "suploss_rpn_box": suploss_rpn_box if epoch > warmup_epoch else 0,
                                # "suploss_rcnn_cls": suploss_rcnn_cls if epoch > warmup_epoch else 0,
                                # "suploss_rcnn_box": suploss_rcnn_box if epoch > warmup_epoch else 0,
                                # "unsuploss_rpn_cls_target": unsuploss_rpn_cls_target if epoch > warmup_epoch else 0,
                                # "unsuploss_rpn_box_target": unsuploss_rpn_box_target if epoch > warmup_epoch else 0,
                                # "unsuploss_rcnn_cls_target": unsuploss_rcnn_cls_target if epoch > warmup_epoch else 0,
                                # "unsuploss_rcnn_box_target": unsuploss_rcnn_box_target if epoch > warmup_epoch else 0,
                                # "pseudo_bbox_num": pseudo_bbox_num if epoch > warmup_epoch else 0,
                            }
                            logger.add_scalars(
                                "logs_s_{}/losses".format(args.session),
                                info,
                                (epoch - 1) * iters_per_epoch + step,
                            )

                        loss_temp = 0

                        start = time.time()

            elif epoch > warmup_epoch and args.enable_unsup==True:
                mutual_checkpoints=[]
                loss_temp = 0
                for step in range(1, iters_per_epoch + 1):
                    loss=0
                    print('epoch',epoch)
                    print('step',step)
                    print('mutual learning')
                    timer.start("iter")
                    # torch.cuda.empty_cache()
                    mutual_flag=True
                    count_iter += 1

                    #### prepare target dataset data(reliable) for mutual learning phase ####
                    if step == 1:
                        t_imdb, t_train_size, t_dataloader = get_dataloader_aut('foggy_cityscape_reliable', args, sequential=True, augment=args.mutual_use_SWAug)
                        data_iter_t = iter(t_dataloader)
                        
                    #### prepare target dataset data(unreliable) for mutual learning phase ####
                    if step == iters_per_epoch // 2:
                        t_imdb, t_train_size, t_dataloader = get_dataloader_aut('foggy_cityscape_unreliable', args, sequential=True, augment=args.mutual_use_SWAug)
                        data_iter_t = iter(t_dataloader)
                        save_checkpoints(args,cfg,epoch,step,model_save_path)
                    if step % 1000 ==0:
                        save_checkpoints(args,cfg,epoch,step,model_save_path)
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
                            im_cls_lb.resize_(data_s_mutual_targetlike[6].size()).copy_(data_s_mutual_targetlike[6])

                        (
                            rois_s_s, # s_s:strong source
                            cls_prob_s_s,
                            bbox_pred_s_s,
                            rpn_loss_cls_s_s,
                            rpn_loss_box_s_s,
                            RCNN_loss_cls_s_s,
                            RCNN_loss_bbox_s_s,
                            rois_label_s_s,
                            category_loss_cls_mtlike,
                            out_d_pixel,
                            out_d,
                            advloss_s_ins
                        ) = student_fasterRCNN(im_data_strong, im_info,im_cls_lb, gt_boxes, num_boxes)
                        loss =  (
                            category_loss_cls_mtlike.mean()
                            + rpn_loss_cls_s_s.mean()
                            + rpn_loss_box_s_s.mean()
                            + RCNN_loss_cls_s_s.mean()
                            + RCNN_loss_bbox_s_s.mean()
                        )

                        domain_s = Variable(torch.zeros(out_d.size(0)).long().cuda()) # domain_predict_source.size(0): batch_size
                        # print('domain_predict_source.size(0)',domain_predict_source.size(0))
                        # print("domain_s:",domain_s)

                        ### adv loss for source sample ####

                        ### pixel adv loss ###
                        advloss_s_p = 0.5 * torch.mean(out_d_pixel ** 2)

                        # ### mid advloss ###
                        # domain_s_mid = Variable(torch.zeros(out_d_mid.size(0)).long().cuda())
                        # advloss_s_mid = 0.5 * F.cross_entropy(out_d_mid, domain_s_mid)

                        ### image level adv ###
                        #                         [1, 2]             1
                        advloss_s_image = 0.5 * FL(out_d, domain_s)

                        adv_loss=advloss_s_p + advloss_s_image + advloss_s_ins 
                        loss = loss + adv_loss * args.adv_loss_weight
                        
                        print('tlike total loss:',loss.item())
                        loss_temp = loss_temp+ loss.item()
                        student_optimizer.zero_grad()
                        loss.backward()
                        if args.net == "vgg16":
                            clip_gradient(student_fasterRCNN, 10.)
                        student_optimizer.step()
                        teacher_fasterRCNN.zero_grad()
                        teacher_optimizer.step(mutual=mutual_flag)
                        # ### inst adv loss ###
                        # domain_gt_ins = Variable(torch.zeros(out_d_ins.size(0)).long().cuda())
                        # advloss_s_ins = 0.5 * FL(out_d_ins, domain_gt_ins)

                    # print('domain_predict_source.shape',domain_predict_source.shape)
                    # print('advloss_source',advloss_s.item())

                    if args.mutual_use_GPA== True:

                        with torch.no_grad():
                            im_data_strong.resize_(data_s_mutual_GPA[1].size()).copy_(data_s_mutual_GPA[1])
                            im_info.resize_(data_s_mutual_GPA[2].size()).copy_(data_s_mutual_GPA[2])
                            gt_boxes.resize_(data_s_mutual_GPA[3].size()).copy_(data_s_mutual_GPA[3])
                            num_boxes.resize_(data_s_mutual_GPA[4].size()).copy_(data_s_mutual_GPA[4])
                            im_cls_lb.resize_(data_s_mutual_GPA[6].size()).copy_(data_s_mutual_GPA[6])
                        (
                            rois_s_s, # s_s:strong source
                            cls_prob_s_s,
                            bbox_pred_s_s,
                            rpn_loss_cls_s_s,
                            rpn_loss_box_s_s,
                            RCNN_loss_cls_s_s,
                            RCNN_loss_bbox_s_s,
                            rois_label_s_s,
                            category_loss_cls_mGPA,
                            out_d_pixel,
                            out_d,
                            advloss_s_ins
                        ) = student_fasterRCNN(im_data_strong, im_info,im_cls_lb, gt_boxes, num_boxes)
                        loss =  (
                            category_loss_cls_mGPA.mean()
                            + rpn_loss_cls_s_s.mean()
                            + rpn_loss_box_s_s.mean()
                            + RCNN_loss_cls_s_s.mean()
                            + RCNN_loss_bbox_s_s.mean()
                        )
                        domain_s = Variable(torch.zeros(out_d.size(0)).long().cuda()) # domain_predict_source.size(0): batch_size
                        # print('domain_predict_source.size(0)',domain_predict_source.size(0))
                        # print("domain_s:",domain_s)

                        ### adv loss for source sample ####

                        ### pixel adv loss ###
                        advloss_s_p = 0.5 * torch.mean(out_d_pixel ** 2)

                        # ### mid advloss ###
                        # domain_s_mid = Variable(torch.zeros(out_d_mid.size(0)).long().cuda())
                        # advloss_s_mid = 0.5 * F.cross_entropy(out_d_mid, domain_s_mid)

                        ### image level adv ###
                        #                         [1, 2]             1
                        advloss_s_image = 0.5 * FL(out_d, domain_s)

                        adv_loss=advloss_s_p + advloss_s_image + advloss_s_ins 
                        loss = loss + adv_loss * args.adv_loss_weight

                        print('GPA total loss:',loss.item())
                        loss_temp = loss_temp+ loss.item()
                        student_optimizer.zero_grad()
                        loss.backward()
                        if args.net == "vgg16":
                            clip_gradient(student_fasterRCNN, 10.)
                        student_optimizer.step()
                        teacher_fasterRCNN.zero_grad()
                        teacher_optimizer.step(mutual=mutual_flag)
                        # ### inst adv loss ###
                        # domain_gt_ins = Variable(torch.zeros(out_d_ins.size(0)).long().cuda())
                        # advloss_s_ins = 0.5 * FL(out_d_ins, domain_gt_ins)

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
                        im_cls_lb.resize_(data_t[6].size()).copy_(data_t[6])
                        
                        # print("im_data_weak.shape:",im_data_weak.shape)
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
                        category_loss_cls_mutual_weakaug,
                        _,
                        _,
                        _
                    ) = teacher_fasterRCNN(im_data_weak, im_info,im_cls_lb, gt_boxes, num_boxes, test=True,target_eval=True)

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
                        try:
                            pred_boxes = bbox_transform_inv(boxes, box_deltas, args.batch_size)
                        except:
                            print(data_t[5])

                        pred_boxes = clip_boxes(pred_boxes, im_info.data, args.batch_size)
                    else:
                        # Simply repeat the boxes, once for each class
                        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

                    # for batchsize>1 do not squeeze()
                    # scores = scores.squeeze()  #[2, 300, 9]
                    # pred_boxes = pred_boxes.squeeze() #[2, 300, 36]

                    

                    gt_boxes_target = [[] for i in range(args.batch_size)]
                    pre_thresh = 0.0
                    thresh = args.threshold #threshold for pseudo label default 0.8
                    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
                    for idx in range(args.batch_size):
                        for j in range(1, len(imdb.classes)):

                            inds = torch.nonzero(scores[idx, :, j] > pre_thresh).view(-1) #[300]
                            
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
                                ## set 30 other than 10
                                for i in range(np.minimum(30, cls_dets_numpy.shape[0])):   #Q:limit the number within 10??  A:per class pseudo_box is limited within 10
                                    bbox = tuple(
                                        int(np.round(x)) for x in cls_dets_numpy[i, :4]
                                    )
                                    score = cls_dets_numpy[i, -1]
                                    if score > thresh:                                    # select proposals with confidence_score > 0.8 as pseudo label for distillation loss
                                        gt_boxes_target[idx].append(list(bbox[0:4]) + [j])
                    # print('real_gt:',data_t[3])
                    # print('image:',data_t[-1])
                    # print('gt_boxes_target:',gt_boxes_target)
                    pseudo_bbox_num=[len(gt_boxes_target[q]) for q in range(args.batch_size)] #len(gt_boxes_target) 
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
                        im_cls_lb.resize_(data_t[6].size()).copy_(data_t[6])
                    (
                        rois_s_t, # s_t: strong aug target
                        cls_prob_s_t,
                        bbox_pred_s_t,
                        rpn_loss_cls_s_t,
                        rpn_loss_box_s_t,
                        RCNN_loss_cls_s_t,
                        RCNN_loss_bbox_s_t,
                        rois_label_s_t,
                        category_loss_cls_mutual_strongaug,
                        out_d_pixel,
                        out_d,
                        advloss_t_ins
                    ) = student_fasterRCNN(im_data_strong, im_info, im_cls_lb, gt_boxes, num_boxes,target_train=True)

                    ### EXP: whether set localization loss for mutual target learning ==0 date:0612 ###
                    rpn_loss_box_s_t = rpn_loss_box_s_t * 0
                    RCNN_loss_bbox_s_t = RCNN_loss_bbox_s_t * 0


                    ### adv loss for target sample ####
                    ### pixel level adv loss ###
                    advloss_t_p = 0.5 * torch.mean((1 - out_d_pixel) ** 2)

                    # ### mid level advloss ###
                    # domain_t_mid = Variable(torch.ones(out_d_mid.size(0)).long().cuda())
                    # advloss_t_mid = 0.5 * F.cross_entropy(out_d_mid, domain_t_mid)

                    ### image level adv loss ###
                    domain_t = Variable(torch.ones(out_d.size(0)).long().cuda())
                    advloss_t_image = 0.5 * FL(out_d, domain_t)

                    # ### inst level adv loss ###
                    # domain_gt_ins = Variable(torch.ones(out_d_ins.size(0)).long().cuda())
                    # advloss_t_ins = 0.5 * FL(out_d_ins, domain_gt_ins)



                    # print('domain_predict_target',domain_predict_target)
                    # print('advloss_target',advloss_t.item())

                    unsup_loss=(
                        rpn_loss_cls_s_t.mean()
                        + rpn_loss_box_s_t.mean()
                        + RCNN_loss_cls_s_t.mean()
                        + RCNN_loss_bbox_s_t.mean()
                    )

                    loss = args.unsup_loss_weight * unsup_loss
                    print("targetimg unsup_loss:",unsup_loss)

                    # adv_loss= advloss_s_p + advloss_t_p + advloss_s_mid  +advloss_t_mid  + advloss_s_image + advloss_t_image + advloss_s_ins  + advloss_t_ins 
                    adv_loss= advloss_t_p  + advloss_t_image + advloss_t_ins 
                    print('target adv_loss:',adv_loss.item())
                    # print((epoch - 1) * iters_per_epoch + step)

                    loss = loss + adv_loss * args.adv_loss_weight

                    if args.use_tfboard:
                        info = {
                            "mean pseudo_bbox_num": mean_pseudo_bbox_num if epoch > warmup_epoch else 0,
                            "adv_loss": adv_loss if epoch > warmup_epoch else 0,
                            "adv_loss_s_p": advloss_s_p.item() if epoch > warmup_epoch else 0,
                            "adv_loss_t_p": advloss_t_p.item() if epoch > warmup_epoch else 0,
                            # "adv_loss_s_mid": advloss_s_mid.item() if epoch > warmup_epoch else 0,
                            # "adv_loss_t_mid": advloss_t_mid.item() if epoch > warmup_epoch else 0,
                            "adv_loss_s_image": advloss_s_image.item() if epoch > warmup_epoch else 0,
                            "adv_loss_t_image": advloss_t_image.item() if epoch > warmup_epoch else 0,
                            "adv_loss_s_inst": advloss_s_ins.item() if epoch > warmup_epoch else 0,
                            "adv_loss_t_inst": advloss_t_ins.item() if epoch > warmup_epoch else 0,
                            # "category_loss_cls_mutual_weakaug": category_loss_cls_mutual_weakaug.item() if epoch > warmup_epoch else 0,
                            # "category_loss_cls_mutual_strongaug": category_loss_cls_mutual_strongaug.item() if epoch > warmup_epoch else 0
                        } 
                        logger.add_scalars(
                            "logs_s_{}/losses".format(args.session),
                            info,
                            (epoch - 1) * iters_per_epoch + step,
                        )


                    print('target-mutual total loss:',loss.item())
                    loss_temp = loss_temp+ loss.item()
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

                    if epoch > warmup_epoch and step in [iters_per_epoch // 3, iters_per_epoch * 2  // 3, iters_per_epoch]:
                        mutual_checkpoints.append(deepcopy(teacher_fasterRCNN))

                    if step == iters_per_epoch:
                        # imdb_target, _, target_dataloader = get_dataloader_aut(args.imdb_name_target, args, sequential=False, augment=args.mutual_use_SWAug)
                        imdbval_name="foggy_cityscape_trainval"
                        select_reliable(imdbval_name,mutual_checkpoints,args,epoch)


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

                        else:
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

            save_checkpoints(args,cfg,epoch,step,model_save_path)
            epoch_end = time.time()
            print("epoch cost time: {} min".format((epoch_end - epoch_start) / 60.0))

        if args.use_tfboard:
            logger.close()
