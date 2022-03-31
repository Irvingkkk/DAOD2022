# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
# from scipy.misc import imread
import imageio
imsave = imageio.imsave
from imageio import imread
# from scipy.misc import imsave
from model.utils.config import cfg
from model.utils.blob import prep_im_for_blob, im_list_to_blob
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from torchvision import transforms as T
from model.utils.augmentation_impl import GaussianBlur
import cv2
import os
import random


def get_minibatch(roidb, num_classes, seg_return=False, augment=False, seed=2020):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)
    assert (cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
            format(num_images, cfg.TRAIN.BATCH_SIZE)

    # Get the input image blob, formatted for caffe
    im_blob, im_scales, gt_boxes = _get_image_blob(roidb, random_scale_inds, augment=augment, seed=seed)

    assert len(im_scales) == 1, "Single batch only"

    blobs = {'data': im_blob}
    blobs['gt_boxes'] = gt_boxes
    blobs['im_info'] = np.array(
        [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
        dtype=np.float32)
    if seg_return:
        blobs['seg_map'] = roidb[0]['seg_map']
    blobs['img_id'] = roidb[0]['img_id']
    blobs['path'] = roidb[0]['image']

    return blobs

def get_minibatch_aut(roidb, num_classes, seg_return=False, seed=2020, augment=False, use_GPA=False, tgt_files= None):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)  #1
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)
    assert (cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
            format(num_images, cfg.TRAIN.BATCH_SIZE)

    # Get the input image blob, formatted for caffe
    im_blob, im_scales, gt_boxes = _get_image_blob_aut(roidb, random_scale_inds, seed=seed,augment=augment,use_GPA=use_GPA, tgt_files=tgt_files)

    assert len(im_scales) == 1, "Single batch only"

    blobs = {'data': im_blob}   # weak:0 strong:1
    blobs['gt_boxes'] = gt_boxes
    blobs['im_info'] = np.array(
        [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],   #height width im_scale(0.58594 for city)
        dtype=np.float32)
    if seg_return:
        blobs['seg_map'] = roidb[0]['seg_map']
    blobs['img_id'] = roidb[0]['img_id']
    blobs['path'] = roidb[0]['image']

    return blobs

# def bbs2numpy(bbs):
#     bboxes = []
#     for bb in bbs.bounding_boxes:
#         x1 = bb.x1 - 1
#         y1 = bb.y1 - 1
#         w = bb.x2 - bb.x1
#         h = bb.y2 - bb.y2
#         label = float(bb.label)
#         bboxes.append([x1, y1, w, h, label])
#     return np.array(bboxes, dtype=np.float32)

def bbs2numpy(bbs):
    bboxes = []
    for bb in bbs.bounding_boxes:
        x1 = bb.x1 - 1
        y1 = bb.y1 - 1
        x2 = bb.x2 - 1
        y2 = bb.y2 - 1
        label = float(bb.label)
        bboxes.append([x1, y1, x2, y2, label])
    return np.array(bboxes, dtype=np.float32)

def _get_image_blob(roidb, scale_inds, augment=False, seed=2020):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    assert len(roidb) == 1, "Single batch only"

    # gt boxes: (x1, y1, x2, y2, cls)
    if cfg.TRAIN.USE_ALL_GT:
        # Include all ground truth boxes
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
    else:
        # For the COCO ground truth boxes, exclude the ones that are ''iscrowd''
        gt_inds = np.where((roidb[0]['gt_classes'] != 0) & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
    gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
    gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :]
    # gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
    gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]

    num_images = len(roidb)

    processed_ims = []
    im_scales = []
    for i in range(num_images):
        # im = cv2.imread(roidb[i]['image'])
        im = imread(roidb[i]['image'])
        # print(roidb[i]['image'])
        if len(im.shape) == 2:
            im = im[:, :, np.newaxis]
            im = np.concatenate((im, im, im), axis=2)
        # flip the channel, since the original one using cv2
        # rgb -> bgr
        im = im[:, :, ::-1]
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        # data augmentation
        if augment:
            im, gt_boxes = augmentor(im, gt_boxes, seed=seed)
        # imsave("target_aug.jpg", im[:, :, ::-1])
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale)
        processed_ims.append(im)
        gt_boxes[:, 0:4] = gt_boxes[:, 0:4] * im_scale


    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales, gt_boxes

def _get_image_blob_aut(roidb, scale_inds, seed=2020, augment=False, use_GPA=False,tgt_files=None):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    assert len(roidb) == 1, "Single batch only"

    # gt boxes: (x1, y1, x2, y2, cls)
    if cfg.TRAIN.USE_ALL_GT:
        # Include all ground truth boxes
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
    else:
        # For the COCO ground truth boxes, exclude the ones that are ''iscrowd''
        gt_inds = np.where((roidb[0]['gt_classes'] != 0) & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
    gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
    gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :]
    # gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
    gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
    gt_boxes_backup=gt_boxes

    num_images = len(roidb)

    processed_ims = []
    im_scales = []
    for i in range(num_images):
        # im = cv2.imread(roidb[i]['image'])
        im = imread(roidb[i]['image'])

        img_name=roidb[i]['image'].split('/')[-1]

        # print(roidb[i]['image'])
        if len(im.shape) == 2:
            im = im[:, :, np.newaxis]
            im = np.concatenate((im, im, im), axis=2)
        # flip the channel, since the original one using cv2
        # rgb -> bgr
        im = im[:, :, ::-1]
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]

        ### apply GPA to align source_img ####  im:source tgt_image:target image
        if use_GPA:
            tgt_image_name = random.choice(tgt_files)
            # print("choose target reference:",tgt_image_name)
            tgt_image = cv2.imread(tgt_image_name) #BGR order 
            source_image=im
            aligned, _ = match_histograms_hybrid(src=source_image, tgt=tgt_image)
            # cv2.imwrite(img_name,aligned)
            im = aligned
            

        # data augmentation -- first apply weak to get a weak-aug img then apply strong to get strong-aug img
        # get two type aug img here
        # im, gt_boxes = weak_augmentor(im, gt_boxes, seed=seed)

        
            
        # different aug has same gt_boxes
        if augment == True:
            im_weak, im_strong, gt_boxes= get_twotype_augmentor(im, gt_boxes, seed=seed, img_name=img_name)
            if len(gt_boxes)==0:
                gt_boxes=np.array([[0,0,0,0,0]], dtype=np.float64)
        else:
            im_weak = im
            im_strong = im 

        # imsave("target_aug.jpg", im[:, :, ::-1])
        # imsave("target_aug2_{}".format(img_name),gt_boxes.draw_on_image(im[:, :, ::-1], size=2))
        # imsave('before_mean_weak{}'.format(img_name),im_weak[:, :, ::-1])
        # imsave('before_mean_strong{}'.format(img_name),im_strong[:, :, ::-1])

        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im_weak, im_scale_weak = prep_im_for_blob(im_weak, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)
        
        im_strong, im_scale_strong = prep_im_for_blob(im_strong, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)
                                        
        # cv2.imwrite("aaaa_{}".format(img_name),im_strong)
        # imsave('after_mean_strong{}'.format(img_name),im_strong[:, :, ::-1])


        # imsave('after_mean_weak{}'.format(img_name),im_weak[:, :, ::-1])
        # imsave('after_mean_strong{}'.format(img_name),im_strong[:, :, ::-1])
        

        im_scales.append(im_scale_weak) # im_scale: 600/1024 for city
        processed_ims.append(im_weak)
        processed_ims.append(im_strong)
        
        try:
            gt_boxes[:, 0:4] = gt_boxes[:, 0:4] * im_scale_weak # adjust bbox
        except:
            print(img_name)
            print('box before:',gt_boxes_backup)
            print('box after:',gt_boxes)
            print("roidb[0]['boxes']",roidb[0]['boxes'])



    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales, gt_boxes

def augmentor(image, bounding_boxes, seed=2020):

    # ia.seed(seed)
    seed=seed.item()
    # print('-------------------seed type:',type(seed))
    ia.seed(int(seed))
    bbxes = []
    # for gt_box in bounding_boxes:
    #     x1 = gt_box[0] + 1
    #     y1 = gt_box[1] + 1
    #     x2 = gt_box[2] + x1
    #     y2 = gt_box[3] + y1
    #     bbxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=str(gt_box[4])))
    for gt_box in bounding_boxes:
        x1 = gt_box[0] + 1
        y1 = gt_box[1] + 1
        x2 = gt_box[2] + 1
        y2 = gt_box[3] + 1
        bbxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=str(gt_box[4])))
    bbs = BoundingBoxesOnImage(bbxes, shape=image.shape)
    seq = iaa.Sequential([
        iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )
    ])
    image, bbs_aug = seq(image=image, bounding_boxes=bbs)
    jitter_param = 0.4
    transform = T.Compose([
        T.ToPILImage(),
        T.ColorJitter(brightness=jitter_param, contrast=jitter_param, saturation=jitter_param),
    ])
    image_pil = transform(image)
    image = np.array(image_pil)
    bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()
    gt_boxes = bbs2numpy(bbs_aug)
    return image, gt_boxes

def weak_augmentor(image, bounding_boxes, seed=2020):

    seed=seed.item()
    # print('-------------------seed type:',type(seed))
    ia.seed(int(seed))
    bbxes = []
    for gt_box in bounding_boxes:
        x1 = gt_box[0] + 1
        y1 = gt_box[1] + 1
        x2 = gt_box[2] + 1
        y2 = gt_box[3] + 1
        bbxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=str(gt_box[4])))
    bbs = BoundingBoxesOnImage(bbxes, shape=image.shape)
    # random_horizontal_flip + random Crop
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Crop(percent=(0,0.1))
    ])
    image, bbs_aug = seq(image=image, bounding_boxes=bbs)
    bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()
    gt_boxes = bbs2numpy(bbs_aug)
    return image, gt_boxes

def get_twotype_augmentor(image, bounding_boxes, seed=2020, img_name=''):

    # ia.seed(seed)
    # seed=seed.item()
    # print('-------------------seed:',seed)
    seed = np.random.randint(4000)
    ia.seed(int(seed))
    bbxes = []
    for gt_box in bounding_boxes:
        x1 = gt_box[0] + 1
        y1 = gt_box[1] + 1
        x2 = gt_box[2] + 1
        y2 = gt_box[3] + 1
        bbxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=str(gt_box[4])))
    bbs = BoundingBoxesOnImage(bbxes, shape=image.shape)
    # 
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Crop(percent=(0,0.1))
    ])
    image_weak, bbs_weak = seq(image=image, bounding_boxes=bbs)
    image_weak = np.array(image_weak)
    bbs_aug = bbs_weak.remove_out_of_image().clip_out_of_image()

    gt_boxes = bbs2numpy(bbs_aug)

    augmentation = []
    augmentation.append(T.ToPILImage())
    augmentation.append(T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8))
    augmentation.append(T.RandomGrayscale(p=0.2))
    augmentation.append(T.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5))

    randcrop_transform = T.Compose(
            [
                T.ToTensor(),
                T.RandomErasing(
                    p=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3), value="random" #scale=(0.05, 0.2)
                ),
                T.RandomErasing(
                    p=0.5, scale=(0.02, 0.2), ratio=(0.1, 6), value="random" #scale=(0.02, 0.2)
                ),
                T.RandomErasing(
                    p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8), value="random" #scale=(0.02, 0.2)
                ),
                T.ToPILImage(),
            ]
        )
    augmentation.append(randcrop_transform)
    strong_transform=T.Compose(augmentation)
    
    image_pil_strong = strong_transform(image_weak)
    image_strong = np.array(image_pil_strong)

    # save the aug images for visulization
    # cv2.imwrite("weak_aug_{}".format(img_name),bbs_aug.draw_on_image(image_weak, size=2))
    # # imsave("weak_aug_{}".format(img_name),bbs_aug.draw_on_image(image_weak[:, :, ::-1], size=2))
    # print('save_weak',img_name)
    # # imsave("strong_aug_{}".format(img_name),bbs_aug.draw_on_image(image_strong[:, :, ::-1], size=2))
    # cv2.imwrite("strong_aug_{}".format(img_name),bbs_aug.draw_on_image(image_strong, size=2))
    # print('save stromg',img_name)
    
    return image_weak, image_strong, gt_boxes
  

def match_histograms_hybrid(src, tgt, bins=256, regularizer=0.01, max_iters = 100, eps=1e-3, step=0.1):
    src = np.array(src)
    tgt = np.array(tgt)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    tgt = cv2.cvtColor(tgt, cv2.COLOR_BGR2LAB)
    src_L, _ = np.histogram(src[:, :, 0], bins=bins, range=[0, 255], density=True)
    src_a, _ = np.histogram(src[:, :, 1], bins=bins, range=[0, 255], density=True)
    src_b, _ = np.histogram(src[:, :, 2], bins=bins, range=[0, 255], density=True)
    tgt_L, _ = np.histogram(tgt[:, :, 0], bins=bins, range=[0, 255], density=True)
    tgt_a, _ = np.histogram(tgt[:, :, 1], bins=bins, range=[0, 255], density=True)
    tgt_b, _ = np.histogram(tgt[:, :, 2], bins=bins, range=[0, 255], density=True)

    src_transform = np.zeros_like(src)

    _, _, n_channels = src.shape

    # gamma correction for L channel
    gamma_list = []
    gamma = 1
    for iter_num in range(max_iters):
        # f = calculate_f(gamma, src_hist, ref_hist, regularizer)
        def calculate_df(gamma, p1, p2, regularizer):
            assert len(p1) == len(p2), 'Length of histograms not match'
            bins = len(p1)
            x = (np.arange(bins) + 1.0)/bins # (1,256) [1 2 3 ... 256]/256
            temp_1 = np.dot(x**gamma, p1) - np.dot(x, p2)
            temp_2 = x**gamma
            temp_2 = np.sum(np.log(x)*temp_2*p1)
            return 2*temp_1*temp_2 + 2*regularizer*(gamma-1.0)
        df = calculate_df(gamma, src_L, tgt_L, regularizer)
        if np.abs(df) <= eps:
            break
        else:
            gamma = gamma - step * df
    if iter_num == max_iters:
        print('Warining! Gamma not converged!')
    gamma_list.append(gamma)
    src_transform[:, :, 0] = (src[:, :, 0]/255.0)**gamma*255.0

    # scale lightness channel
    trasformed_L, _ = np.histogram(src_transform[:, :, 0], bins=bins, range=[0, 255], density=True)
    def calculate_alpha(p1, p2, regularizer=0.0):
        assert len(p1) == len(p2)
        bins = len(p1)
        x = (np.arange(bins) + 1.0) / bins
        mean_p1 = np.dot(x, p1)
        mean_p2 = np.dot(x, p2)
        var1 = np.dot(p1, (x - mean_p1) ** 2)
        var2 = np.dot(p2, (x - mean_p2) ** 2)
        return np.sqrt(var2 / var1), mean_p1
    scale_factor, transformed_mean = calculate_alpha(trasformed_L, tgt_L)
    scaled_L = (scale_factor * (src_transform[:, :, 0] / 255.0 - transformed_mean) + transformed_mean) * 255.0
    src_transform[:, :, 0] = np.uint8(np.clip(scaled_L, a_min=0.0, a_max=255.0))

    # color histogram specification for a* and b* channel
    src_cdf_a = calculate_cdf(src_a)
    src_cdf_b = calculate_cdf(src_b)
    ref_cdf_a = calculate_cdf(tgt_a)
    ref_cdf_b = calculate_cdf(tgt_b)
    a_lookup_table = calculate_lookup(src_cdf_a, ref_cdf_a)
    b_lookup_table = calculate_lookup(src_cdf_b, ref_cdf_b)
    src_transform[:, :, 1] = cv2.LUT(src[:, :, 1], a_lookup_table)
    src_transform[:, :, 2] = cv2.LUT(src[:, :, 2], b_lookup_table)
    src_transform = src_transform.astype(np.uint8)
    src_transform = cv2.cvtColor(src_transform, cv2.COLOR_LAB2BGR)

    return src_transform, gamma_list


def calculate_cdf(histogram):
    """
    This method calculates the cumulative distribution function
    :param array histogram: The values of the histogram
    :return: normalized_cdf: The normalized cumulative distribution function
    :rtype: array
    """
    # Get the cumulative sum of the elements
    cdf = histogram.cumsum()

    # Normalize the cdf
    normalized_cdf = cdf / float(cdf.max())

    return normalized_cdf

def calculate_lookup(src_cdf, ref_cdf):
    """
    This method creates the lookup table
    :param array src_cdf: The cdf for the source image
    :param array ref_cdf: The cdf for the reference image
    :return: lookup_table: The lookup table
    :rtype: array
    """
    lookup_table = np.zeros(256)
    lookup_val = 0
    for src_pixel_val in range(len(src_cdf)):
        for ref_pixel_val in range(len(ref_cdf)):
            if ref_cdf[ref_pixel_val] >= src_cdf[src_pixel_val]:
                lookup_val = ref_pixel_val
                break
        lookup_table[src_pixel_val] = lookup_val
    return lookup_table