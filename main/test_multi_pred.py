#!/usr/bin/env python
from base.utilities import get_parser, get_logger


#!/usr/bin/env python
import os
import pdb

import cv2
import torch
import numpy as np
import librosa
import pickle

from transformers import Wav2Vec2Processor
from base.utilities import get_parser
from models import get_model
from base.baseTrainer import load_state_dict

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
cfg = get_parser()

import tempfile
from subprocess import call

os.environ['PYOPENGL_PLATFORM'] = 'osmesa'  # egl
# os.environ['PYOPENGL_PLATFORM'] = 'egl'

cfg = get_parser()
#os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in cfg.test_gpu)

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

mouth_map = np.array([
    1576, 1577, 1578, 1579, 1580, 1581, 1582, 1583, 1590, 1590, 1591, 1593, 1593,
    1657, 1658, 1661, 1662, 1663, 1667, 1668, 1669, 1670, 1686, 1687, 1691, 1693,
    1694, 1695, 1696, 1697, 1700, 1702, 1703, 1704, 1709, 1710, 1711, 1712, 1713,
    1714, 1715, 1716, 1717, 1718, 1719, 1720, 1721, 1722, 1723, 1728, 1729, 1730,
    1731, 1732, 1733, 1734, 1735, 1736, 1737, 1738, 1740, 1743, 1748, 1749, 1750,
    1751, 1758, 1763, 1765, 1770, 1771, 1773, 1774, 1775, 1776, 1777, 1778, 1779,
    1780, 1781, 1782, 1787, 1788, 1789, 1791, 1792, 1793, 1794, 1795, 1796, 1801,
    1802, 1803, 1804, 1826, 1827, 1836, 1846, 1847, 1848, 1849, 1850, 1865, 1866,
    2712, 2713, 2714, 2715, 2716, 2717, 2718, 2719, 2726, 2726, 2727, 2729, 2729,
    2774, 2775, 2778, 2779, 2780, 2784, 2785, 2786, 2787, 2803, 2804, 2808, 2810,
    2811, 2812, 2813, 2814, 2817, 2819, 2820, 2821, 2826, 2827, 2828, 2829, 2830,
    2831, 2832, 2833, 2834, 2835, 2836, 2837, 2838, 2839, 2840, 2843, 2844, 2845,
    2846, 2847, 2848, 2849, 2850, 2851, 2852, 2853, 2855, 2858, 2863, 2864, 2865,
    2866, 2869, 2871, 2873, 2878, 2879, 2880, 2881, 2882, 2883, 2884, 2885, 2886,
    2887, 2888, 2889, 2890, 2891, 2892, 2894, 2895, 2896, 2897, 2898, 2899, 2904,
    2905, 2906, 2907, 2928, 2929, 2934, 2935, 2936, 2937, 2938, 2939, 2948, 2949,
    3503, 3504, 3506, 3509, 3511, 3512, 3513, 3531, 3533, 3537, 3541, 3543, 3546,
    3547, 3790, 3791, 3792, 3793, 3794, 3795, 3796, 3797, 3798, 3799, 3800, 3801,
    3802, 3803, 3804, 3805, 3806, 3914, 3915, 3916, 3917, 3918, 3919, 3920, 3921,
    3922, 3923, 3924, 3925, 3926, 3927, 3928
])

def main():
    global cfg, logger

    logger = get_logger()
    logger.info(cfg)
    logger.info("=> creating model ...")
    model = get_model(cfg)
    model = model.cuda()

    if os.path.isfile(cfg.model_path):
        logger.info("=> loading checkpoint '{}'".format(cfg.model_path))
        checkpoint = torch.load(cfg.model_path, map_location=lambda storage, loc: storage.cpu())
        load_state_dict(model, checkpoint['state_dict'], strict=False)
        logger.info("=> loaded checkpoint '{}'".format(cfg.model_path))
    else:
        raise RuntimeError("=> no checkpoint flound at '{}'".format(cfg.model_path))

    # ####################### Data Loader ####################### #
    from dataset.data_loader_multi import get_dataloaders
    dataset = get_dataloaders(cfg, test_config=True)
    test_loader = dataset['test']

    test(model, test_loader, cfg)


def evaluate_lve(gt_path, pred_path):
    vertices_gt_all = []
    vertices_pred_all = []

    for pred_vert in sorted(os.listdir(pred_path)):
        if not os.path.isfile(os.path.join(gt_path, pred_vert)):
            continue
        gt = np.load(os.path.join(gt_path, pred_vert), allow_pickle=False)
        pred = np.load(os.path.join(pred_path, pred_vert))
        pred = pred.reshape(-1, 5023, 3)
        gt = gt.reshape(-1, 5023, 3)

        vertices_gt_all.extend(list(gt))
        vertices_pred_all.extend(list(pred))

    vertices_gt_all = np.array(vertices_gt_all)
    vertices_pred_all = np.array(vertices_pred_all)
    L2_dis_mouth_max = np.array([np.square(vertices_gt_all[:, v, :] - vertices_pred_all[:, v, :]) for v in mouth_map])
    L2_dis_mouth_max = np.transpose(L2_dis_mouth_max, (1, 0, 2))
    L2_dis_mouth_max = np.sum(L2_dis_mouth_max, axis=2)
    L2_dis_mouth_max = np.max(L2_dis_mouth_max, axis=1)
    lve = np.mean(L2_dis_mouth_max)
    print("Lip Vertex Error on test set: ", lve)

def test(model, test_loader, cfg):
    model.eval()
    save_folder, gt_save_folder = os.path.join(cfg.save_folder, 'npy'), os.path.join(cfg.gt_save_folder, 'npy')

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(gt_save_folder):
        os.makedirs(gt_save_folder)

    train_subjects_list = [i for i in cfg.train_subjects.split(" ")]

    with torch.no_grad():
        for i, (audio, vertice, template, one_hot_all, file_name) in enumerate(test_loader):
            audio = audio.cuda(non_blocking=True)
            one_hot_all = one_hot_all.cuda(non_blocking=True)
            vertice = vertice.cuda(non_blocking=True)
            template = template.cuda(non_blocking=True)

            train_subject = file_name[0].split("_")[0].capitalize()
            np.save(os.path.join(gt_save_folder, file_name[0].split(".")[0] + ".npy"),vertice.squeeze().detach().cpu().numpy())
            if train_subject in train_subjects_list:
                one_hot = one_hot_all
                prediction = model.predict(audio, template, one_hot, gt_frame_num=vertice.shape[1])
                prediction = prediction.squeeze()
                np.save(os.path.join(save_folder, file_name[0].split(".")[0]+".npy"), prediction.detach().cpu().numpy())

            else:
                one_hot = one_hot_all

                prediction = model.predict(audio, template, one_hot, gt_frame_num=vertice.shape[1])
                prediction = prediction.squeeze()
                np.save(os.path.join(save_folder, file_name[0].split(".")[0] + ".npy"),prediction.detach().cpu().numpy())

    evaluate_lve(gt_save_folder, save_folder)

if __name__ == '__main__':
    main()
