"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import os
import numpy as np
import torch
import torchvision
from tqdm import tqdm
from pyquaternion import Quaternion
from PIL import Image
from functools import reduce
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.map_expansion.map_api import NuScenesMap
import cv2
import math
import copy


def get_lidar_data(nusc, sample_rec, nsweeps, min_distance):
    """
    Returns at most nsweeps of lidar in the ego frame.
    Returned tensor is 5(x, y, z, reflectance, dt) x N
    Adapted from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py#L56
    """
    points = np.zeros((5, 0))

    # Get reference pose and timestamp.
    ref_sd_token = sample_rec['data']['LIDAR_TOP']
    ref_sd_rec = nusc.get('sample_data', ref_sd_token)
    ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
    ref_time = 1e-6 * ref_sd_rec['timestamp']

    # Homogeneous transformation matrix from global to _current_ ego car frame.
    car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                        inverse=True)

    # Aggregate current and previous sweeps.
    sample_data_token = sample_rec['data']['LIDAR_TOP']
    current_sd_rec = nusc.get('sample_data', sample_data_token)
    for _ in range(nsweeps):
        # Load up the pointcloud and remove points close to the sensor.
        current_pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, current_sd_rec['filename']))
        current_pc.remove_close(min_distance)

        # Get past pose.
        current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
        global_from_car = transform_matrix(current_pose_rec['translation'],
                                            Quaternion(current_pose_rec['rotation']), inverse=False)

        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
        car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                            inverse=False)

        # Fuse four transformation matrices into one and perform transform.
        trans_matrix = reduce(np.dot, [car_from_global, global_from_car, car_from_current])
        current_pc.transform(trans_matrix)

        # Add time vector which can be used as a temporal feature.
        time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']
        times = time_lag * np.ones((1, current_pc.nbr_points()))

        new_points = np.concatenate((current_pc.points, times), 0)
        points = np.concatenate((points, new_points), 1)

        # Abort if there are no previous sweeps.
        if current_sd_rec['prev'] == '':
            break
        else:
            current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

    return points


def ego_to_cam(points, rot, trans, intrins):
    """Transform points (3 x N) from ego frame into a pinhole camera
    """
    points = points - trans.unsqueeze(1)
    points = rot.permute(1, 0).matmul(points)

    points = intrins.matmul(points)
    points[:2] /= points[2:3]

    return points


def cam_to_ego(points, rot, trans, intrins):
    """Transform points (3 x N) from pinhole camera with depth
    to the ego frame
    """
    points = torch.cat((points[:2] * points[2:3], points[2:3]))
    points = intrins.inverse().matmul(points)

    points = rot.matmul(points)
    points += trans.unsqueeze(1)

    return points


def get_only_in_img_mask(pts, H, W):
    """pts should be 3 x N
    """
    return (pts[2] > 0) &\
        (pts[0] > 1) & (pts[0] < W - 1) &\
        (pts[1] > 1) & (pts[1] < H - 1)


def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])

# TODO：看看该函数如何优化
def met_resize(image, resizeW, resizeH):
    roiH, roiW, _ = image.shape
    # out_img = np.zeros((resizeH, resizeW, 3), np.uint8)
    out_img = copy.deepcopy(image)
    rx_rate = float(roiW) / resizeW
    ry_rate = float(roiH) / resizeH

    down_x_scale_times = int(math.log(rx_rate, 2))
    down_y_scale_times = int(math.log(ry_rate, 2))

    target_rz_w = roiW
    target_rz_h = roiH

    #max_down_scale_times = down_x_scale_times > down_y_scale_times ? down_x_scale_times : down_y_scale_times;
    max_down_scale_times = 0
    if down_x_scale_times > down_y_scale_times:
       max_down_scale_times = down_x_scale_times
    else:
       max_down_scale_times = down_y_scale_times
    if max_down_scale_times >= 1:
       for i in range(max_down_scale_times):
           if i < down_x_scale_times and max_down_scale_times >= 1:
              target_rz_w = (int)((target_rz_w + 1) / 2.0)
           if i <= down_y_scale_times and max_down_scale_times >= 1:
              target_rz_h = (int)((target_rz_h + 1) / 2.0)
           out_img = cv2.resize(out_img, (target_rz_w,target_rz_h), interpolation=cv2.INTER_LINEAR)
       if out_img.shape[1]!=resizeW or out_img.shape[0]!=resizeH:
          out_img = cv2.resize(out_img, (resizeW, resizeH), interpolation=cv2.INTER_LINEAR)
    else:
       out_img = cv2.resize(image, (resizeW, resizeH), interpolation=cv2.INTER_LINEAR)
    return out_img

# 数据增强函数
def cvimg_transform(img, post_rot, post_tran, vp,
                  resize, resize_dims, crop,
                  flip, rotate, flag = 0):

    W,H = resize_dims    # 568*174
    if flag == 0:
       img = cv2.resize(img, (W,H))
    else:
       img = met_resize(img, W, H) 
    vp_h = int(vp[1]*resize)
    # print ('resize11 = ', img.shape, crop)
    # cv2.imwrite('resize.jpg', img)
    top = 0
    bot = 0
    left = 0
    rt = 0
    crop1 = crop    # crop (83, 9, 435, 137)
    crop = list(crop)
    # vp_h = max(20, vp_h)
    if crop[1] > vp_h and vp_h > 10:
        fH = crop[3] - crop[1]
        crop[1] = vp_h - 10
        crop[3] = crop[1] + fH

    if  crop[0] < 0:
        left = -crop[0]
    if crop[1] < 0:
        top = -crop[1]
    if crop[2] > W:
        rt = crop[2] - W
    if crop[3] > H:
        bot = crop[3] - H

    # print(max(0,crop[1]), min(H,crop[3]), max(0,crop[0]), min(W,crop[2]))
    img = img[max(0,crop[1]): min(H,crop[3]), max(0,crop[0]): min(W,crop[2])]  # 128*352*3
    img = cv2.copyMakeBorder(img, top, bot, left, rt, borderType=cv2.BORDER_CONSTANT, value = 0)
    # print ('crop1 = ', img.shape)
    if flip:
        img = cv2.flip(img, 1)

    h, w, _ = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rotate, 1)
    # print ('img.shape = ', img.shape)
    img = cv2.warpAffine(img, M, (w, h))

    # post-homography transformation
    post_rot *= resize
    post_tran -= torch.Tensor(crop[:2])
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
    A = get_rot(rotate/180*np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    post_rot = A.matmul(post_rot)
    post_tran = A.matmul(post_tran) + b
    hh,ww, _ = img.shape
    if hh > 128:
        print(crop, crop1, vp_h)
    # cv2.imwrite('img.jpg', img)
    return img, post_rot, post_tran


def img_transform(img, post_rot, post_tran,
                  resize, resize_dims, crop,
                  flip, rotate):
    # adjust image
    img = img.resize(resize_dims)
    img = img.crop(crop)
    if flip:
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    img = img.rotate(rotate)

    # post-homography transformation
    post_rot *= resize
    post_tran -= torch.Tensor(crop[:2])
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
    A = get_rot(rotate/180*np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    post_rot = A.matmul(post_rot)
    post_tran = A.matmul(post_tran) + b

    return img, post_rot, post_tran


class NormalizeInverse(torchvision.transforms.Normalize):
    #  https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


# denormalize_img = torchvision.transforms.Compose((
#             NormalizeInverse(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225]),
#             torchvision.transforms.ToPILImage(),
#         ))


# normalize_img = torchvision.transforms.Compose((
#                 torchvision.transforms.ToTensor(),
#                 torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225]),
# ))

denormalize_img = torchvision.transforms.Compose((
            NormalizeInverse(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
            torchvision.transforms.ToPILImage(),
        ))


normalize_img = torchvision.transforms.Compose((
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
))


def gen_dx_bx(xbound, ybound, zbound, alpha = 1.0):
    dx = torch.Tensor([row[2] / alpha for row in [xbound, ybound, zbound]]) # dx([0.8500, 0.5000, 1.0000]) 分别为x, y, z三个方向上的网格间距
    bx = torch.Tensor([row[0] + row[2]/alpha/2.0 for row in [xbound, ybound, zbound]]) # bx=tensor([ 0.4250, -9.7500, -1.5000])  分别为x, y, z三个方向上第一个格子的坐标
    nx = torch.LongTensor([(row[1] - row[0]) / (row[2]/alpha) for row in [xbound, ybound, zbound]]) # nx=tensor([120,  40,   6])  分别为x, y, z三个方向上格子的数量
    
    # dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    # bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    # nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    # print('22222 = ', dx, bx, nx)
    return dx, bx, nx


def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])

    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    return x, geom_feats


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


class SimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super(SimpleLoss, self).__init__()
        # sigmoid+二值交叉熵损失, pos_weight是给正样本乘的权重系数，防止正样本过少，用于平衡precision和recall。
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))

    def forward(self, ypred, ytgt):
        ypred = ypred.view((-1, 1))
        ytgt = ytgt.view((-1, 1))
        mask = ytgt > -0.5
        ypred = ypred[mask]
        ytgt = ytgt[mask]
        if ytgt.shape[0] == 0:
            loss = torch.zeros(1, requires_grad=True).to(ypred.device)
        else:
            loss = self.loss_fn(ypred, ytgt)
        return loss

class SimpleLoss1(torch.nn.Module):
    def __init__(self, pos_weight):
        super(SimpleLoss1, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))

    def forward(self, ypred, ytgt):
        ypred = ypred.view((-1, 1))
        yout = ypred.sigmoid()
        ytgt = ytgt.view((-1, 1))
        pos_mask = (ytgt > 0.5) & (yout < 1.8)
        neg_mask = (ytgt < 0.5) & (ytgt > -0.5) & (yout > 0.2)
        mask = neg_mask|pos_mask
        ypred = ypred[mask]
        ytgt = ytgt[mask]
        if ytgt.shape[0] == 0:
            loss = torch.zeros(1, requires_grad=True).to(ypred.device)
        else:
            loss = self.loss_fn(ypred, ytgt)
        return loss

class RegLoss(torch.nn.Module):
    def __init__(self, limit):
        super(RegLoss, self).__init__()
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.limit = limit

    def forward(self, ypred, ytgt):
        mask = torch.abs(ypred-ytgt)  > self.limit
        ypred = ypred[mask]
        ytgt = ytgt[mask]
        
        if ypred.shape[0] == 0:
            loss = torch.zeros(1, requires_grad=True).to(ypred.device)
        else:
            loss = self.loss_fn(ypred, ytgt)
        return loss


class SegLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super(SegLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))

    # def forward(self, ypred, ytgt):
    #     ypred = ypred.view((-1, 1))
    #     yout = ypred.sigmoid()
    #     ytgt = ytgt.view((-1, 1))
    #     pos_mask = (ytgt > 0.5) & (yout < 1.8)
    #     neg_mask = (ytgt <0.5) & (ytgt > -0.5) & (yout > 0.25)
    #     mask = neg_mask|pos_mask
    #     ypred = ypred[mask]
    #     ytgt = ytgt[mask]
    #     loss = self.loss_fn(ypred, ytgt)
    #     return loss
    def forward(self, ypred, ytgt):
        ypred = ypred.view((-1, 1))
        ytgt = ytgt.view((-1, 1))
        #pos_mask = (ytgt > 0.5) & (yout < 1.8)
        #neg_mask = (ytgt <0.5) & (ytgt > -0.5) & (yout > 0.25)
        mask = ytgt > -0.5#neg_mask|pos_mask
        ypred = ypred[mask]
        ytgt = ytgt[mask]
        loss = self.loss_fn(ypred, ytgt)
        return loss

class SegLoss1(torch.nn.Module):
    def __init__(self, pos_weight):
        super(SegLoss1, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))

    def forward(self, ypred, ytgt):
        ypred = ypred.view((-1, 1))
        yout = ypred.sigmoid()
        ytgt = ytgt.view((-1, 1))
        pos_mask = (ytgt > 0.5) & (yout < 1.8)
        neg_mask = (ytgt <0.5) & (ytgt > -0.5) & (yout > 0.2)
        mask = neg_mask|pos_mask
        ypred = ypred[mask]
        ytgt = ytgt[mask]
        loss = self.loss_fn(ypred, ytgt)
        return loss


class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, ypred, ytgt):
        ypred = ypred.view((-1, 1))
        yout = ypred.sigmoid()
        ytgt = ytgt.view((-1, 1))
        pos_mask = (ytgt > 0.5) & (yout < 1.8)
        neg_mask = (ytgt <0.5) & (ytgt > -0.5) & (yout > 0.25)
        mask = neg_mask|pos_mask
        ypred = ypred[mask]
        ytgt = ytgt[mask]

        pt = torch.sigmoid(ypred) # sigmoide获取概率
        #在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        loss = - self.alpha * (1 - pt) ** self.gamma * ytgt * torch.log(pt) - (1 - self.alpha) * pt ** self.gamma * (1 - ytgt) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

def get_batch_iou(preds, binimgs):
    """Assumes preds has NOT been sigmoided yet
    """
    with torch.no_grad():
        preds = preds.view((-1, 1))
        binimgs = binimgs.view((-1, 1))
        mask = binimgs > -0.5
        preds = preds[mask]
        binimgs = binimgs[mask]
        pred = (preds > 0)
        tgt = binimgs.bool()
        intersect = (pred & tgt).sum().float().item()
        union = (pred | tgt).sum().float().item()
    return intersect, union, intersect / union if (union > 0) else 1.0


def get_val_info(model, valloader, loss_fn, device, counter, val_sampler):
    model.eval()
    total_loss = 0.0
    total_intersect = 0.0
    total_union = 0
    last_preds = None
    last_binimgs = None
    last_allimgs = None
    print('running eval...')
    loader = valloader
    with torch.no_grad():
        val_sampler.set_epoch(counter)
        cnt = 0
        for batch in loader:
            #fork_label_gt
            cnt+=1
            # if cnt < 1162:
            #     print('cnt =', cnt)
            #     continue
            allimgs, rots, trans, intrins, post_rots, post_trans, binimgs, lf_label_gt, lf_norm_gt, lf_kappa_gt, fork_patch_gt, fork_scales_gt, fork_offsets_gt, fork_oris_gt = batch
            seg_preds, seg_ipreds, lf_preds, _ = model(allimgs.to(device), rots.to(device),
                          trans.to(device), intrins.to(device), post_rots.to(device),
                          post_trans.to(device),fork_scales_gt.to(device),fork_offsets_gt.to(device),fork_oris_gt.to(device))
            binimgs = binimgs.to(device)

           
            # if cnt == 1162:
            # if 1:
            #     # imgname = img_paths[-1][0][img_paths[-1][0].rfind('/')+1 :]
            #     # print ('imgname = ', imgname)
            #     binimgs = binimgs.cpu().numpy()
            #     img_gt = np.ones((480, 160, 3), dtype=np.uint8)
            #     colors = [(255, 255, 255), (102, 102, 255), (255, 255, 255), (179, 255, 102), (102, 222, 255), (255, 102, 102), (0, 255, 255)]
            #     for class_id in range(6):
            #         result = binimgs[0][4][class_id]
            #         img_gt[result> 0.5] = np.array(colors[class_id])
            #         # img_gt[result< -0.5] = np.array((128,128,128))
            #     img_gt = cv2.flip(cv2.flip(img_gt, 0), 1)
            #     cv2.imwrite('gt.png', img_gt)

            # loss
            total_loss += loss_fn(seg_preds[:, :, :6].contiguous(), binimgs).item() * seg_preds.shape[0]
            # print('val_loss = ', cnt, total_loss, len(valloader.dataset))
            # iou
            intersect, union, _ = get_batch_iou(seg_preds, binimgs)
            total_intersect += intersect
            total_union += union
            last_preds = allimgs
            last_binimgs = binimgs
            last_preds = seg_preds
            
    model.train()
    return {
            'loss': total_loss / len(valloader.dataset),
            'iou': total_intersect / total_union,
            'last_allimgs': allimgs,
            'last_binimgs': binimgs,
            'last_preds': seg_preds,
            }


def add_ego(bx, dx):
    # approximate rear axel
    W = 1.85
    pts = np.array([
        [-4.084/2.+0.5, W/2.],
        [4.084/2.+0.5, W/2.],
        [4.084/2.+0.5, -W/2.],
        [-4.084/2.+0.5, -W/2.],
    ])
    pts = (pts - bx) / dx
    pts[:, [0,1]] = pts[:, [1,0]]
    plt.fill(pts[:, 0], pts[:, 1], '#76b900')


def get_nusc_maps(map_folder):
    nusc_maps = {map_name: NuScenesMap(dataroot=map_folder,
                map_name=map_name) for map_name in [
                    "singapore-hollandvillage",
                    "singapore-queenstown",
                    "boston-seaport",
                    "singapore-onenorth",
                ]}
    return nusc_maps


def plot_nusc_map(rec, nusc_maps, nusc, scene2map, dx, bx):
    egopose = nusc.get('ego_pose', nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
    map_name = scene2map[nusc.get('scene', rec['scene_token'])['name']]
    rot = Quaternion(egopose['rotation']).rotation_matrix
    rot = np.arctan2(rot[1, 0], rot[0, 0])
    center = np.array([egopose['translation'][0], egopose['translation'][1], np.cos(rot), np.sin(rot)])
    poly_names = ['road_segment', 'lane']
    line_names = ['road_divider', 'lane_divider']
    lmap = get_local_map(nusc_maps[map_name], center,
                         50.0, poly_names, line_names)
    for name in poly_names:
        for la in lmap[name]:
            pts = (la - bx) / dx
            plt.fill(pts[:, 1], pts[:, 0], c=(1.00, 0.50, 0.31), alpha=0.2)
    for la in lmap['road_divider']:
        pts = (la - bx) / dx
        plt.plot(pts[:, 1], pts[:, 0], c=(0.0, 0.0, 1.0), alpha=0.5)
    for la in lmap['lane_divider']:
        pts = (la - bx) / dx
        plt.plot(pts[:, 1], pts[:, 0], c=(159./255., 0.0, 1.0), alpha=0.5)

def draw_nusc_map(rec, nusc_maps, nusc, scene2map, dx, bx, nx):
    egopose = nusc.get('ego_pose', nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
    map_name = scene2map[nusc.get('scene', rec['scene_token'])['name']]
    rot = Quaternion(egopose['rotation']).rotation_matrix
    rot = np.arctan2(rot[1, 0], rot[0, 0])
    center = np.array([egopose['translation'][0], egopose['translation'][1], np.cos(rot), np.sin(rot)])
    poly_names = ['road_segment', 'lane']
    line_names = ['road_divider', 'lane_divider']
    lmap = get_local_map(nusc_maps[map_name], center,
                         50.0, poly_names, line_names)

    img = np.zeros((nx[0], nx[1], 3), dtype=np.uint8)+255
    for name in poly_names:
        for la in lmap[name]:
            pts = (la - bx) / dx
            pts = np.round(pts).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(img, [pts], (128, 128, 128))
    for la in lmap['road_divider']:
        pts = (la - bx) / dx
        pts = np.round(pts).astype(np.int32)
        pts[:, [1, 0]] = pts[:, [0, 1]]
        cv2.polylines(img, [pts], 0, (0, 0, 255), 1)
    for la in lmap['lane_divider']:
        pts = (la - bx) / dx
        pts = np.round(pts).astype(np.int32)
        pts[:, [1, 0]] = pts[:, [0, 1]]
        cv2.polylines(img, [pts], 0, (255, 0, 0), 1)
    return img


def get_local_map(nmap, center, stretch, layer_names, line_names):
    # need to get the map here...
    box_coords = (
        center[0] - stretch,
        center[1] - stretch,
        center[0] + stretch,
        center[1] + stretch,
    )

    polys = {}

    # polygons
    records_in_patch = nmap.get_records_in_patch(box_coords,
                                                 layer_names=layer_names,
                                                 mode='intersect')
    for layer_name in layer_names:
        polys[layer_name] = []
        for token in records_in_patch[layer_name]:
            poly_record = nmap.get(layer_name, token)
            if layer_name == 'drivable_area':
                polygon_tokens = poly_record['polygon_tokens']
            else:
                polygon_tokens = [poly_record['polygon_token']]

            for polygon_token in polygon_tokens:
                polygon = nmap.extract_polygon(polygon_token)
                polys[layer_name].append(np.array(polygon.exterior.xy).T)

    # lines
    for layer_name in line_names:
        polys[layer_name] = []
        for record in getattr(nmap, layer_name):
            token = record['token']

            line = nmap.extract_line(record['line_token'])
            if line.is_empty:  # Skip lines without nodes
                continue
            xs, ys = line.xy

            polys[layer_name].append(
                np.array([xs, ys]).T
                )

    # convert to local coordinates in place
    rot = get_rot(np.arctan2(center[3], center[2])).T
    for layer_name in polys:
        for rowi in range(len(polys[layer_name])):
            polys[layer_name][rowi] -= center[:2]
            polys[layer_name][rowi] = np.dot(polys[layer_name][rowi], rot)

    return polys
