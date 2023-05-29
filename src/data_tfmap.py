"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import os
import os.path as Path
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
import numpy as np
from PIL import Image
import cv2
from pyquaternion import Quaternion
from glob import glob
import time
import json
import glob
import math
import shapely
from shapely.geometry import Polygon
from shapely.geometry import LineString
from shapely.geometry import box
import re
import open3d as o3d
import torch
import shutil
import random
from scipy.spatial.transform import Rotation as R
from torch.utils.data.distributed import DistributedSampler
from src.tools import get_lidar_data, img_transform, cvimg_transform, normalize_img, gen_dx_bx, get_nusc_maps, get_local_map
# import pysnooper
from time import time
# import bev
# import kornia as Ka
import torch.nn as nn
# import cvcuda
# import jpeg4py as jpeg

import lmdb
import pickle

pi = 3.1415926

def euler_angles_to_rotation_matrix(angle, is_dir=False):
    """Convert euler angels to quaternions.
    Input:
        angle: [roll, pitch, yaw]
        is_dir: whether just use the 2d direction on a map
    """
    roll, pitch, yaw = angle[0], angle[1], angle[2]

    rollMatrix = np.matrix([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]])

    pitchMatrix = np.matrix([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]])

    yawMatrix = np.matrix([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]])

    R = yawMatrix * pitchMatrix * rollMatrix
    R = np.array(R)

    if is_dir:
        R = R[:, 2]

    return R

def convert_quat_to_pose_mat(xyzquat):
    xyz = xyzquat[:3]
    quat = xyzquat[3:]
    ret = R.from_quat(quat)
    #print('xxx = ', xyz, quat, Rot)
    T = np.array(xyz).reshape(3, 1)
    pose_mat = np.eye(4, dtype= np.float64)
    pose_mat[:3, :3] = ret.as_matrix()
    pose_mat[:3, 3] = T.T
    return np.matrix(pose_mat)

def convert_6dof_to_pose_mat(dof6):
    xyz = dof6[:3]
    angle = dof6[3:]
    R = euler_angles_to_rotation_matrix(angle)
    T = np.array(xyz).reshape(3, 1)
    pose_mat = np.eye(4, dtype= np.float64)
    pose_mat[:3, :3] = R
    pose_mat[:3, 3] = T.T
    return np.matrix(pose_mat)

def convert_rvector_to_pose_mat(dof6):
    xyz = dof6[:3]
    angle = dof6[3:]
    #R = euler_angles_to_rotation_matrix(angle)
    R, _ = cv2.Rodrigues(np.array(angle))
    T = np.array(xyz).reshape(3, 1)
    pose_mat = np.eye(4, dtype= np.float64)
    pose_mat[:3, :3] = R
    pose_mat[:3, 3] = T.T
    return np.matrix(pose_mat)

def convert_rollyawpitch_to_pose_mat(roll, yaw, pitch, x, y, z):
    roll *= pi/180.
    yaw *= pi/180.
    pitch *= pi/180.
    Rr = np.array([[0.0, -1.0, 0.0],
                   [0.0, 0.0, -1.0],
                   [1.0, 0.0, 0.0]], dtype=np.float32)
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, np.cos(roll), np.sin(roll)],
                   [0.0, -np.sin(roll), np.cos(roll)]], dtype=np.float32)
    Ry = np.array([[np.cos(pitch), 0.0, -np.sin(pitch)],
                   [0.0, 1.0, 0.0],
                   [np.sin(pitch), 0.0, np.cos(pitch)]], dtype=np.float32)
    Rz = np.array([[np.cos(yaw), np.sin(yaw), 0.0],
                   [-np.sin(yaw), np.cos(yaw), 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float32)
    R = np.matrix(Rr) * np.matrix(Rx) * np.matrix(Ry) * np.matrix(Rz)
    T = np.array([x,y,z]).reshape(3, 1)
    pose_mat = np.eye(4, dtype= np.float64)
    pose_mat[:3, :3] = R
    pose_mat[:3, 3] = T.T
    return np.matrix(pose_mat)

def convert_rollyawpitch_to_rot(roll, yaw, pitch):
    roll *= pi/180.
    yaw *= pi/180.
    pitch *= pi/180.
    Rr = np.array([[0.0, -1.0, 0.0],
                   [0.0, 0.0, -1.0],
                   [1.0, 0.0, 0.0]], dtype=np.float32)
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, np.cos(roll), np.sin(roll)],
                   [0.0, -np.sin(roll), np.cos(roll)]], dtype=np.float32)
    Ry = np.array([[np.cos(pitch), 0.0, -np.sin(pitch)],
                   [0.0, 1.0, 0.0],
                   [np.sin(pitch), 0.0, np.cos(pitch)]], dtype=np.float32)
    Rz = np.array([[np.cos(yaw), np.sin(yaw), 0.0],
                   [-np.sin(yaw), np.cos(yaw), 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float32)
    R = np.matrix(Rr) * np.matrix(Rx) * np.matrix(Ry) * np.matrix(Rz)
    return R


class TFmapData(torch.utils.data.Dataset):
    def __init__(self, data_idxs, param_infos_list, data_infos_list, vp_infos_list, mesh_objs_list, roadmap_data_list, roadmap_samples_list, roadmap_forks_list, map_paths_list, ignore_map_paths_list, ob_map_paths_list, image_paths_list, is_train, data_aug_conf, grid_conf, seq_len, used_fcodes, crop_size, dataroot, dataset_list):
        self.data_idxs = data_idxs
        self.data_infos_list = data_infos_list
        self.vp_infos_list = vp_infos_list
        self.roadmap_data_list = roadmap_data_list
        self.roadmap_samples_list = roadmap_samples_list
        self.roadmap_forks_list = roadmap_forks_list
        self.mesh_objs_list = mesh_objs_list
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf
        self.seq_len = seq_len
        self.used_fcodes = used_fcodes
        self.crop_size = crop_size
        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'], 4.0)
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()
        self.image_paths_list = image_paths_list
        self.map_paths_list = map_paths_list
        self.ignore_map_paths_list = ignore_map_paths_list
        self.ob_map_paths_list = ob_map_paths_list
        
        K_list = []
        rot_list = []
        tran_list = []
        T_lidar2cam_list = []
        for param_infos in param_infos_list:
            K_list.append(param_infos['K'])
            rvector = param_infos['rvector']
            T_lidar2cam_list.append(convert_rvector_to_pose_mat(rvector))
            yaw = param_infos['yaw']
            pitch = param_infos['pitch']
            roll = param_infos['roll']
            # tran = param_infos['xyz']
            # roll = -0.753848
            # pitch = -0.560495
            # yaw = 0.169544

            rot_list.append(convert_rollyawpitch_to_rot(roll, yaw, pitch).I)
            tran = param_infos['xyz']
            print('roll = ', roll, 'pitch = ', pitch, 'yaw = ', yaw)
            print('tran = ', tran)
            # tran = [-1.650000, 0.000000, 1.400000]
            tran_list.append(np.array(tran))
        self.K_list = K_list
        self.rot_list = rot_list
        self.tran_list = tran_list
        self.T_lidar2cam_list = T_lidar2cam_list

        self.maps_list, self.ignore_maps_list, self.ob_maps_list  = self.get_maps()
        # self.scenes = self.get_scenes()
        self.ixes = self.prepro()

        self.lf_env_path = dataroot+'/lf.lmdb'   # LMDB环境的路径
        max_dbs = 10  # 新的最大数据库数量

        # 打开LMDB环境
        lf_env = lmdb.open(self.lf_env_path, readonly=True, max_dbs=max_dbs)  # 只读模式打开LMDB环境
        
        # 初始化时读取所有数据
        data_list = []
        # 打开LMDB事务和数据库
        with lf_env.begin() as txn:
            for path in dataset_list:
                    lf_key = path.encode()  # 要读取的lf的键
                    lf_data = txn.get(lf_key)
                     # 反序列化数据为列表
                    data_list += pickle.loads(lf_data)
                    # print('===dataset', path, ' len is: ', len(data_list))

        # 初始化时只读取一个数据集的数据，训练过程中再换
        # self.sce_id = 0
        # self.dataset_list = dataset_list

        # path = dataset_list[0]
        # with lf_env.begin() as txn:            
        #     lf_key = path.encode()  # 要读取的lf的键
        #     lf_data = txn.get(lf_key)
        #     # 反序列化数据为列表
        #     data_list = pickle.loads(lf_data)

        # # 关闭LMDB环境
        lf_env.close()
       
        self.data_list = data_list

        image_env_path = dataroot+'/img.lmdb'  # LMDB环境的路径
        # 打开LMDB环境
        self.env = lmdb.open(image_env_path, readonly=True, max_dbs=max_dbs)



    def get_maps(self):
        maps_list = []
        ignore_maps_list = []
        ob_maps_list  = []
        for map_paths in self.map_paths_list:
            tf_maps = []
            for ii, path in enumerate(map_paths):
                with open(path, encoding='utf-8') as fp:
                    geojson = json.load(fp)
                    features = geojson['features']
                    map_data = []
                    for ii, feature in enumerate(features):
                        poly = feature['geometry']['coordinates']
                        type = feature['geometry']['type']
                        name = feature['properties']['name']
                        fcode = feature['properties']['FCode']
                        if not fcode in self.used_fcodes:
                            continue
                        if type == 'MultiLineString':
                            for pp in poly:
                                if len(pp) == 0:
                                    continue
                                data = np.array(pp, dtype=np.float64)
                                dist = np.sum(np.abs(data[-1]-data[0]))
                                label = self.used_fcodes[fcode]
                                if label == 6:
                                    map_data.append([data, 5])
                                elif name == '大面积车辆遮挡区域或路面不清晰' or fcode == '444300' or fcode == '401434' or fcode == '401453' or dist > 0.001:
                                    #print(path, name, fcode)
                                    map_data.append([data, -1])
                                else:
                                    if label == 3:
                                        map_data.append([data, 1])
                                    elif label == 4:
                                        map_data.append([data, 3])
                                    elif label == 5:
                                        map_data.append([data, 4])
                                    else:
                                        map_data.append([data, label])

                        elif type == "LineString":
                            if len(poly) == 0:
                                continue
                            data = np.array(poly, dtype=np.float64)
                            dist = np.sum(np.abs(data[-1]-data[0]))
                            label = self.used_fcodes[fcode]
                            if label == 6:
                                map_data.append([data, 5])
                            elif name == '大面积车辆遮挡区域或路面不清晰' or fcode == '444300' or fcode == '401434' or fcode == '401453' or dist > 0.001:
                                #print(path, name, fcode)
                                map_data.append([data, -1])
                            else:
                                if label == 3:
                                    map_data.append([data, 1])
                                elif label == 4:
                                    map_data.append([data, 3])
                                elif label == 5:
                                    map_data.append([data, 4])
                                else:
                                    map_data.append([data, label])
                        else:
                            print(type)
                    tf_maps += map_data
            maps_list.append(tf_maps)
        
        for ignore_map_paths in self.ignore_map_paths_list:    
            tf_ignore_maps = []
            for ii, path in enumerate(ignore_map_paths):
                with open(path, encoding='utf-8') as fp:
                    geojson = json.load(fp)
                    features = geojson['features']
                    map_data = []
                    for ii, feature in enumerate(features):
                        poly = feature['geometry']['coordinates']
                        type = feature['geometry']['type']
                        name = feature['properties']['name']
                        fcode = feature['properties']['FCode']
                        if type == 'MultiLineString':
                            for pp in poly:
                                if len(pp) == 0:
                                    continue
                                data = np.array(pp, dtype=np.float64)
                                if name == '过渡区' or fcode == '99502102':
                                   map_data.append([data, -2])
                                else:
                                   map_data.append([data, -1])
                        elif type == "LineString":
                            if len(poly) == 0:
                                continue
                            data = np.array(poly, dtype=np.float64)
                            if name == '过渡区' or fcode == '99502102':
                                map_data.append([data, -2])
                            else:
                                map_data.append([data, -1])
                        else:
                            print(type)
                    tf_ignore_maps += map_data     
            ignore_maps_list.append(tf_ignore_maps) 

        for ob_map_paths in self.ob_map_paths_list:    
            tf_ob_maps = []
            for ii, path in enumerate(ob_map_paths):
                with open(path, encoding='utf-8') as fp:
                    geojson = json.load(fp)
                    features = geojson['features']
                    map_data = []
                    for ii, feature in enumerate(features):
                        poly = feature['geometry']['coordinates']
                        type = feature['geometry']['type']
                        name = feature['properties']['name']
                        fcode = feature['properties']['FCode']
                        if type == 'MultiLineString':
                            for pp in poly:
                                if len(pp) == 0:
                                    continue
                                data = np.array(pp, dtype=np.float64)
                                map_data.append([data, 1])
                        elif type == "LineString":
                            if len(poly) == 0:
                                continue
                            data = np.array(poly, dtype=np.float64)
                            map_data.append([data, 1])
                        else:
                            print(type)
                    tf_ob_maps += map_data     
            ob_maps_list.append(tf_ob_maps)    
        return maps_list, ignore_maps_list, ob_maps_list

    def get_scenes(self):
        scenes = []
        # for i, image_paths in enumerate(self.image_paths_list):
        #     image_list = sorted(glob.glob(image_paths))
        #     length = 0
        #     for j, imagename in enumerate(image_list):
        #         data_info = self.data_infos_list[i][imagename.split('/')[-1]]
        #         length += 1
        #         if "end_path" not in data_info:
        #             break
        #     scenes.append([0, length])
        for i, data_infos in enumerate(self.data_infos_list):
            # # image_list = sorted(glob.glob(self.image_paths[i]))
            # length = 0
            # for j, data_info in enumerate(data_infos):
            #     if "end_path" not in data_info:
            #         break
            #     length+= 1
            scenes.append([0, len(data_infos)])
        return scenes

    def prepro(self):
        sample_list = {}
        for i in range(len(self.image_paths_list)):
            # if i == 0:
            #     continue
            sub_samples = []
            # start_idx, end_idx = self.scenes[i]
            data_infos = self.data_infos_list[i]
            vp_infos = self.vp_infos_list[i]
            image_paths = self.image_paths_list[i]
            image_list = sorted(glob.glob(image_paths))
            rot = self.rot_list[i]
            tran = self.tran_list[i]
            K = self.K_list[i]
            end_idx = len(data_infos) - self.seq_len + 1
            end_idx = len(data_infos)
            for j in range(0, end_idx, 1):
                imgname = os.path.split(image_list[j])[-1]
                ipose = data_infos[imgname]['ipose']
                pose = convert_quat_to_pose_mat(ipose[1:8])
                vp = vp_infos[imgname]['vp_pose']
                sub_samples.append((image_list[j], pose, vp))
                # sample_list.append([[0, end_idx - self.seq_len + 1], sub_samples, rot, tran])
            sample_list[image_paths] = [sub_samples, rot, tran, K]
        return sample_list

    def sample_augmentation(self, cam_idx):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
       
        if self.is_train:
           # if 0:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'][cam_idx])
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim'][cam_idx]))*newH) - fH
            # crop_h = min(max(0, crop_h), newH - fH-1)
            crop_w = int(np.random.uniform(-fW/8., fW/8.)+(newW - fW)/2.)
            # crop_w = min(max(0, crop_w), newW - fW-1)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            # if crop_w + fW > newW or crop_h + fH > newH:
            #    print ('crop = ', crop, newW, newH, fH, fW)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = np.mean(self.data_aug_conf['resize_lim'][cam_idx])
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim'][cam_idx]))*newH) - fH
            # crop_w = int(max(0, newW - fW) / 2)
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            #print ('crop = ', crop_w, crop_h, crop, resize, resize_dims)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_image_data_tf(self, recs, cams, extrin_rot, extrin_tran, K):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        aug_params = []

        # 打开LMDB事务和数据库
        env = self.env
        with env.begin() as txn:
            image_shape = (1080, 3520, 3)  # 根据实际图像尺寸设置
            # a1 = time()
            idxs = [0,1]
            np.random.shuffle(idxs)
            for jj in idxs:
                aug_params.append(self.sample_augmentation(jj))
            for ii, sample in enumerate(recs):
                image_path, _, vp = sample

                # a1 = time()     
                # image = cv2.imread(image_path)
                image_key = image_path.encode()  # 要读取的图像的键
                image_data = txn.get(image_key)

                image_array = np.frombuffer(image_data, np.uint8)  # 将图像数据解码为NumPy数组
                image = image_array.reshape(image_shape)

                # image = jpeg.JPEG(image_path).decode()
                # print("imread耗时：",  time()-a1)
            
                # image = Ka.io.load_image(image_path, Ka.io.ImageLoadType.RGB32)[None, ...]  # BxCxHxW

                # print("image_path：", image_path)
                # cv2.imwrite('img_jpeg.jpg', image)
                for jj in range(2):
                    # a3 = time()
                    post_rot = torch.eye(2)
                    post_tran = torch.zeros(2)

                    intrin = torch.Tensor(K)
                    rot = torch.Tensor(extrin_rot)
                    tran = torch.Tensor(extrin_tran)

                    # augmentation (resize, crop, horizontal flip, rotate)
                    resize, resize_dims, crop, flip, rotate = aug_params[jj] # 数据增强的参数                   
                    
                    img, post_rot2, post_tran2 = cvimg_transform(image, post_rot, post_tran, vp,
                                                            resize=resize,
                                                            resize_dims=resize_dims,
                                                            crop=crop,
                                                            flip=flip,
                                                            rotate=rotate,
                                                            flag = 1) # 数据增强后的相机参数
                

                    # cv2.imwrite('img_amba.jpg', img)
                    # for convenience, make augmentation matrices 3x3
                    # post_tran2 = post_tran
                    # post_rot2 = post_rot
                    # img = image

                
                    post_tran = torch.zeros(3)
                    post_rot = torch.eye(3)
                    post_tran[:2] = post_tran2
                    post_rot[:2, :2] = post_rot2
                    #print ("trans222222:: post_rot = ", post_rot)
                    imgs.append(normalize_img(img))

                    # imgs.append(normalize_img(image))
                    # imgs.append(torch.Tensor(image))
                    intrins.append(intrin)
                    rots.append(rot)
                    trans.append(tran)
                    post_rots.append(post_rot)
                    post_trans.append(post_tran)
                    # print("cvimg_transform:", time() - a3)
        # env.close()
        return (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans))
        # print("cvimg_transform：", time() - a1)
        # return (imgs, torch.stack(rots), torch.stack(trans),
        #             torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans))

    def get_localmap_seg(self, sce_id, recs, extrin_rot, extrin_tran):
        # print('sce_id = ', sce_id)
        local_maps = []
        data_infos = self.data_infos_list[sce_id]
        T_lidar2cam = self.T_lidar2cam_list[sce_id]
        roadmap_data = self.roadmap_data_list[sce_id]
        mesh_objs = self.mesh_objs_list[sce_id]
        ignore_maps = self.ignore_maps_list[sce_id]
        ob_maps = []
        if len(self.ob_maps_list) > 0:
           ob_maps = self.ob_maps_list[sce_id]
        maps = self.maps_list[sce_id]
        valid_lf_areas = []
        for ii, sample in enumerate(recs):
            image_path, pose, _ = sample
            # print('imagpath = ', image_path)
            data_info = data_infos[os.path.split(image_path)[-1]]
            concerned_obj_idxs = data_info['concerned_obj_idxs']
            concerned_map_idxs = data_info['concerned_map_idxs']
            concerned_roadmap_idxs = data_info['concerned_roadmap_idxs']
            concerned_ignore_map_idxs = []
            isExist = data_info.get('concerned_ignore_map_idxs', 'not exist')
            if isExist != 'not exist':
               concerned_ignore_map_idxs = data_info['concerned_ignore_map_idxs']#guodu & ignore
            # concerned_ignore_map_idxs = data_info['concerned_ignore_map_idxs']#guodu & ignore
            concerned_ob_map_idxs = []
            isExist = data_info.get('concerned_ob_map_idxs', 'not exist')
            if isExist != 'not exist':
                concerned_ob_map_idxs = data_info['concerned_ob_map_idxs']#zhangaiwu
            local_map = [np.zeros((self.nx[0], self.nx[1])) for jj in range(6)]
            # show_seg_map = np.zeros((self.nx[0], self.nx[1], 3), dtype=np.uint8)*255
            # print('nx,ny = ', self.nx[0], self.nx[1])
            for map_idx in concerned_map_idxs:
                poly, label = maps[map_idx]
                # print ('label = ', label)
                if 1:
                    poly = np.concatenate([poly, np.ones_like(poly[:, :1])], axis=-1)
                    poly2lidar = pose.getI().dot(poly.T)
                    poly2cam = T_lidar2cam.dot(poly2lidar)[:3, :]
                    poly2car = extrin_rot.dot(poly2cam).T + extrin_tran
                    pts = poly2car[:, :2]
                    # print ('pts = ', pts,  self.bx[:2], self.dx[:2])
                    pts = np.round(
                    (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                    ).astype(np.int32)
                    #print ('pts00000 = ', pts)
                    pts[:, [1, 0]] = pts[:, [0, 1]]
                    #print ('pts11111 = ', pts)
                    if label == 5: # 路沿
                        # cv2.polylines(local_map[label-1], [pts], 0, 1, 2)
                        # cv2.fillPoly(show_seg_map, [pts], (0, 0, 255))
                        cv2.polylines(local_map[label-1], [pts], 0, 1, 2)
                    elif label > 0: # 除了路沿之外的几个真值
                        cv2.fillPoly(local_map[label-1], [pts], 1)
                        # cv2.fillPoly(show_seg_map, [pts], (0, 0, 255))
                    elif label == -1:#ignore
                        for jj in range(5):
                            cv2.polylines(local_map[jj], [pts], 1, -1, 10)
                            # cv2.polylines(local_map[jj], [pts], 1, -1, 1)
                            cv2.fillPoly(local_map[jj], [pts], -1)
            # cl_map = np.zeros((self.nx[0], self.nx[1], 3), dtype=np.uint8)                
            for roadmap_idx in concerned_roadmap_idxs:
                roadmap_idx= int(roadmap_idx)
                # print(self.roadmap_data,  roadmap_idx)
                road = roadmap_data[roadmap_idx]
                if 'cl' in road:
                    poly = road['cl']
                    poly = np.concatenate([poly, np.ones_like(poly[:, :1])], axis=-1)
                    poly2lidar = pose.getI().dot(poly.T)
                    poly2cam = T_lidar2cam.dot(poly2lidar)[:3, :]
                    poly2car = extrin_rot.dot(poly2cam).T + extrin_tran
                    pts = poly2car[:, :2]
                    pts = np.round(
                    (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                    ).astype(np.int32)
                    pts[:, [1, 0]] = pts[:, [0, 1]]
                    # print ('pts = ', pts)
                    # cv2.polylines(local_map[5], [pts], 0, 1, 2)
                    # cv2.polylines(show_seg_map, [pts], 0, (255, 255, 255), 1)
                    cv2.polylines(local_map[5], [pts], 0, 1, 1)
            # cv2.imwrite('show_seg_map.jpg', show_seg_map)
            #mesh
            valid_local_area = np.zeros((self.nx[0], self.nx[1]))
            valid_lf_area = np.zeros((self.nx[0], self.nx[1]))
            for obj_idx in concerned_obj_idxs:
               #    if obj_idx >= len(mesh_objs):
               #   print  (len(mesh_objs), obj_idx, image_path)
               mesh_obj = mesh_objs[obj_idx]
               vertices = np.asarray(mesh_obj.vertices)
               triangles= np.asarray(mesh_obj.triangles)
               if triangles.shape[0] == 0:
                  continue

               vertices = np.concatenate([vertices, np.ones_like(vertices[:, :1])], axis=-1)
               vertices2lidar = pose.getI().dot(vertices.T)
               vertices2cam = T_lidar2cam.dot(vertices2lidar)[:3, :]
               vertices2car = extrin_rot.dot(vertices2cam).T + extrin_tran
               pts = vertices2car[:, :2]
               pts = np.round(
                    (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                    ).astype(np.int32)
               pts[:, [1, 0]] = pts[:, [0, 1]]
               triangles = list(pts[triangles])
               #for kk in range(triangles.shape[0]):
               #    triangle = pts[triangles[kk]]
               cv2.fillPoly(valid_local_area, triangles, 1)
               cv2.fillPoly(valid_lf_area, triangles, 1)

            #ob
            for map_idx in concerned_ob_map_idxs:
                poly, label = ob_maps[map_idx]
                if 1:
                    if not np.all(poly[0] == poly[-1]):
                       continue
                    poly1 = Polygon(poly)
                    if not poly1.is_valid:
                       continue
                    poly = np.concatenate([poly, np.ones_like(poly[:, :1])], axis=-1)
                    poly2lidar = pose.getI().dot(poly.T)
                    poly2cam = T_lidar2cam.dot(poly2lidar)[:3, :]
                    poly2car = extrin_rot.dot(poly2cam).T + extrin_tran
                    pts = poly2car[:, :2]
                    pts = np.round(
                    (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                    ).astype(np.int32)
                    pts[:, [1, 0]] = pts[:, [0, 1]]
                    if label == 1:
                        cv2.fillPoly(valid_local_area, [pts], 1)
                        cv2.fillPoly(valid_lf_area, [pts], 2)

            for map_idx in concerned_ignore_map_idxs:
                poly, label = ignore_maps[map_idx]
                if 1:
                    poly = np.concatenate([poly, np.ones_like(poly[:, :1])], axis=-1)
                    poly2lidar = pose.getI().dot(poly.T)
                    poly2cam = T_lidar2cam.dot(poly2lidar)[:3, :]
                    poly2car = extrin_rot.dot(poly2cam).T + extrin_tran
                    pts = poly2car[:, :2]
                    pts = np.round((pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]).astype(np.int32)
                    pts[:, [1, 0]] = pts[:, [0, 1]]
                    # if label == 1:
                    #     cv2.fillPoly(valid_local_area, [pts], 1)
                    #     cv2.fillPoly(valid_lf_area, [pts], 1)
                    if label == -2:
                        cv2.polylines(valid_lf_area, [pts], 1, 0, 10)
                        cv2.fillPoly(valid_lf_area, [pts], 0)
                    if label == -1:
                        cv2.polylines(valid_local_area, [pts], 1, 0, 10)
                        cv2.fillPoly(valid_local_area, [pts], 0)

                        cv2.polylines(valid_lf_area, [pts], 1, 0, 10)
                        cv2.fillPoly(valid_lf_area, [pts], 0)
            
            # cv2.imwrite('valid_local.jpg', valid_local_area)
            # cv2.imwrite('valid_lf.jpg', valid_lf_area)
            local_map[5][valid_lf_area==2] = 0
            local_map[5][valid_lf_area==0] = -1
            for jj in range(5):
                local_map[jj][valid_local_area==0] = -1
 
            local_maps.append(torch.Tensor(np.stack(local_map)))
            valid_lf_areas.append(valid_lf_area)
        return torch.stack(local_maps, axis= 0), valid_lf_areas

    def generate_lf(self, sce_id, rec, extrin_rot, extrin_tran):
        data_infos = self.data_infos_list[sce_id]
        roadmap_samples = self.roadmap_samples_list[sce_id]
        T_lidar2cam = self.T_lidar2cam_list[sce_id]

        lf_label = np.zeros((1, self.nx[0], self.nx[1]))-1
        lf_norm = np.zeros((2, 2, self.nx[0], self.nx[1]))-999
        lf_kappa = np.zeros((2, self.nx[0], self.nx[1]))
        image_path, pose, _ = rec
        data_info = data_infos[os.path.split(image_path)[-1]]
        concerned_roadmap_idxs = data_info['concerned_roadmap_idxs']
        ipose = pose.getI()
        sample_pts = []
        sample_idxs =[]
        # a2 = time()
        for roadmap_idx in concerned_roadmap_idxs:
            roadmap_idx = int(roadmap_idx)
            if roadmap_idx not in roadmap_samples:
                continue
            if "left" in roadmap_samples[roadmap_idx]:
                # a1 =time()
                mesh = np.load(roadmap_samples[roadmap_idx]["left"])

                # print("left np.load耗时：",  time()-a1)
                ys = np.arange(mesh.shape[0])
                np.random.shuffle(ys)
                ys = ys[:3]
                mesh = mesh[ys]
                xs, ys = np.meshgrid(np.arange(mesh.shape[1]), ys)
                sample_pts.append(np.reshape(mesh, (-1, 6)))
                xs = np.reshape(xs, (-1, 1))
                ys = np.reshape(ys, (-1, 1))
                idxs = np.concatenate([np.zeros_like(ys, dtype=np.int32)+roadmap_idx, np.zeros_like(ys, dtype=np.int32), ys, xs], axis=-1)
                sample_idxs.append(idxs)                
            if "right" in roadmap_samples[roadmap_idx]:
                # a1 =time()
                mesh = np.load(roadmap_samples[roadmap_idx]["right"])
                # print("right np.load耗时：",  time()-a1)
                ys = np.arange(mesh.shape[0])
                np.random.shuffle(ys)
                ys = ys[:3]
                mesh = mesh[ys]
                xs, ys = np.meshgrid(np.arange(mesh.shape[1]), ys)
                sample_pts.append(np.reshape(mesh, (-1, 6)))
                xs = np.reshape(xs, (-1, 1))
                ys = np.reshape(ys, (-1, 1))
                idxs = np.concatenate([np.zeros_like(ys, dtype=np.int32)+roadmap_idx, np.ones_like(ys, dtype=np.int32), ys, xs], axis=-1)
                sample_idxs.append(idxs)
        # print("=====222222222222==：",  time()-a2)
        # a3 = time()
        data2car = None
        idxs = None
        if len(sample_pts) > 0:
            sample_pts = np.concatenate(sample_pts, axis=0)
            idxs = np.concatenate(sample_idxs, axis=0)

            data = sample_pts[..., :3]
            data= np.concatenate([data, np.ones_like(data[:, :1])], axis=-1)
            kappa = np.reshape(sample_pts[..., 3], (-1, 1))
            norm = np.reshape(sample_pts[..., 4:], (-1, 2))
            norm = np.concatenate([norm, np.zeros_like(kappa)], axis=-1)

            data2lidar = ipose.dot(data.T)
            data2cam = T_lidar2cam.dot(data2lidar)[:3, :]
            data2car = (extrin_rot.dot(data2cam).T + extrin_tran).A
            pts = data2car[..., :2]
            pts = np.round(
                        (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                        ).astype(np.int32)
            #print(key, data.shape, pts.shape)
            pts[:, [1, 0]] = pts[:, [0, 1]]

            norm2lidar= ipose[:3, :3].dot(norm.T)
            norm2cam = T_lidar2cam[:3, :3].dot(norm2lidar)[:3, :]
            norm2car = extrin_rot.dot(norm2cam).T.A
            norms = norm2car[..., :2]
            norms = norms/np.linalg.norm(norms, axis=-1).reshape((-1, 1))
            norms[:, [1, 0]] = norms[:, [0, 1]]
            norms[:,1] -= 1

            mask = np.logical_and(np.logical_and(pts[:, 0] >=0, pts[:, 0] <self.nx[1]), np.logical_and(pts[:, 1] >=0, pts[:, 1] <self.nx[0]))
            xs = pts[:, 0][mask]
            ys = pts[:, 1][mask]
            if xs.shape[0] > 0:
                norms = norms[mask]
                kappas = kappa[mask]
                lf_label[:, ys, xs] = 0
                lf_norm[0, :, ys, xs] = norms
                lf_norm[1, :, ys, xs] = norms
                lf_kappa[0, ys, xs] = kappas.flatten()
                lf_kappa[1, ys, xs] = kappas.flatten()
        # print("=====3333333333333==：",  time()-a3)
        return lf_label, lf_norm, lf_kappa, data2car, idxs


    def generate_lf_chatgpt(self, sce_id, rec, extrin_rot, extrin_tran):
        data_infos = self.data_infos_list[sce_id]
        roadmap_samples = self.roadmap_samples_list[sce_id]
        T_lidar2cam = self.T_lidar2cam_list[sce_id]

        lf_label = np.full((1, self.nx[0], self.nx[1]), -1)
        lf_norm = np.full((2, 2, self.nx[0], self.nx[1]), -999)
        lf_kappa = np.zeros((2, self.nx[0], self.nx[1]))
        image_path, pose, _ = rec
        data_info = data_infos[os.path.split(image_path)[-1]]
        concerned_roadmap_idxs = data_info['concerned_roadmap_idxs']
        ipose = pose.getI()
        sample_pts = []
        sample_idxs = []

        for roadmap_idx in concerned_roadmap_idxs:
            roadmap_idx = int(roadmap_idx)
            if roadmap_idx not in roadmap_samples:
                continue
            for side in ["left", "right"]:
                if side in roadmap_samples[roadmap_idx]:
                    mesh = np.load(roadmap_samples[roadmap_idx][side])
                    ys = np.random.choice(mesh.shape[0], size=3, replace=False)
                    mesh = mesh[ys]
                    xs, ys = np.meshgrid(np.arange(mesh.shape[1]), ys)
                    sample_pts.append(np.reshape(mesh, (-1, 6)))
                    xs = np.reshape(xs, (-1, 1))
                    ys = np.reshape(ys, (-1, 1))
                    idxs = np.concatenate([np.full_like(ys, roadmap_idx, dtype=np.int32),
                                        np.ones_like(ys, dtype=np.int32),
                                        ys, xs], axis=-1)
                    sample_idxs.append(idxs)

        data2car = None
        idxs = None
        if sample_pts:
            sample_pts = np.concatenate(sample_pts, axis=0)
            idxs = np.concatenate(sample_idxs, axis=0)

            data = sample_pts[..., :3]
            data = np.concatenate([data, np.ones_like(data[:, :1])], axis=-1)
            kappa = np.reshape(sample_pts[..., 3], (-1, 1))
            norm = np.reshape(sample_pts[..., 4:], (-1, 2))
            norm = np.concatenate([norm, np.zeros_like(kappa)], axis=-1)

            data2lidar = ipose.dot(data.T)
            data2cam = T_lidar2cam.dot(data2lidar)[:3, :]
            data2car = (extrin_rot.dot(data2cam).T + extrin_tran).A
            pts = data2car[..., :2]
            pts = np.round((pts - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]

            norm2lidar = ipose[:3, :3].dot(norm.T)
            norm2cam = T_lidar2cam[:3, :3].dot(norm2lidar)[:3, :]
            norm2car = extrin_rot.dot(norm2cam).T.A
            norms = norm2car[..., :2]
            norms = norms / np.linalg.norm(norms, axis=-1).reshape((-1, 1))
            norms[:, [1, 0]] = norms[:, [0, 1]]
            norms[:, 1] -= 1

            mask = np.logical_and.reduce((
                pts[:, 0] >= 0,
                pts[:, 0] < self.nx[1],
                pts[:, 1] >= 0,
                pts[:, 1] < self.nx[0]
            ))
            xs = pts[:, 0][mask]
            ys = pts[:, 1][mask]
            if xs.shape[0] > 0:
                norms = norms[mask]
                kappas = kappa[mask]
                lf_label[:, ys, xs] = 0
                lf_norm[0, :, ys, xs] = norms
                lf_norm[1, :, ys, xs] = norms
                lf_kappa[0, ys, xs] = kappas.flatten()
                lf_kappa[1, ys, xs] = kappas.flatten()

        return lf_label, lf_norm, lf_kappa, data2car, idxs


    def generate_lf_lmdb(self, sce_id, rec, extrin_rot, extrin_tran):
        data_infos = self.data_infos_list[sce_id]
        roadmap_samples = self.roadmap_samples_list[sce_id]
        T_lidar2cam = self.T_lidar2cam_list[sce_id]

        lf_label = np.zeros((1, self.nx[0], self.nx[1]))-1
        lf_norm = np.zeros((2, 2, self.nx[0], self.nx[1]))-999
        lf_kappa = np.zeros((2, self.nx[0], self.nx[1]))
        image_path, pose, _ = rec
        data_info = data_infos[os.path.split(image_path)[-1]]
        concerned_roadmap_idxs = data_info['concerned_roadmap_idxs']
        ipose = pose.getI()
        sample_pts = []
        sample_idxs =[]
        # a2 = time()
        for roadmap_idx in concerned_roadmap_idxs:
            roadmap_idx = int(roadmap_idx)
            if roadmap_idx not in roadmap_samples:
                continue
            if "left" in roadmap_samples[roadmap_idx]:
                # a1 =time()
                # mesh = np.load(roadmap_samples[roadmap_idx]["left"])
                # print('=====index: ', roadmap_samples[roadmap_idx]["left"])
                mesh = self.data_list[roadmap_samples[roadmap_idx]["left"]]
                # print("left lmdb耗时：",  time()-a1)
                ys = np.arange(mesh.shape[0])
                np.random.shuffle(ys)
                ys = ys[:3]
                mesh = mesh[ys]
                xs, ys = np.meshgrid(np.arange(mesh.shape[1]), ys)
                sample_pts.append(np.reshape(mesh, (-1, 6)))
                xs = np.reshape(xs, (-1, 1))
                ys = np.reshape(ys, (-1, 1))
                idxs = np.concatenate([np.zeros_like(ys, dtype=np.int32)+roadmap_idx, np.zeros_like(ys, dtype=np.int32), ys, xs], axis=-1)
                sample_idxs.append(idxs)                
            if "right" in roadmap_samples[roadmap_idx]:
                # a1 =time()
                # mesh = np.load(roadmap_samples[roadmap_idx]["right"])
                mesh = self.data_list[roadmap_samples[roadmap_idx]["right"]]
                # print("right np.load耗时：",  time()-a1)
                ys = np.arange(mesh.shape[0])
                np.random.shuffle(ys)
                ys = ys[:3]
                mesh = mesh[ys]
                xs, ys = np.meshgrid(np.arange(mesh.shape[1]), ys)
                sample_pts.append(np.reshape(mesh, (-1, 6)))
                xs = np.reshape(xs, (-1, 1))
                ys = np.reshape(ys, (-1, 1))
                idxs = np.concatenate([np.zeros_like(ys, dtype=np.int32)+roadmap_idx, np.ones_like(ys, dtype=np.int32), ys, xs], axis=-1)
                sample_idxs.append(idxs)
        # print("=====222222222222==：",  time()-a2)
        # a3 = time()
        data2car = None
        idxs = None
        if len(sample_pts) > 0:
            sample_pts = np.concatenate(sample_pts, axis=0)
            idxs = np.concatenate(sample_idxs, axis=0)

            data = sample_pts[..., :3]
            data= np.concatenate([data, np.ones_like(data[:, :1])], axis=-1)
            kappa = np.reshape(sample_pts[..., 3], (-1, 1))
            norm = np.reshape(sample_pts[..., 4:], (-1, 2))
            norm = np.concatenate([norm, np.zeros_like(kappa)], axis=-1)

            data2lidar = ipose.dot(data.T)
            data2cam = T_lidar2cam.dot(data2lidar)[:3, :]
            data2car = (extrin_rot.dot(data2cam).T + extrin_tran).A
            pts = data2car[..., :2]
            pts = np.round(
                        (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                        ).astype(np.int32)
            #print(key, data.shape, pts.shape)
            pts[:, [1, 0]] = pts[:, [0, 1]]

            norm2lidar= ipose[:3, :3].dot(norm.T)
            norm2cam = T_lidar2cam[:3, :3].dot(norm2lidar)[:3, :]
            norm2car = extrin_rot.dot(norm2cam).T.A
            norms = norm2car[..., :2]
            norms = norms/np.linalg.norm(norms, axis=-1).reshape((-1, 1))
            norms[:, [1, 0]] = norms[:, [0, 1]]
            norms[:,1] -= 1

            mask = np.logical_and(np.logical_and(pts[:, 0] >=0, pts[:, 0] <self.nx[1]), np.logical_and(pts[:, 1] >=0, pts[:, 1] <self.nx[0]))
            xs = pts[:, 0][mask]
            ys = pts[:, 1][mask]
            if xs.shape[0] > 0:
                norms = norms[mask]
                kappas = kappa[mask]
                lf_label[:, ys, xs] = 0
                lf_norm[0, :, ys, xs] = norms
                lf_norm[1, :, ys, xs] = norms
                lf_kappa[0, ys, xs] = kappas.flatten()
                lf_kappa[1, ys, xs] = kappas.flatten()
        # print("=====3333333333333==：",  time()-a3)
        return lf_label, lf_norm, lf_kappa, data2car, idxs

    def get_localmap_lf(self, sce_id, recs, extrin_rot, extrin_tran, valid_lf_areas):
        # angles = np.linspace(-np.pi/4+np.pi,np.pi/4+np.pi,8).tolist()
        # vectors = np.array([(math.cos(q), math.sin(q)) for q in angles])
        # a2 = time()
        bb = box(self.bx[0]-self.dx[0]/2., self.bx[1]-self.dx[1]/2., self.bx[0]-self.dx[0]/2.+self.nx[0]*self.dx[0], self.bx[1]-self.dx[1]/2.+self.nx[1]*self.dx[1])
        fork_labels = []
        fork_patchs = []
        fork_scales = []
        fork_offsets = []
        fork_oris = []

        lf_labels = []
        lf_norms = []
        lf_kappas = []

        data_infos = self.data_infos_list[sce_id]
        T_lidar2cam = self.T_lidar2cam_list[sce_id]
        roadmap_forks = self.roadmap_forks_list[sce_id]


        # lf切换数据集
        # if self.sce_id != sce_id:
        #     self.sce_id = sce_id
        #     path = self.dataset_list[sce_id]
        #     max_dbs = 10
        #     lf_env = lmdb.open(self.lf_env_path, readonly=True, max_dbs=max_dbs)  # 只读模式打开LMDB环境
        #     with lf_env.begin() as lf_txn:            
        #         lf_key = path.encode()  # 要读取的lf的键
        #         lf_data = lf_txn.get(lf_key)
        #         # 反序列化数据为列表
        #         self.data_list = pickle.loads(lf_data)
            
        #     lf_env.close()

        for ii, sample in enumerate(recs):
            # print ('ii = ', ii)
            
            valid_lf_area = valid_lf_areas[ii]
            image_path, pose, _ = sample
            data_info = data_infos[os.path.split(image_path)[-1]]
            # concerned_roadmap_idxs = data_info['concerned_roadmap_idxs']
            ipose = pose.getI()
            # a1 = time()
            lf_label, lf_norm, lf_kappa, lf_sample_pts, lf_sample_idxs = self.generate_lf_lmdb(sce_id, sample, extrin_rot, extrin_tran)
            # lf_label, lf_norm, lf_kappa, lf_sample_pts, lf_sample_idxs = self.generate_lf(sce_id, sample, extrin_rot, extrin_tran)
            # print("generate_lf_lmdb耗时：",  time()-a1)
            fork_label = np.zeros((self.nx[0], self.nx[1]))
            fork_patch = np.zeros((2, self.crop_size, self.crop_size)) - 1.
            fork_scale = np.ones((1))
            fork_offset = np.zeros((2))
            fork_ori = np.zeros((1))
            fork_sample_idxs = []
            fork_sample_pts = []
            concerned_roadmap_forks = data_info['concerned_roadmap_forks']
            

            # a3 = time()

            for key in concerned_roadmap_forks:
                # a2 = time()
                area = np.array(roadmap_forks[key]['area'], dtype=np.float64)
                # print("np.array耗时：",  time()-a2)
                area = np.concatenate([area, np.ones_like(area[:, :1])], axis=-1)
                area2lidar = ipose.dot(area.T)
                area2cam = T_lidar2cam.dot(area2lidar)[:3, :]
                area2car = extrin_rot.dot(area2cam).T + extrin_tran
                area2car = area2car[:, :2]
                area = Polygon(area2car)
                if bb.intersects(area):
                    pts = np.round(
                        (area2car - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                        ).astype(np.int32)
                    pts[:, [1, 0]] = pts[:, [0, 1]]
                    cv2.fillPoly(fork_label, [pts], 1)
                    fork_sample_idxs.append(roadmap_forks[key]['idxs'])
                    fork_sample_pts.append(roadmap_forks[key]['pts'])

            # 
            lf_label[:, valid_lf_area == 1] = 0
            lf_label[:, fork_label==1] = -1

            if len(fork_sample_idxs) > 0: # 计算分叉汇入区域的真值
                fork_sample_idxs = np.concatenate(fork_sample_idxs)
                fork_sample_pts = np.concatenate(fork_sample_pts)
                # print('fork_sample_pts = ', fork_sample_pts, fork_sample_pts.shape)
                fork_sample_kappa0 = fork_sample_pts[..., 3]
                fork_sample_norm0 = np.reshape(fork_sample_pts[..., 4:6], (-1, 2))
                fork_sample_norm0 = np.concatenate([fork_sample_norm0, np.zeros_like(fork_sample_norm0[..., 0:1])], axis=-1)

                fork_sample_kappa1 = fork_sample_pts[..., 6]
                fork_sample_norm1 = np.reshape(fork_sample_pts[..., 7:9], (-1, 2))
                fork_sample_norm1 = np.concatenate([fork_sample_norm1, np.zeros_like(fork_sample_norm1[..., 0:1])], axis=-1)

                fork_sample_pts = np.concatenate([fork_sample_pts[..., :3], np.ones_like(fork_sample_pts[:, :1])], axis=-1)

                fork_sample_pts2lidar = ipose.dot(fork_sample_pts.T)
                fork_sample_pts2cam = T_lidar2cam.dot(fork_sample_pts2lidar)[:3, :]
                fork_sample_pts2car = (extrin_rot.dot(fork_sample_pts2cam).T + extrin_tran).A

                fork_pts = fork_sample_pts2car[..., :2]
                fork_pts = np.round(
                                    (fork_pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                                    ).astype(np.int32)
                #print(key, data.shape, pts.shape)
                fork_pts[:, [1, 0]] = fork_pts[:, [0, 1]]


                fork_sample_norm0_2lidar= ipose[:3, :3].dot(fork_sample_norm0.T)
                fork_sample_norm0_2cam = T_lidar2cam[:3, :3].dot(fork_sample_norm0_2lidar)[:3, :]
                fork_sample_norm0_2car = extrin_rot.dot(fork_sample_norm0_2cam).T.A
                fork_sample_norm0_2car = fork_sample_norm0_2car[..., :2]
                fork_sample_norm0_2car = fork_sample_norm0_2car/np.linalg.norm(fork_sample_norm0_2car, axis=-1).reshape((-1, 1))
                fork_sample_norm0_2car[:, [1, 0]] = fork_sample_norm0_2car[:, [0, 1]]

                fork_sample_norm1_2lidar= ipose[:3, :3].dot(fork_sample_norm1.T)
                fork_sample_norm1_2cam = T_lidar2cam[:3, :3].dot(fork_sample_norm1_2lidar)[:3, :]
                fork_sample_norm1_2car = extrin_rot.dot(fork_sample_norm1_2cam).T.A
                fork_sample_norm1_2car = fork_sample_norm1_2car[..., :2]
                fork_sample_norm1_2car = fork_sample_norm1_2car/np.linalg.norm(fork_sample_norm1_2car, axis=-1).reshape((-1, 1))
                fork_sample_norm1_2car[:, [1, 0]] = fork_sample_norm1_2car[:, [0, 1]]

                mask = np.logical_and(np.logical_and(fork_pts[:, 0] >=0, fork_pts[:, 0] <self.nx[1]), np.logical_and(fork_pts[:, 1] >=0, fork_pts[:, 1] <self.nx[0]))
                fork_sample_norm0_2car=fork_sample_norm0_2car[mask]
                fork_sample_norm1_2car=fork_sample_norm1_2car[mask]
                fork_sample_kappa0 =fork_sample_kappa0[mask]
                fork_sample_kappa1 =fork_sample_kappa1[mask]

                fork_pts=fork_pts[mask]
                tmp = np.cross(fork_sample_norm0_2car, fork_sample_norm1_2car)
                ys = fork_pts[..., 1]
                xs = fork_pts[..., 0]
                fork_sample_norm0_2car[:,1] -= 1
                fork_sample_norm1_2car[:,1] -= 1
                lf_label[:, ys, xs] = 1
                lf_norm[(tmp>0).astype(np.int32), :, ys, xs] = fork_sample_norm0_2car # Headings
                lf_norm[(tmp<=0).astype(np.int32), :, ys, xs] = fork_sample_norm1_2car
                lf_kappa[(tmp>0).astype(np.int32), ys, xs] = fork_sample_kappa0 # 曲率
                lf_kappa[(tmp<=0).astype(np.int32), ys, xs] = fork_sample_kappa1

            #     mask = np.logical_and(np.logical_and(fork_sample_pts2car[:, 0] >=10., fork_sample_pts2car[:, 0] <110.), np.logical_and(fork_sample_pts2car[:, 1] >=-10., fork_sample_pts2car[:, 1] <10.))
            #     idxs = np.where(mask)[0]
            #     if idxs.shape[0] > 0:
            #         fork_patch = fork_patch*0.
            #         idx = idxs[random.randint(0, idxs.shape[0]-1)]
            #         sample_idx = fork_sample_idxs[idx]
            #         sample_pt2car = fork_sample_pts2car[idx][:2].reshape((1, -1))

            #         fork_offset = sample_pt2car[0]

            #         poly0 = np.load(self.roadmap_samples[sample_idx[0]]["left" if sample_idx[1]==0 else "right"])[sample_idx[2]][..., :3]
            #         poly1 = np.load(self.roadmap_samples[sample_idx[3]]["left" if sample_idx[4]==0 else "right"])[sample_idx[5]][..., :3]

            #         poly0 = np.concatenate([poly0, np.ones_like(poly0[:, :1])], axis=-1)
            #         poly0_2lidar = ipose.dot(poly0.T)
            #         poly0_2cam = self.T_lidar2cam.dot(poly0_2lidar)[:3, :]
            #         poly0_2car = extrin_rot.dot(poly0_2cam).T + extrin_tran
            #         poly0_2offset = poly0_2car[:, :2] - sample_pt2car
            #         pts0 = np.round(
            #                     poly0_2offset/fork_scale/ self.dx[:2] + np.array([[self.crop_size/2., self.crop_size/2.]], dtype=np.float64)
            #                     ).astype(np.int32)
            #         pts0[:, [1, 0]] = pts0[:, [0, 1]]

            #         poly1 = np.concatenate([poly1, np.ones_like(poly1[:, :1])], axis=-1)
            #         poly1_2lidar = ipose.dot(poly1.T)
            #         poly1_2cam = self.T_lidar2cam.dot(poly1_2lidar)[:3, :]
            #         poly1_2car = extrin_rot.dot(poly1_2cam).T + extrin_tran
            #         poly1_2offset = poly1_2car[:, :2] - sample_pt2car
            #         pts1 = np.round(
            #                     poly1_2offset / self.dx[:2] +  np.array([[48., 48.]], dtype=np.float64)
            #                     ).astype(np.int32)
            #         pts1[:, [1, 0]] = pts1[:, [0, 1]]

            #         cross = np.cross(poly0_2offset[-1]-poly0_2offset[0], poly1_2offset[-1]-poly1_2offset[0])
            #         if cross > 0:
            #             cv2.polylines(fork_patch[0], [pts0], 0, 1, 1)
            #             cv2.polylines(fork_patch[1], [pts1], 0, 1, 1)
            #         else:
            #             cv2.polylines(fork_patch[1], [pts0], 0, 1, 1)
            #             cv2.polylines(fork_patch[0], [pts1], 0, 1, 1)

            #         min_y0 = pts0[:, 1].min()
            #         max_y0 = pts0[:, 1].max()
            #         min_y1 = pts1[:, 1].min()
            #         max_y1 = pts1[:, 1].max()
            #         min_y = max(min_y0, min_y1)
            #         max_y = min(max_y0, max_y1)
            #         if min_y > 0:
            #             fork_patch[:, :min_y] = -1
            #         if max_y < self.crop_size:
            #             fork_patch[:, max_y:] = -1
            #     # else:
            #     #     print("fork outside!!")

            # elif not lf_sample_pts is None:
            #     mask = np.logical_and(np.logical_and(lf_sample_pts[:, 0] >=10., lf_sample_pts[:, 0] <110.), np.logical_and(lf_sample_pts[:, 1] >=-10., lf_sample_pts[:, 1] <10.))
            #     idxs = np.where(mask)[0]
            #     if idxs.shape[0] > 0:
            #         idx = idxs[random.randint(0, idxs.shape[0]-1)]
            #         sample_idx = lf_sample_idxs[idx]
            #         sample_pt = lf_sample_pts[idx]
            #         mask = np.logical_and(np.logical_and(lf_sample_idxs[..., 0] == sample_idx[0], lf_sample_idxs[..., 1] == sample_idx[1]), lf_sample_idxs[..., 2] == sample_idx[2])
            #         #print(self.roadmap_samples[sample_idx[0]]["left"].shape, self.roadmap_samples[sample_idx[0]]["right"].shape, np.sum(mask), sample_idx, np.sum(mask1))
            #         poly2car = lf_sample_pts[mask]

            #         if poly2car.shape[0] > 0:
            #             fork_patch = fork_patch*0.
            #             sample_pt2car = sample_pt[:2].reshape((1, -1))
            #             fork_offset = sample_pt2car[0]
            #             pts = poly2car[:, :2] - sample_pt2car
            #             pts = np.round(
            #                         pts/fork_scale/ self.dx[:2] + np.array([[self.crop_size/2., self.crop_size/2.]], dtype=np.float64)
            #                         ).astype(np.int32)

            #             pts[:, [1, 0]] = pts[:, [0, 1]]

            #             cv2.polylines(fork_patch[0], [pts], 0, 1, 1)
            #             cv2.polylines(fork_patch[1], [pts], 0, 1, 1)
            #             min_y = pts[:, 1].min()
            #             max_y = pts[:, 1].max()
            #             if min_y > 0:
            #                 fork_patch[:, :min_y] = -1
            #             if max_y < self.crop_size:
            #                 fork_patch[:, max_y:] = -1
            #         else:
            #             print("road outside error!!")
            #     # else:
            #     #     print("road outside!!")
            lf_label[:, valid_lf_area == 0] = -1
            lf_label[:, valid_lf_area == 2] = 0

            lf_norm[0, :, valid_lf_area == 0] = -999
            lf_norm[1, :, valid_lf_area == 0] = -999

            lf_norm[0, :, valid_lf_area == 2] = -999
            lf_norm[1, :, valid_lf_area == 2] = -999

            lf_labels.append(torch.Tensor(lf_label))
            lf_norms.append(torch.Tensor(lf_norm).view(-1, self.nx[0], self.nx[1])) # 趋势线方向
            lf_kappas.append(torch.Tensor(lf_kappa))
            
            fork_patchs.append(torch.Tensor(fork_patch))
            fork_scales.append(torch.Tensor(fork_scale))
            fork_offsets.append(torch.Tensor(fork_offset))
            fork_oris.append(torch.Tensor(fork_ori))
            # print("处理耗时:", time() - a3)
            
        return torch.stack(lf_labels, axis=0), torch.stack(lf_norms, axis=0), torch.stack(lf_kappas, axis=0), torch.stack(fork_patchs, axis=0), torch.stack(fork_scales, axis=0), torch.stack(fork_offsets, axis=0), torch.stack(fork_oris, axis=0)



    def choose_cams(self):
        if self.is_train and self.data_aug_conf['Ncams'] < len(self.data_aug_conf['cams']):
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.data_aug_conf['cams']
        return cams

    def __str__(self):
        return f"""TFmapData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.data_idxs)


class SegmentationData(TFmapData):
    def __init__(self, *args, **kwargs):
        super(SegmentationData, self).__init__(*args, **kwargs)
        self.buf = {}
        

    def __getitem__(self, index):
        sce_id, sce_id_ind = self.data_idxs[index] # sce_id 表示哪一段数据，sce_id_ind 表示第几帧图片
        sce_name = self.image_paths_list[sce_id] # 数据名称
        cams = self.choose_cams() # 对于训练集且data_aug_conf中Ncams<6的，随机选择摄像机通道，否则选择全部相机通道

        max_stride = min((len(self.ixes[sce_name][0]) - sce_id_ind)/self.seq_len, 5) # 计算最大的取帧间隔
  
        # print('sce_id= ', sce_id, sce_id_ind, index)
        # max_stride = min(int((len(sub_ixes)-index)/self.seq_len), 2)
        stride = np.random.randint(1, max_stride+1)
        # stride = 10
        # recs = [sub_ixes[ii] for ii in range(index, index+self.seq_len*stride, stride)]
        recs = [self.ixes[sce_name][0][ii] for ii in range(sce_id_ind, sce_id_ind+self.seq_len*stride, stride)] # 取出的十帧图像作为一个样本
        rot = self.ixes[sce_name][1] # 按索引取出sample # rot /tran/ K 相机参数
        tran = self.ixes[sce_name][2]
        K = self.ixes[sce_name][3]
        #runs
        # 扰动
        axisangle_limits = [[-4./180.*np.pi, 4./180.*np.pi], [-4./180.*np.pi, 4./180.*np.pi], [-4./180.*np.pi, 4./180.*np.pi]]
        #roll pitch yaw
        # axisangle_limits = [[-2./180.*np.pi, 2./180.*np.pi], [-3./180.*np.pi, 3./180.*np.pi], [-2./180.*np.pi, 2./180.*np.pi]]
        tran_limits = [[-2., 2.], [-1., 1.], [-1., 1.]]

        axisangle_noises = [np.random.uniform(*angle_limit) for angle_limit in axisangle_limits]
        tran_noises = [np.random.uniform(*tran_limit) for tran_limit in tran_limits]
        noise_rot = euler_angles_to_rotation_matrix(axisangle_noises)
        noise_tran = np.array(tran_noises)
        extrin_rot = noise_rot.dot(rot)
        extrin_tran = noise_rot.dot(tran).T + noise_tran

        # 下面三个函数是为了获取真值
        a1 = time()
        # imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data_tf(recs, cams, rot, tran, K)#extrin_rot, extrin_tran)
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data_tf(recs, cams, extrin_rot, extrin_tran, K) # 读取图像数据、相机参数和数据增强的像素坐标映射关系
        # imgs: 6,3,128,352  图像数据
        # rots: 6,3,3  相机坐标系到自车坐标系的旋转矩阵
        # trans: 6,3  相机坐标系到自车坐标系的平移向量
        # intrins: 6,3,3  相机内参
        # post_rots: 6,3,3  数据增强的像素坐标旋转映射关系
        # post_trans: 6,3  数据增强的像素坐标平移映射关系

        # print("get_image_data_tf：", time() - a1)

        # binimg = self.get_localmap_seg(sce_id, recs, rot, tran)#extrin_rot, extrin_tran)
        a2 = time()
        binimg, valid_lf_areas = self.get_localmap_seg(sce_id, recs, extrin_rot, extrin_tran)
        # print("get_localmap_seg：", time() - a2)

        a3 = time()
        # lf_label, lf_norm, lf_kappa, fork_patch, fork_scale, fork_offset, fork_ori = self.get_localmap_lf(sce_id, recs, rot, tran)
        lf_label, lf_norm, lf_kappa, fork_patch, fork_scale, fork_offset, fork_ori = self.get_localmap_lf(sce_id, recs, extrin_rot, extrin_tran, valid_lf_areas)
        # print("get_localmap_lf：", time() - a3)
        # print("preprocess：", time() - a1)
        #fork_offset[..., 0] = 20.
        #print(fork_offset)
        return imgs, rots, trans, intrins, post_rots, post_trans, binimg, lf_label, lf_norm, lf_kappa, fork_patch, fork_scale, fork_offset, fork_ori

class Segmentation1Data(TFmapData):
    def __init__(self, *args, **kwargs):
        super(Segmentation1Data, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        # index = self.data_idxs[index]
        # index = 1161
        sce_id, sce_id_ind = self.data_idxs[index]
        sce_name = self.image_paths_list[sce_id]
        cams = self.choose_cams()
        # sub_ixes = []

        # for i in range(len(self.ixes)):
        #     if index >= self.ixes[i][0][0] and index < self.ixes[i][0][1]:
        #         sub_ixes = self.ixes[i][1]
        #         index -= self.ixes[i][0][0]

        # sces_len = [(len(self.ixes[i][0]) - self.seq_len) for i in self.ixes.keys()]
        # scenes = [i for i in self.ixes.keys()]
        # sce_id = [i for i in range(len(sces_len)) if (sum(sces_len[:i]) <= index and sum(sces_len[:i+1]) > index)][0]
        # sce_id_ind = index - sum(sces_len[:sce_id])
        max_stride = min((len(self.ixes[sce_name][0]) - sce_id_ind)/self.seq_len, 5)

        # print('sce_id= ', sce_id, len(sces_len), index)
        stride = np.random.randint(1, max_stride+1)
        stride = 1
        # recs = [sub_ixes[ii] for ii in range(index, index+self.seq_len*stride, stride)]
        recs = [self.ixes[sce_name][0][ii] for ii in range(sce_id_ind, sce_id_ind+self.seq_len*stride, stride)]
        rot = self.ixes[sce_name][1]
        tran = self.ixes[sce_name][2]
        K = self.ixes[sce_name][3]
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data_tf(recs, cams, rot, tran, K)#extrin_rot, extrin_tran)

        binimg, valid_lf_areas = self.get_localmap_seg(sce_id, recs, rot, tran)#extrin_rot, extrin_tran)
        lf_label, lf_norm, lf_kappa, fork_patch, fork_scale, fork_offset, fork_ori = self.get_localmap_lf(sce_id, recs, rot, tran, valid_lf_areas)

        idx = torch.Tensor([index])
        img_paths = [rec[0] for rec in recs]

        return imgs, rots, trans, intrins, post_rots, post_trans, binimg, lf_label, lf_norm, lf_kappa, fork_patch, fork_scale, fork_offset, fork_ori, idx, img_paths

def worker_rnd_init(x): # x是线程id
    np.random.seed(13 + x)

def get_data_param(path_idxs, dataroot, seq_len, is_train = True):
    data_infos_list = []
    param_infos_list= []
    vp_infos_list = []
    mesh_objs_list = []
    road_map_data_list = []
    roadmap_samples_list = []
    roadmap_forks_list = []
    map_paths_list = []
    ignore_map_paths_list = []
    ob_map_paths_list = []
    image_paths_list = []
    idxs = []

    # 使用lmdb进行批量转化
    max_dbs = 10  # 新的最大数据库数量
    lf_path = dataroot+'/lf.lmdb'  
    image_path = dataroot+'/img.lmdb'
    datasets_lmdb = []  
    lmdb_exist_if = False   
    # 判断lf_path是否已存在
    # 检查LMDB数据库是否存在
    if os.path.exists(lf_path):
        # LMDB数据库已存在，打开现有数据库
        env = lmdb.open(lf_path, readonly=False)
        lmdb_exist_if = True
    else:
        # 创建LMDB环境
        env = lmdb.open(lf_path, map_size=1099511627776 * 6, max_dbs=max_dbs)  # 创建新的LMDB环境

    if os.path.exists(image_path):
        # LMDB数据库已存在，打开现有数据库
        env_img = lmdb.open(image_path, readonly=False)
    else:
        # 创建LMDB环境
        env_img = lmdb.open(image_path, map_size=1099511627776 * 6, max_dbs=max_dbs)  # 创建新的LMDB环境
    
    print("open env============")

    roadmap_sample_idx = 0
    
    convert_to_lmdb = True # 数据集的lf 是否要转化
    img_convert_to_lmdb = True # 数据集的图片是否要转化
    
    for sce_id, path in enumerate(path_idxs):
        param_path = path + '/gen/param_infos.json'
        with open(param_path, 'r') as ff :
            param_infos = json.load(ff)
            param_infos_list.append(param_infos)

        geojson_path = path + '/org/geojson'
        map_list0 = sorted(glob.glob(geojson_path+'/标线标注*.geojson'))
        map_list1 = sorted(glob.glob(geojson_path+'/箭头标注*.geojson'))
        map_list2 = sorted(glob.glob(geojson_path+'/路沿标注*.geojson'))
        # map_list3 = sorted(glob.glob(geojson_path+'/问题路段范围*.geojson'))
        map_paths = map_list0 + map_list1 + map_list2
        # ignore_map_paths = map_list3
        map_paths_list.append(map_paths)
        # ignore_map_paths_list.append(ignore_map_paths)

        ignore_map_path = geojson_path + '/问题路段范围.geojson'
        if os.path.exists(ignore_map_path):
            ignore_map_paths_list.append(sorted(glob.glob(ignore_map_path)))
        else:
            ignore_map_paths_list.append([])

        ob_map_path = geojson_path + '/非可行驶区域.geojson'
        if os.path.exists(ob_map_path):
            ob_map_paths_list.append(sorted(glob.glob(ob_map_path)))
        else:
            ob_map_paths_list.append([])
        roadmap_path = path + '/org/hdmap'
        roadmap_list0 = sorted(glob.glob(roadmap_path+'/SideLines.geojson'))
        roadmap_list1 = sorted(glob.glob(roadmap_path+'/LaneLines.geojson'))
        road_map_data = {}
        for map_file in roadmap_list0:
            with open(map_file, encoding='utf-8') as fp:
                geojson = json.load(fp)
                print(geojson['name'])
                features = geojson['features']
                for ii, feature in enumerate(features):
                    poly = feature['geometry']['coordinates']
                    type = feature['geometry']['type']
                    if type == 'MultiLineString':
                        for pp in poly:
                            data = np.array(pp, dtype=np.float64)
                            print(type, data.shape)
                    elif type == "LineString":
                        llid = feature['properties']['LLID']
                        if llid is None:
                            continue
                        line_type = feature['properties']['LineType']
                        if line_type == 4:
                            continue
                        line_id = feature['properties']['Line_ID']
                        handside = feature['properties']['HandSide']
                        data = np.array(poly, dtype=np.float64)
                        if llid not in road_map_data:
                            road_map_data[llid]={}
                            road_map_data[llid]["leftside"] = []
                            road_map_data[llid]["rightside"] = []

                        if handside == 1:
                            road_map_data[llid]["leftside"].append([line_id, data])
                        elif handside == 2:
                            road_map_data[llid]["rightside"].append([line_id, data])

                        print(type, data.shape)
                    elif type == "Polygon":
                        print(type)
                    elif type == "MultiPolygon":
                        print(type)
                        #for pp in poly:
                        #    data = np.array(pp, dtype=np.float64)
                            #print(type, data.shape)
                            #all_map_data.append(data)
                    elif type == "Point":
                        print(type)
                    else:
                        print(type)

        for map_file in roadmap_list1:
            with open(map_file, encoding='utf-8') as fp:
                geojson = json.load(fp)
                print(geojson['name'])
                features = geojson['features']
                for ii, feature in enumerate(features):
                    poly = feature['geometry']['coordinates']
                    type = feature['geometry']['type']
                    if type == 'MultiLineString':
                        for pp in poly:
                            data = np.array(pp, dtype=np.float64)
                            print(type, data.shape)
                    elif type == "LineString":
                        llid = int(feature['properties']['LLID'])
                        if llid is None:
                            continue
                        if llid not in road_map_data:
                            road_map_data[llid]={}
                            road_map_data[llid]["leftside"] = []
                            road_map_data[llid]["rightside"] = []

                        data = np.array(poly, dtype=np.float64)
                        road_map_data[llid]['cl'] = data
                        print(type, data.shape)
                    elif type == "Polygon":
                        print(type)
                    elif type == "MultiPolygon":
                        print(type)
                        #for pp in poly:
                        #    data = np.array(pp, dtype=np.float64)
                            #print(type, data.shape)
                            #all_map_data.append(data)
                    elif type == "Point":
                        print(type)
                    else:
                        print(type)
        road_map_data_list.append(road_map_data)
        
        roadmap_sample_path = path+'/gen/samples'
        roadmap_samples = {}
        # 原始的lf处理代码
        # for key in road_map_data.keys():
        #     left_name = roadmap_sample_path+ '/' +str(key)+"_l.npy"
        #     right_name = roadmap_sample_path + '/' +str(key)+"_r.npy"
        #     roadmap_samples[key] = {}
        #     if os.path.exists(left_name):
        #         roadmap_samples[key]["left"] = left_name
        #     if os.path.exists(right_name):
        #         roadmap_samples[key]["right"] = right_name
  
        
        print('==============lmdb_exist_if: ', lmdb_exist_if)
        if lmdb_exist_if:
            # 已存在lmdb
            # 读取 datasets_lmdb
            with env.begin() as txn:
                data_bytes = txn.get(b'datasets_lmdb')
                if data_bytes == None:
                    datasets_lmdb = []
                else:
                    datasets_lmdb = pickle.loads(data_bytes)
            # 判断该数据集是否已经转化过
            if path in datasets_lmdb:
                # 已存在，不用再转化
                convert_to_lmdb = False
                img_convert_to_lmdb = False
           
        mesh_list = []
        for key in road_map_data.keys():
            left_name = roadmap_sample_path+ '/' +str(key)+"_l.npy"
            right_name = roadmap_sample_path + '/' +str(key)+"_r.npy"
            
            roadmap_samples[key] = {}
            if os.path.exists(left_name):
                if convert_to_lmdb:
                    mesh = np.load(left_name)
                    mesh_list.append(mesh)
                roadmap_samples[key]["left"] = roadmap_sample_idx
                roadmap_sample_idx +=1
            if os.path.exists(right_name):
                if convert_to_lmdb:
                    mesh = np.load(right_name)
                    mesh_list.append(mesh)
                roadmap_samples[key]["right"] = roadmap_sample_idx
                roadmap_sample_idx +=1

        if convert_to_lmdb:        
            # 将整个列表转换为字节类似对象
            byte_mesh_data = pickle.dumps(mesh_list)
            #将已转化的数据集记录下来
            datasets_lmdb.append(path)
            # 将数据列表序列化为字节流
            data_bytes = pickle.dumps(datasets_lmdb)
            # # 打开LMDB事务和数据库
            with env.begin(write=True) as txn:           
                print("22222222222============")           
                # 将npy数据存储到LMDB数据库中
                # txn.put(u'{}'.format(sce_id).encode('ascii'), byte_mesh_data) #以索引为键
                txn.put(path.encode(), byte_mesh_data) # 以数据集名称为键
                # 将已转化的数据集记录下来
                txn.put(b'datasets_lmdb', data_bytes)
            

        roadmap_samples_list.append(roadmap_samples)


  
        fork_path = path + '/gen/lf_fork.json'
        fork_sample_path =  path + '/gen/fork_samples'
        roadmap_forks = {}
        with open(fork_path, "r") as ff:
            forks = json.load(ff)
            roadmap_idxs = []
            for key in forks.keys():
                key0, key1 = key.split("_")
                key0 = int(key0)
                key1 = int(key1)
                roadmap_idxs.append(key0)
                roadmap_idxs.append(key1)
                re1 = os.path.exists(fork_sample_path+'/'+key+"_idxs.npy")
                re2 = os.path.exists(fork_sample_path+'/'+key+"_pts.npy")
                if re1 and re2:
                    fork_sample_idxs = np.load(fork_sample_path+'/'+key+"_idxs.npy")
                    fork_sample_pts = np.load(fork_sample_path+'/'+key+"_pts.npy")
                    roadmap_forks[key] = {}
                    roadmap_forks[key]['area'] = forks[key]
                    roadmap_forks[key]['idxs'] = fork_sample_idxs
                    roadmap_forks[key]['pts'] = fork_sample_pts
            roadmap_forks_list.append(roadmap_forks)

        obj_list = sorted(glob.glob(path + '/org/mesh/*.obj'))       
        mesh_objs = []
        for ii, obj_file in enumerate(obj_list):
            obj_mesh= o3d.io.read_triangle_mesh(obj_file)
            if np.asarray(obj_mesh.vertices).shape[0] == 0:
                continue
            mesh_objs.append(obj_mesh)
        mesh_objs_list.append(mesh_objs)

        image_paths = os.path.join(path+'/org/jpg' ,"*.jpg" )

        img_convert_to_lmdb = False # 后面记得注释掉
        # image转lmdb
        if img_convert_to_lmdb:
            image_list = sorted(glob.glob(image_paths))
            # 打开LMDB事务和数据库
            with env_img.begin(write=True) as txn_img:
                for image_path in image_list:
                    image = cv2.imread(image_path)
                    print(image_path)
                    # 将图像数据存储到LMDB数据库中
                    txn_img.put(image_path.encode(), image)



        image_paths_list.append(image_paths)
        data_infos = {}
        with open(path + '/gen/bev_infos_roadmap_fork.json', 'r') as ff:
            data_infos = json.load(ff)
            data_infos_list.append(data_infos)

        vp_infos = {}
        with open(path + '/gen/vp_infos.json', 'r') as ff:
            vp_infos = json.load(ff)
            vp_infos_list.append(vp_infos)

        if is_train == True:
            with open(path + '/gen/train_fork20.lst', 'r') as ff:
                 lines = ff.readlines()
                 idxs += [[sce_id, int(item)] for item in lines]
        else:
            idxs += [[sce_id, index] for index in range(len(data_infos) - seq_len)]

    # 关闭LMDB环境
    env.close()
    env_img.close()

    return param_infos_list, data_infos_list, vp_infos_list, mesh_objs_list, road_map_data_list, roadmap_samples_list, roadmap_forks_list, map_paths_list, ignore_map_paths_list, ob_map_paths_list, image_paths_list, idxs


# 新建DataLoaderX类
# from prefetch_generator import BackgroundGenerator

# class DataLoaderX(torch.utils.data.DataLoader):

#     def __iter__(self):
#         return BackgroundGenerator(super().__iter__())

def compile_data(version, dataroot, data_aug_conf, grid_conf, bsz, seq_len,
                 nworkers, parser_name):
    parser = {
        'segmentationdata': SegmentationData,
        'segmentation1data': Segmentation1Data,
    }[parser_name] # 根据传入的参数选择数据解析器

    # train_datasets = ['20230223_3.54km_zj228_0310_A12_1']
    # train_datasets = ['20230308_0.66km_zj251_0404_A12_1', '20230308_0.66km_zj251_0404_A12_2', '20230308_0.66km_zj251_0402_A12_1',
    #                   '20230308_0.66km_zj251_0402_A12_6', '20230306_1.56km_zj245_0402_A12_1', '20230306_1.56km_zj245_0402_A12_6',
    #                   '20230306_1.56km_zj245_0404_A12_1', '20230216_2.11km_zj227_0402_A12_0', '20230306_1.56km_zj245_0404_A12_2']

    # val_datasets = ['20230223_3.54km_zj228_0310_A12_1']
    # val_datasets = ['20221017_6.22km','20220926_6.29km', '20221008_6.7km_1002', '20220927_5.15km_1003', '20220929_5.65km_00_1002', '20221109_1.90km', '20221110_6.35km']
    train_datasets = ['20220928_6.02km_1001_21', '20230222_4.43km_zj229_0301_3', '20221110_6km']
    val_datasets = ['20220928_6.02km_1001_21', '20230222_4.43km_zj229_0301_3', '20221110_6km']

    train_list = [dataroot + '/' + dataset for dataset in train_datasets]
    val_list = [dataroot + '/' + dataset for dataset in val_datasets]
    # train_list = data_list[0:14]
    # train_list = data_list[0:5]
    # val_list = data_list[14:17]
    train_idxs= []
    val_idxs = []

    lines = [fcode.strip().split() for fcode in open("/cailirong/bev22-clr/tf_dataset/used_fcode.txt", 'r').readlines()]
    used_fcodes = {}
    for line in lines:
        used_fcodes[line[0]] = int(line[1])

    param_infos_list, data_infos_list, vp_infos_list,  mesh_objs_list, road_map_data_list, roadmap_samples_list, roadmap_forks_list, map_paths_list, ignore_map_paths_list, ob_map_paths_list, image_paths_list, train_idxs = get_data_param(train_list, dataroot, seq_len, is_train = True)
    # for ii in range(len(train_list)):
    #     train_idxs += range(0, len(data_infos_list[ii])-5)
    # b1 = time()
    traindata = parser(train_idxs, param_infos_list, data_infos_list, vp_infos_list, mesh_objs_list, road_map_data_list, roadmap_samples_list, roadmap_forks_list, map_paths_list, ignore_map_paths_list, ob_map_paths_list, image_paths_list, is_train=True, data_aug_conf=data_aug_conf,
                         grid_conf=grid_conf, seq_len=seq_len, used_fcodes=used_fcodes, crop_size=96, dataroot = dataroot, dataset_list = train_list)
    # print("parser：", time() - b1)

    param_infos_list, data_infos_list, vp_infos_list, mesh_objs_list, road_map_data_list, roadmap_samples_list, roadmap_forks_list, map_paths_list, ignore_map_paths_list, ob_map_paths_list, image_paths_list, val_idxs = get_data_param(val_list, dataroot, seq_len, is_train = False)
    # for ii in range(len(val_list)):
    #     val_idxs += range(0, len(data_infos_list[ii])-5)
    valdata = parser(val_idxs, param_infos_list, data_infos_list, vp_infos_list, mesh_objs_list, road_map_data_list, roadmap_samples_list, roadmap_forks_list, map_paths_list, ignore_map_paths_list, ob_map_paths_list, image_paths_list, is_train=False, data_aug_conf=data_aug_conf,
                         grid_conf=grid_conf, seq_len=seq_len, used_fcodes=used_fcodes, crop_size=96, dataroot = dataroot, dataset_list = val_list)
    
    train_sampler = DistributedSampler(traindata) # 为各个进程切分数据，以保证训练数据不重叠。
    val_sampler = DistributedSampler(valdata)
    
    if parser_name == "segmentation1data":
        print("segmentation1data!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        trainloader = torch.utils.data.DataLoader(traindata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=nworkers,
                                            pin_memory=False, persistent_workers=True)
    else:
        trainloader = torch.utils.data.DataLoader(traindata, sampler = train_sampler, batch_size=bsz,
                                                    shuffle=False,
                                                    num_workers=nworkers,
                                                    drop_last=True,
                                                    worker_init_fn=worker_rnd_init,
                                                    pin_memory=True, prefetch_factor = 32)
        
        # trainloader = DataLoaderX(traindata, sampler=train_sampler, batch_size=bsz,
        #                                           shuffle=False,
        #                                           num_workers=nworkers,
        #                                           drop_last=True,
        #                                           worker_init_fn=worker_rnd_init,
        #                                           pin_memory=False, prefetch_factor=32)
        

    # sampler = val_sampler, 
        
    # valloader = torch.utils.data.DataLoader(valdata, sampler = val_sampler,  batch_size=bsz,
    #                                         shuffle=False,
    #                                         drop_last=True,
    #                                         num_workers=nworkers,
    #                                         pin_memory=True, persistent_workers=True)
    valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=nworkers,
                                            pin_memory=False, persistent_workers=True)

    return train_sampler, val_sampler, trainloader, valloader



class BatchIndex:
    def __init__(self, size, batch_size, shuffle=False, drop_last=True):
        if drop_last:
            self.index_list = [(x, x + batch_size,) for x in range(size) if
                               not x % batch_size and x + batch_size <= size]
        else:
            li = [(x, x + batch_size,) for x in range(size) if not x % batch_size]
            x, y = li[-1]
            li[-1] = (x, size)
            self.index_list = li
            self.shuffle = shuffle
            self.drop_last = drop_last

        if shuffle:
            import random as r
            r.shuffle(self.index_list)

    def __next__(self):
        self.pos += 1
        if self.pos >= len(self.index_list):
            raise StopIteration

        return self.index_list[self.pos]

    def __iter__(self):
        self.pos = -1
        return self

    def __len__(self):
        return len(self.index_list)
    

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.imgs, self.rots, self.trans, self.intrins, self.post_rots, self.post_trans, self.binimgs, self.lf_label_gt, self.lf_norm_gt, self.lf_kappa_gt, self.fork_patch_gt, self.fork_scales_gt, self.fork_offsets_gt, self.fork_oris_gt = next(self.loader)
        except StopIteration:
            self.imgs = None
            self.rots = None
            self.trans = None 
            self.intrins = None 
            self.post_rots = None
            self.post_trans = None
            self.binimgs = None
            self.lf_label_gt = None
            self.lf_norm_gt = None
            self.lf_kappa_gt = None
            self.fork_patch_gt = None
            self.fork_scales_gt = None
            self.fork_offsets_gt = None
            self.fork_oris_gt = None

            return
    
        with torch.cuda.stream(self.stream):
            self.imgs = self.imgs.cuda(non_blocking=True)
            self.rots = self.rots.cuda(non_blocking=True)
            self.trans = self.trans.cuda(non_blocking=True)
            self.intrins = self.intrins.cuda(non_blocking=True)
            self.post_rots = self.post_rots.cuda(non_blocking=True)
            self.post_trans = self.post_trans.cuda(non_blocking=True)
            self.binimgs = self.binimgs.cuda(non_blocking=True)
            self.lf_label_gt = self.lf_label_gt.cuda(non_blocking=True)
            self.lf_norm_gt = self.lf_norm_gt.cuda(non_blocking=True)
            self.lf_kappa_gt = self.lf_kappa_gt.cuda(non_blocking=True)
            self.fork_patch_gt = self.fork_patch_gt.cuda(non_blocking=True)
            self.fork_scales_gt = self.fork_scales_gt.cuda(non_blocking=True)
            self.fork_offsets_gt = self.fork_offsets_gt.cuda(non_blocking=True)
            self.fork_oris_gt = self.fork_oris_gt.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        imgs = self.imgs
        rots = self.rots
        trans = self.trans
        intrins = self.intrins
        post_rots = self.post_rots
        post_trans = self.post_trans
        binimgs = self.binimgs
        lf_label_gt = self.lf_label_gt
        lf_norm_gt = self.lf_norm_gt
        lf_kappa_gt = self.lf_kappa_gt
        fork_patch_gt = self.fork_patch_gt
        fork_scales_gt = self.fork_scales_gt
        fork_offsets_gt = self.fork_offsets_gt
        fork_oris_gt = self.fork_oris_gt

        if imgs is not None:
            imgs.record_stream(torch.cuda.current_stream())
        if rots is not None:
            rots.record_stream(torch.cuda.current_stream())
        if trans is not None:
            trans.record_stream(torch.cuda.current_stream())
        if intrins is not None:
            intrins.record_stream(torch.cuda.current_stream())
        if post_rots is not None:
            post_rots.record_stream(torch.cuda.current_stream())
        if post_trans is not None:
            post_trans.record_stream(torch.cuda.current_stream())
        if binimgs is not None:
            binimgs.record_stream(torch.cuda.current_stream())
        if lf_label_gt is not None:
            lf_label_gt.record_stream(torch.cuda.current_stream())
        
        if lf_norm_gt is not None:
            lf_norm_gt.record_stream(torch.cuda.current_stream())
        if lf_kappa_gt is not None:
            lf_kappa_gt.record_stream(torch.cuda.current_stream())
        if fork_patch_gt is not None:
            fork_patch_gt.record_stream(torch.cuda.current_stream())
        if fork_scales_gt is not None:
            fork_scales_gt.record_stream(torch.cuda.current_stream())
        if fork_offsets_gt is not None:
            fork_offsets_gt.record_stream(torch.cuda.current_stream())
        if fork_oris_gt is not None:
            fork_oris_gt.record_stream(torch.cuda.current_stream())

        self.preload()
        return imgs, rots, trans, intrins, post_rots, post_trans, binimgs, lf_label_gt, lf_norm_gt, lf_kappa_gt, fork_patch_gt, fork_scales_gt, fork_offsets_gt, fork_oris_gt


def custom_collate_fn(batch):
    # 将批次中的数据转移到CUDA设备上
    batch = list(zip(*batch))
    imgs = [item.to(torch.device("cuda")) for item in batch[0]]
    rots = [item.to(torch.device("cuda")) for item in batch[1]]
    trans = [item.to(torch.device("cuda")) for item in batch[2]]
    intrins = [item.to(torch.device("cuda")) for item in batch[3]]
    post_rots = [item.to(torch.device("cuda")) for item in batch[4]]

    post_trans = [item.to(torch.device("cuda")) for item in batch[5]]
    binimg = [item.to(torch.device("cuda")) for item in batch[6]]
    trans = [item.to(torch.device("cuda")) for item in batch[2]]
    intrins = [item.to(torch.device("cuda")) for item in batch[3]]
    post_rots = [item.to(torch.device("cuda")) for item in batch[4]]

    return imgs, rots, trans, intrins, post_rots, post_trans, binimg, lf_label, lf_norm, lf_kappa, fork_patch, fork_scale, fork_offset, fork_ori

# dataset = CustomDataset()  # 自定义数据集

# data_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn,
#                          num_workers=4, pin_memory=True)
