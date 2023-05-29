"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""
# from fire import Fire
import argparse
import torch
import src
"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import os
import numpy as np
from time import time
from torch import nn
from src.models_goe_1129 import compile_model
from tensorboardX import SummaryWriter
from src.data_tfmap import compile_data, BatchIndex, data_prefetcher
from src.tools import SimpleLoss, RegLoss, SegLoss, SegLoss1, BCEFocalLoss, get_batch_iou, get_val_info, denormalize_img, SimpleLoss1
import sys
import cv2
from collections import OrderedDict
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # only one GPU
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
# os.environ['LOCAL_RANK'] = "0, 1, 2, 3"
# torch.set_num_threads(8)
import argparse
import cProfile
import re

# import multiprocessing as mp


def main():
    # cProfile.run('compile_data', filename='restats')
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default = -1, type=int)
    args = parser.parse_args()
    print("sssss",args.local_rank)
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device=torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')

    # mp.set_start_method('spawn', force=True)

    version = "0"
    dataroot='/defaultShare/user-data/bev_data'
    # nepochs=10000
    nepochs = 10
    H=1080
    W=3520
    final_dim=(128, 352)
    max_grad_norm=5.0
    pos_weight=2.13 # 损失函数中给正样本项损失乘的权重系数
    logdir = './runs0421_bs8_2243km_100m_depth'

    xbound=[0.0, 102., 0.85] # 限制x方向的范围并划分网格
    ybound=[-10.0, 10.0, 0.5] # 限制Y方向的范围并划分网格
    zbound=[-2.0, 4.0, 1] # 限制z方向的范围并划分网格
    dbound=[3.0, 103.0, 2.] # 限制深度方向的范围并划分网格

    # bsz=8
    seq_len=10
    bsz=12
    #seq_len=30
    nworkers=2 # 线程数
    lr=1e-3
    weight_decay=1e-7 # 权重衰减系数
        
    torch.backends.cudnn.benchmark = True
    grid_conf = { # 网格配置
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    # data_aug_conf = {
    #                 'resize_lim': [(0.12, 0.15), (0.34, 0.50)],
    #                 'final_dim': (128, 352),
    #                 'rot_lim': (-5.4, 5.4),
    #                 'H': H, 'W': W,
    #                 'rand_flip': False,
    #                 'bot_pct_lim': [(0.04, 0.10), (0.2, 0.4)],
    #                 'cams': ['CAM_FRONT0', 'CAM_FRONT1'],
    #                 'Ncams': 2,
    #             }
    # data_aug_conf = {
    #                 'resize_lim': [(0.1, 0.3), (0.3, 0.60)],
    #                 'final_dim': (128, 352),
    #                 'rot_lim': (-5.4, 5.4),
    #                 'H': H, 'W': W,
    #                 'rand_flip': False,
    #                 'bot_pct_lim': [(0.04, 0.35), (0.15, 0.4)],
    #                 'cams': ['CAM_FRONT0', 'CAM_FRONT1'],
    #                 'Ncams': 2,
    #             }
    data_aug_conf = {
                'resize_lim': [(0.05, 0.4), (0.3, 0.90)],#(0.3-0.9) # resize的范围
                'final_dim': (128, 352), # 数据预处理之后最终的图片大小
                'rot_lim': (-5.4, 5.4), # 训练时旋转图片的角度范围
                'H': H, 'W': W,
                'rand_flip': False,
                'bot_pct_lim': [(0.04, 0.35), (0.15, 0.4)],# 裁剪图片时，图像底部裁剪掉部分所占比例范围
                'cams': ['CAM_FRONT0', 'CAM_FRONT1'],
                'Ncams': 2, # 训练时选择的相机通道数
            }
    
    train_sampler, val_sampler, trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, seq_len=seq_len, nworkers=nworkers,
                                          parser_name='segmentationdata') # 获取训练数据和测试数据
    
    
    print("train lengths: ", len(trainloader))
    print("val lengths: ", len(valloader))
    # device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')
    model = compile_model(grid_conf, data_aug_conf, seq_len=seq_len, batchsize=int(bsz)) # 获取模型

    if 0:
        print('==> loading existing model')
        # model_info = torch.load('./model_met_latest_20221126_1511.pt')
        # model_info = torch.load('/xiedongmei/pytorch_models/BEVnet12/runs_1205_391km_modelmcn/model0.pt')
        # model_info = torch.load('/xiedongmei/pytorch_models/BEVnet12/model_latest_1208.pt')
        # model_info = torch.load('/xiedongmei/pytorch_models/BEVnet16/model_bev16_vp_1213.pt')
        model_info = torch.load('/xiedongmei/pytorch_models/BEVnet22/runs0407_bs8_2147km_100m_depth/model0.pt')
        # model_info.pop('module.dx')
        # model_info.pop('module.bx')
        # model_info.pop('module.nx')
        # model_info.pop('module.frustum')
        # model_info.pop('module.voxels')
        new_state_dict = OrderedDict()
        for k, v in model_info.items():
            name = k[7:]  # remove "module."
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
    model.to(device)

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                            output_device=args.local_rank,find_unused_parameters=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) # 优化器

    loss_fn = SegLoss(pos_weight).cuda(args.local_rank)
    # 类别的顺序： 0-laneline, 1-stopline, 2-zebra 斑马线, 3-arrow, 4-roadedge, 5-centerline
    loss_fn_ll = SegLoss(pos_weight).cuda(args.local_rank) # 下面几个是斑马线/车道线/车道中线/箭头/悬挂物
    loss_fn_sl = SegLoss(pos_weight).cuda(args.local_rank)
    loss_fn_zc = SegLoss(pos_weight).cuda(args.local_rank)
    loss_fn_ar = SegLoss(pos_weight).cuda(args.local_rank)
    loss_fn_rs = SegLoss(pos_weight).cuda(args.local_rank)
    loss_fn_cl = SimpleLoss(pos_weight).cuda(args.local_rank) # 路沿
    loss_fn_lf_pred = SimpleLoss1(pos_weight).cuda(args.local_rank) # 分叉点和汇入点 重叠三角区域
    loss_fn_lf_norm = RegLoss(0).cuda(args.local_rank) # 损失函数  # Heading 趋势线方向
    # loss_fn_patch = SimpleLoss(pos_weight).cuda(args.local_rank)


    writer = SummaryWriter(logdir=logdir)
    val_step = 1000 # 每隔多少个iter验证一次
    t1 = time()
    t2 = time()
    model.train()
    counter = 0
    # scaler = torch.cuda.amp.GradScaler()
    #                                                  
    # 运行一个 epoch  796.510126 s/804.03616 s/809.14043/810.61527
    for epoch in range(nepochs):
        np.random.seed()
        train_sampler.set_epoch(epoch) # 为了保证每个epoch的划分是不同的
        dataloder_time = 0
        start_time = time() # 每个epoch起始时间
        ep1 = time()

         
        prefetcher = data_prefetcher(trainloader)
        imgs, rots, trans, intrins, post_rots, post_trans, binimgs, lf_label_gt, lf_norm_gt, lf_kappa_gt, fork_patch_gt, fork_scales_gt, fork_offsets_gt, fork_oris_gt = prefetcher.next()

        while imgs is not None:
            


        # for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs, lf_label_gt, lf_norm_gt, lf_kappa_gt, fork_patch_gt, fork_scales_gt, fork_offsets_gt, fork_oris_gt) in enumerate(trainloader):
            # imgs: 1 x 20 x 3 x 128 x 352  
            # rots: 1 x 20 x 3 x 3
            # trans: 1 x 20 x 3
            # intrins: 1x 20 x 3 x 3
            # post_rots: 1 x 20 x 3 x 3
            # post_trans: 1 x 20 x 3
            # binimgs: 1 x 10x6 x 480 x 160 # 480 x 160是像素坐标大小
            b1 = time()
            dataloder_time +=(b1 - ep1)
            # print(f"第{counter} counter 数据加载时间：", b1 - ep1)

            t0 = time()
            t =  t0 - t1
            tt = t0 - t2
            t1 = time()
            # print('imgs.shape: ', len(imgs))
            # print('rots.shape: ', rots.shape)

            # img_trans = []
            # train_transform = Transforms_cvcuda().to(device)
            # for img in imgs:
            #     img = train_transform(img.to(device))
            #     img_trans.append(img)

            # img_trans = torch.stack(img_trans, dim =1)

            # train_transform = Transforms_cvcuda_batch().to(device)
            # img_trans = train_transform(imgs.to(device))


            # print(f"Transforms 数据加载时间：", time() - t1)


            # print('img_trans.shape: ', img_trans.shape)

            # with torch.cuda.amp.autocast():
                # seg_preds, seg_ipreds, lf_preds, lf_crop_preds = model(imgs.to(device), rots.to(device), trans.to(device), intrins.to(device), post_rots.to(device),
                        #  post_trans.to(device),fork_scales_gt.to(device),fork_offsets_gt.to(device),fork_oris_gt.to(device))

            seg_preds, seg_ipreds, lf_preds, _= model(imgs.to(device, non_blocking=True),
                                                    #   imgs,
                                                    #   img_trans,
                                                      rots.to(device, non_blocking=True),
                                                      trans.to(device, non_blocking=True),
                                                      intrins.to(device, non_blocking=True),
                                                      post_rots.to(device, non_blocking=True),
                                                      post_trans.to(device, non_blocking=True),
                                                      fork_scales_gt.to(device, non_blocking=True),
                                                      fork_offsets_gt.to(device, non_blocking=True),
                                                      fork_oris_gt.to(device, non_blocking=True)) # 推理  preds: 4 x 1 x 200 x 200
            


            # seg_preds, seg_ipreds, lf_preds, _= model(imgs,
            #                                           rots,
            #                                           trans,
            #                                           intrins,
            #                                           post_rots,
            #                                           post_trans,
            #                                           fork_scales_gt,
            #                                           fork_offsets_gt,
            #                                           fork_oris_gt) # 推理  preds: 4 x 1 x 200 x 200
            



            # seg_preds, seg_ipreds, lf_preds, _= model(imgs_data,
            #                                           rots_data,
            #                                           trans_data,
            #                                           intrins_data,
            #                                           post_rots_data,
            #                                           post_trans_data,
            #                                           fork_scales_gt_data,
            #                                           fork_offsets_gt_data,
            #                                           fork_oris_gt_data) # 推理  preds: 4 x 1 x 200 x 200

            # print(f"第{counter} counter =============模型计算时间：", time() - b1)

            lf_pred = lf_preds[:, :, :1].contiguous() # 趋势线预测结果
            lf_norm = lf_preds[:, :, 1:(1+4)].contiguous()

            lf_out = lf_pred.sigmoid()
            out = seg_preds.sigmoid()

            # s1 = time()
            binimgs = binimgs.to(device)

            # 计算二值交叉熵损失
            loss_ll = loss_fn_ll(seg_preds[:, :, 0].contiguous(), binimgs[:, :, 0].contiguous()) + loss_fn_ll(seg_ipreds[:, :, 0].contiguous(), binimgs[:, :, 0].contiguous())
            loss_sl = loss_fn_sl(seg_preds[:, :, 1].contiguous(), binimgs[:, :, 1].contiguous()) + loss_fn_sl(seg_ipreds[:, :, 1].contiguous(), binimgs[:, :, 1].contiguous())
            loss_zc = loss_fn_zc(seg_preds[:, :, 2].contiguous(), binimgs[:, :, 2].contiguous()) + loss_fn_zc(seg_ipreds[:, :, 2].contiguous(), binimgs[:, :, 2].contiguous())
            loss_ar = loss_fn_ar(seg_preds[:, :, 3].contiguous(), binimgs[:, :, 3].contiguous()) + loss_fn_ar(seg_ipreds[:, :, 3].contiguous(), binimgs[:, :, 3].contiguous())
            loss_rs = loss_fn_rs(seg_preds[:, :, 4].contiguous(), binimgs[:, :, 4].contiguous()) + loss_fn_rs(seg_ipreds[:, :, 4].contiguous(), binimgs[:, :, 4].contiguous())
            loss_cl = loss_fn_cl(seg_preds[:, :, 5].contiguous(), binimgs[:, :, 5].contiguous()) + loss_fn_cl(seg_ipreds[:, :, 5].contiguous(), binimgs[:, :, 5].contiguous())

            norm_mask = (lf_norm_gt > -500)

            scale_lf = 5.
            loss_lf = loss_fn_lf_pred(lf_pred, lf_label_gt.to(device)) / 5. + loss_fn_lf_norm(lf_norm[norm_mask], scale_lf*lf_norm_gt[norm_mask].to(device))
            loss = loss_lf + loss_ll + loss_sl + loss_zc + loss_ar + loss_rs + loss_cl# + loss_ilf

            opt.zero_grad()
            loss.backward()
            # s2 = time()
            # # print(f"第{counter} counter ==================================梯度计算时间：", s2 - s1)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) # 梯度裁剪 优化技术 加速收敛
            opt.step()

            t2 = time()
            # print(f"第{counter} counter =================================================参数更新：", t2 - s2)

            counter += 1

            # print(f"第{batchi} counter 总运行时间：", t2 - ep1)
            ep1 = time()
            # if args.local_rank==0:
            #     print('counter = ', counter, "total_time:", t, "data_time",tt)
            if counter % 20 == 0 and args.local_rank==0: # 每10个iter打印并记录一次loss
                print(counter, loss.item())
                print(f"第{counter} counter代码运行时间：", time() - start_time)
                # print(f"第{counter} counter 数据加载平均时间：", dataloder_time/(time() - start_time))
                writer.add_scalar('train/loss', loss, counter)
                writer.add_scalar('train/loss_ll', loss_ll, counter)
                writer.add_scalar('train/loss_sl', loss_sl, counter)
                writer.add_scalar('train/loss_zc', loss_zc, counter)
                writer.add_scalar('train/loss_ar', loss_ar, counter)
                writer.add_scalar('train/loss_rs', loss_rs, counter)
                writer.add_scalar('train/loss_cl', loss_cl, counter)
                writer.add_scalar('train/loss_lf', loss_lf, counter)
                # writer.add_scalar('train/loss_lf_crop', loss_lf_crop, counter)

            if counter % 50 == 0 and args.local_rank==0: # 每50个iter打印并记录一次iou和一次优化的时间
                _, _, iou_ll = get_batch_iou(seg_preds[:, :, 0].contiguous(), binimgs[:, :, 0].contiguous())
                _, _, iou_sl = get_batch_iou(seg_preds[:, :, 1].contiguous(), binimgs[:, :, 1].contiguous())
                _, _, iou_zc = get_batch_iou(seg_preds[:, :, 2].contiguous(), binimgs[:, :, 2].contiguous())
                _, _, iou_ar = get_batch_iou(seg_preds[:, :, 3].contiguous(), binimgs[:, :, 3].contiguous())
                _, _, iou_rs = get_batch_iou(seg_preds[:, :, 4].contiguous(), binimgs[:, :, 4].contiguous())
                _, _, iou_cl = get_batch_iou(seg_preds[:, :, 5].contiguous(), binimgs[:, :, 5].contiguous())
                writer.add_scalar('train/iou_ll', iou_ll, counter)
                writer.add_scalar('train/iou_sl', iou_sl, counter)
                writer.add_scalar('train/iou_zc', iou_zc, counter)
                writer.add_scalar('train/iou_ar', iou_ar, counter)
                writer.add_scalar('train/iou_rs', iou_rs, counter)
                writer.add_scalar('train/iou_cl', iou_cl, counter)
                writer.add_scalar('train/epoch', epoch, counter)
                writer.add_scalar('train/step_time', t, counter)
                writer.add_scalar('train/data_time', tt, counter)

            if counter % 200 == 0 and args.local_rank==0:
                fH = final_dim[0]
                fW = final_dim[1]

                # imgs type is tensor
                image0 =np.array(denormalize_img(imgs[0, 0]))
                image1 =np.array(denormalize_img(imgs[0, 1]))

                # data augmentation after dataloader
                # image0 =np.array(denormalize_img(img_trans[0, 0]))
                # image1 =np.array(denormalize_img(img_trans[0, 1]))
                # image2 =np.array(denormalize_img(imgs[0, 2]))
                # image3 =np.array(denormalize_img(imgs[0, 3]))
                writer.add_image('train/image/00', image0, global_step=counter, dataformats='HWC')
                writer.add_image('train/image/01', image1, global_step=counter, dataformats='HWC')
                # writer.add_image('train/image/02', image2, global_step=counter, dataformats='HWC')
                # writer.add_image('train/image/03', image3, global_step=counter, dataformats='HWC')
                writer.add_image('train/binimg/0', (binimgs[0, 1, 0:1]+1.)/2.01, global_step=counter)

                writer.add_image('train/binimg/1', (binimgs[0, 1, 1:2]+1.)/2.01, global_step=counter)
                writer.add_image('train/binimg/2', (binimgs[0, 1, 2:3]+1.)/2.01, global_step=counter)
                writer.add_image('train/binimg/3', (binimgs[0, 1, 3:4]+1.)/2.01, global_step=counter)
                writer.add_image('train/binimg/4', (binimgs[0, 1, 4:5]+1.)/2.01, global_step=counter)
                writer.add_image('train/binimg/5', (binimgs[0, 1, 5:6]+1.)/2.01, global_step=counter)
                writer.add_image('train/out/0', out[0, 1, 0:1], global_step=counter)
                writer.add_image('train/out/1', out[0, 1, 1:2], global_step=counter)
                writer.add_image('train/out/2', out[0, 1, 2:3], global_step=counter)
                writer.add_image('train/out/3', out[0, 1, 3:4], global_step=counter)
                writer.add_image('train/out/4', out[0, 1, 4:5], global_step=counter)
                writer.add_image('train/out/5', out[0, 1, 5:6], global_step=counter)

                writer.add_image('train/lf_label_gt/0', (lf_label_gt[0, 1]+1.)/2.01, global_step=counter)
                writer.add_image('train/lf_out/0', lf_out[0, 1], global_step=counter)
                writer.add_image('train/fork_patch/0', (fork_patch_gt[0, 1, 0:1]+1.)/2.01, global_step=counter)
                writer.add_image('train/fork_patch/1', (fork_patch_gt[0, 1, 1:2]+1.)/2.01, global_step=counter)
                # writer.add_image('train/lf_crop_out/0', lf_crop_out[0, 1, 0:1], global_step=counter)
                # writer.add_image('train/lf_crop_out/1', lf_crop_out[0, 1, 1:2], global_step=counter)

                seg_ll_data = binimgs[0, 1, 0].cpu().detach().numpy()
                seg_cl_data = binimgs[0, 1, 5].cpu().detach().numpy()

                # lf_label_data_gt = lf_label_gt[0, 1, 0].numpy()
                # lf_norm_data_gt = lf_norm_gt[0, 1].numpy()

                lf_label_data_gt = lf_label_gt[0, 1, 0].cpu().detach().numpy()
                lf_norm_data_gt = lf_norm_gt[0, 1].cpu().detach().numpy()

                lf_norm_show = np.zeros((480, 160, 3), dtype=np.uint8)
                ys, xs = np.where(seg_ll_data > 0.5)
                lf_norm_show[ys, xs, :] = 200

                ys, xs = np.where(lf_label_data_gt> -0.5)
                lf_norm_show[ys, xs, :] = 128

                labels = np.logical_or(seg_ll_data[ys, xs] > 0.5, seg_cl_data[ys, xs] > 0.5)
                ys = ys[labels]
                xs = xs[labels]
                scale = 1.7

                if ys.shape[0] > 0:
                    for mm in range(0, ys.shape[0], 10):
                        y = ys[mm]
                        x = xs[mm]
                        norm0 = lf_norm_data_gt[0:2, y, x]
                        cv2.line(lf_norm_show, (x, y), (x+int(round(norm0[0]*50)), y + int(round(scale * (norm0[1]+1)*50))), (0, 0, 255))
                        norm1 = lf_norm_data_gt[2:4, y, x]
                        cv2.line(lf_norm_show, (x, y), (x+int(round(norm1[0]*50)), y + int(round(scale * (norm1[1]+1)*50))), (255, 0, 0))
                writer.add_image('train/lf_norm_gt/0',  lf_norm_show, global_step=counter, dataformats='HWC')

                lf_norm_data = lf_norm[0, 1].detach().cpu().numpy()
                ys, xs = np.where(np.logical_or(seg_ll_data > 0.5, seg_cl_data > 0.5))
                lf_norm_show = np.zeros((480, 160, 3), dtype=np.uint8)
                if ys.shape[0] > 0:
                    for mm in range(0, ys.shape[0], 10):
                        y = ys[mm]
                        x = xs[mm]
                        norm0 = lf_norm_data[0:2, y, x]/scale_lf
                        cv2.line(lf_norm_show, (x, y), (x+int(round(norm0[0]*50)), y+int(round(scale * (norm0[1]+1)*50))), (0, 0, 255))
                        norm1 = lf_norm_data[2:4, y, x]/scale_lf
                        cv2.line(lf_norm_show, (x, y), (x+int(round(norm1[0]*50)), y+int(round(scale * (norm1[1]+1)*50))), (255, 0, 0))
                writer.add_image('train/lf_norm/0',  lf_norm_show, global_step=counter, dataformats='HWC')

            if counter % (2*val_step) == 0 and args.local_rank==0: # 记录checkpoint
                model.eval()
                mname = os.path.join(logdir, "model{}.pt".format(0))
                #mname = os.path.join(logdir, "model{}.pt".format(counter))#counter))
                #print('saving', mname)
                torch.save(model.state_dict(), mname)
                model.train()

            imgs, rots, trans, intrins, post_rots, post_trans, binimgs, lf_label_gt, lf_norm_gt, lf_kappa_gt, fork_patch_gt, fork_scales_gt, fork_offsets_gt, fork_oris_gt = prefetcher.next()

        end_time = time() # epoch结束
        print(f"第{epoch} epoch代码运行时间：", end_time - start_time)

if __name__ == '__main__':
    main()
