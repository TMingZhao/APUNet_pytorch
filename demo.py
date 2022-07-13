import argparse
import torch
from model import TPU
import os
from utils.pc_utils import read_ply, save_ply
from utils import pc_utils
from utils.point_util import farthest_point_sample, index_points, knn_point, normalize_point_batch
from tqdm import tqdm
from torch.autograd import Variable
from utils import operations
import config


def points2patches(xyz, npoint, nsample):
    # N, C = xyz.shape
    xyz = torch.from_numpy(xyz).unsqueeze(0)   # [1, N, 3]
    fps_idx = farthest_point_sample(xyz, npoint) # [1, npoint]
    new_xyz = index_points(xyz, fps_idx)    #[1, npoint, C]
    idx = knn_point(nsample, xyz, new_xyz)  # [1, npoint, nsample]
    grouped_xyz = index_points(xyz, idx) # [1, npoint, nsample, C]
    grouped_xyz = grouped_xyz.squeeze(0)    # [npoint, nsample, C]
    return grouped_xyz


# 点云上采样函数，将输入点云上采样两倍后输出
def upsamplingX2(net, points, num_patch, device, batch_size=8, detial_visual=False):
    # input : [N, 3]
    # output : [rN, 3]

    # point cloud pretreat
    num_points = points.shape[0]
    points, all_centroid, all_radius = pc_utils.normalize_point_cloud(points)   # [N, 3] [1, 3] [1, 1]
    net.eval()
    
    a = int(num_points / num_patch) * config.repeatability
    patches = points2patches(points, a, num_patch)
    
    up_patch_list = []

    use_tqdm = False
    if use_tqdm:
        pbar = tqdm(total=len(patches), desc='process')
        pbar.update(0)
    for i in range(0, len(patches), batch_size):
        
        
        i_s = i
        i_e = (i+batch_size) if ((i+batch_size) < len(patches)) else len(patches)
        patch = patches[i_s:i_e]

        if use_tqdm:
            pbar.update(len(patch))
        # save input data
        if detial_visual:
            for patch_i, p in enumerate(patch):
                save_ply(p, "output/visual/demo/%d_input.ply"%(patch_i+i))
        patch = patch.to(device)
        norm_patch, centroid, radius = normalize_point_batch(patch, False)# [B, N, 3]  [B, 1, 3]   [B, 1, 1]
        patch_pos = torch.cat((centroid, radius), dim=2)   # [B, 1, 4]
        input_points = norm_patch.permute(0,2,1)
        patch_pos = patch_pos.permute(0, 2, 1)   # [B, 4, 1]
        input_points = Variable(input_points.to(device), requires_grad=False)
        patch_pos = Variable(patch_pos.to(device), requires_grad=False)


        with torch.no_grad():
            up_patch = net(input_points, patch_pos=patch_pos)
        up_ratio = up_patch.shape[2] // input_points.shape[2]
        up_patch = up_patch.permute(0,2,1)
        up_patch = up_patch * radius + centroid

        # save output data
        if detial_visual:
            for patch_i, p in enumerate(up_patch):
                save_ply(p.cpu(), "output/visual/demo/%d_output.ply"%(patch_i+i))
        up_patch_list.extend(up_patch)
    if use_tqdm:
        pbar.close()
    up_points = torch.cat(up_patch_list, dim=0)

    # downsample
    up_points = up_points.unsqueeze(0)   # [1, N, 3]
    _, up_points = operations.furthest_point_sample(up_points.permute(0, 2, 1), up_ratio*num_points)
    up_points = up_points.permute(0, 2, 1).cpu()

    up_points = up_points[0, ...]   # [rN, 3]
    up_points = up_points.numpy() * all_radius + all_centroid

    return up_points

def upsamplingX(net, ply_path, num_patch, step_up_ratio, iteration, device, batch_size, detial_visual=False):
    
    points = read_ply(ply_path)[:, 0:3]
    mode_name = os.path.splitext(ply_path)[0]
    up_ratio = 1
    for iter in range(iteration):
        up_ratio = int(up_ratio * step_up_ratio)
        points_num = points.shape[0]
        points = upsamplingX2(net, points, num_patch, device, batch_size, detial_visual)
        save_path = mode_name + "X%d.ply"%(up_ratio)
        save_ply(points, save_path)
        print("X%d result: %d --> %d"%(up_ratio, points_num, points.shape[0]))



if __name__ == "__main__":
    parser = argparse.ArgumentParser('TPUNet')
    parser.add_argument('--ply_path', type=str, default="data/model4.ply", help='the file of trian data')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size in test')   
    parser.add_argument('--pretrained_weights', type=str, default='pretrain_model/model_epoch.pth', help='pretrain models file')  # 416
    args = parser.parse_args()
    device = torch.device("cuda:0")
    net = TPU().to(device)
    net.load_state_dict(torch.load(args.pretrained_weights))
    print("load model: {}".format(args.pretrained_weights))


    # Upsample point cloud and save to the same directory 
    upsamplingX(net, 
        args.ply_path, 
        num_patch=config.num_point, 
        step_up_ratio=config.up_ratio, 
        iteration=2, 
        device=device, 
        batch_size=args.batch_size, 
        detial_visual=False)



