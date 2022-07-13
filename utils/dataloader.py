import torch.utils.data as data
import torch
from utils.point_util import index_points, farthest_point_sample
import numpy as np
from utils import operations
import random
from utils.pc_utils import normalize_point_cloud, jitter_perturbation_point_cloud, rotate_point_cloud_and_gt
import h5py

class H5Dataset(data.Dataset):
    def __init__(self, h5_filepath, numOfPoints=256, ratio=2, sub_batch_size=64, data_augmentation=10):
        super(H5Dataset, self).__init__()
        self.h5_filepath = h5_filepath
        self.sub_batch_size = sub_batch_size
        self.ratio = ratio
        self.isUniformity = False
        self.data_augmentation = data_augmentation
        self.input_patch_nums = numOfPoints
        self.gt_patch_nums = self.input_patch_nums * self.ratio
        self.model_num_list = [1250, 2500, 5000, 10000, 20000, 40000, 80000]

    def downsample_points(self, pts, k):
        # pts : [B, N, 3]
        # k : int
        # return : [B, K, 3]
        pts = torch.from_numpy(pts)
        fps_idx = farthest_point_sample(pts, k) # [B, k]
        new_xyz = index_points(pts, fps_idx)    # [B, k, 3]

        return new_xyz.numpy()

    def random_downsample_points(self, points, num, k=0):
        # pts : [B, N, 3]
        # k : int
        # return : [B, K, 3]

        if k > 0 and points.shape[0] >= num * k : 
            points = self.downsample_points(points, num * k)
        out_point = points[:, np.random.choice(points.shape[1], num, replace=(num < points.shape[1])), :]
        return out_point


    def groupKNN(self, pts, k, batch_size, random_sample=False):
        # Select k points with seed_pt as the center
        # pts: [N, 3]
        # seed_pt: [3]
        # random_sample : bool random_Sample/fartest sample
        # return: [k, 3]

        # numpy --> torch
        pts = torch.from_numpy(pts).unsqueeze(0)  # [1, N, 3]
        if not random_sample:
            fps_idx = farthest_point_sample(pts, batch_size) # [1, B]
            seed_pt = index_points(pts, fps_idx)    #[1, B, 3]
        else:
            rand_idx = np.random.randint(0, pts.shape[1], size=(1,batch_size))   # [1, B]
            rand_idx = torch.from_numpy(rand_idx)   # [1, B]
            seed_pt = index_points(pts, rand_idx)    # [1, B, 3]

        grouped_xyz, _, _ = operations.group_knn(k, seed_pt, pts, NCHW=False)    # [1, B, k, 3]
        grouped_xyz = grouped_xyz.squeeze(0) # [B, K, 3]

        return grouped_xyz.numpy()

    def __getitem__(self, index):
        f = h5py.File(self.h5_filepath, "r")
        
        # Randomly select the number of point clouds of the model to control the sparsity of the input point cloud
        model_num = self.model_num_list[random.randint(0, len(self.model_num_list)-1 )]
        data = f["poisson_%d"%(model_num)]
        
        index = index % len(data)
        model = data[index,:,0:3]
        f.close()

        # Randomly select the uniformity k of the input point cloud
        uniformity_k = random.randint(1,2) if model.shape[0] <= 5000 else random.randint(1,4)
        model, _, _ = normalize_point_cloud(model)
        
        # select a big patch with gt_point_nums * uniformity_k  points
        patch = self.groupKNN(model, min(model.shape[0], self.gt_patch_nums * uniformity_k), self.sub_batch_size, random_sample=True)   # [B,k, 3]
        # if model.shape[0] > self.gt_patch_nums * uniformity_k:
        #     patch = self.groupKNN(model, self.gt_patch_nums * uniformity_k, self.sub_batch_size)   # [k, 3]
        # elif model.shape[0] < self.gt_patch_nums:
        #     print(model_path, " is little")
        # else:
        #     patch = model
        # patch normalization
        point, centroid, furthest_distance = normalize_point_cloud(patch)  # [B, N, 3]  [B, 1, 3]   [B, 1, 1]
        patch_pos = np.concatenate((centroid, furthest_distance), axis=2)   # [B, 1, 4]
        # uniform downsample patch to get gt_patch
        gt_point = self.downsample_points(point, self.gt_patch_nums)
        if self.isUniformity:
            # uniform downsample patch to get input_patch
            input_point = self.downsample_points(point, self.input_patch_nums)
        else:
            # non-uniform downsample patch to get input_patch
            input_point = self.random_downsample_points(point, self.input_patch_nums, uniformity_k)    # 对输入点云进行不均匀下采样, k值从0-4之间随机旋转
        # rotate
        input_point, gt_point = rotate_point_cloud_and_gt(input_point, gt_point)
        # jitter
        input_point = jitter_perturbation_point_cloud(input_point)
        # numpy --> torch
        input_point = torch.from_numpy(input_point)     # [B, K, 3]
        gt_point = torch.from_numpy(gt_point)   # [B, K, 3]
        input_point = input_point.permute(0, 2, 1)     # [B, 3, K]
        gt_point = gt_point.permute(0, 2, 1)   # [B, 3, K]
        patch_pos = torch.from_numpy(patch_pos)     # [B, 1, 4]
        patch_pos = patch_pos.permute(0, 2, 1)   # [B, 4, 1]


        return input_point, gt_point, patch_pos, "cat:%d---%d"%(index, model_num), uniformity_k

    def collate_fn(self, batch):
        input_point, gt_point, patch_pos, info, uniformity_k = list(zip(*batch))
        input_point = torch.cat(input_point, 0)
        gt_point = torch.cat(gt_point, 0)
        patch_pos = torch.cat(patch_pos, 0)

        return input_point, gt_point, patch_pos, info, uniformity_k


    def __len__(self):
        return 90 * len(self.model_num_list) * self.data_augmentation // self.sub_batch_size