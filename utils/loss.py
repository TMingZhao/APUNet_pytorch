import torch
import torch.nn.functional as F
from emd_pytorch_implement.pkg.emd_loss_layer import EMDLoss
from utils.point_util import knn_point, index_points, farthest_point_sample
def get_emd_loss(pred, gt):
    '''
    pred, gt: torch.tensor[batch_size, npoint, 3]
    '''
    dist =  EMDLoss()
    cost = dist(pred, gt)
    loss = (torch.sum(cost))/(pred.size()[1]*gt.size()[0])

    return loss

def get_repulsion_loss(pred, nsample=20, radius=0.7):
    ''' 
    pred : torch.tensor[batch_size, npoint, 3]
    '''
    assert(pred.shape[1] >= nsample), 'rep loss : point number is less than nsample'
    fps_idx = farthest_point_sample(pred, pred.shape[1]) # [B, npoint]
    new_xyz = index_points(pred, fps_idx)
    idx = knn_point(nsample, pred, new_xyz)
    grouped_pred = index_points(pred, idx)      # [batch_size, nponts, nsample, C]
    grouped_pred -= pred.unsqueeze(2)           # [B, N, nsample, 3 ]

    # h = CalPtsDensity(pred.shape[1])
    # h = h / 5
    h = 0.15
    dist_square = torch.sum(grouped_pred ** 2, dim=-1)      # [B, N, nsample]
    assert(dist_square.shape[2] >= 6), 'rep loss : group point number is less than k'
    dist_square, idx = torch.topk(dist_square, k=6, dim=-1, largest=False, sorted=True)    # 得到最小的5个点     # [B, N, 6]
    dist_square = dist_square[:, :, 1:]  # 移除自身点,剩下五个离自己最近的点        # [B, N, 5]

    dist = torch.sqrt(dist_square)  # [B, N, 5]
    # loss = torch.max(h-dist, torch.tensor(0.0).to(pred.device))
    # h = torch.tensor(h, requires_grad=False).to(pred.device)
    # dist = h - dist
    loss = F.relu(h-dist)       # [B, N, 5]
    uniform_loss = torch.mean(loss)
    return uniform_loss