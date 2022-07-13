from torch.autograd import Variable
import torch
from utils.point_util import merge2dict
from utils.pc_utils import save_ply
from tqdm import tqdm
import argparse
from model import TPU
import os
from utils.dataloader import H5Dataset

def evaluate(model, dataloader, device, visual=True, out_dir=""):
    model.eval()
    test_loss = []
    detail_loss = {}
    for batch_i, (input_points, gt_points, patch_pos, _, _) in enumerate(tqdm(dataloader, desc="test")):
        patch_pos = Variable(patch_pos.to(device))
        input_points = Variable(input_points.to(device))
        gt_points = Variable(gt_points.to(device), requires_grad=False)

        with torch.no_grad():
            loss, coords, log = model(input_points, gt_points, patch_pos)

        # print(path[0], " : ", log)

        test_loss.append(loss.cpu())
        merge2dict(log, detail_loss)

        if(visual):
            # save ply file
            for i, (pred, input_, gt) in enumerate(zip(coords, input_points, gt_points)):
                save_ply(pred.transpose(1, 0).cpu().numpy(), out_dir + "/res_{}x{}-pred.ply".format(batch_i, i))
                save_ply(input_.transpose(1, 0).cpu().numpy(), out_dir + "/res_{}x{}-input.ply".format(batch_i, i))
                save_ply(gt.transpose(1, 0).cpu().numpy(), out_dir + "/res_{}x{}-gt.ply".format(batch_i, i))
            break

        
    test_loss = torch.mean(torch.stack(test_loss))
    for key, value in detail_loss.items():
            detail_loss[key] = "%.6f"% (detail_loss[key] / (batch_i+1))

    return test_loss, detail_loss



