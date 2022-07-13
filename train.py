import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"    # 该行代码必须在所有访问GPU代码之前
from utils import logger
import torch
from model import TPU
from utils.dataloader import H5Dataset
from torch.autograd import Variable
from tqdm import tqdm
from test import evaluate
from utils.point_util import merge2dict
import config


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser('TPU')
    # training parameters
    parser.add_argument('--batch_size', type=int, default=256, help='batch size in training')
    parser.add_argument('--epochs', type=int, default=500, help='number of epoch in training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate in training')
    parser.add_argument('--adjust_lr_time', type=int, default=80, help='learning rate in training')
    parser.add_argument('--n_cpu', type=int, default=8, help='how many process')
    parser.add_argument('--data_dir', type=str, default="data/train_data/poisson.hdf5", help='the file of trian data')
    parser.add_argument('--pretrained_weights', type=str, default='pretrain_model/model_epoch.pth')
    parser.add_argument('--gpu_id', type=int, default=0, help='cuda GPI ID')
    parser.add_argument("--out_dir", type=str, default="output", help="the out of program")
    parser.add_argument("--session", type=str, default="train_76_1", help="the out of folder")    
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    out_dir = os.path.join(args.out_dir, args.session)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    save_model_dir = os.path.join(out_dir, 'models')
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)

    log_dir = os.path.join(out_dir, "log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if os.path.exists(os.path.join(out_dir, "lastlogpath")):
        logger_path = logger.getStrFromFile(os.path.join(out_dir, "lastlogpath"))
    else:
        now_str = logger.getFileNameBasedTime()
        logger_path = os.path.join(log_dir, now_str + ".log")
        logger.saveStr2File(os.path.join(out_dir, "lastlogpath"), logger_path)
    
    logger = logger.Logger(logger_path).get_logger()
    logger.info(" opt : {}".format(args))

    logger.info(" same as train_76")
    

    if not torch.cuda.is_available():
        print("CUDA is not available, the current version does not support CPU")
        exit()

    if (args.gpu_id == 0):
        device = torch.device('cuda:0')
    elif(args.gpu_id == 1):
        device = torch.device('cuda:1')
    else:
        device = torch.device("cuda")

    dataset = H5Dataset(args.data_dir, numOfPoints=config.mini_point, ratio=config.up_ratio, sub_batch_size=args.batch_size, data_augmentation=5, isUniformity=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size//dataset.sub_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        )
    test_dataset = H5Dataset(args.data_dir, numOfPoints=config.mini_point, ratio=config.up_ratio, sub_batch_size=args.batch_size, data_augmentation=1, isUniformity=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size//dataset.sub_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        )

    
    model = TPU().to(device)

    # load model weights
    if args.pretrained_weights is not '':
        model.load_state_dict(torch.load(args.pretrained_weights), strict=True)
        logger.info("load model: {}".format(args.pretrained_weights))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)  

    for epoch in range(args.epochs):
        model.train()
        train_loss = []
        detail_loss = {}
        lr = args.learning_rate * (0.4 ** (epoch // args.adjust_lr_time))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if (epoch) % args.adjust_lr_time == 0:
            logger.info(" adjust lr: %f"%(lr))

        for batch_i, (input_points, gt_points, patch_pos, path, unis) in enumerate(tqdm(dataloader, desc="train")):
            patch_pos = Variable(patch_pos.to(device))
            input_points = Variable(input_points.to(device))
            gt_points = Variable(gt_points.to(device), requires_grad=False)

            loss, _, loss_log = model(input_points, gt_points, patch_pos)
            loss.backward()


            optimizer.step()
            optimizer.zero_grad() 

            train_loss.append(loss.cpu())
            merge2dict(loss_log, detail_loss)

        train_loss = torch.mean(torch.stack(train_loss))
        for key, value in detail_loss.items():
            detail_loss[key] = "%.6f"% (detail_loss[key] / (batch_i+1))

        test_loss, test_log = evaluate(model, test_dataloader, device, False)
        logger.info("==> EPOCH [{}], train_loss: {}, detail_loss: {}, test_loss: {}".format(epoch, train_loss, detail_loss, test_log))
        if epoch % 2 == 0 :
            model_path = os.path.join(save_model_dir, 'model_epoch_{}.pth'.format(epoch))
            torch.save(model.state_dict(), model_path)
            print("model saved : ", model_path)
