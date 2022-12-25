'''
Generating probability map
import torch
outputs = torch.tensor([[1, 2, 3], [4, 5, 6]])
#targets = torch.tensor([[0, 1, 1], [1, 1, 1]])
targets = [[0, 1, 1], [1, 1, 1]]
probs = outputs.sigmoid()
predicts = (probs >= 0.5).type(torch.FloatTensor)
acc_data = (predicts == targets).sum().item()
print(probs)
print(predicts)
print(acc_data)'''

import sys
import os
import argparse
import logging
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from testWSI_producer import GridWSIPatchDataset  # noqa
from mobilenet_V2_crf_model import MobileNetV2


parser = argparse.ArgumentParser(description='Get the probability map of tumor'
                                 ' patch predictions given a WSI')
parser.add_argument('--wsi_path', default=r'D:\self_study\medical_imaging\digital_slide_sample\003548-1b - 1.tif', metavar='WSI_PATH', type=str,
                    help='Path to the input WSI file')
parser.add_argument('--ckpt_path', default=r'D:\self_study\medical_imaging\ckpt\mobilenet_crf.pth', metavar='CKPT_PATH', type=str,
                    help='Path to the saved ckpt file of a pytorch model')
parser.add_argument('--cfg_path', default='D:\self_study\medical_imaging\config/resnet18_crf.json', metavar='CFG_PATH', type=str,
                    help='Path to the config file in json format related to'
                    ' the ckpt file')
parser.add_argument('--mask_path', default='D:\self_study\medical_imaging\mask/testslide.npy', metavar='MASK_PATH', type=str,
                    help='Path to the tissue mask of the input WSI file')
parser.add_argument('--probs_map_path', default=r'D:\self_study\medical_imaging\prob_map\prob_003548-1b-1.npy', metavar='PROBS_MAP_PATH',
                    type=str, help='Path to the output probs_map numpy file')
parser.add_argument('--GPU', default='0', type=str, help='which GPU to use'
                    ', default 0')
parser.add_argument('--num_workers', default=0, type=int, help='number of '
                    'workers to use to make batch, default 5')
parser.add_argument('--eight_avg', default=0, type=int, help='if using average'
                    ' of the 8 direction predictions for each patch,'
                    ' default 0, which means disabled')
def get_probs_map(model, dataloader, device):
    #device = torch.device("cuda:0")
    xid_num = dataloader.dataset._mask.shape[0] + 768
    yid_num = dataloader.dataset._mask.shape[1] + 768
    probs_map = np.zeros((xid_num, yid_num))
    print(probs_map.shape)
    num_batch = len(dataloader)
    # only use the prediction of the center patch within the grid
    idx_center = dataloader.dataset._grid_size // 2

    count = 0
    time_now = time.time()
    with torch.no_grad():
        for (data, x_mask, y_mask) in dataloader:
            data = data.to(device)
            output = model(data)
            # because of torch.squeeze at the end of forward in resnet.py, if the
            # len of dim_0 (batch_size) of data is 1, then output removes this dim.
            # should be fixed in resnet.py by specifying torch.squeeze(dim=2) later
            if len(output.shape) == 1:
                probs = output[idx_center].sigmoid().cpu().data.numpy().flatten()
            else:
                probs = output[:,
                        idx_center].sigmoid().cpu().data.numpy().flatten()

            for i in range(len(x_mask)):
                prob_area = np.full((36, 36), probs[i]) #9:56   25:36
                xmin = (x_mask[i] - 18).to(dtype=torch.int, device=device)
                xmax = (x_mask[i] + 18).to(dtype=torch.int, device=device)
                ymin = (y_mask[i] - 18).to(dtype=torch.int, device=device)
                ymax = (y_mask[i] + 18).to(dtype=torch.int, device=device)
                probs_map[xmin:xmax, ymin:ymax] = prob_area
            #probs_map[x_mask-112:x_mask+112, y_mask-112:y_mask+112] = probs
            count += 1

            time_spent = time.time() - time_now
            time_now = time.time()
            logging.info(
                '{}, flip : {}, rotate : {}, batch : {}/{}, Run Time : {:.2f}'
                    .format(
                    time.strftime("%Y-%m-%d %H:%M:%S"), dataloader.dataset._flip,
                    dataloader.dataset._rotate, count, num_batch, time_spent))

    return probs_map


def make_dataloader(args, cfg, flip='NONE', rotate='NONE'):
    batch_size = cfg['batch_size'] * 2
    num_workers = args.num_workers

    dataloader = DataLoader(
        GridWSIPatchDataset(args.wsi_path, args.mask_path,
                            image_size=cfg['image_size'],
                            patch_size=cfg['patch_size'],
                            crop_size=cfg['crop_size'], normalize=True,
                            flip=flip, rotate=rotate),
        batch_size=batch_size, num_workers=num_workers, drop_last=False)

    return dataloader



def run(args):
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    device = torch.device("cuda:0")
    logging.basicConfig(level=logging.INFO)

    with open(args.cfg_path) as f:
        cfg = json.load(f)

    #if cfg['image_size'] % cfg['patch_size'] != 0:
    #        raise Exception('Image size / patch size != 0 : {} / {}'.
    #                        format(cfg['image_size'], cfg['patch_size']))

    patch_per_side = cfg['image_size'] // cfg['patch_size']
    grid_size = patch_per_side * patch_per_side

    mask = np.load(args.mask_path)
    ckpt = torch.load(args.ckpt_path, map_location="cuda:0")
    #model = MobileNetV2(num_classes=1, use_crf=True, num_nodes=grid_size)
    model = MobileNetV2(num_nodes=9, use_crf=True)
    model.load_state_dict(ckpt)
    if torch.cuda.is_available():
        model = model.to(device)
    model = model.cuda().eval()

    dataloader = make_dataloader(args, cfg, flip='NONE', rotate='NONE')
    x_len = dataloader.dataset._mask.shape[0]
    y_len = dataloader.dataset._mask.shape[1]
    probs_map = get_probs_map(model, dataloader, device)

    np.save(args.probs_map_path, probs_map[cfg['image_size']//2:x_len-cfg['image_size']//2, cfg['image_size']//2:y_len-cfg['image_size']//2])


def main():
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()