import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import pandas as pd
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder



DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)



def save_results(flo, timestamp, save_dir, csv_path):
    os.makedirs(save_dir, exist_ok=True)

    # tensor [1, 2, H, W] → numpy [H, W, 2]
    flo_np = flo[0].permute(1, 2, 0).cpu().numpy()
    H, W, _ = flo_np.shape

    # optical flow npy 저장 (frame 단위) -> 추후 필요하면 합치는 코드로 하나의 .npy로 저장
    # np.save(os.path.join(save_dir, f"flow_{idx:05d}.npy"), flo_np)
    np.save(os.path.join(save_dir, f"flow_{timestamp}.npy"), flo_np)

    # optical flow csv 저장 (frame 단위로 append)
    with open(csv_path, 'a') as f:
        for y in range(H):
            for x in range(W):
                u, v = flo_np[y, x]
                f.write(f"{timestamp},{x},{y},{u:.6f},{v:.6f}\n")




def extract_timestamp(filename):
    """frame_1695195876301234.png → 1695195876301234"""
    base = os.path.basename(filename)
    stamp_str = base.replace("frame", "").replace(".png", "").replace(".jpg", "")
    return stamp_str



def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    trained_dataset = args.model.split('-')[-1].split('.')[0]
    model = model.module

    model.to(DEVICE)
    model.eval()
    
    # save_dir = os.path.join(args.path, 'flow_output' + f'_{trained_dataset}')
    save_dir = os.path.join('/dev/ssd2/hyungyu/datasets/WVN/2023-09-20-09-43-57_Hiking_Utilberg-001', 'optical_flow_raft' + f'_{trained_dataset}')
    os.makedirs(save_dir, exist_ok=True)


    csv_path = os.path.join(save_dir, 'optical_flow.csv')
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write("timestamp,x,y,u,v\n")

    save_dir_npy = os.path.join(save_dir, 'npy_files')

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)

        total_frames = len(images) -1
        for idx, (imfile1, imfile2) in enumerate(zip(images[:-1], images[1:])):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)


            if idx == 0:
                print(f"Start Optical flow calculation")
            
            if idx % 50 == 0 or idx == total_frames - 1:
                percent = int((idx + 1) / total_frames * 100)
                print(f"{percent:3d}% | Processing frame {idx + 1}/{total_frames}")
            

            timestamp = extract_timestamp(imfile1) 
            # print(timestamp)

            save_results(flow_up, timestamp, save_dir_npy, csv_path)
            #visualize(image1, flow_up, save_dir, idx)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="image frames for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)






# all_flows = []
# all_records = []

# def save_results(flo, idx, total, save_dir):
#     os.makedirs(save_dir, exist_ok=True)

#     # tensor -> numpy 변환 후 저장
#     # flo_np: [1, 2, H, W] -> [H, W, 2]
#     flo_np = flo[0].permute(1,2,0).cpu().numpy()

#     # all_flows: [l, H, W, 2] 
#     H, W, _ = flo_np.shape
#     all_flows.append(flo_np)

#     for y in range(H):
#         for x in range(W):
#             u, v = flo_np[y, x]
#             all_records.append([idx, x, y, u, v])


#     if idx == total - 1:
#         flows = np.stack(all_flows, axis=0)                   # [l, H, W, 2] 리스트 탈출
#         np.save(os.path.join(save_dir, 'lhw2.npy'), flows)

#         flows_flat = flows.reshape(flows.shape[0], -1, 2)     # [l, H*W, 2]
#         np.save(os.path.join(save_dir, 'lhwflat2.npy'), flows_flat)

#         df = pd.DataFrame(all_records, columns=['frame', 'x', 'y', 'u', 'v'])
#         df.to_csv(os.path.join(save_dir, 'optical_flow.csv'), index=False)

#         print(f"Optical flow 저장 : \n  - {flows.shape} → .npy\n  - {df.shape[0]} rows → .csv")
