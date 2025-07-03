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

from utils import flow_viz

DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def load_flow_from_csv(csv_path, timestamp):
    df = pd.read_csv(csv_path)

    df_frame = df[df['timestamp'] == float(timestamp)]

    if df_frame.empty:
        raise ValueError(f"No flow found for timestamp: {timestamp}")

    H = df_frame['y'].max() + 1
    W = df_frame['x'].max() + 1

    flow = np.zeros((H, W, 2), dtype=np.float32)

    for _, row in df_frame.iterrows():
        x, y, u, v = int(row['x']), int(row['y']), row['u'], row['v']
        flow[y, x] = [u, v]  # [H, W, 2]

    return flow




def visualize(args):
     
     start = float(args.start)
     if args.stop:
          stop = float(args.stop)
     else:
          stop = start
     
     raft_csv_path = args.raft_csv_path

     save_dir = raft_csv_path.replace('/optical_flow.csv', '')
     save_dir = os.path.join(save_dir, f'viz-{args.start}-{args.stop}')
     os.makedirs(save_dir, exist_ok=True)


     png_path = args.png_path
     png_files = sorted(glob.glob(os.path.join(png_path, '*.png')))

     csv_path = args.raft_csv_path
     
     # Filter files based on start and stop timestamps
     for i, file in enumerate(png_files):   
        timestamp = float(os.path.basename(file).replace("frame", "").replace(".png", ""))
        if timestamp >= start and timestamp <= stop:
            
            img = load_image(file)
            img = img[0].permute(1,2,0).cpu().numpy()  # [1, 3, H, W] -> [H, W, 3] image output of RAFT

            flow = load_flow_from_csv(csv_path, timestamp)
            flo_rgb = flow_viz.flow_to_image(flow)

            H, W, _ = img.shape
            flo_rgb = flo_rgb[:H, :W] # Ensure flo_rgb matches img dimensions   

            img_flo = np.concatenate([img, flo_rgb], axis=0)
            cv2.imwrite(os.path.join(save_dir, f"flow{timestamp}.png"), img_flo[:, :, [2,1,0]])
            print(f"Saved visualization for timestamp {timestamp} to {save_dir}")




if __name__ == '__main__':
     parser = argparse.ArgumentParser()
     parser.add_argument('--start', help="start timestamp")
     parser.add_argument('--stop', help="stop timestamp", default=None)
     parser.add_argument('--png_path', help="path to png files")
     parser.add_argument('--raft_csv_path', help="path to RAFT csv file")
     args = parser.parse_args()

     visualize(args)