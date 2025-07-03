#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python demo.py --model=models/raft-kitti.pth --path=/dev/ssd2/hyungyu/datasets/WVN/2023-09-20-09-43-57_Hiking_Utilberg-001/rgb_png


# CUDA_VISIBLE_DEVICES=0 python demo.py --model=models/raft-things.pth --path=demo-frames
# CUDA_VISIBLE_DEVICES=0 python demo.py --model=models/raft-kitti.pth --path=demo-frames
# CUDA_VISIBLE_DEVICES=0 python demo.py --model=models/raft-sintel.pth --path=demo-frames
# CUDA_VISIBLE_DEVICES=0 python demo.py --model=models/raft-small.pth --path=demo-frames