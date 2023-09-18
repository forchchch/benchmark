import diffusers
import torch
import torch.nn as nn
from torchvision.io import read_video
from PIL import Image
import clip

device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolov5_model = torch.hub.load(source='local', 
                              repo_or_dir='/root/.cache/torch/hub/ultralytics_yolov5_master', 
                              model = 'yolov5s', 
                              pretrained = True,
                              device = device)
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# video file -> (F, C, H, W)
def Video2ImageList(video_file: str) -> torch.Tensor:
    video, _, _ = read_video(video_file)
    return video.permute(0, 3, 1, 2)

# Number of objects
def Yolov5(video: torch.Tensor):
    return yolov5_model(video).xyxy[0]

def CLIP_T(video: torch.Tensor, encoded_text: torch.Tensor):
    

# print(tokenizer(['import numpy as np']).shape)