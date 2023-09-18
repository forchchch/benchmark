import diffusers
import torch
import torch.nn as nn
from torchvision.io import read_video
import torchvision.transforms as transforms
from PIL import Image
import clip

device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolov5_model = torch.hub.load(source='local', 
                              repo_or_dir='/root/.cache/torch/hub/ultralytics_yolov5_master', 
                              model = 'yolov5s', 
                              pretrained = True,
                              device = device)
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ]
)

# video file -> Tuple[tensor(T, C, H, W), tensor(T, C, H, W)]
def Video2FrameTensor(video_file: str):
    video, _, _ = read_video(video_file, output_format = 'TCHW')
    return video

def benchmark2class_list(benchmark: dict) -> list:
    class_list = []
    for key in benchmark.keys():
        if key.startwith('cls'):
            class_list.append(benchmark[key])
    return class_list

def process_class(xyxy_list):
    pass
    # reference = {
        
    # }
    # for xyxy in xyxy_list:


    # return xyxy_list

def benchmark2instance_path_list(benchmark):
    class_list = []
    for key in benchmark.keys():
        if key.startwith('ins'):
            class_list.append(benchmark[key])
    return class_list