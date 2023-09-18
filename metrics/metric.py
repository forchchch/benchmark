import utils
import torch
import clip
import torch.nn.functional as F
import random
import torchvision.transforms as transforms
import pandas as pd
import os
from PIL import Image

def metrics(video_path: str, text: str, benchmark: dict):
    class_list: list(str) = utils.benchmark2class_list(benchmark)
    instance_path_list: list(str) = utils.benchmark2instance_path_list(benchmark)

    # load video
    video_tensor = utils.Video2FrameTensor(video_path)
    video_tensor4clip = torch.stack([utils.transform(frame) for frame in video_tensor])
    frame_len = video_tensor.shape[0]
    original_num_of_object = len(class_list)

    # choose a random frame
    f = random.randint(0, frame_len - 1)
    frame_pil = transforms.ToPILImage()(video_tensor[f])

    # yolov5
    yolov5_output = utils.yolov5_model([frame_pil])

    # drill the detect result
    yolov5_output_pandas = yolov5_output.pandas().xyxy
    # utils.process_class(yolov5_output_pandas)
    for idx, xyxy in enumerate(yolov5_output_pandas):
        mask = xyxy['name'].isin(class_list)
        yolov5_output_pandas[idx] = xyxy[mask]
        
    # number of objects
    noo = 0
    for xyxy in yolov5_output_pandas:
        noo += abs(xyxy.shape[0] - original_num_of_object)
    metric_number_of_objects = noo/frame_len

    # prepare for cutting pictures
    # for idx, xyxy in enumerate(yolov5_output_pandas):
    #     groups = xyxy.groupby('class')
    #     top_n_rows = []
    #     for cls, group_df in groups:
    #         top_n_rows.append(group_df.head(class_list.count(cls)))
    #     yolov5_output_pandas[idx] = pd.concat(top_n_rows)

    # encode image and text
    image_feature = utils.clip_model.encode_image(video_tensor4clip)
    text_feature = utils.clip_model.encode_image([text])

    # clip-t
    image_feature_norm = F.normalize(image_feature, p = 2, dim = 1)
    text_feature_norm = F.normalize(text_feature, p = 2, dim = 1)
    metric_CLIP_T = image_feature_norm @ text_feature_norm.T

    # frame consistency
    metric_frame_consistency = torch.mean(image_feature_norm @ image_feature_norm.T)

    # encode the instance images
    instance_feature = []
    for instance_path in instance_path_list:
        img_path = random.choice(os.listdir(instance_path))
        instance_img_tensor = utils.transform(Image.open(img_path))
        instance_feature.append(instance_img_tensor)
    instance_feature = utils.clip_model.encode_image(torch.stack(instance_feature))

    # cut the objects off the frame
    def cut_picture(picture_pil, xyxy: pd.DataFrame) -> list:
        # groups = xyxy.groupby('class')
        obj_list = []
        black_image = Image.new('RGB', (224, 224), color='black')
        for index, row in xyxy.iterrows():
            coordinate = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])
            obj = picture_pil.crop(coordinate)
            obj_list.append(obj)
        if original_num_of_object > xyxy.shape[0]:
            obj_list.append([black_image]*())
        # for cls, group_df in groups:
        #     original_num_of_object_in_this_class = class_list.count(cls)
        #     boxes: pd.DataFrame = group_df.head(original_num_of_object_in_this_class)
        #     for index, row in boxes.iterrows():
        #         xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        #         obj = picture_pil.crop((xmin, ymin, xmax, ymax))
        #         obj_list.append(obj)
        #     balance = original_num_of_object_in_this_class - boxes.shape[0]
        #     obj_list.extend([black_image]*balance)
        
    frame_obj_pils = cut_picture(frame_pil, yolov5_output_pandas[f])
    frame_obj_tensor = torch.stack([utils.transform(obj) for obj in frame_obj_pils])

    # encode the objects
    object_feature = utils.clip_model.encode_image(frame_obj_tensor)

    assert object_feature.shape == instance_feature.shape

    instance_feature_norm = F.normalize(instance_feature)
    object_feature_norm = F.normalize(object_feature)

    match_matrix = instance_feature_norm @ object_feature_norm.T

    def max_sum_from_matrix(matrix):
        n = matrix.size(0)  # Assuming 'matrix' is a n x n torch.Tensor
        max_sum = float('-inf')  # Initialize max_sum as negative infinity

        # Define a recursive function to explore all possible combinations
        def backtrack(row, col, current_sum, count):
            nonlocal max_sum

            # Base case: When we have selected 'n' elements, update max_sum
            if count == n:
                max_sum = max(max_sum, current_sum)
                return

            # Explore all possible choices for the next element
            for i in range(n):
                for j in range(n):
                    # Ensure the selected element is not from the same row or column
                    if i != row and j != col:
                        # Recursively explore the next element
                        backtrack(i, j, current_sum + matrix[i][j], count + 1)

        # Start the backtracking process for each cell in the matrix
        for i in range(n):
            for j in range(n):
                backtrack(i, j, matrix[i][j], 1)

        return max_sum
            
    # clip-i
    metric_CLIP_I = max_sum_from_matrix(match_matrix)
    
    # dino-i
    metric_DINO_I = 0

    return metric_number_of_objects, metric_CLIP_T, metric_CLIP_I, metric_DINO_I, metric_frame_consistency