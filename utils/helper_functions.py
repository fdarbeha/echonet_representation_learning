import torch
import torchvision.transforms as transforms
import os
import random
import csv
import tqdm
import cv2
import numpy as np
import matplotlib
matplotlib.use('agg')

to_pil_image = transforms.ToPILImage()
to_tensor = transforms.ToTensor()

def get_next_model_folder(prefix, path = '', run=None):

    model_folder = lambda prefix, run_idx: f"{prefix}_run_{run_idx}"
    if run == None:
        run_idx = 1
        while os.path.isdir(os.path.join(path, model_folder(prefix, run_idx))):
            run_idx += 1
    else:
        run_idx = run

    model_path = os.path.join(path, model_folder(prefix, run_idx))
    print(f"STARTING {prefix} RUN {run_idx}! Storing the models at {model_path}")

    return model_path


def inspect_model(model):
    param_count = 0
    for param_tensor_str in model.state_dict():
        tensor_size = model.state_dict()[param_tensor_str].size()
        print(f"{param_tensor_str} size {tensor_size} = {model.state_dict()[param_tensor_str].numel()} params")
        param_count += model.state_dict()[param_tensor_str].numel()

    print(f"Number of parameters: {param_count}")

def get_image_patch_tensor_from_video_batch(img_batch):
    """
    extracts frames of a video and lists them in the same order
    returns a list of PIL images
    """
    # Input of the function is a tensor [C, frame, H, W]
    # Output of the functions is a tensor [frame, C, H, W]

    all_frames_list = []

    for i in range(img_batch.shape[1]): #frames
        all_frames_list.append(to_pil_image(img_batch[:, i, :, :]))
        # print(to_tensor(all_frames_list[i]).shape)
    
    # print(len(all_frames_list))
    return all_frames_list

def reshape_videos_cnn_input(video_batch): 
    batch_size, C, D, H, W = video_batch.shape
    
    all_frames_list = []
    for i in range(batch_size): #batch_size
        for j in range(D):
            print(video_batch[i, :, j, :, :].shape)
            all_frames_list.append(video_batch[i, :, j, :, :])
            # all_frames_list.append(to_pil_image(video_batch[i, :, j, :, :]))

    # all_frames_tensor = torch.stack(all_frames_list, dim=0)
    # print(all_frames_tensor.shape)
    # return all_frames_tensor
    return all_frames_list

def reshape_videos_cnn_input_eval(video_batch): 
    batch_size, C, D, H, W = video_batch.shape
    
    all_frames_list = []
    for i in range(batch_size): #batch_size
        for j in range(D):
            all_frames_list.append(video_batch[i, :, j, :, :])

    all_frames_tensor = torch.stack(all_frames_list, dim=0)
    # print(all_frames_tensor.shape)
    return all_frames_tensor
    # return all_frames_list


def write_csv_stats(csv_path, stats_dict):

    if not os.path.isfile(csv_path):
        with open(csv_path, "w") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(stats_dict.keys())

    for key, value in stats_dict.items():
        if isinstance(value, float):
            precision = 0.001
            stats_dict[key] =  ((value / precision ) // 1.0 ) * precision

    with open(csv_path, "a") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(stats_dict.values())


