import torch
from skimage import transform
import os
import numpy as np 
import torch.nn.functional as F

# return embeddings path for dataloader to load embeddings from file
def to_embeddings_path(caption_path, directory_path):
    cls, fn = caption_path.split('/')[-2:], caption_path.split('/')[-1]
    embeddings_path = os.path.join(os.path.join(directory_path, cls), fn)
    return embeddings_path

# caption data path architecture
# -- data path
#   -- class
#     -- file
# read caption data from caption path and split file path 
def read_caption(caption_path, split_file):
    # if caption path does not exist, return empty list
    if not os.path.exists(caption_path): 
        print('Caption path does not exist!')
        return []
    
    class_list = []
    caption_list = []
    file = open(split_file, 'r')
    line = file.readline().strip('\n')

    # read class name and file name from split file and append to class list
    while line:
        _, class_name, file_name = line.split(' ')
        class_list.append(os.path.join(caption_path, class_name))   
        line = file.readline().strip('\n')
    file.close()

    # read caption data from class list and append to caption list
    for class_path in class_list:
        if not os.path.isdir(class_path):
            print('Class path does not exist!', class_path)
            continue
        for caption in os.listdir(class_path):
            caption = os.path.join(class_path, caption)
            if os.path.isfile(caption):
                caption_list.append(caption)
            else:
                print('Caption file does not exist!', caption)
    

    return caption_list


