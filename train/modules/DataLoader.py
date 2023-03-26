from __future__ import print_function, division
from utils.util import read_caption, rescale, random_crop, center_crop, to_tensor, to_embeddings_path
from torch.utils.serialization import load_lua
import torch
import os
from torch.utils.data import Dataset, DataLoader
from skimage import io, color
import matplotlib as plt
from torchvision import transforms, utils


class CMAR_Dataset(Dataset):
    def __init__(self, root_directory, caption_directory, image_directory, embeds_directory, split, transform=None):
        # check if caption directory and split file exist
        assert(os.path.isdir(os.path.join(root_directory, caption_directory)))
        assert(os.path.isfile(os.path.join(root_directory, split)))
        # pass in root directory, caption directory, image directory, and split file
        self.root_directory, self.image_directory, self.transform = root_directory, image_directory, transform
        # get caption data and image data paths
        self.caption_directory, self.split_file, self.embeds_directory = os.path.join(root_directory,caption_directory), 
        os.path.join(root_directory, split), os.path.join(root_directory, embeds_directory)

        # read caption data from caption data path and split file path
        self.caption_list = read_caption(self.caption_directory, self.split_file)
    
    
    def __len__(self):
        # return length of caption data list, equal to number of images in dataset
        return len(self.caption_list)

    def __getitem__(self, index):

        assert(os.path.join(self.caption_directory))
        # get caption data path and load caption data
        caption = load_lua(self.caption_list[index])
        # get image data path and load image data
        image = io.imread(os.path.join(self.root_directory, self.image_directory, caption['img']))
        #get embeds data path and load embeds data
        embeds = torch.load(to_embeddings_path(self.caption_list[index], self.embeds_directory))['embeds']
        # if image is grayscale, convert to rgb
        if len(image.shape) == 2:
            image = color.gray2rgb(image)
        # create sample dictionary with image and caption data
        sample = {'image': image, 'embeds': embeds}
        # if transform is not None, apply transform to image and caption data
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
