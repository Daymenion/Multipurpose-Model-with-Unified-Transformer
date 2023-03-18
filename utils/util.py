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
        # read caption data from class path
        for caption in os.listdir(class_path):
            caption = os.path.join(class_path, caption)
            # if caption data is a file, append to caption list
            if os.path.isfile(caption):
                caption_list.append(caption)
            else:
                print('Caption file does not exist!', caption)

    return caption_list

# cosine similarity between two tensors
def cos_sim(x, y, dim=1):
    # assert x and y have the same shape
    assert (x.shape() == y.shape())
    # if x and y are 3D tensors, calculate cosine similarity between each row
    if len(x.shape()) >2:
        return F.cosine_similarity(x, y, dim=dim)
    # if x and y are 2D tensors, calculate cosine similarity between each column
    else:
        return F.cosine_similarity(x.view(1,-1), y.view(1,-1))


class to_tensor(object):
    #Convert ndarrays in sample to Tensors.
    def __call__(self, sample):
        image, embeds = sample['image'], sample['embeds']

        # we swap color axis
        #numpy image: H x W x C to torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        return {'embeds': embeds, 'image': image}


class random_crop(object):
    # Crop randomly the image.
    def __init__(self, crop_size):
        # crop_size: int or tuple (H, W) of desired output size of crop image
        assert isinstance(crop_size, (int, tuple))
        # if crop_size is int, crop_size = (crop_size, crop_size) because we want to crop square image
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            assert len(crop_size) == 2
            self.crop_size = crop_size

    def __call__(self, sample):
        image, embeds = sample['image'], sample['embeds']
        # get image height and width and crop size height and width
        height, width = image.shape[:2]
        new_height, new_width = self.crop_size

        # if image is smaller than crop size, resize image to crop size
        if height < new_height or width < new_width:
            # resize image to crop size using skimage.transform.resize function with mode = 'constant' to fill empty pixels with 0
            image = transform.resize(image, self.crop_size, mode='constant')
            return {'embeds': embeds, 'image': image}
        
        # if image is larger than crop size, crop image randomly
        top = np.random.randint(0, height - new_height)
        left = np.random.randint(0, width - new_width)

        # crop image from top left corner to top + new_height, left + new_width
        image = image[top: top + new_height, left: left + new_width, :]
        return {'embeds': embeds, 'image': image}
    

class center_crop(object):
    # Crop the center of the image.
    def __init__(self, crop_size):
        # crop_size: int or tuple (H, W) of desired output size of crop image
        assert isinstance(crop_size, (int, tuple))
        # if crop_size is int, crop_size = (crop_size, crop_size) because we want to crop square image
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            assert len(crop_size) == 2
            self.crop_size = crop_size

    def __call__(self, sample):
        image, embeds = sample['image'], sample['embeds']
        # get image height and width and crop size height and width
        height, width = image.shape[:2]
        new_height, new_width = self.crop_size

        # if image is smaller than crop size, resize image to crop size
        if height < new_height or width < new_width:
            # resize image to crop size using skimage.transform.resize function with mode = 'constant' to fill empty pixels with 0
            image = transform.resize(image, self.crop_size, mode='constant')
            return {'embeds': embeds, 'image': image}
        
        # if image is larger than crop size, crop image from center
        top = (height - new_height) // 2
        left = (width - new_width) // 2

        # crop image from top left corner to top + new_height, left + new_width
        image = image[top: top + new_height, left: left + new_width, :]
        return {'embeds': embeds, 'image': image}
    

class rescale(object):
    # Rescale the image to a given size.
    def __init__(self, output_size):
        # output_size: int or tuple (H, W) of desired output size of image
        # if tuple, output is matched to output_size exactly,
        # if int, it is matched to output_size which is smaller than the image edges, keeping the aspect ratio the same

        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, embeds = sample['image'], sample['embeds']
        # get image height and width
        height, width = image.shape[:2]
        # if output size is int, calculate new height and width
        if isinstance(self.output_size, int):
            if height > width:
                new_height, new_width = self.output_size * height / width, self.output_size
            else:
                new_height, new_width = self.output_size, self.output_size * width / height
        # if output size is tuple, new height and width are output size
        else:
            new_height, new_width = self.output_size

        new_height, new_width = int(new_height), int(new_width)

        # resize image to output size using skimage.transform.resize function
        image = transform.resize(image, (new_height, new_width))
        return {'embeds': embeds, 'image': image}


