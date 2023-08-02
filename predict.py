import sys
import os
import argparse
import pickle
import json
import cv2
import torch
import numpy as np
import libs.architecture as architecture
from typing import Any
import warnings
warnings.filterwarnings("ignore")

from PIL import Image
from torchvision import transforms


penn_vocab_file='conf/penn_vocab.json'
vqa_vocab_file='conf/vqa_vocab.pkl'
hmdb_labels_file='conf/hmdblabels.txt'

video_resize_height=300
video_resize_width=300
clip_len=16
crop_size=224


def extract_frames_from_video(video_file):
    """
    Extracts frames from a video file and returns a tensor of shape (1, 16, 3, 224, 224).

    Args:
    video_file: str, path to the video file.

    Returns:
    A tensor of shape (1, 16, 3, 224, 224) containing the extracted frames.
    """
    capture = cv2.VideoCapture(video_file)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    EXTRACT_FREQUENCY = 4

    # Adjust the extract frequency to ensure that at least 16 frames are extracted
    if frame_count // EXTRACT_FREQUENCY <= 16:
        EXTRACT_FREQUENCY -= 1
        if frame_count // EXTRACT_FREQUENCY <= 16:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= 16:
                EXTRACT_FREQUENCY -= 1

    count = 0
    i = 0
    retaining = True
    frames = []

    # Extract frames from the video
    while (count < frame_count and retaining):
        retaining, frame = capture.read()
        if frame is None:
            continue

        if count % EXTRACT_FREQUENCY == 0:
            if (frame_height != video_resize_height) or (frame_width != video_resize_width):
                frame = cv2.resize(frame, (video_resize_width, video_resize_height))
            frames.append(frame)
            i += 1
        count += 1

    capture.release()

    # Convert the frames to a tensor
    frame_count = len(frames)
    buffer = np.empty((frame_count, video_resize_height, video_resize_width, 3), np.dtype('float32'))
    for i, frame in enumerate(frames):
        buffer[i] = np.array(frame)

    # Crop and normalize the frames
    time_index = 0
    height_index = 0
    width_index = 0
    buffer = buffer[time_index:time_index + clip_len,
                height_index:height_index + crop_size,
                width_index:width_index + crop_size, :]
    buffer = buffer / 255
    for i, frame in enumerate(buffer):
        frame -= np.array([[[0.485, 0.456, 0.406]]])
        frame /= np.array([[[0.229, 0.224, 0.225]]])
        buffer[i] = frame
    buffer = buffer.transpose((0, 3, 1, 2))
    buffer = torch.from_numpy(buffer)

    return buffer.unsqueeze(0)



def extract_pixels_from_image(image_path: str) -> Any:
    """
    Extracts pixels from an image and applies normalization and transforms to it.

    Args:
        image_path (str): The path to the image file.

    Returns:
        Any: The transformed image tensor.
    """
    # Open the image and convert it to RGB format
    img = Image.open(image_path)
    img = img.convert('RGB')

    # Define the normalization and transforms to be applied to the image
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(int(224*1.14)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # Apply the transforms to the image and return the transformed image tensor
    img = transform(img)
    img = img.unsqueeze(0) # type: ignore
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UnifiedTransformer prediction script.')
    parser.add_argument('model_file', help='Location to pretrained model file.')
    parser.add_argument('task', help='Task to predict for.')
    parser.add_argument('--image', default=None, help='Image file to encode')
    parser.add_argument('--video', default=None, help='Video file to encodre')
    parser.add_argument('--text', default=None, help='Text to encode')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    model_file=args.model_file
    task=args.task
    image=args.image
    video=args.video
    text=str(args.text)
    verbose=args.verbose
    if verbose==False:
        sys.stdout = open(os.devnull, 'w')
    #Load UnifiedTransformer model
    model = architecture.UnifiedTransformer(gpu_id=0)
    model.restore_file(model_file)
    model=model.to(0)
    model=model.eval()
    model.reset(1)
    if image is not None:
        image=extract_pixels_from_image(image)
        image=image.to(0)
        model.encode_images(image)
    if text is not None:
        model.encode_englishtexts([text])
    if video is not None:
        video=extract_frames_from_video(video)
        video=video.to(0) # type: ignore
        model.encode_videos(video)
    if verbose==False:
        sys.stdout = sys.__stdout__
    if task=='caption':
        prediction=model.decode_greedy('IMAGE_CAPTION',num_steps=100)
        prediction = prediction.argmax(-1)
        prediction = model.english_language_perph.decode_tokens(prediction)
        print('Caption Prediction: %s'%prediction[0])
    elif task=='hmdb':
        prediction = model.decode_greedy('HMDB', num_steps=1)
        prediction = prediction.argmax(-1).cpu().tolist()[0][0]
        with open(hmdb_labels_file,'r') as  f:
            lines=f.readlines()
        id_to_label=dict()
        for l in lines:
            id,label=l.split(' ')
            id_to_label[id]=label
        prediction=id_to_label[str(prediction)]
        print('Action recognition prediction: %s'%prediction)

    elif task=='vqa':
        prediction = model.decode_greedy('VQA', num_steps=1)
        prediction = prediction.argmax(-1).cpu().tolist()[0][0]
        with open(vqa_vocab_file,'rb') as f:
            ans_to_id,id_to_ans=pickle.loads(f.read())
        prediction=id_to_ans[prediction]
        print('VQA Prediction: %s'%prediction)

    elif task=='penn':
        if text is None:
            raise Exception('No text has been provided. POS tagging cannot proceed.')
        prediction= model.decode_greedy('PENN', num_steps=len(text.split(' ')))
        prediction=prediction.argmax(-1).cpu().tolist()[0]
        with open(penn_vocab_file,'r') as f:
            data=json.loads(f.read())
        id_to_tag=data['id_to_tag']
        penn_text=''
        for p in prediction:
            penn_text='%s %s'%(penn_text,id_to_tag[str(p)])
        print('POS tagging Prediction: %s'%penn_text)


        

        

        