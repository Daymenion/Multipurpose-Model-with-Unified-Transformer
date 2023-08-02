import sys
import os
import argparse
import torch
import pickle
import time
import json
import numpy as np
import glob
import libs.architecture as architecture
from libs.utils import dataloaders as dl
from tensorboardX import SummaryWriter
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import libs.architecture.routines as r
from libs.architecture.util import ScheduledOptim
from torch.optim.adam import Adam
import random
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from PIL import Image
from libs.utils.train_util import *
from libs.utils.cocoapi.coco import COCO
from libs.utils.vqa.vqa import VQA
from libs.utils.vqa.vqaEval import VQAEval
from libs.utils.cocoapi.eval import COCOEvalCap
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from libs.utils.bleu import compute_bleu

from typing import List, Optional

# Define file paths and directories
coco_images_dir = 'data/coco/train_val'
caption_dir = 'data/coco'
vqa_dir = 'data/vqa'
hmdb_data_dir = 'data/hmdb'
hmdb_process_dir = 'data/hmdbprocess'
model_save_path = 'checkpoints'

# Define image parameters
dropout = 0.5
image_height = 224
image_width = 224


class ImageDataset(Dataset):
    """
    A PyTorch Dataset class for loading images from a list of file paths and applying a given transformation.
    """
    def __init__(self, image_file_paths: List[str], transform: Optional[transforms.Compose] = None):
        """
        Args:
            image_file_paths (list): A list of file paths to the images.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.image_file_paths = image_file_paths
        self.num_images = len(self.image_file_paths)
        self.transform = transform

    def __len__(self) -> int:
        """
        Returns:
            The number of images in the dataset.
        """
        return self.num_images

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Args:
            index (int): Index of the image to retrieve.

        Returns:
            The transformed image at the given index.
        """
        try:
            image = Image.open(self.image_file_paths[index])
            image = image.convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image).convert("RGB")
            return image
        except Exception as e:
            print(f"Error loading image at index {index}: {e}")
            return None


def vqa_evaluate(model, batch_size):
    """
    Evaluate a VQA model on the validation set and print the results.

    Args:
        model (nn.Module): The VQA model to evaluate.
        batch_size (int): The batch size to use for evaluation.
    """
    predictions = []
    vqa_val_questions_path = os.path.join(vqa_dir, 'v2_OpenEnded_mscoco_val2014_questions.json')
    vqa_val_annotations_path = os.path.join(vqa_dir, 'v2_mscoco_val2014_annotations.json')
    vocab_file_path = os.path.join('conf/vqa_vocab.pkl')
    with open(vocab_file_path, 'rb') as f:
        ans_to_id, id_to_ans = pickle.loads(f.read())
    vqa = VQA(annotation_file=vqa_val_annotations_path, question_file=vqa_val_questions_path)
    questions = []
    image_paths = []
    question_ids = []
    answers = []
    
    # Load questions and answers from the validation set
    for q in vqa.questions['questions']:
        image_paths.append(os.path.join(coco_images_dir, '%012d.jpg' % (q['image_id'])))
        questions.append(q['question'])
        question_ids.append(q['question_id'])
        answer = vqa.loadQA(q['question_id'])
        answers.append(answer[0]['multiple_choice_answer'])
    answers = np.array(answers)
    
    # Define validation set transformations
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_tfms = transforms.Compose([
        transforms.Resize(int(image_height * 1.14)),
        transforms.CenterCrop(image_height),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Load validation set images
    val_dataset = ImageDataset(image_paths, val_tfms)
    val_dl = DataLoader(val_dataset, num_workers=8, batch_size=batch_size,
                         drop_last=False, pin_memory=True, shuffle=False)
    counter = 0
    result_json = []
    
    # Make predictions on the validation set
    for batch in tqdm(val_dl):
        images = batch.cuda(0)
        batch_questions = questions[counter:counter + images.shape[0]]
        preds, _, _ = r.vqa(model, images, batch_questions, mode='predict', return_str_preds=True, num_steps=1)
        preds = preds.reshape([-1]).cpu().numpy()
        for p in preds:
            result_json.append({'question_id': question_ids[counter], 'answer': id_to_ans[p]})
            counter += 1
    with open('results/vqa_prediction.json', 'w') as outfile:
        json.dump(result_json, outfile)
    
    # Evaluate the predictions
    predictions = []
    vqa_res = vqa.loadRes('results/vqa_prediction.json', vqa_val_questions_path)
    vqa_eval = VQAEval(vqa, vqa_res, n=2)   # n is precision of accuracy (number of places after decimal), default is 2
    with open('results/vqa_prediction.json', 'r') as f:
        json_ans = json.load(f)
    for j in json_ans:
        predictions.append(j['answer'])
    predictions = np.array(predictions)
    print(np.sum(predictions == answers) / predictions.shape[0])
    
    # Print evaluation results
    vqa_eval.evaluate()
    print('\n\nVQA Evaluation results')
    print('-' * 50)
    print("\n")
    print("Overall Accuracy is: %.02f\n" % (vqa_eval.accuracy['overall']))
    print("Per Question Type Accuracy is the following:")
    for quesType in vqa_eval.accuracy['perQuestionType']:
        print("%s : %.02f" % (quesType, vqa_eval.accuracy['perQuestionType'][quesType]))
    print("\n")
    print("Per Answer Type Accuracy is the following:")
    for ansType in vqa_eval.accuracy['perAnswerType']:
        print("%s : %.02f" % (ansType, vqa_eval.accuracy['perAnswerType'][ansType]))
    print("\n")
   
        
        
def coco_evaluate(model: torch.nn.Module, batch_size: int) -> None:
    """
    Evaluates a given image captioning model on the COCO validation set and prints the evaluation results.

    Args:
        model (torch.nn.Module): The image captioning model to evaluate.
        batch_size (int): The batch size to use for evaluation.
    """
    # Load validation data
    val_ann_file = os.path.join(caption_dir, 'captions_val2017.json')
    coco = COCO(val_ann_file)
    img_ids = coco.getImgIds()
    img_list = [os.path.join(coco_images_dir, '%012d.jpg' % (i)) for i in img_ids]

    # Define validation transformations
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_tfms = transforms.Compose([
        transforms.Resize(int(224*1.14)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # Create validation dataset and dataloader
    val_dataset = ImageDataset(img_list, val_tfms)
    val_dl = DataLoader(val_dataset, num_workers=8, batch_size=batch_size,
                         drop_last=False, pin_memory=True, shuffle=False)

    # Generate predictions for each image in the validation set
    counter = 0
    result_json = []
    for b in tqdm(val_dl):
        imgs = b.cuda(0)
        preds, _, _ = r.image_caption(model, imgs, mode='predict', return_str_preds=True)
        for p in preds:
            result_json.append({'image_id': img_ids[counter], 'caption': p})
            counter += 1

    # Save predictions to file
    with open('results/caption_prediction.json', 'w') as outfile:
        json.dump(result_json, outfile)

    # Evaluate the predictions using COCO metrics
    cocoRes = coco.loadRes('results/caption_prediction.json')
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # Print evaluation results
    print('\n\nCOCO Evaluation results')
    print('-'*50)
    for metric, score in cocoEval.eval.items():
        print('%s: %.3f'%(metric, score))
                


def hmdb_evaluate(model, batch_size):
    """
    Evaluates a given action recognition model on the HMDB51 test set and prints the evaluation results.

    Args:
        model (torch.nn.Module): The action recognition model to evaluate.
        batch_size (int): The batch size to use for evaluation.
    """
    # Load test data
    _, test_dl = dl.hmdb_batchgen(hmdb_data_dir, hmdb_process_dir, num_workers=8, batch_size=batch_size,
                                  test_batch_size=batch_size, clip_len=16)

    # Evaluate the model on the test set
    correct = 0
    total = 0
    for b in tqdm(test_dl):
        vid, labels = b
        vid = vid.cuda(device=0)
        preds, _, _ = r.hmdb(model, vid, mode='predict', return_str_preds=True, num_steps=1)
        preds = preds.reshape([-1]).cpu().numpy()
        labels = labels.reshape([-1]).cpu().numpy()
        correct += np.sum(preds == labels)
        total += labels.shape[0]

    # Calculate and print the accuracy
    accuracy = (correct / total) * 100
    print('\n\nHMDB Evaluation results')
    print('-' * 50)
    print('Accuracy: %f%%' % (accuracy))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UnifiedTransformer Evaluation script.')
    parser.add_argument('task', help='Task for which to evaluate for.')
    parser.add_argument('model_file', help='Model file to evaluate with.')
    parser.add_argument('--batch_size', default=32, help='Batch size')
    args = parser.parse_args()
    torch.manual_seed(47)
    task=args.task
    batch_size=int(args.batch_size)
    model_file=args.model_file
    
    model = architecture.UnifiedTransformer(gpu_id=0)
    model.restore_file(model_file)
   
    model=model.to(0)
    model=model.eval()
    if task=='caption':
        coco_evaluate(model,batch_size)
    elif task=='vqa':
        vqa_evaluate(model,batch_size)
    elif task=='hmdb':
        hmdb_evaluate(model, batch_size)
    else:
        print('Invalid task provided')
    