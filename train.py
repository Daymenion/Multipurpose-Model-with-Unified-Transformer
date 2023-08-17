import sys
import os
import argparse
import torch
import time
import glob
import numpy as np
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
from tqdm import tqdm
from libs.utils.train_util import *

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", FutureWarning)

coco_images = 'data/coco/train_val'
caption_dir = 'data/coco'
vqa_dir = 'data/vqa'
model_save_path = 'checkpoints'
hmdb_data_dir='data/hmdb'
hmdb_process_dir='data/hmdbprocess'
penn_data_dir='data/penn'



def train(shared_model, task, batch_size, train_steps, gpu_id, start, restore, counter, barrier=None, save_interval=None,
          eval_interval=None, log=True):
    """
    Train function that trains a given shared model on a given task.

    Args:
        shared_model (torch.nn.Module): The shared model to be trained.
        task (str): The task to be trained on.
        batch_size (int): The batch size to be used during training.
        train_steps (int): The number of training steps to be performed.
        gpu_id (int): The ID of the GPU to be used for training.
        start (int): The starting step for training.
        restore (bool): Whether to restore the model from a previous checkpoint.
        counter (torch.multiprocessing.Value): A shared counter to keep track of the current training step.
        barrier (torch.multiprocessing.Barrier, optional): A synchronization barrier to be used during training.
        save_interval (int, optional): The interval at which to save the model during training.
        eval_interval (int, optional): The interval at which to evaluate the model during training.
        log (bool, optional): Whether to log the training progress.

    Returns:
        None
    """
    # Create log directory if it does not exist
    log_dir = 'logs/%s' % task
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create summary writer if logging is enabled
    if log:
        summary_writer = SummaryWriter(log_dir)

    # Set random seed
    torch.manual_seed(int(random.random() * 1000))

    # Create local model
    if gpu_id > 0:
        model = architecture.UnifiedTransformer(gpu_id=gpu_id)
        model = model.cuda(gpu_id)
    else:
        # For GPU 0, use the shared model always
        model = shared_model

    # Get data loaders and optimizer based on task
    if task == 'caption':
        train_dl, val_dl = dl.coco_cap_batchgen(caption_dir='data/coco', image_dir='data/coco/train_val',
                                                num_workers=8, batch_size=batch_size)
        optimizer = ScheduledOptim(
            Adam(
                filter(lambda x: x.requires_grad, shared_model.parameters()),
                betas=(0.9, 0.98), eps=1e-09),
            512, 16000, restore, init_lr=0.02)
    elif task == 'vqa':
        train_dl, val_dl = dl.vqa_batchgen(vqa_dir='data/vqa', image_dir='data/coco/train_val', num_workers=8,
                                           batch_size=batch_size)
        optimizer = ScheduledOptim(
            Adam(
                filter(lambda x: x.requires_grad, shared_model.parameters()),
                betas=(0.9, 0.98), eps=1e-09),
            512, 16000, restore, max_lr=0.0001, init_lr=0.02)
    elif task == 'hmdb':
        train_dl, val_dl = dl.hmdb_batchgen(hmdb_data_dir, hmdb_process_dir,
                                            num_workers=8, batch_size=batch_size, test_batch_size=int(batch_size / 4),
                                            clip_len=16)
        optimizer = ScheduledOptim(
            Adam(
                filter(lambda x: x.requires_grad, shared_model.parameters()),
                betas=(0.9, 0.98), eps=1e-09),
            512, 16000, restore, max_lr=0.0001, init_lr=0.02)
    elif task == 'penn':
        train_dl, val_dl, test_dl = dl.penn_dataloader(penn_data_dir, batch_size=batch_size,
                                                        test_batch_size=int(batch_size / 2), num_workers=4,
                                                        vocab_file='conf/penn_vocab.json')
        optimizer = ScheduledOptim(
            Adam(
                filter(lambda x: x.requires_grad, shared_model.parameters()),
                betas=(0.9, 0.98), eps=1e-09),
            512, 16000, restore, init_lr=0.02)
        
    # Set model to training mode
    model = model.train()


    for i in range(start, train_steps):
        model.zero_grad()
        if barrier is not None:
            barrier.wait()
        if gpu_id > 0:
            with torch.cuda.device(gpu_id):
                model.load_state_dict(shared_model.state_dict())
                       
        # Calculate loss
        step = counter.increment()
        if task == 'caption':
            if (log and eval_interval is not None and i % eval_interval == 0):
                model = model.eval()
                val_loss=0
                val_acc=0
                print('-' * 100)
                print('Evaluation step')
                for b in tqdm(val_dl):
                    imgs = b['img']
                    if gpu_id>=0:
                        imgs=imgs.cuda(device=gpu_id)
                    captions = b['cap']
                    # In val mode we do not pass the targets for prediction. We use it only for loss calculation
                    _,loss,acc = r.image_caption(model, imgs, targets=captions, mode='val',return_str_preds=True)
                    val_loss += float(loss.detach().cpu().numpy())
                    val_acc+=acc
                val_loss/=len(val_dl)
                val_acc=(val_acc/len(val_dl))
                summary_writer.add_scalar('Val_loss', val_loss, step)
                print('Step %d, COCO validation loss: %f, Accuracy %f %%' % (step, val_loss,val_acc))
                print('-' * 100)
                model = model.train()
            batch = next(DL)
            if gpu_id >= 0:
                imgs = batch['img'].cuda(device=gpu_id)
            else:
                imgs = batch['img']
            captions = batch['cap']
            _, loss,acc = r.image_caption(model, imgs, targets=captions)
            loss.backward()
            loss=loss.detach()
            if log:
                summary_writer.add_scalar('Loss', loss, step)
            print('Step %d, Caption Loss: %f, Accuracy:  %f %%' % (step, loss,acc))
            
        elif task == 'vqa':
            
            # If it's time to evaluate the model
            if (log and eval_interval is not None and i % eval_interval == 0):
                
                # Set the model to evaluation mode
                model = model.eval()
                
                # Initialize the validation loss and accuracy
                val_loss = 0
                val_acc = 0
                
                # Print the evaluation step
                print('-' * 100)
                print('Evaluation step')
                
                # Loop through the validation data loader
                for val_batch in tqdm(val_dl):
                    
                    # Get the images, questions, and answers from the batch
                    val_imgs = val_batch['img']
                    val_answers = val_batch['ans']
                    
                    # Move the data to the GPU if necessary
                    if gpu_id >= 0:
                        val_imgs = val_imgs.cuda(device=gpu_id)
                        val_answers = val_answers.cuda(device=gpu_id)
                    
                    # Get the predicted answers, loss, and accuracy for the batch
                    val_preds, val_batch_loss, val_batch_acc = r.vqa(model, val_imgs, val_batch['ques'], 
                                                                      targets=val_answers, mode='val', 
                                                                      return_str_preds=True)
                    
                    # Add the batch loss to the total validation loss
                    val_loss += float(val_batch_loss.detach().cpu().numpy())
                    
                    # Add the batch accuracy to the total validation accuracy
                    val_acc += val_batch_acc
                
                # Calculate the average validation loss and accuracy
                val_loss /= len(val_dl)
                val_acc = (val_acc / len(val_dl))
                
                # Log the validation loss
                summary_writer.add_scalar('Val_loss', val_loss, step)
                
                # Print the validation loss and accuracy
                print('Step %d, VQA validation loss: %f, Accuracy %f %%' % (step, val_loss, val_acc))
                print('-' * 100)
                
                # Set the model back to training mode
                model = model.train()
                
                # Continue to the next iteration
                continue
            
            # Get the images, questions, and answers from the batch
            train_imgs = batch['img']
            train_answers = batch['ans']
            train_questions = batch['ques']
            
            # Move the data to the GPU if necessary
            if gpu_id >= 0:
                train_imgs = train_imgs.cuda(device=gpu_id)
                train_answers = train_answers.cuda(device=gpu_id)
            
            # Get the predicted answers, loss, and accuracy for the batch
            train_preds, train_batch_loss, train_batch_acc = r.vqa(model, train_imgs, train_questions, 
                                                                    targets=train_answers, return_str_preds=True)
            
            # Backpropagate the loss
            train_batch_loss.backward()
            
            # Detach the loss
            train_batch_loss = train_batch_loss.detach()
            
            # Log the loss if necessary
            if log:
                summary_writer.add_scalar('Loss', train_batch_loss, step)
            
            # Print the loss and accuracy
            print('Step %d, VQA Loss: %f, Accuracy:  %f %%' % (step, train_batch_loss, train_batch_acc))
        
        # If the task is HMDB
        elif task == 'hmdb':
            
            # If it's time to evaluate the model
            if (log and eval_interval is not None and i % eval_interval == 0):
                
                # Set the model to evaluation mode
                model = model.eval()
                
                # Initialize the validation loss and accuracy
                val_loss = 0
                val_acc = 0
                
                # Print the evaluation step
                print('-' * 100)
                print('Evaluation step')
                
                # Loop through the validation data loader
                for val_batch in tqdm(val_dl):
                    
                    # Get the videos and labels from the batch
                    val_vids, val_labels = val_batch
                    
                    # Move the data to the GPU if necessary
                    if gpu_id >= 0:
                        val_vids = val_vids.cuda(device=gpu_id)
                        val_labels = val_labels.cuda(device=gpu_id)
                    
                    # Get the predicted labels, loss, and accuracy for the batch
                    val_preds, val_batch_loss, val_batch_acc = r.hmdb(model, val_vids, targets=val_labels, mode='val')
                    
                    # Add the batch loss to the total validation loss
                    val_loss += float(val_batch_loss.detach().cpu().numpy())
                    
                    # Add the batch accuracy to the total validation accuracy
                    val_acc += val_batch_acc
                
                # Calculate the average validation loss and accuracy
                val_loss /= len(val_dl)
                val_acc = (val_acc / len(val_dl))
                
                # Log the validation loss
                summary_writer.add_scalar('Val_loss', val_loss, step)
                
                # Print the validation loss and accuracy
                print('Step %d, HMDB validation loss: %f, Accuracy %f %%' % (step, val_loss, val_acc))
                print('-' * 100)
                
                # Set the model back to training mode
                model = model.train()
                
                # Continue to the next iteration
                continue
            
            # Get the videos and labels from the batch
            train_vids, train_labels = batch
            
            # Move the data to the GPU if necessary
            if gpu_id >= 0:
                train_vids = train_vids.cuda(device=gpu_id)
                train_labels = train_labels.cuda(device=gpu_id)
            
            # Get the predicted labels, loss, and accuracy for the batch
            train_preds, train_batch_loss, train_batch_acc = r.hmdb(model, train_vids, targets=train_labels, 
                                                                      return_str_preds=True)
            
            # Backpropagate the loss
            train_batch_loss.backward()
            
            # Detach the loss
            train_batch_loss = train_batch_loss.detach()
            
            # Log the loss if necessary
            if log:
                summary_writer.add_scalar('Loss', train_batch_loss, step)
            
            # Print the loss and accuracy
            print('Step %d, HMDB Loss: %f, Accuracy:  %f %%' % (step, train_batch_loss, train_batch_acc))
        
        # If the task is PENN
        elif task == 'penn':
            
            # If it's time to evaluate the model
            if (log and eval_interval is not None and i % eval_interval == 0):
                
                # Set the model to evaluation mode
                model = model.eval()
                
                # Initialize the validation loss and accuracy
                val_loss = 0
                val_acc = 0
                
                # Print the evaluation step
                print('-' * 100)
                print('Evaluation step')
                
                # Loop through the validation data loader
                for val_batch in tqdm(test_dl):
                    
                    # Get the inputs and targets from the batch
                    val_inputs = val_batch['text']
                    val_targets = val_batch['tokens']
                    val_pad_id = val_batch['pad_id']
                    val_pad_mask = val_batch['pad_mask']
                    
                    # Move the data to the GPU if necessary
                    if gpu_id >= 0:
                        val_targets = val_targets.to(gpu_id)
                        val_pad_mask = val_pad_mask.to(gpu_id)
                    
                    # Get the predicted targets, loss, and accuracy for the batch
                    val_preds, val_batch_loss, val_batch_acc = r.penn(model, val_inputs, target_pad_mask=val_pad_mask,
                                                                      pad_id=val_pad_id, targets=val_targets, 
                                                                      mode='val', return_str_preds=True)
                    
                    # Detach the loss
                    val_batch_loss = val_batch_loss.detach()
                    
                    # Add the batch loss to the total validation loss
                    val_loss += float(val_batch_loss.cpu().numpy())
                    
                    # Add the batch accuracy to the total validation accuracy
                    val_acc += val_batch_acc
                
                # Calculate the average validation loss and accuracy
                val_loss /= len(val_dl)
                val_acc = (val_acc / len(val_dl))
                
                # Log the validation loss
                summary_writer.add_scalar('Val_loss', val_loss, step)
                
                # Print the validation loss and accuracy
                print('Step %d, PENN validation loss: %f, Accuracy %f %%' % (step, val_loss, val_acc))
                print('-' * 100)
                
                # Set the model back to training mode
                model = model.train()
            
            # Get the inputs and targets from the batch
            train_inputs = batch['text']
            train_targets = batch['tokens']
            train_pad_id = batch['pad_id']
            train_pad_mask = batch['pad_mask']
            
            # Move the data to the GPU if necessary
            if gpu_id >= 0:
                train_targets = train_targets.to(gpu_id)
                train_pad_mask = train_pad_mask.to(gpu_id)
            
            # Get the predicted targets, loss, and accuracy for the batch
            train_preds, train_batch_loss, train_batch_acc = r.penn(model, train_inputs, pad_id=train_pad_id, 
                                                                      targets=train_targets, 
                                                                      target_pad_mask=train_pad_mask)
            
            # Backpropagate the loss
            train_batch_loss.backward()
            
            # Detach the loss
            train_batch_loss = train_batch_loss.detach()
            
            # Log the loss if necessary
            if log:
                summary_writer.add_scalar('Loss', train_batch_loss, step)
            
            # Print the loss and accuracy
            print('Step %d, PENN Loss: %f, Accuracy:  %f %%' % (step, train_batch_loss, train_batch_acc))
        
        # If the GPU ID is greater than 0, ensure shared gradients
        if gpu_id > 0:
            ensure_shared_grads(model, shared_model, gpu_id)
        
        # Take a step with the optimizer
        optimizer.step()
        
        # Save the model if necessary
        if (save_interval is not None and (i + 1) % save_interval == 0):
            shared_model.save(model_save_path, step)
        
        # Flush the stdout buffer
        sys.stdout.flush()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UnifiedTransformer training script.')
    parser.add_argument('n_iters', help='Number of iterations to train.')
    parser.add_argument('tasks', help='List of tasks seperated by comma.')
    parser.add_argument('batch_sizes', help='List of batch size for each task seperated by comma')
    parser.add_argument('--n_jobs', default=1, help='Number of asynchronous jobs to run for each task.')
    parser.add_argument('--n_gpus', default=1, help='Number of GPUs to use')
    parser.add_argument('--save_interval', default=100, help='Number of iterations after which to save the model.')
    parser.add_argument('--restore', default=-1, help='Step from which to restore model training')
    parser.add_argument('--restore_last', help='Restore the latest version of the model.', action='store_true')
    parser.add_argument('--eval_interval', help='Interval after which to evaluate on the test/val set.', default=1000)
    
    args = parser.parse_args()
    torch.manual_seed(47)
    mp.set_start_method('spawn',force=True)
    n_iters = int(args.n_iters)
    n_jobs = int(args.n_jobs)
    tasks=args.tasks
    batch_sizes=args.batch_sizes
    save_interval = int(int(args.save_interval) / n_jobs)
    eval_interval = int(int(args.eval_interval) / n_jobs)

    if args.restore_last == True:
        ckpts = glob.glob(os.path.join(model_save_path, '*'))
        iters = [int(os.path.basename(c)) for c in ckpts]
        if len(iters) != 0:
            restore = max(iters)
        else:
            restore = 0
    else:
        restore = int(args.restore)
    tasks=tasks.split(',')
    tasks=[t.strip() for t in tasks]
    batch_sizes=batch_sizes.split(',')
    batch_sizes=[int(b.strip()) for b in batch_sizes]

    if len(tasks)!=len(batch_sizes):
        raise Exception('Number of tasks provided does not match the number of batch sizes provided.')

    n_gpus = int(args.n_gpus)
    n_tasks = len(tasks) * n_jobs

    shared_model = architecture.UnifiedTransformer(gpu_id=0)
    if restore != -1:
        shared_model.restore(model_save_path, restore)
    else:
        restore=0
        
    shared_model=shared_model.to(0)
    shared_model.share_memory()
    counters = [Counter(restore) for i in range(len(tasks))]
    barrier = mp.Barrier(n_tasks)
    start = int(restore / n_jobs)
    # Declare training processes for multi-gpu hogwild training
    processes = []
    for i in range(n_tasks):
        #If more than one GPU is used, use first GPU only for model sharing
        if n_gpus>1:
            gpu_id=i%n_gpus
        else:
            gpu_id=0
        process = mp.Process(target=train, args=(shared_model, tasks[i % len(tasks)], batch_sizes[i % len(tasks)],
                                                 int(n_iters / n_jobs),
                                                 gpu_id, start, restore, counters[i % len(tasks)], barrier,
                                                 (save_interval if i == 0 else None),
                                                 (eval_interval if i < len(tasks) else None),
                                                 (True if i < len(tasks) else False)))
        process.start()
        processes.append(process)
    for p in processes:
        p.join()
