import torch.nn as nn
import torch as T
import torch.nn.functional as F
from torch.autograd import Variable as var
import numpy as np
import torch
from torch.autograd import Variable


def get_subsequent_mask(shape,gpu_id):
  ''' 
  Returns a mask tensor for masking out the subsequent info.
  
  Args:
  - shape: tuple of int, shape of the mask tensor to be returned.
  - gpu_id: int, the id of the gpu to be used. If -1, cpu is used instead.
  
  Returns:
  - subsequent_mask: tensor of shape (sz_b, len_s, len_s), where sz_b is the batch size and len_s is the length of the sequence.
  '''
  sz_b, len_s = shape
  if gpu_id>=0:
    subsequent_mask = torch.triu(
      torch.ones((len_s, len_s), device=gpu_id, dtype=torch.uint8), diagonal=1)
  else:
    subsequent_mask = torch.triu(
      torch.ones((len_s, len_s), dtype=torch.uint8), diagonal=1)
  subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

  return subsequent_mask



def get_attn_key_pad_mask(pad_mask, seq_q):
  ''' 
  Returns a mask tensor for masking out the padding part of key sequence.
  
  Args:
  - pad_mask: tensor of shape (batch_size, seq_len_k), where seq_len_k is the length of the key sequence.
  - seq_q: tensor of shape (batch_size, seq_len_q, d_model), where seq_len_q is the length of the query sequence and d_model is the dimension of the model.
  
  Returns:
  - pad_mask: tensor of shape (batch_size, seq_len_q, seq_len_k), where seq_len_q is the length of the query sequence and seq_len_k is the length of the key sequence.
  '''
  len_q = seq_q.size(1)
  pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
  return pad_mask



def get_non_pad_mask(seq, pad_mask):
  '''
  Returns a mask tensor for masking out the padding part of a sequence.

  Args:
  - seq: tensor of shape (batch_size, seq_len), where seq_len is the length of the sequence.
  - pad_mask: tensor of shape (batch_size, seq_len), where seq_len is the length of the sequence.

  Returns:
  - non_pad_mask: tensor of shape (batch_size, seq_len, 1), where seq_len is the length of the sequence.
  '''
  if pad_mask is None:
    return None
  else:
    return pad_mask.ne(1).type(torch.float).unsqueeze(-1)



def get_sinusoid_encoding_table(num_positions, hidden_dim, padding_idx=None):
  ''' 
  Generates a sinusoidal encoding table for use in positional encoding.

  Args:
  - num_positions: int, the maximum number of positions to be encoded.
  - hidden_dim: int, the number of dimensions in the hidden layer.
  - padding_idx: int, the index of the padding token. If None, no padding is used.

  Returns:
  - sinusoid_table: tensor of shape (n_position, d_hid), the sinusoidal encoding table.
  '''
  def calculate_angle(pos, hid_idx):
    return pos / np.power(10000, 2 * (hid_idx // 2) / hidden_dim)

  def get_position_angle_vector(pos):
    return [calculate_angle(pos, hid_j) for hid_j in range(hidden_dim)]

  sinusoid_table = np.array([get_position_angle_vector(pos_i) for pos_i in range(num_positions)])

  sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
  sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

  if padding_idx is not None:
    # zero vector for padding dimension
    sinusoid_table[padding_idx] = 0.

  return torch.FloatTensor(sinusoid_table)



def cuda(x, grad=False, gpu_id=-1):
  if gpu_id == -1:
    return var(x, requires_grad=grad)
  else:
    return var(x.pin_memory(), requires_grad=grad).cuda(gpu_id, non_blocking=True)



class ScheduledOptim():
  '''A simple wrapper class for learning rate scheduling'''

  def __init__(self, optimizer, hidden_size, n_warmup_steps, n_current_steps=0, init_lr=0.1, max_lr=None):
    '''
    Initializes the ScheduledOptim class.

    Args:
    - optimizer: optimizer object, the optimizer to be used for training.
    - hidden_size: int, the number of dimensions in the hidden layer.
    - n_warmup_steps: int, the number of warmup steps.
    - n_current_steps: int, the current number of steps.
    - init_lr: float, the initial learning rate.
    - max_lr: float, the maximum learning rate. If None, no maximum is used.
    '''
    self.optimizer = optimizer
    self.n_warmup_steps = n_warmup_steps
    self.n_current_steps = n_current_steps
    self.init_lr = init_lr
    self.hidden_size = hidden_size
    self.max_lr = max_lr

  def step(self):
    "Step with the inner optimizer"
    self._update_learning_rate()
    self.optimizer.step()

  def zero_grad(self):
    "Zero out the gradients by the inner optimizer"
    self.optimizer.zero_grad()

  def _get_lr_scale(self):
    '''
    Calculates the learning rate scale.

    Returns:
    - lr_scale: float, the learning rate scale.
    '''
    learning_rate_scale = np.min([1.0, self.n_current_steps / self.n_warmup_steps])
    rsqrt_decay = np.power((np.max([self.n_current_steps, self.n_warmup_steps])), -0.5)
    return learning_rate_scale * rsqrt_decay

  def _update_learning_rate(self):
    ''' 
    Updates the learning rate based on the current number of steps.
    '''
    self.n_current_steps += 1
    if self.max_lr is not None:
      learning_rate = min(self.init_lr * self._get_lr_scale(), self.max_lr)
    else:
      learning_rate = self.init_lr * self._get_lr_scale()
    for param_group in self.optimizer.param_groups:
      param_group['lr'] = learning_rate
