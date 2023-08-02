import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class ScaledDotProductAttention(nn.Module):
    '''
    A module that represents the Scaled Dot-Product Attention mechanism.
    '''

    def __init__(self, temperature, attn_dropout=0.1):
        '''
        Initializes a ScaledDotProductAttention object.

        Args:
            temperature (float): The temperature parameter.
            attn_dropout (float, optional): The dropout rate. Defaults to 0.1.
        '''
        super().__init__()

        # Set model parameters
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, query, key, value, mask=None, k_gate=None):
        '''
        Performs a forward pass through the Scaled Dot-Product Attention model.

        Args:
            query (torch.Tensor): The query tensor.
            key (torch.Tensor): The key tensor.
            value (torch.Tensor): The value tensor.
            mask (torch.Tensor, optional): The mask tensor. Defaults to None.
            k_gate (torch.Tensor, optional): The k_gate tensor. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The output tensor and the attention tensor.
        '''
        # Compute attention scores
        attn_scores = torch.bmm(query, key.transpose(1, 2))
        attn_scores = attn_scores / self.temperature

        # Apply k_gate if provided
        if k_gate is not None:
            attn_scores = torch.mul(attn_scores, k_gate)

        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, -np.inf)

        # Apply softmax
        attn_scores = self.softmax(attn_scores)

        # Apply dropout
        attn_scores = self.dropout(attn_scores)

        # Compute output
        output = torch.bmm(attn_scores, value)

        return output, attn_scores


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, num_heads, model_dim, key_dim, value_dim, dropout=0.1):
        '''
        Initializes a MultiHeadAttention object.

        Args:
            n_head (int): The number of attention heads.
            model_dim (int): The dimensionality of the model.
            key_dim (int): The dimensionality of the keys.
            value_dim (int): The dimensionality of the values.
            dropout (float, optional): The dropout rate. Defaults to 0.1.
        '''
        super().__init__()
        
        # Set model parameters
        self.num_head = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim

        # Initialize linear layers
        self.w_qs = nn.Linear(model_dim, num_heads * key_dim)
        self.w_ks = nn.Linear(model_dim, num_heads * key_dim)
        self.w_vs = nn.Linear(model_dim, num_heads * value_dim)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (model_dim + key_dim)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (model_dim + key_dim)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (model_dim + value_dim)))

        # Initialize attention mechanism
        self.attention = ScaledDotProductAttention(temperature=np.power(key_dim, 0.5))
        # Initialize normalization layer
        self.layer_norm = nn.LayerNorm(model_dim)

        # Initialize output linear layer
        self.fc = nn.Linear(num_heads * value_dim, model_dim)
        nn.init.xavier_normal_(self.fc.weight)
        # Initialize dropout layer
        self.dropout = nn.Dropout(dropout)

 
    def forward(self, query, key, value, mask=None,k_gate=None):
        '''
        Computes the forward pass of the MultiHeadAttention module.

        Args:
            query (torch.Tensor): The query tensor of shape (batch_size, query_len, model_dim).
            key (torch.Tensor): The key tensor of shape (batch_size, key_len, model_dim).
            value (torch.Tensor): The value tensor of shape (batch_size, value_len, model_dim).
            mask (torch.Tensor, optional): The mask tensor of shape (batch_size, query_len, key_len). Defaults to None.
            k_gate (torch.Tensor, optional): The k_gate tensor of shape (key_len, batch_size). Defaults to None.

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, query_len, model_dim).
            torch.Tensor: The attention tensor of shape (batch_size, num_heads, query_len, key_len).
        '''
        dim_key, dim_value, n_head = self.key_dim, self.value_dim, self.num_head
        # Get tensor sizes
        batch_size, query_len, _ = query.size()
        batch_size, key_len, _ = key.size()
        batch_size, value_len, _ = value.size()

        # Apply k_gate if provided
        if k_gate is not None:
            k_gate = k_gate.transpose(0, 1)
            k_gate=k_gate.reshape(n_head*batch_size,query_len,value_len)
        # Save residual connection
        residual = query

        # Apply linear transformations
        query_lt = self.w_qs(query).view(batch_size, query_len, n_head, dim_key)
        key_lt = self.w_ks(key).view(batch_size, key_len, n_head, dim_key)
        value_lt = self.w_vs(value).view(batch_size, value_len, n_head, dim_value)
        
        #A Weighting score for the keys is provided
        # Transpose dimensions for attention computation
        transpose_query = query_lt.permute(2, 0, 1, 3).contiguous().view(-1, query_len, dim_key) # (n*b) x lq x dk
        transpose_key = key_lt.permute(2, 0, 1, 3).contiguous().view(-1, key_len, dim_key) # (n*b) x lk x dk
        transpose_value = value_lt.permute(2, 0, 1, 3).contiguous().view(-1, value_len, dim_value) # (n*b) x lv x dv
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        
        # Compute attention and output
        output, attention = self.attention(transpose_query, transpose_key, transpose_value, mask=mask,k_gate=k_gate)

        # Reshape output tensor
        output = output.view(n_head, batch_size, query_len, dim_value)
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, query_len, -1) # b x lq x (n*dv)

        # Apply output linear layer and residual connection
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        # Reshape attention tensor
        attention=attention.view(n_head,batch_size,query_len,value_len).transpose(0,1)
        return output, attention



class PositionwiseFeedForward(nn.Module):
    '''
    A module that represents the Positionwise Feed Forward layer.
    '''

    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        '''
        Initializes a PositionwiseFeedForward object.

        Args:
            input_dim (int): The dimensionality of the input.
            hidden_dim (int): The dimensionality of the hidden layer.
            dropout (float, optional): The dropout rate. Defaults to 0.1.
        '''
        super().__init__()
        # Initialize linear layers
        self.w_1 = nn.Linear(input_dim, hidden_dim) # position-wise
        self.w_2 = nn.Linear(hidden_dim, input_dim) # position-wise
        # Initialize normalization layer
        self.layer_norm = nn.LayerNorm(input_dim)
        # Initialize dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        Performs a forward pass through the Positionwise Feed Forward layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        '''
        # Save residual connection
        residual = x
        output = x
        # Apply linear transformations and activation function
        output = self.w_2(F.relu(self.w_1(output)))
        # Apply dropout and residual connection
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

