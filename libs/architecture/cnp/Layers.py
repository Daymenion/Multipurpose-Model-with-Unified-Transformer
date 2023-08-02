import torch.nn as nn
import torch
from libs.architecture.cnp.SubLayers import MultiHeadAttention, PositionwiseFeedForward

class EncoderLayer(nn.Module):
    """
    A module that represents a single layer of the Encoder.
    """
    def __init__(self, model_dim, inner_dim, num_heads, key_dim, value_dim, dropout=0.1):
        """
        Initializes an EncoderLayer object.

        Args:
            model_dim (int): The dimension of the model.
            inner_dim (int): The dimension of the inner feedforward layer.
            num_heads (int): The number of attention heads.
            key_dim (int): The dimension of the attention key vectors.
            value_dim (int): The dimension of the attention value vectors.
            dropout (float, optional): The dropout rate. Defaults to 0.1.
        """
        super().__init__()

        # Set model parameters
        self.slf_attn = MultiHeadAttention(num_heads, model_dim, key_dim, value_dim, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(model_dim, inner_dim, dropout=dropout)

    def forward(self, input_seq, non_pad_mask, slf_attn_mask=None):
        """
        Performs a forward pass through the EncoderLayer model.

        Args:
            input_seq (torch.Tensor): The input sequence tensor.
            non_pad_mask (torch.Tensor): The non-padding mask tensor.
            slf_attn_mask (torch.Tensor, optional): The self-attention mask tensor. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The encoded sequence tensor and the self-attention weights tensor.
        """
        # Apply self-attention
        enc_output, enc_slf_attn = self.slf_attn(input_seq, input_seq, input_seq, mask=slf_attn_mask)

        # Apply non-padding mask
        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        # Apply position-wise feedforward layer
        enc_output = self.pos_ffn(enc_output)

        # Apply non-padding mask
        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        return enc_output, enc_slf_attn



class DecoderLayer(nn.Module):
    """
    A module that represents a single layer of the Decoder.
    """
    def __init__(self, model_dim, inner_dim, num_heads, key_dim, value_dim, temporal_dim, spatial_dim, dropout=0.1, gpu_id=-1):
        """
        Initializes a DecoderLayer object.

        Args:
            model_dim (int): The dimension of the model.
            inner_dim (int): The dimension of the inner feedforward layer.
            num_heads (int): The number of attention heads.
            key_dim (int): The dimension of the attention key vectors.
            value_dim (int): The dimension of the attention value vectors.
            temporal_dim (int): The dimension of the temporal cache.
            spatial_dim (int): The dimension of the spatial cache.
            dropout (float, optional): The dropout rate. Defaults to 0.1.
            gpu_id (int, optional): The ID of the GPU to use. Defaults to -1.
        """
        super().__init__()

        # Set model parameters
        self.gpu_id = gpu_id
        self.slf_attn = MultiHeadAttention(num_heads, model_dim, key_dim, value_dim, dropout=dropout)
        self.temporal_cache_attn = MultiHeadAttention(num_heads, temporal_dim, key_dim, value_dim, dropout=dropout)
        self.temporal_proj = nn.Linear(model_dim, temporal_dim)
        self.spatial_proj = nn.Linear(temporal_dim, spatial_dim)
        self.spatial_cache_attn = MultiHeadAttention(num_heads, spatial_dim, key_dim, value_dim, dropout=dropout)
        self.spat_dec_proj = nn.Linear(spatial_dim, model_dim)
        self.pos_ffn = PositionwiseFeedForward(model_dim, inner_dim, dropout=dropout)

    def forward(self, dec_input, temporal_cache, spatial_cache, temporal_spatial_link, non_pad_mask, slf_attn_mask=None, dec_enc_attn_mask=None):
        """
        Performs a forward pass through the DecoderLayer model.

        Args:
            dec_input (torch.Tensor): The decoder input tensor.
            temporal_cache (torch.Tensor): The temporal cache tensor.
            spatial_cache (torch.Tensor): The spatial cache tensor.
            temporal_spatial_link (List[Tuple[int, int]]): The list of tuples representing the temporal-spatial link.
            non_pad_mask (torch.Tensor): The non-padding mask tensor.
            slf_attn_mask (torch.Tensor, optional): The self-attention mask tensor. Defaults to None.
            dec_enc_attn_mask (torch.Tensor, optional): The decoder-encoder attention mask tensor. Defaults to None.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: The decoder output tensor and the list of attention weights tensors.
        """
        # Apply self-attention
        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input, mask=slf_attn_mask)

        # Apply non-padding mask
        if non_pad_mask is not None:
            dec_output *= non_pad_mask

        # Apply temporal cache attention
        dec_temp = self.temporal_proj(dec_output)
        dec_temp, dec_temp_attn = self.temporal_cache_attn(dec_temp, temporal_cache, temporal_cache, mask=dec_enc_attn_mask)

        # Apply non-padding mask
        if non_pad_mask is not None:
            dec_temp *= non_pad_mask

        # Apply spatial cache attention
        dec_spat = self.spatial_proj(dec_temp)
        dec_spat_attn = None
        if spatial_cache is not None:
            # Process the spatial cache and add the respective weightings
            spatial_gate = []
            idx_start = 0
            for l in temporal_spatial_link:
                t, s = l
                if s > 1:
                    temp_sel = dec_temp_attn[:, :, :, idx_start:idx_start + t]
                    b, nh, dq, t = temp_sel.shape
                    temp_sel = temp_sel.unsqueeze(4).expand(b, nh, dq, t, s).transpose(3, 4)
                    temp_sel = temp_sel.reshape(b, nh, dq, t * s)
                    spatial_gate.append(temp_sel)
                idx_start = idx_start + t
            spatial_gate = torch.cat(spatial_gate, dim=3)
            dec_spat, dec_spat_attn = self.spatial_cache_attn(dec_spat, spatial_cache, spatial_cache, k_gate=spatial_gate)

        # Apply non-padding mask
        if non_pad_mask is not None:
            dec_spat *= non_pad_mask
        dec_output=self.spat_dec_proj(dec_spat)
        dec_output = self.pos_ffn(dec_output)
        if non_pad_mask is not None: dec_output*=non_pad_mask
        return dec_output,[dec_slf_attn,dec_spat_attn,dec_temp_attn]
       
