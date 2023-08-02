from .Layers import *
from ..util import *
from torch.nn.functional import log_softmax, softmax


class CNP(nn.Module):
    """
    A class representing a Central Neural Processor (CNP) model.
    """
    def __init__(self, tasks, config=None, domains=['EMPTY'], gpu_id=-1):
        """
            Initializes a Central Neural Processor (CNP) model.

            Args:
                tasks (dict): A dictionary containing the tasks and output classifier dimension.
                config (dict, optional): A dictionary containing the configuration parameters for the CNP model. Defaults to None.
                domains (list, optional): A list of domain names. Defaults to ['EMPTY'].
                gpu_id (int, optional): The ID of the GPU to use. Defaults to -1.
        """
        super(CNP, self).__init__()
        default_config = self.__default_config__()

        # Load the configuration
        if config is not None:
            for k in config.keys():
                if k not in default_config:
                    raise ValueError("The provided configuration does not contain %s" % k)
        else:
            config = default_config

        self.gpu_id = gpu_id
        self.input_dim = config['input_dim']
        self.control_dim = config['control_dim']
        self.output_dim = config['output_dim']
        self.spatial_dim = config['spatial_dim']
        self.temporal_dim = config['temporal_dim']
        self.temporal_n_layers = config['temporal_n_layers']
        self.temporal_n_heads = config['temporal_n_heads']
        self.temporal_d_k = config['temporal_d_k']
        self.temporal_d_v = config['temporal_d_v']
        self.temporal_hidden_dim = config['temporal_hidden_dim']
        self.decoder_dim = config['decoder_dim']
        self.decoder_n_layers = config['decoder_n_layers']
        self.decoder_n_heads = config['decoder_n_heads']
        self.decoder_d_k = config['decoder_d_k']
        self.decoder_d_v = config['decoder_d_v']
        self.decoder_hidden_dim = config['decoder_hidden_dim']
        self.max_seq_len = config['max_seq_len']
        self.output_embedding_dim = config['output_embedding_dim']
        self.dropout = config['dropout']
        self.batch_size = -1  # Uninitialized CNP memory


        # Prepare the task lists and various output classifiers and embeddings
        if isinstance(tasks, dict):
            self.task_clflen = list(tasks.values())
            self.task_dict = {t: i for i, t in enumerate(tasks.keys())}
        else:
            raise ValueError('Tasks must be of type dict containing the tasks and output classifier dimension')

        self.output_clfs = nn.ModuleList([nn.Linear(self.output_dim, t) for t in self.task_clflen])
        #Use one extra to define padding
        self.output_embs = nn.ModuleList([nn.Embedding(t+1,self.output_embedding_dim,padding_idx=t) for t in self.task_clflen])

        #Initialize the various sublayers of the CNP
        control_states=domains+list(tasks.keys())
        self.control_peripheral=ControlPeripheral(self.control_dim,control_states,gpu_id=gpu_id)
        self.temporal_encoder = TemporalCacheEncoder(self.max_seq_len,self.temporal_n_layers,
                                                     self.temporal_n_heads,self.temporal_d_k,self.temporal_d_v,
                                                    self.temporal_dim,self.temporal_hidden_dim,dropout=self.dropout,
                                                     gpu_id=self.gpu_id)
        self.decoder=Decoder(self.max_seq_len,self.decoder_n_layers,self.decoder_n_heads,self.decoder_d_k,
                             self.decoder_d_v,self.decoder_dim,self.decoder_hidden_dim,self.temporal_dim,
                             self.spatial_dim,self.output_dim, dropout=self.dropout,gpu_id=self.gpu_id)

        #Initialize the various CNP caches as empty
        self.spatial_cache=None
        self.temporal_cache=None
        self.decoder_cache=None
        self.temporal_spatial_link=[]
        self.pad_cache=None    #Used to store the padding values so that it can be used later in enc dec attn


        #Various projection layers
        self.spatial_pool=nn.AdaptiveAvgPool1d(1)
        self.inpcont_input_proj=nn.Linear(self.input_dim+self.control_dim,self.input_dim)
        self.input_spatial_proj=nn.Linear(self.input_dim,self.spatial_dim)
        self.input_temporal_proj=nn.Linear(self.input_dim,self.temporal_dim)
        self.emb_decoder_proj=nn.Linear(self.output_embedding_dim,self.decoder_dim)
        self.cont_decoder_proj=nn.Linear(self.control_dim,self.decoder_dim)
        
        #freeze layers
        


    def decode(self, task, targets=None, num_steps=100, recurrent_steps=1, pad_mask=None, beam_width=1):
        """
        Decodes the given task.

        Args:
        - task (str): The task to decode.
        - targets (torch.Tensor): The target tensor.
        - num_steps (int): The number of steps to take.
        - recurrent_steps (int): The number of recurrent steps to take.
        - pad_mask (torch.Tensor): The pad mask.
        - beam_width (int): The beam width.

        Returns:
        - log_softmax(predictions,dim=2) (torch.Tensor): The log softmax predictions.
        """
        if targets is not None:
            # Use teacher forcing to generate predictions. The graph is kept in memory during this operation.
            if (len(targets.shape) != 2 or targets.shape[0] != self.batch_size):
                raise ValueError(
                    "Target tensor must be of shape (batch_size,length of sequence).")
            if task not in self.task_dict.keys():
                raise ValueError('Invalid task %s'%task)

            # Get the task-specific output embeddings
            output_embeddings = self.output_embs[self.task_dict[task]](targets)

            # Project the output embeddings
            projected_embeddings = self.emb_decoder_proj(output_embeddings)

            # Get the control vector for the given task
            control = self.control_peripheral(task, (self.batch_size))
            control = control.unsqueeze(1)
            control = self.cont_decoder_proj(control)

            # Concatenate the control vector and the output embeddings
            concatenated_embeddings = torch.cat([control, projected_embeddings], 1)

            # Increase the length of the pad_mask to match the size after adding the control vector
            if pad_mask is not None:
                pad_extra = torch.zeros((self.batch_size, 1), device=torch.device(self.gpu_id), dtype=pad_mask.dtype)
                pad_mask = torch.cat([pad_extra, pad_mask], 1)

            # Get output from decoder
            logits, = self.decoder(concatenated_embeddings, self.spatial_cache, self.temporal_cache, self.temporal_spatial_link,
                                 self.pad_cache,
                                 recurrent_steps=recurrent_steps, pad_mask=pad_mask)

            # Predict using the task-specific classifier
            predictions = self.output_clfs[self.task_dict[task]](logits)
            predictions = predictions[:, 0:targets.shape[1], :]
            return log_softmax(predictions, dim=2)
        else:
            # Get the control vector for the given task
            control = self.control_peripheral(task, (self.batch_size))
            control = control.unsqueeze(1)
            control = self.cont_decoder_proj(control)

            # Set the decoder inputs to the control vector
            decoder_inputs = control

            # Generate predictions
            for i in range(num_steps-1):
                logits, = self.decoder(decoder_inputs, self.spatial_cache, self.temporal_cache, self.temporal_spatial_link,
                                       self.pad_cache, recurrent_steps=recurrent_steps)
                prediction = self.output_clfs[self.task_dict[task]](logits)
                prediction = prediction[:, -1, :].unsqueeze(1)
                prediction = log_softmax(prediction, dim=2).argmax(-1)
                prediction = self.output_embs[self.task_dict[task]](prediction)
                prediction = self.emb_decoder_proj(prediction).detach()
                if beam_width > 1:
                    p = torch.topk(softmax(prediction), beam_width)

                decoder_inputs = torch.cat([decoder_inputs, prediction], 1)

            logits, = self.decoder(decoder_inputs, self.spatial_cache, self.temporal_cache, self.temporal_spatial_link, 
                                   self.pad_cache, recurrent_steps=recurrent_steps)
            predictions = self.output_clfs[self.task_dict[task]](logits)
            return log_softmax(predictions, dim=2)
        

    def encode(self, input_data, pad_mask=None, domain='EMPTY', recurrent_steps=1):
        """
        Encodes a sequence of temporal and spatial data into a fixed-length vector representation.

        Args:
            input_data (torch.Tensor): The input data tensor.
            pad_mask (torch.Tensor, optional): The padding mask tensor. Defaults to None.
            domain (str, optional): The domain of the input data. Defaults to 'EMPTY'.
            recurrent_steps (int, optional): The number of recurrent steps to perform. Defaults to 1.

        Raises:
            Exception: If the input dimensions are invalid or the batch size does not match.

        Returns:
            Tuple[torch.Tensor, List[Tuple[int, int]]]: The encoded sequence tensor and a list of tuples representing the temporal-spatial link.
        """
        if len(input_data.shape) != 4:
            raise Exception('Invalid input dimensions.')
        batch_size, seq_len, spatial_dim, feature_dim = list(input_data.size())
        self.temporal_spatial_link.append((seq_len, spatial_dim))
        if batch_size != self.batch_size:
            raise Exception('Input batch size does not match.')

        # Spatial encode. Spatial encodes encodes both spatial and time dimension features together
        control_vecs = self.control_peripheral(domain, (batch_size, seq_len, spatial_dim))
        input_data = torch.cat([input_data, control_vecs], 3)
        input_data = self.inpcont_input_proj(input_data)

        # Project the spatial data, into the query dimension and add it to the existing cache
        if spatial_dim > 1:
            spatial_features = torch.reshape(input_data, [batch_size, seq_len * spatial_dim, feature_dim])
            spatial_features = self.input_spatial_proj(spatial_features)
            if self.spatial_cache is None:
                self.spatial_cache = spatial_features
            else:
                self.spatial_cache = torch.cat([self.spatial_cache, spatial_features], 1)

        # Feed the time features. First AVGPool the spatial features.
        temp_data = input_data.transpose(2, 3).reshape(batch_size * seq_len, feature_dim, spatial_dim)
        temp_data = self.spatial_pool(temp_data).reshape(batch_size, seq_len, feature_dim)
        temp_data = self.input_temporal_proj(temp_data)

        # Create a control state and concat with the temporal data
        # Add data to temporal cache
        temp_data, = self.temporal_encoder(temp_data, pad_mask=pad_mask, recurrent_steps=recurrent_steps)

        if self.temporal_cache is None:
            self.temporal_cache = temp_data
        else:
            self.temporal_cache = torch.cat([self.temporal_cache, temp_data], 1)

        # Add pad data to pad cache
        if pad_mask is None:
            pad_mask = torch.zeros((batch_size, seq_len), device=torch.device(self.gpu_id), dtype=torch.uint8)
        if self.pad_cache is None:
            self.pad_cache = pad_mask
        else:
            self.pad_cache = torch.cat([self.pad_cache, pad_mask], 1)
            
            
    def clear_spatial_cache(self):
        self.spatial_cache=None

    def clear_temporal_cache(self):
        self.temporal_raw_cache=None
        self.temporal_cache=None

    def reset(self,batch_size=1):
        self.attn_scores=[]
        self.batch_size=batch_size
        self.temporal_spatial_link=[]
        self.pad_cache=None
        self.clear_spatial_cache()
        self.clear_temporal_cache()
    
    @staticmethod
    def __default_config__():
        conf={
            'input_dim':128,
            'control_dim':32,
            'output_dim':128,
            'spatial_dim':128,
            'temporal_dim':512,
            'temporal_n_layers':6,
            'temporal_n_heads':8,
            'temporal_d_k':64,
            'temporal_d_v':64,
            'temporal_hidden_dim':2048,
            'decoder_dim':512,
            'decoder_n_layers':6,
            'decoder_n_heads':8,
            'decoder_d_k':64,
            'decoder_d_v':64,
            'decoder_hidden_dim':2048,
            'max_seq_len':1000,
            'output_embedding_dim':300,
            'dropout':0.1
        }
        return conf


class ControlPeripheral(nn.Module):
    """
    A special peripheral used to help the Central Neural Processor (CNP) identify the data domain or specify the context of the current operation.
    """
    def __init__(self, control_dim, control_states, gpu_id=-1):
        """
        Initializes a ControlPeripheral object.

        Args:
            control_dim (int): The dimension of the control embeddings.
            control_states (list): A list of control states as strings.
            gpu_id (int, optional): The ID of the GPU to use. Defaults to -1.
        """
        super(ControlPeripheral, self).__init__()

        # Set model parameters
        self.control_dim = control_dim
        self.gpu_id = gpu_id
        self.control_dict = {}
        for i, control_state in enumerate(control_states):
            self.control_dict[control_state] = i
        self.control_embeddings = nn.Embedding(len(control_states) + 1, self.control_dim)

    def forward(self, control_state, shape=()):
        """
        Performs a forward pass through the ControlPeripheral model.

        Args:
            control_state (str): The control state to embed.
            shape (tuple, optional): The shape of the output tensor. Defaults to ().

        Returns:
            torch.Tensor: The embedded control state tensor.
        """
        if self.gpu_id >= 0:
            control_ids = torch.ones(shape, dtype=torch.long, device=torch.device(self.gpu_id)) * self.control_dict[control_state]
        else:
            control_ids = torch.ones(shape, dtype=torch.long) * self.control_dict[control_state]
        return self.control_embeddings(control_ids)


class TemporalCacheEncoder(nn.Module):
    """
    A module that encodes a sequence of temporal data into a fixed-length vector representation.
    """
    def __init__(
            self,
            max_seq_len,
            num_layers, num_heads, key_dim, value_dim,
            model_dim, inner_dim, dropout=0.1, gpu_id=-1):
        """
        Initializes a TemporalCacheEncoder object.

        Args:
            max_seq_len (int): The maximum length of the input sequence.
            num_layers (int): The number of layers in the encoder.
            num_heads (int): The number of attention heads in each layer.
            key_dim (int): The dimension of the attention key vectors.
            value_dim (int): The dimension of the attention value vectors.
            model_dim (int): The dimension of the model.
            inner_dim (int): The dimension of the inner feedforward layer.
            dropout (float, optional): The dropout rate. Defaults to 0.1.
            gpu_id (int, optional): The ID of the GPU to use. Defaults to -1.
        """
        super().__init__()

        # Set model parameters
        position_dim = model_dim
        num_positions = max_seq_len + 1
        self.dropout_emb = nn.Dropout(dropout)
        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(num_positions, position_dim, padding_idx=0),
            freeze=True)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(model_dim, inner_dim, num_heads, key_dim, value_dim, dropout=dropout)
            for _ in range(num_layers)])
        self.gpu_id = gpu_id

    def forward(self, input_seq, return_attns=False, recurrent_steps=1, pad_mask=None):
        """
        Performs a forward pass through the TemporalCacheEncoder model.

        Args:
            input_seq (torch.Tensor): The input sequence tensor.
            return_attns (bool, optional): Whether to return the attention weights. Defaults to False.
            recurrent_steps (int, optional): The number of recurrent steps to perform. Defaults to 1.
            pad_mask (torch.Tensor, optional): The padding mask tensor. Defaults to None.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, List[torch.Tensor]]: The encoded sequence tensor or a tuple of the encoded sequence tensor and a list of attention weights tensors.
        """
        attn_weights_list = []
        batch_size, seq_len, _ = input_seq.shape

        # Generate positional encodings
        if self.gpu_id >= 0:
            pos_ids = torch.arange(1, seq_len + 1, device=self.gpu_id).repeat(batch_size, 1)
        else:
            pos_ids = torch.arange(1, seq_len + 1).repeat(batch_size, 1)
        pos_encodings = self.position_enc(pos_ids)

        # Apply positional encodings and dropout
        enc_output = input_seq + pos_encodings
        enc_output = self.dropout_emb(enc_output)

        # Apply padding mask
        if pad_mask is not None:
            slf_attn_mask = get_attn_key_pad_mask(pad_mask, input_seq)
        else:
            slf_attn_mask = None
        non_pad_mask = get_non_pad_mask(input_seq, pad_mask)

        # Apply recurrent steps and encoder layers
        for i in range(recurrent_steps):
            for enc_layer in self.layer_stack:
                enc_output, attn_weights = enc_layer(
                    enc_output, non_pad_mask, slf_attn_mask=slf_attn_mask)

                if return_attns:
                    attn_weights_list += [attn_weights]

        # Return encoded sequence tensor and attention weights list (if requested)
        if return_attns:
            return enc_output, attn_weights_list
        return enc_output,


class Decoder(nn.Module):
    """
    A module that decodes a sequence of data into a fixed-length vector representation.
    """
    def __init__(
            self,
            max_seq_len,
            num_layers, num_heads, key_dim, value_dim,
            model_dim, inner_dim, temporal_dim, spatial_dim, output_dim,
            dropout=0.1, gpu_id=-1):
        """
        Initializes a Decoder object.

        Args:
            max_seq_len (int): The maximum length of the input sequence.
            num_layers (int): The number of layers in the decoder.
            num_heads (int): The number of attention heads in each layer.
            key_dim (int): The dimension of the attention key vectors.
            value_dim (int): The dimension of the attention value vectors.
            model_dim (int): The dimension of the model.
            inner_dim (int): The dimension of the inner feedforward layer.
            temporal_dim (int): The dimension of the temporal data.
            spatial_dim (int): The dimension of the spatial data.
            output_dim (int): The dimension of the output.
            dropout (float, optional): The dropout rate. Defaults to 0.1.
            gpu_id (int, optional): The ID of the GPU to use. Defaults to -1.
        """
        super().__init__()

        # Set model parameters
        position_dim = model_dim
        num_positions = max_seq_len + 1
        self.dropout_emb = nn.Dropout(dropout)
        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(num_positions, position_dim, padding_idx=0),
            freeze=True)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(model_dim, inner_dim, num_heads, key_dim, value_dim, temporal_dim,
                         spatial_dim, dropout=dropout, gpu_id=gpu_id)
            for _ in range(num_layers)])
        self.output_fc = nn.Linear(model_dim, output_dim)
        self.gpu_id = gpu_id

    def forward(self, input_seq, spatial_cache, temporal_cache, temporal_spatial_link,
                pad_cache, pad_mask=None, return_attns=False, recurrent_steps=1):
        """
        Performs a forward pass through the Decoder model.

        Args:
            input_seq (torch.Tensor): The input sequence tensor.
            spatial_cache (torch.Tensor): The spatial cache tensor.
            temporal_cache (torch.Tensor): The temporal cache tensor.
            temporal_spatial_link (torch.Tensor): The temporal-spatial link tensor.
            pad_cache (torch.Tensor): The padding cache tensor.
            pad_mask (torch.Tensor, optional): The padding mask tensor. Defaults to None.
            return_attns (bool, optional): Whether to return the attention weights. Defaults to False.
            recurrent_steps (int, optional): The number of recurrent steps to perform. Defaults to 1.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, List[torch.Tensor]]: The decoded sequence tensor or a tuple of the decoded sequence tensor and a list of attention weights tensors.
        """
        attn_weights_list = []
        batch_size, seq_len, _ = input_seq.shape

        # Generate positional encodings
        if self.gpu_id >= 0:
            pos_ids = torch.arange(1, seq_len + 1, device=self.gpu_id).repeat(batch_size, 1)
        else:
            pos_ids = torch.arange(1, seq_len + 1).repeat(batch_size, 1)
        pos_encodings = self.position_enc(pos_ids)

        # Apply positional encodings and dropout
        dec_output = input_seq + pos_encodings
        dec_output = self.dropout_emb(dec_output)

        # Apply padding mask
        slf_attn_mask_subseq = get_subsequent_mask((batch_size, seq_len), self.gpu_id)
        if pad_mask is not None:
            slf_attn_mask_keypad = get_attn_key_pad_mask(pad_mask, input_seq)
            slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        else:
            slf_attn_mask = slf_attn_mask_subseq

        # Apply recurrent steps and decoder layers
        dec_enc_attn_mask = get_attn_key_pad_mask(pad_cache, input_seq)
        non_pad_mask = get_non_pad_mask(input_seq, pad_mask)
        for i in range(recurrent_steps):
            for dec_layer in self.layer_stack:
                dec_output, attn_weights = dec_layer(
                    dec_output, temporal_cache, spatial_cache, temporal_spatial_link,
                    non_pad_mask, slf_attn_mask=slf_attn_mask, dec_enc_attn_mask=dec_enc_attn_mask)

                if return_attns:
                    attn_weights_list += [attn_weights]

        # Apply output layer
        dec_output = self.output_fc(dec_output)

        # Return decoded sequence tensor and attention weights list (if requested)
        if return_attns:
            return dec_output, attn_weights_list
        return dec_output,


