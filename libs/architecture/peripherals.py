from bpemb import BPEmb
from torch.nn.functional import relu, log_softmax
from .base_models.resnet import resnet152
from .util import *

class base_peripheral(nn.Module):
    """
        The base standard non recursive perpheral
        All base peripherals must implement the following functions:
            __init__()
            run_cycle()

    """
    def __init__(self):
        super(base_peripheral,self).__init__()


# Definition of standard peripherals for most common tasks


class ImageInputPeripheral(nn.Module):
    '''
    A module that represents an image input peripheral.
    '''

    def __init__(self, output_dim, dropout=0, weights_preload=True, freeze_layers=True):
        '''
        Initializes an ImageInputPeripheral object.

        Args:
            output_dim (int): The dimensionality of the output.
            dropout (float, optional): The dropout rate. Defaults to 0.
            weights_preload (bool, optional): Whether to preload weights. Defaults to True.
            freeze_layers (bool, optional): Whether to freeze layers. Defaults to True.
        '''
        super().__init__()

        # Set model parameters
        self.feature_dim = 2048
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout)
        self.output_fc = nn.Linear(self.feature_dim, output_dim)

        # Initialize image model
        self.image_model = resnet152(pretrained=weights_preload)

        # Freeze layers if specified
        if freeze_layers:
            self.image_model = self.image_model.eval()
            self.image_model.train = self.empty_fun # type: ignore
            self.image_model.eval = self.empty_fun # type: ignore
            for param in self.image_model.parameters():
                param.requires_grad = False

    def encode(self, image_tensor):
        '''
        Encodes an image tensor.

        Args:
            image_tensor (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The encoded image tensor.
        '''
        # Reshape tensor if necessary
        shape = image_tensor.shape
        t_dim = None
        if len(shape) == 5:
            t_dim = image_tensor.shape[1]
            image_tensor = torch.reshape(image_tensor, (-1, 3, shape[3], shape[4]))
        batch_size = image_tensor.shape[0]

        # Compute image encoding
        image_enc = self.image_model(image_tensor)
        enc_reshape = torch.reshape(image_enc, [batch_size, self.feature_dim, -1])
        enc_transposed = torch.transpose(enc_reshape, 1, 2)
        drp_enc = self.dropout(enc_transposed)
        output_enc = self.output_fc(drp_enc)

        # Reshape tensor if necessary
        if t_dim is not None:
            output_enc = torch.reshape(output_enc, (-1, t_dim, output_enc.shape[1], output_enc.shape[2]))
        else:
            output_enc = output_enc.unsqueeze(1)

        return output_enc

    def empty_fun(self, mode):
        '''
        An empty function used to override the train and eval modes of the image model.

        Args:
            mode (str): The mode to override.
        '''
        pass



class LanguagePeripheral(base_peripheral):
    '''
    A module that represents a language peripheral.
    '''

    def __init__(self, output_dim, vocab_size=10000, embed_dim=50, lang='en', embedding_preload=True, gpu_id=-1, dropout=0):
        '''
        Initializes a LanguagePeripheral object.

        Args:
            output_dim (int): The dimensionality of the output.
            vocab_size (int, optional): The size of the vocabulary. Defaults to 10000.
            embed_dim (int, optional): The dimensionality of the embeddings. Defaults to 50.
            lang (str, optional): The language to use. Defaults to 'en'.
            embedding_preload (bool, optional): Whether to preload embeddings. Defaults to True.
            gpu_id (int, optional): The ID of the GPU to use. Defaults to -1.
            dropout (float, optional): The dropout rate. Defaults to 0.
        '''
        super().__init__()

        # Set model parameters
        self.gpu_id = gpu_id
        self.pad_token = vocab_size
        self.bpe_encoder = BPEmb(lang=lang, vs=vocab_size, dim=embed_dim, add_pad_emb=True)
        self.embed_layer = nn.Embedding(vocab_size+1, embed_dim, padding_idx=self.pad_token)
        if embedding_preload:
            self.embed_layer.load_state_dict({'weight': torch.tensor(self.bpe_encoder.emb.vectors)})
            print("Loading pretrained word embeddings.")
        self.enc_dropout = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, output_dim)

    def forward(self, tokens):
        '''
        Forward pass of the LanguagePeripheral.

        Args:
            tokens (torch.Tensor): The input tokens.
        Returns:
            torch.Tensor: The output tensor.
        '''
        pad_mask = tokens.eq(self.pad_token)
        embeddings = self.embed_layer(tokens)
        embeddings = self.enc_dropout(embeddings)
        output = self.output(embeddings)
        return output.unsqueeze(2)

    def embed_sentences(self, sentences):
        '''
        Embeds a list of sentences.

        Args:
            sentences (list): The input sentences.
        Returns:
            torch.Tensor: The embedded sentences.
            torch.Tensor: The padding mask.
        '''
        tokens, pad_mask = self.tokenize_sentences(sentences)
        return self.forward(tokens), pad_mask

    def decode_tokens(self, tokens):
        '''
        Decodes a list of tokens.

        Args:
            tokens (list): The input tokens.
        Returns:
            str: The decoded tokens.
        '''
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy().astype(int).tolist()
        elif isinstance(tokens, np.ndarray):
            tokens = tokens.astype(int).tolist()

        # Filter out all tokens which have values larger than vocab_size and filter all elements after EOS
        filtered_tokens = []
        for t in tokens:
            values = []
            for i in t:
                if i == self.id_EOS:
                    break
                elif i < self.pad_token:
                    values.append(i)
            filtered_tokens.append(values)

        # Remove all the padding characters in a list
        return self.bpe_encoder.decode_ids(filtered_tokens)

    def tokenize_sentences(self, sentences):
        '''
        Tokenizes a list of sentences.

        Args:
            sentences (list): The input sentences.
        Returns:
            torch.Tensor: The tokenized sentences.
            torch.Tensor: The padding mask.
        '''
        tokens = self.bpe_encoder.encode_ids_with_bos_eos(sentences)

        # Pad the tokens with the pad_token
        max_len = 0
        for t in tokens:
            max_len = max(max_len, len(t)) # type: ignore
        for i in range(len(tokens)):
            tok_len = len(tokens[i]) # type: ignore
            tokens[i].extend([self.pad_token]*(max_len-tok_len)) # type: ignore
        tokens = torch.tensor(np.array(tokens))
        if self.gpu_id > -1:
            tokens = tokens.cuda(self.gpu_id)
        pad_mask = tokens.eq(self.pad_token)
        return tokens, pad_mask

    @property
    def id_PAD(self):
        '''
        Returns the ID of the padding token.
        Returns:
            int: The ID of the padding token.
        '''
        return self.pad_token

    @property
    def id_GO(self):
        '''
        Returns the ID of the GO token.
        Returns:
            int: The ID of the GO token.
        '''
        return 1

    @property
    def id_EOS(self):
        '''
        Returns the ID of the EOS token.
        Returns:
            int: The ID of the EOS token.
        '''
        return 2
