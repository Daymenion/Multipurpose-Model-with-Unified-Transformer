import torch
import os
import torch.nn as nn

from .peripherals import *
from .util import *
from .cnp import CNP


class UnifiedTransformer(nn.Module):
    '''
    A module that represents a unified transformer.
    '''

    def __init__(self, config=None, gpu_id=-1, dropout=None):
        '''
        Initializes a UnifiedTransformer object.
        Args:
            config (tuple, optional): The configuration. Defaults to None.
            gpu_id (int, optional): The GPU ID. Defaults to -1.
            dropout (float, optional): The dropout rate. Defaults to None.
        '''
        super(UnifiedTransformer, self).__init__()

        # Set configuration parameters
        if config is None:
            config_c, config_p, domains = self.__defaultconf__()
        else:
            config_c, config_p, domains = config
        if dropout is not None:
            config_c['dropout'] = dropout
            config_p['dropout'] = dropout
        self.gpu_id = gpu_id

        # Set tasks
        tasks = {'PENN': config_p['penn_output_classes'], 'HMDB': config_p['hmdb_output_classes'],
                 'IMAGE_CAPTION': config_p['english_language_output_vocab'], 'VQA': config_p['vqa_output_vocab']}

        # Initialize CNP
        self.cnp = CNP(tasks, config=config_c, domains=domains, gpu_id=gpu_id)

        # Initialize image input peripheral
        self.image_input_perph = ImageInputPeripheral(output_dim=config_c['input_dim'],
                                                            dropout=config_p['dropout'], freeze_layers=True)

        # Initialize English language peripheral
        self.english_language_perph = LanguagePeripheral(vocab_size=config_p['english_language_input_vocab'],
                                                               embed_dim=config_p['english_language_input_embed'],
                                                               output_dim=config_c['input_dim'],
                                                               lang='en',
                                                               gpu_id=gpu_id, dropout=config_p['dropout'])

        # Initialize German language peripheral
        self.german_language_perph = LanguagePeripheral(vocab_size=config_p['german_language_input_vocab'],
                                                              embed_dim=config_p['german_language_input_embed'],
                                                              output_dim=config_c['input_dim'],
                                                              lang='de',
                                                              gpu_id=gpu_id)

    def reset(self, batch_size):
        '''
        Resets the state of the model.
        Args:
            batch_size (int): The batch size.
        '''
        self.cnp.reset(batch_size)

    def encode_videos(self, videos, domain='IMAGE'):
        '''
        Encodes videos.
        Args:
            videos (torch.Tensor): The videos to encode.
            domain (str, optional): The domain. Defaults to 'IMAGE'.
        '''
        video_encodings = self.image_input_perph.encode(videos)
        self.cnp.encode(video_encodings, domain=domain)


    def encode_images(self, images, domain='IMAGE'):
        '''
        Encodes images.
        Args:
            images (torch.Tensor): The images to encode.
            domain (str, optional): The domain. Defaults to 'IMAGE'.
        '''
        image_encodings = self.image_input_perph.encode(images)
        self.cnp.encode(image_encodings, domain=domain)
    
    def encode_englishtexts(self, texts, domain='ENGLISH'):
        '''
        Encodes English texts.
        Args:
            texts (list): The texts to encode.
            domain (str, optional): The domain. Defaults to 'ENGLISH'.
        Returns:
            tuple: A tuple containing the sentence encodings and the input pad mask.
        '''
        sent_encodings, input_pad_mask = self.english_language_perph.embed_sentences(texts)
        self.cnp.encode(sent_encodings, pad_mask=input_pad_mask, domain=domain)
    
    def decode_from_targets(self, task, targets, target_pad_mask=None):
        '''
        Decodes from targets.
        Args:
            task (str): The task.
            targets (torch.Tensor): The targets.
            target_pad_mask (torch.Tensor, optional): The target pad mask. Defaults to None.
        Returns:
            torch.Tensor: The decoded output.
        '''
        return self.cnp.decode(task, targets=targets, pad_mask=target_pad_mask)
    
    def decode_greedy(self, task, num_steps):
        '''
        Decodes using a greedy algorithm.
        Args:
            task (str): The task.
            num_steps (int): The number of steps.
        Returns:
            torch.Tensor: The decoded output.
        '''
        return self.cnp.decode(task, targets=None, num_steps=num_steps)



    def save(self, checkpoint_dir, iterations):
        '''
        Saves the model.
        Args:
            checkpoint_dir (str): The checkpoint directory.
            iterations (int): The number of iterations.
        '''
        save_dir = os.path.join(checkpoint_dir, str(iterations))
        try:
            os.stat(save_dir)
        except:
            os.mkdir(save_dir)
        torch.save(self.state_dict(), os.path.join(save_dir, 'model.pth'))
        print('Model saved, iterations: %d' % iterations)


        
    def restore(self, checkpoint_dir, iterations):
        '''
        Restores the model.
        Args:
            checkpoint_dir (str): The checkpoint directory.
            iterations (int): The number of iterations.
        '''
        save_dir = os.path.join(checkpoint_dir, str(iterations), 'model.pth')
        pretrained_dict = torch.load(save_dir)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
        self.load_state_dict(pretrained_dict, strict=False)
        print('Restored existing model with iterations: %d' % (iterations))
    

    def restore_file(self, file):
        '''
        Restores the model from a file.
        Args:
            file (str): The file to restore from.
        '''
        pretrained_dict = torch.load(file)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
        self.load_state_dict(pretrained_dict, strict=False)
        

    
    @staticmethod
    def __defaultconf__():
        """
        The default configuration as specified in the original paper.
        Returns:
            tuple: A tuple containing the configuration for the CNP, the configuration for the peripherals, and the domains.
        """

        cnp_conf = {
            'input_dim':512,
            'control_dim':32,
            'output_dim':512,
            'spatial_dim':512,
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
            'max_seq_len':500,
            'output_embedding_dim':300,
            'dropout':0.1}
        perph_conf = {
            'german_language_input_vocab': 25000,
            'german_language_input_embed': 300,
            'english_language_input_vocab': 25000,
            'english_language_input_embed': 300,
            'english_language_output_vocab': 25000,
            'german_language_output_vocab': 25000,
            'dropout': 0.1 ,
            'vqa_output_vocab':3500,
            'hmdb_output_classes':52,
            'penn_output_classes':48
        }

        domains = ['ENGLISH','GERMAN','IMAGE']

        return cnp_conf, perph_conf, domains
