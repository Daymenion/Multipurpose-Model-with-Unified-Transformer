import gensim
import torch
from utils.util import read_caption
from torch.utils.serialization import load_lua
import os
import numpy as np 

# use Google model to convert sentence to embedding matrix
def word2vec(model, sentence, sentence_size, embedding_size):
    # sentence: string of words separated by space (e.g. 'this is a sentence') 
    # embedding: [sentence_size, embedding_size]
    words = sentence.rstrip().split(' ')
    removeable_words = []

    # remove punctuation and check if word is in vocabulary
    for i, word in enumerate(words):
        cleaned = word.strip('.,')
        words[i] = cleaned
        if cleaned not in model.vocab.keys():
            print('Word not in vocabulary: ', cleaned)
            removeable_words.append(i)
    
    # remove words not in vocabulary from sentence
    for word in removeable_words:
        while word in words:
            words.remove(word)
    
    # vocabulary is a dictionary of words and their index in the embedding matrix.
    len_vocab_words = len(words)
    vocabulary = {words[i]: i for i in range(len_vocab_words)}
    embedding = np.zeros((sentence_size, embedding_size))

    # model is a dictionary of words and their word vector, fill embedding matrix with word vectors from vocabulary
    for j,k in vocabulary.items():
        if k> 15:
            break
        embedding[k] = model[j]

    return torch.from_numpy(embedding)


# convert character tensor to sentence string
def to_sentence(alphabet, character):
    # character: [sentence_size] 
    # sentence: string of words separated by space (e.g. 'this is a sentence') 
    sentence = None
    multi_sentence = len(character.stride()) > 1

    # fill sentence with words from alphabet using character tensor as index
    if multi_sentence:
        for i_sentence_th in torch.t(character):
            sentence = [alphabet[int(idx.item()-1)] for idx in i_sentence_th]  # -1 for padding index, index strats from 1
    
    # return sentence string 
    return ' '.join(list(map(str, sentence)))



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
        for caption in os.listdir(class_path):
            caption = os.path.join(class_path, caption)
            if os.path.isfile(caption):
                caption_list.append(caption)
            else:
                print('Caption file does not exist!', caption)
    

    return caption_list


# save embedding matrix to file
def save_embeddings(file_path, file_name, embeds):
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    target = os.path.join(file_path, file_name)
    torch.save({'embeds': embeds}, target)
    return True


# Word_Embeddings class to convert caption data to embedding matrix and save to file
class Word_Embeddings():

    def __init__(self, root_directory, caption_directory, split_file, alphabet):
        self.caption_directory = caption_directory
        self.split_file = split_file
        self.alphabet = alphabet

        self.dir_path = os.path.join(root_directory, 'pretrained_embeddings')
        
        # load Google model for word2vec conversion 
        self.model = gensim.models.KeyedVectors.load_word2vec_format('../models/GoogleNews-vectors-negative300.bin',
                                                                     binary=True)

    # convert caption data to embedding matrix and save to file
    def process(self, cap):
        assert (os.path.isfile(cap))
        
        # get file path and file name from caption data path 
        caption_class, file_name = cap.split('/')[-2], cap.split('/')[-1]
        file_path = os.path.join((self.dir_path, caption_class))

        # load caption data from caption data path
        caption = load_lua(cap)
        char = caption['char']
        
        # convert character tensor to sentence string
        sentence = to_sentence(self.alphabet, char)

        # convert sentence string to embedding matrix
        embeds = word2vec(self.model, sentence, sen_size=16, emb_size=300)

        # return file path, file name, and embedding matrix
        return file_path, file_name, embeds



alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} " 
data_directory = '../datasets/CUB_200_2011/'
split_file = os.path.join(data_directory, 'train_val.txt') 
caption_directory = os.path.join(data_directory, 'cub_icml') 
caption_list = read_caption(caption_directory, split_file) # read caption data from caption path and split file path

WE = Word_Embeddings(data_directory, caption_directory, split_file, alphabet) # create Word_Embeddings object

for caption in caption_list:
    file_path, file_name, embeds = WE.process(caption)
    # save embedding matrix to file
    if save_embeddings(file_path, file_name, embeds):
        print(file_path, file_name, 'saved')


# # check load
# data = torch.load(os.path.join(WE.dir_path, '007.Parakeet_Auklet/Parakeet_Auklet_0064_795954.t7'))
# print(data['embeds'])