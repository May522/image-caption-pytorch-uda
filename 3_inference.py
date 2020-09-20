# _*_ coding utf-8 _*_
# 开发人员： RUI
# 开发时间： 2020/9/20
# 文件名称： 3_inference
# 开发工具： PyCharm


from data_loader import get_loader
from torchvision import transforms

# TODO #1: Define a transform to pre-process the testing images.
transform_test = transforms.Compose([
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.RandomCrop(224),                      # get 224x224 crop from random location
    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])
#-#-#-# Do NOT modify the code below this line. #-#-#-#

# Create the data loader.
data_loader = get_loader(transform=transform_test,
                         mode='test')

import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from model import EncoderCNN, DecoderRNN
# Obtain sample image before and after pre-processing.
orig_image, image = next(iter(data_loader))



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder_file = 'encoder-1.pkl'
decoder_file = 'decoder-1.pkl'

# TODO #3: Select appropriate values for the Python variables below.
embed_size = 512
hidden_size = 512

# The size of the vocabulary.
vocab_size = len(data_loader.dataset.vocab)
print('vocab_size: {}'.format(vocab_size))
# Initialize the encoder and decoder, and set each to inference mode.
encoder = EncoderCNN(embed_size)
encoder.eval()
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
decoder.eval()

# Load the trained weights.
encoder.load_state_dict(torch.load(os.path.join('./models', encoder_file)))
decoder.load_state_dict(torch.load(os.path.join('./models', decoder_file)))

# Move models to GPU if CUDA is available.
encoder.to(device)
decoder.to(device)


def clean_sentence(output):
    if output[0] == 0:
        del output[0]
    if output[-1] == 1:
        del output[-1]

    sentence = []
    for index in output:
        word = data_loader.dataset.vocab.idx2word[index]
        sentence.append(word)
    sentence = ' '.join(sentence)
    return sentence


def get_prediction():
    orig_image, image = next(iter(data_loader))
    image = image.to(device)
    features = encoder(image).unsqueeze(1)
    output = decoder.sample(features)
    sentence = clean_sentence(output)
    print(sentence)

    plt.imshow(np.squeeze(orig_image))
    plt.title('Sample Image')
    plt.show()


get_prediction()