# _*_ coding utf-8 _*_
# 开发人员： RUI
# 开发时间： 2020/9/14
# 文件名称： 1_preli
# 开发工具： PyCharm

import sys
#sys.path.append('/opt/cocoapi/PythonAPI')
from pycocotools.coco import COCO
import nltk
nltk.download('punkt')
from data_loader import get_loader
from torchvision import transforms

transform_train = transforms.Compose([
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.RandomCrop(224),                      # get 224x224 crop from random location
    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])

vocab_threshold = 5
batch_size = 10

data_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=True)

import numpy as np
import torch.utils.data as data

# Randomly sample a caption length, and sample indices with that length.
indices = data_loader.dataset.get_train_indices()
print('sampled indices:', indices)

# Create and assign a batch sampler to retrieve a batch with the sampled indices.
new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
data_loader.batch_sampler.sampler = new_sampler

# Obtain the batch.
images, captions = next(iter(data_loader))

print('images.shape:', images.shape)
print('captions.shape:', captions.shape)

import torch
from model import EncoderCNN, DecoderRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_size = 256
encoder = EncoderCNN(embed_size)
encoder.to(device)
images = images.to(device)
features = encoder(images)
print('type(features):', type(features))
print('features.shape:', features.shape)

assert type(features)==torch.Tensor, "Encoder output needs to be a PyTorch Tensor."
assert (features.shape[0]==batch_size) & (features.shape[1]==embed_size), "The shape of the encoder output is incorrect."

hidden_size = 512
vocab_size = len(data_loader.dataset.vocab)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
decoder.to(device)
captions = captions.to(device)
print(captions.shape,type(captions))
outputs = decoder(features, captions)
print('type(outputs):', type(outputs))
print('outputs.shape:', outputs.shape)



print('end')