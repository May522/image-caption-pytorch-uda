import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN,self).__init__()
        #super().__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        #super().__init__()
        super(DecoderRNN, self).__init__()
        self.num_layers=num_layers
        self.hidden_size=hidden_size
        self.vocab_size=vocab_size
        self.Wemb=nn.Embedding(vocab_size,embed_size)
        self.lstm=nn.LSTM(input_size=embed_size,hidden_size=hidden_size,
                          num_layers=num_layers,batch_first=True)
        self.fc=nn.Linear(hidden_size,vocab_size)
        self.init_weights()
        

    
    def forward(self, features, captions):
        
        batch_size=features.shape[0]
        # 初始化状态h,c
        weight=next(self.parameters()).data
        self.hc=(weight.new(self.num_layers,batch_size,self.hidden_size).zero_(),
        weight.new(self.num_layers,batch_size,self.hidden_size).zero_())

        features=torch.unsqueeze(features,1) # features变成 10*1*256
        captions=self.Wemb(captions) # 输入10*14 输出：10*14*256


        inputs=torch.cat((features,captions),1) # inputs 10*15*256
        inputs=inputs[:,:-1,:] # 10*14*256
        lstm_output,self.hc=self.lstm(inputs,self.hc)
        # inputs:10*14*256  lstm_output:10*14*512 (batch_size,seq_len,hidden_size)
        outputs=self.fc(lstm_output) # outputs: (10,14,8852)
        
        

        return outputs



    
    
    def init_weights(self):
        # initrange=0.1
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-1, 1)
    
    

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "       
        weight=next(self.parameters()).data
        hc=(weight.new(self.num_layers,1,self.hidden_size).zero_(),
        weight.new(self.num_layers,1,self.hidden_size).zero_())
        
        predict_caption=[]
        
        for count in range(1,max_len):
            if count>1:
                inputs=outputs_index#inuts is 2 dimensional
                inputs=self.Wemb(inputs)#inputs become 3 dimensional
                
            lstm_output,hc=self.lstm(inputs,hc)
            outputs=self.fc(lstm_output)#outputs:[1,1,vocab_size]
            _,outputs_index=torch.max(outputs,2)#outputs_index shape is [1,1],eg:[[855]]
            
            word_index=outputs_index.cpu().numpy().item()
            predict_caption.append(word_index)
            if word_index==1:
                return predict_caption

    
        return predict_caption
        
        
        
        
        