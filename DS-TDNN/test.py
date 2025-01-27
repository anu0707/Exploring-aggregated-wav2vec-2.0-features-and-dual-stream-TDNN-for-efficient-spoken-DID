import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy
import os
import numpy as np
import pandas as pd
import glob, os, random, soundfile, torch
import glob
import random
from torch.autograd import Variable
from torch.autograd import Function
from torch import optim
import gc
import traceback
from functools import partial
import torch, torchaudio, math
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import torch.fft
from modules import *
#import soundfile as sf
import wave
from sklearn.metrics import accuracy_score
import sklearn.metrics
import h5py
from sklearn.metrics import roc_curve
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LANGUAGE = {'HK':0,'KK':1,'MK':2,'OM':3}
n_epoch=100
class LocalBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, dilation=1, scale=4, drop_path=0.):
        super().__init__()
        self.res2conv = res2conv1d(dim, kernel_size, dilation, scale)     
        # self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)   
           
        self.norm1 = nn.BatchNorm1d(dim)   
        self.norm2 = nn.BatchNorm1d(dim)   
        self.norm3 = nn.BatchNorm1d(dim)   
        self.proj1 = nn.Conv1d(dim, dim, kernel_size=1)  
        self.proj2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.act   = nn.ReLU()
        # self.act = nn.GELU()
        self.se    = SEModule(dim)

    def forward(self, x):
        skip = x
        
        x = self.proj1(x)
        x = self.act(x)
        x = self.norm1(x)
        
        x = self.res2conv(x)
        # x = self.dwconv(x)
        # x = self.act(x)
        # x = self.norm2(x)
  
        x = self.proj2(x)
        x = self.act(x)
        x = self.norm3(x)
        
        x = skip + self.se(x)
        
        return x    
    
    

class GlobalBlock(nn.Module):
    """ 
     Global block: if global modules = MSA or LSTM, need to permute the dimension of input tokens
    """
    def __init__(self, dim, T=200, dropout=0.2, K=4):
        super().__init__()
        self.gf = SparseDGF(dim, T, dropout=dropout, K=K) # Dynamic global-aware filters with sparse regularization
#         self.gf = SparseGF(dim, T, dropout=dropout) # Global-aware filters with sparse regularization
#         self.gf = DGF(dim, T, K=K) # Dynamic global-aware filters
#         self.gf = GF(dim, T) # Global-aware filters
#         self.gf = MSA(num_attention_heads=K, input_size=dim, hidden_size=dim) # Multi-head self-attention
#         self.gf = LSTM(input_size=dim, hidden_size=dim, bias=False, bidirectional=False) # LSTM
        
        self.norm1 = nn.BatchNorm1d(dim)  
        self.norm2 = nn.BatchNorm1d(dim)  
        self.norm3 = nn.BatchNorm1d(dim)  
        self.proj1 = nn.Conv1d(dim, dim, kernel_size=1)  
        self.proj2 = nn.Conv1d(dim, dim, kernel_size=1)  
        self.act   = nn.ReLU()

    def forward_(self, x):
        skip = x
        
        x = self.proj1(x)
        x = self.act(x)
        x = self.norm1(x)
        
        x = self.gf.forward_(x) 
        x = self.act(x)    
        x = self.norm2(x)
        
        x = self.proj2(x)
        x = self.act(x) 
        x = self.norm3(x) 
        
        x = skip + x
        
        return x    
    
    def forward(self, x):
        skip = x
        
        x = self.proj1(x)
        x = self.act(x)
        x = self.norm1(x) 
        
        # if self.gf == SA or LSTM, we need to transform the input tokens x
        # x = x.permute(0,2,1)
        x = self.gf(x) 
        # x = x.permute(0,2,1)
       
        x = self.act(x)      
        x = self.norm2(x)
        
        x = self.proj2(x)
        x = self.act(x) 
        x = self.norm3(x) 
        
        x = skip + x
        
        return x    
  
     
    
class PreEmphasis(torch.nn.Module):
    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)

    
    
class FbankAug(nn.Module):
    def __init__(self, freq_mask_width = (0, 8), time_mask_width = (0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)
        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)
            
        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):    
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x
                 
                    
                    
class DS_TDNN(nn.Module):
    def __init__(self, C, uniform_init=True, if_sparse=True):
        super(DS_TDNN, self).__init__()
        self.sparse=True
        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80),
            )

        self.specaug = FbankAug() # Spec augmentation

        self.conv1  = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.gelu   = nn.GELU()
        self.bn1    = nn.BatchNorm1d(C)
        
        # local branch
        self.llayer1 = LocalBlock(C//2, kernel_size=3, scale=4, dilation=1) #默认8 8 8 C=1024; 尝试4 6 8 C=960
        self.llayer2 = LocalBlock(C//2, kernel_size=3, scale=4, dilation=1)
        self.llayer3 = LocalBlock(C//2, kernel_size=3, scale=8, dilation=1)
         
        #global branch
        self.glayer1 = GlobalBlock(C//2, T=200, dropout=0.3,  K=4)
        self.glayer2 = GlobalBlock(C//2, T=200, dropout=0.1,  K=8)
        self.glayer3 = GlobalBlock(C//2, T=200, dropout=0.1,  K=8)
        
        
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
        # ASP
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(), # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
            )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)
        self.fc8 = nn.Linear(192,192)
        self.fc9 = nn.Linear(192,64)
        self.fc7 = nn.Linear(64,4)
        self.uniform_init = uniform_init

    def forward_(self, x, aug=None):    
        '''
         Sparse forward
        '''
        assert self.sparse==True       
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            x = x.log()   
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug == True:
                x = self.specaug(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        
        lx, gx = torch.chunk(x, 2, dim=1)
        
        #Dual branch:
        lx1 = self.llayer1(lx)
        gx1 = self.glayer1.forward_(gx)
                
        lx2 = self.llayer2(0.8*lx1+0.2*gx1)
        gx2 = self.glayer2.forward_(0.8*gx1+0.2*lx1)
        
        lx3 = self.llayer3(0.8*lx2+0.2*gx2)
        gx3 = self.glayer3.forward_(0.8*gx2+0.2*lx2)    
        
        x = self.layer4(torch.cat((lx1,gx1, lx2,gx2, lx3,gx3),dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        
        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)
        x = self.fc8(x)
        x = self.fc7(x)
        return x   
    
    
    
    def forward(self, x, aug=None):
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            x = x.log()   
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug == True:
                x = self.specaug(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        
        lx, gx = torch.chunk(x, 2, dim=1)
        
        #Dual branch:
        lx1 = self.llayer1(lx)
        gx1 = self.glayer1(gx)
                
        lx2 = self.llayer2(0.8*lx1+0.2*gx1)
        gx2 = self.glayer2(0.8*gx1+0.2*lx1)
        
        lx3 = self.llayer3(0.8*lx2+0.2*gx2)
        gx3 = self.glayer3(0.8*gx2+0.2*lx2)   

        x = self.layer4(torch.cat((lx1,gx1, lx2,gx2, lx3,gx3),dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        
        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),1)
        #print(f'x shape {x.shape}')
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)
        x = self.fc8(x)
        x = self.fc9(x)
        print(f'x shape {x.shape}')
        x = self.fc7(x)
        #print(f'x shape {x.shape}')
        return x

      
    def hook(self, x, aug=None):
        '''
         hook for different-scale feature maps
        '''
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            x = x.log()   
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug == True:
                x = self.specaug(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        
        stem_o = x
        
        lx, gx = torch.chunk(x, 2, dim=1)
        
        #Dual branch:
        lx1 = self.llayer1(lx)
        gx1 = self.glayer1(gx)
                
        lx2 = self.llayer2(0.8*lx1+0.2*gx1)
        gx2 = self.glayer2(0.8*gx1+0.2*lx1)
        
        lx3 = self.llayer3(0.8*lx2+0.2*gx2)
        gx3 = self.glayer3(0.8*gx2+0.2*lx2)        

        x = self.layer4(torch.cat((lx1,gx1, lx2,gx2, lx3,gx3),dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        
        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x, stem_o, [lx, lx1, lx2, lx3], [gx, gx1, gx2, gx3], [lx1+gx1, lx2+gx2]
    
    
    def _init_weights(self, m):
        if not self.uniform_init:
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        else:
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)     
                    
                    
le=len('/mnt/Disk12TB/DID/slt/test/')
lan1id={'HK':0,'KK':1,'MK':2,'OM':3}

model=DS_TDNN(2048)
model.cuda()

#optimizer =  optim.Adam(model.parameters(), lr=0.0001)
#loss_lang = torch.nn.CrossEntropyLoss()  # cross entropy loss function for the softmax output

#####for deterministic output set manual_seed ##############
manual_seed = random.randint(1,10000) #randomly seeding
random.seed(manual_seed)
torch.manual_seed(manual_seed)

params = {'batch_size': 64,
          'shuffle': True}

X_test,y_test=[],[]


folders = glob.glob('/mnt/Disk12TB/DID/slt/test/*')
print("Folder=", folders)
for folder in folders:
    for f in glob.glob(folder+'/*.wav'):
        X_test.append(f) 
        f1 = os.path.splitext(f)[0]     
        lang = f1[le:le+2] 
        y_test.append(LANGUAGE[lang])

print('Total Training files: ',len(X_test))

l=len(X_test)

class train_loader(object):
    def __init__(self, file_paths, labels,num_frames=200,**kwargs):
        self.file_paths = file_paths
        self.labels = labels
        self.num_frames = num_frames
    def __len__(self):
        return len(self.file_paths)
        
    def lstm_data(self,file_path):
        lang = file_path.split('/')[-2][:3]
        Y1 = LANGUAGE[lang]
        #print('file path',file_path)
        #print('y1 in lstm',Y1)
        #Y2 = np.array([Y1])
        #Ydata1 = torch.from_numpy(Y2).long()
        return Y1  
        
    def __getitem__(self, index):
        audio, sr = soundfile.read(self.file_paths[index])		
        length = self.num_frames * 160 + 240
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = np.pad(audio, (0, shortage), 'wrap')
        start_frame = np.int64(random.random()*(audio.shape[0]-length))
        audio = audio[start_frame:start_frame + length]
        audio = np.stack([audio],axis=0)
        file_path = self.file_paths[index]
        label = self.lstm_data(file_path)
        #print(file_path)
        #print(label)
        #print('shape',torch.FloatTensor(audio[0]).shape)
        #print(audio[0])
        return torch.FloatTensor(audio[0]), label  
 
training_set = train_loader(X_test, y_test)
training_generator = torch.utils.data.DataLoader(training_set, **params)
cnt = len(X_test)
#model.load_state_dict(torch.load('/mnt/Disk12TB/DID/slt/ds-tdnn/model/e_12.pth'))
txtfl = open('/mnt/Disk12TB/DID/slt/ds-tdnn/test_model4.txt', 'w')
for e in range(200): # repeat for n_epoch
    i = 0
    model = DS_TDNN(2048)
    model.cuda()
    path =  "/mnt/Disk12TB/DID/slt/ds-tdnn/model4/e_"+str(e+1)+".pth"
    model.load_state_dict(torch.load(path))
    full_gts = []
    full_preds = []
    train_acc = 0
    for (Xdata1,Y1) in training_generator:                          
        #model1.zero_grad() # setting model gradient to zero before calculating gradient
        Xdata1 = Xdata1.cuda()
        Y1 = Y1.cuda()
        lang_op = model.forward(Xdata1)   # forward propagation the input to the model
        i = i+1    
        print("x-vec-bnf.py: Epoch = ",e+1,"  completed files  "+str(i))
        #print(f'language output {lang_op.shape}')
        #X = Variable(Xdata1, requires_grad=True).cuda()
        #Y = Variable(Y1, requires_grad=True).cuda()
        predictions = np.argmax(lang_op.detach().cpu().numpy(),axis=1) #Convert one hot coded result to label
        #print(predictions)
        for pred in predictions:
            full_preds.append(pred)
        for lab in Y1.detach().cpu().numpy():
            full_gts.append(lab)
            ############################
    #print(train_acc/cnt)
    #print('Actual \n')
    #print(Actual)
    #print('Pred \n')
    #print(Pred)
    nc1 =4
    fpr = dict()
    tpr = dict()
    fnr = dict()
    EER = dict()
    y_test  = F.one_hot(torch.as_tensor(full_preds), num_classes=nc1)
    y_score = F.one_hot(torch.as_tensor(full_gts), num_classes=nc1)
  
    for i in range(nc1):
      fpr[i], tpr[i],_ = roc_curve(y_test[:, i], y_score[:, i],pos_label=1)
      fnr[i] = 1-tpr[i]
      EER[i] = (fpr[i][np.nanargmin(np.absolute((fnr[i]-fpr[i])))] + 
              fnr[i][np.nanargmin(np.absolute((fnr[i]-fpr[i])))])/2
            
    txtfl.write(path)
    txtfl.write('\n')
    mean_acc = accuracy_score(full_gts,full_preds) 
    txtfl.write("acc: "+str(mean_acc))
    CM2=sklearn.metrics.confusion_matrix(full_gts, full_preds)
    print(CM2)
 # accuracy calculation over the true label and predicted label
    txtfl.write('\n')
    txtfl.write(str(CM2))
    txtfl.write('\n')
    print(EER)
    print('EER dialect:\t',np.mean(list(EER.values())))
    txtfl.write("EER:"+str(EER))
    txtfl.write('\n')
    txtfl.write("Mean EER:"+str(np.mean(list(EER.values()))))
    txtfl.write('\n')
    txtfl.write('########################################################################################################################')
    txtfl.write('\n')
    #mean_loss = np.mean(np.asarray(train_loss_list)) # average loss calculation
    print('Total testing Accuracy {} after {} epochs'.format(mean_acc,e+1))      
    #path = "/mnt/Disk12TB/DID/multif/stat_pooling/e_"+str(e+1)+".pth"
    #torch.save(model1.state_dict(),path) # saving the model parameters 
    #txtfl.write(path)


###############################################################


