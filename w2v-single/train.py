import torch
import torchaudio
import torchaudio.pipelines
from torch import nn
#from transformers import Wav2Vec2Model
import soundfile as sf
from torch import Tensor
import os 
from torch import optim
import glob
import random
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from Attention_Pooling import Classic_Attention
le = len('/media/data1/ananya/TAMIL-SLT-extension/large/train/')
n_epoch = 100
num_classes = 4
lan1id = {'MT':0,'NT':1,'OT':2,'ST':3,'TT':4,'UT':5}
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
class StatPooling(nn.Module):
    def __init__(self, num_classes=6):
        super(StatPooling, self).__init__()
        #self.bundle = torchaudio.pipelines.WAV2VEC2_LARGE
        #self.model = self.bundle.get_model().to(device)
        self.segment6 = nn.Linear(2048, 1024).to(device)
        self.segment7 = nn.Linear(1024, 1024).to(device)
        self.output = nn.Linear(1024, num_classes).to(device)
        self.softmax = nn.Softmax(dim=1).to(device)
        self.ll = nn.Linear(1024,1024).to(device)
        self.attention = Classic_Attention(1024,1024).to(device)
    def forward(self, inputs):
        #features = self.extract_wav2vec2(filename)
        #print(inputs.shape)
        #mean = torch.mean(inputs,1)
        #print(f'shape of mean is {mean.shape}')
        #std = torch.var(inputs,1,)
        # print(f'shape of std is {std.shape}')
        #stat_pooling = torch.cat((mean,std),1)
        #print(f'shape of stat_pooling is {stat_pooling.shape}')
        attn_weights = self.attention(inputs)
        stat_pool_out = self.stat_attn_pool(inputs,attn_weights)
        segment6_out = self.segment6(stat_pool_out)
        segment1 = self.segment7(segment6_out)
        op2 = self.segment7(segment1)
        prediction = self.output(op2)
        return prediction
    def weighted_sd(self,inputs,attention_weights, mean):
        el_mat_prod = torch.mul(inputs,attention_weights)
        #print(f'el_mat_prod {el_mat_prod.shape}')
        hadmard_prod = torch.mul(inputs,el_mat_prod)
        variance = torch.sum(hadmard_prod,1) - torch.mul(mean,mean)
        return variance
    
    
    def stat_attn_pool(self,inputs,attention_weights):
        el_mat_prod = torch.mul(inputs,attention_weights)
        mean = torch.mean(el_mat_prod,1)
        variance = self.weighted_sd(inputs,attention_weights,mean)
        stat_pooling = torch.cat((mean,variance),1)
        return stat_pooling

##########################  Model ####################################
model1 = StatPooling()
model1.cuda()
optimizer =  optim.Adam(model1.parameters(), lr=0.0001)
loss_lang = torch.nn.CrossEntropyLoss()  # cross entropy loss function for the softmax output
loss_lang.cuda()


#####for deterministic output set manual_seed ##############
manual_seed = random.randint(1,10000) #randomly seeding
random.seed(manual_seed)
torch.manual_seed(manual_seed)

files_list=[]
folders = glob.glob('/media/data1/ananya/TAMIL-SLT-extension/large/train/*')
print("Folder=", folders)
for folder in folders:
    for f in glob.glob(folder+'/*.pt'):
        files_list.append(f)
print('Total Training files: ',len(files_list))
txtfl = open('/media/data1/ananya/TAMIL-SLT-extension/large/train.txt', 'w')

def lstm_data(f):
    #print(f)
    '''hf = h5py.File(f, 'r')
    X = np.array(hf.get('feature'))
    y = np.array(hf.get('target'))
    print(X.shape, "---", y.shape)
    hf.close()'''
    
    X = torch.load(f)
    #print("Y[0]=", y[0])
    #Y1 = y[0]
    #Y1 = np.array([Y1])  
    
    f1 = os.path.splitext(f)[0]     
    lang = f1[le:le+2]  
    #print('lang',lang)
    Y1 = lan1id[lang]    
    Y1 = np.array([Y1]) 
    Y1 = torch.from_numpy(Y1).long()
    #print('Y1 in lstm',Y1)
    #print('shape',Y1.shape)
    #Xdata1 = np.array(X)    
    #Xdata1 = torch.from_numpy(X).float() 
    
    #Y1 = torch.from_numpy(Y1).long() 
    return X, Y1  # Return the data and true label
# Example usage
'''folder_path ="/mnt/Disk12TB/DID/kannada/mono/train/KK"
for filename in os.listdir(folder_path):
    if filename.endswith('.wav'):  # Check if the file is a WAV file
        file_path = os.path.join(folder_path, filename)
        
        # Create an instance of StatPooling for each file
        f = StatPooling(4)
        
        # Call the forward method to process the file
        op = f.forward(file_path)
        
        # Print the prediction or do something with it
        print(f"Prediction for {filename}: {op}")'''
#model1.load_state_dict(torch.load('/home/iit/disk_10TB/ananya/multif/model/e_8.pth'))
for e in range(n_epoch): # repeat for n_epoch
    i = 0
    cost = 0.0
    random.shuffle(files_list)
    train_loss_list=[]
    full_preds=[]
    full_gts=[]

    for fn in files_list:   
        XX1,YY1 = lstm_data(fn)  
        #print(YY1)    
        XX1 = torch.unsqueeze(XX1, 1) # Adding one additional dimension at specified position
        #print("shape of xx1",XX1.shape)
  #Counting the number of files

        X1 = np.swapaxes(XX1,0,1)  # changing the axis(similar to transpose)

        # Enable cuda
        X1 = Variable(X1,requires_grad=False).cuda() 
        Y1 = Variable(YY1,requires_grad=False).cuda()               
        model1.zero_grad() # setting model gradient to zero before calculating gradient
        
        lang_op = model1.forward(X1)   # forward propagation the input to the model
        #print(lang_op)
        T_err = loss_lang(lang_op,Y1)  # loss calculation over the model output with true label
               
        T_err.backward()  # calculating the gradient on loss obtained
        
        optimizer.step() # parameter updation based on gradient calculated in previous step 
        
        train_loss_list.append(T_err.item())

        cost = cost + T_err.item()
        
        i = i+1    
        print("x-vec-bnf.py: Epoch = ",e+1,"  completed files  "+str(i)+"/"+str(len(files_list))+" Loss= %.7f"%(cost/i))
        predictions = np.argmax(lang_op.detach().cpu().numpy(),axis=1) #Convert one hot coded result to label
        for pred in predictions:
            full_preds.append(pred)
        for lab in Y1.detach().cpu().numpy():
            full_gts.append(lab)

            ############################

    mean_acc = accuracy_score(full_gts,full_preds)  # accuracy calculation over the true label and predicted label
    mean_loss = np.mean(np.asarray(train_loss_list)) # average loss calculation
    print('Total training loss {} and training Accuracy {} after {} epochs'.format(mean_loss,mean_acc,e+1))      
    path = "/media/data1/ananya/TAMIL-SLT-extension/large/model/e_"+str(e+1)+".pth"
    torch.save(model1.state_dict(),path) # saving the model parameters 
    txtfl.write(path)
    txtfl.write('\n')
    txtfl.write("acc: "+str(mean_acc))
    txtfl.write('\n')
    txtfl.write("loss: "+str(mean_loss))
    txtfl.write('\n')

###############################################################

