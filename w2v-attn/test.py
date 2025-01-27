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
import sklearn.metrics
from sklearn.metrics import roc_curve
import torch.nn.functional as F
from Attention_Pooling import Classic_Attention

le = len('/mnt/Disk12TB/DID/slt/test1/')
n_epoch = 50
lan1id = {'HK':0,'KK':1,'MK':2,'OM':3}
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
class StatPooling(nn.Module):
    def __init__(self, num_classes=4):
        super(StatPooling, self).__init__()
        self.bundle = torchaudio.pipelines.WAV2VEC2_LARGE
        self.model = self.bundle.get_model().to(device)
        self.attention = Classic_Attention(1024,1024).to(device)
        #### Frame levelPooling
        self.segment6 = nn.Linear(2048, 1024).to(device)
        self.segment7 = nn.Linear(1024, 1024).to(device)
        self.output = nn.Linear(1024, num_classes).to(device)
        self.softmax = nn.Softmax(dim=1).to(device)
        self.ll = nn.Linear(1024,1024).to(device)
    def forward(self, filename):
        features = self.extract_wav2vec2(filename)
        stacked = []
        for i, layer_features in enumerate(features):
            #print(f"Layer {i + 1} features shape:", layer_features.shape) 
            attn_weights = self.attention(layer_features)
            #print(f'attention weights {attn_weights.shape}')
            stat_pool_out = self.stat_attn_pool(layer_features,attn_weights)
            #print(f'stat pool layer {}
            segment6_out = self.segment6(stat_pool_out)
            #print(f'mean {segment6_out.shape}')   
            vec = self.segment7(segment6_out) 
            #print(f'vec {vec.shape}')  
            stacked.append(vec)
        stacked_tensor = torch.stack(stacked, dim=1)
        linear_l = self.ll(stacked_tensor)
        #mean_l = torch.mean(linear_l,1)
        #var_l = torch.var(linear_l,1)
        #stat_pooling_l = torch.cat((mean_l,var_l),1)
        attn_weights1 = self.attention(linear_l)
        stat_pooling_l = self.stat_attn_pool(linear_l,attn_weights1)
        segment6_1 = self.segment6(stat_pooling_l)
        vec1 = self.segment7(segment6_1) 
        #print(f'vector after processing {vec1.shape}')
        prediction = self.output(vec1)
        return prediction

    def extract_wav2vec2(self, aud_path):
        X, sample_rate = sf.read(aud_path)
        waveform = Tensor(X)
        waveform = waveform.unsqueeze(0)
        waveform = waveform.to(device)
        if sample_rate != self.bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.bundle.sample_rate)
        features, _ = self.model.extract_features(waveform)
        return features


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
#model1 = StatPooling()
#model1.cuda()
#optimizer =  optim.Adam(model1.parameters(), lr=0.0001, weight_decay=5e-5, betas=(0.9, 0.98), eps=1e-9)
loss_lang = torch.nn.CrossEntropyLoss()  # cross entropy loss function for the softmax output
loss_lang.cuda()


#####for deterministic output set manual_seed ##############
manual_seed = random.randint(1,10000) #randomly seeding
random.seed(manual_seed)
torch.manual_seed(manual_seed)

files_list=[]
folders = glob.glob('/mnt/Disk12TB/DID/slt/test1/*')
print("Folder=", folders)
for folder in folders:
    for f in glob.glob(folder+'/*.wav'):
        files_list.append(f)
print('Total Training files: ',len(files_list))
txtfl = open('/mnt/Disk12TB/DID/slt/large_wofinetune/stat_att_test_complete.txt', 'w')

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

for e in range(20,n_epoch): # repeat for n_epoch
    i = 0
    model1 = StatPooling()
    model1.cuda()
    random.shuffle(files_list)
    path =  "/mnt/Disk12TB/DID/slt/large_wofinetune/att_stat_pool/e_"+str(e+1)+".pth"
    model1.load_state_dict(torch.load(path))
    Actual = []
    Pred = []

    for fn in files_list:                          
        #model1.zero_grad() # setting model gradient to zero before calculating gradient

        lang_op = model1.forward(fn)   # forward propagation the input to the model
        #print("lang_op=", lang_op)
        #print("lang_op shape=", lang_op.shape)
        f1 = os.path.splitext(fn)[0]     
        lang = f1[le:le+2]  
        #print('lang',lang)
        Y1 = lan1id[lang]    
        Y1 = np.array([Y1]) 
        Y1 = torch.from_numpy(Y1).long()
        #print('Y1',Y1)
        i = i+1    
        print("x-vec-bnf.py: Epoch = ",e+1,"  completed files  "+str(i)+"/"+str(len(files_list)))
        predictions = np.argmax(lang_op.detach().cpu().numpy(),axis=1) #Convert one hot coded result to label
        Actual.append(int(Y1))
        Pred.append(int(predictions))

            ############################
    nc1 =4
    fpr = dict()
    tpr = dict()
    fnr = dict()
    EER = dict()
    y_test  = F.one_hot(torch.as_tensor(Pred), num_classes=nc1)
    y_score = F.one_hot(torch.as_tensor(Actual), num_classes=nc1)
  
    for i in range(nc1):
      fpr[i], tpr[i],_ = roc_curve(y_test[:, i], y_score[:, i],pos_label=1)
      fnr[i] = 1-tpr[i]
      EER[i] = (fpr[i][np.nanargmin(np.absolute((fnr[i]-fpr[i])))] + 
              fnr[i][np.nanargmin(np.absolute((fnr[i]-fpr[i])))])/2
            
    txtfl.write(str(path))   
    txtfl.write('\n')  
    print(EER)
    print('EER dialect:\t',np.mean(list(EER.values())))
    txtfl.write("EER:"+str(EER))
    txtfl.write('\n')
    txtfl.write("Mean EER:"+str(np.mean(list(EER.values()))))
    txtfl.write('\n')
    #txtfl.write('########################################################################################################################')
    #txtfl.write('\n')
    CM2=sklearn.metrics.confusion_matrix(Actual, Pred)
    print(CM2)
    mean_acc = accuracy_score(Actual,Pred)  # accuracy calculation over the true label and predicted label
    #mean_loss = np.mean(np.asarray(train_loss_list)) # average loss calculation
    print('Total testing Accuracy {} after {} epochs'.format(mean_acc,e+1))      
    #path = "/mnt/Disk12TB/DID/multif/stat_pooling/e_"+str(e+1)+".pth"
    #torch.save(model1.state_dict(),path) # saving the model parameters 
    #txtfl.write(path)
    #txtfl.write('\n')
    txtfl.write("acc: "+str(mean_acc))
    txtfl.write('\n')
    txtfl.write("CM: "+str(CM2))
    txtfl.write('########################################################################################################################')
    txtfl.write('\n')

###############################################################

