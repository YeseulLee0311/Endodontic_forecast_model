import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from dental_datasets import DentalDataset,get_pm_label
from torch import optim
from torch.optim import lr_scheduler
from datetime import datetime
import pandas as pd
from Attention_ResNet import Ensemble, Net, Net2,ClinicalModel,ResNet18
import numpy as np
import pandas as pd
import time
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def get_clinical_features(path='/home/NAS_mount/yslee/dataset/clinical_features_0811.csv'):
    df=pd.read_csv(path,encoding='CP949')
    df=df[['P_No','CV','PF','ST','PA','FV','CD','RR','PS','TM','BRG']] #
    
    label_file=pd.read_csv('/home/NAS_mount/yslee/dataset/premolar_labels_0810.csv')
    label_file=label_file.set_index('PatientID_new')
    
    P_No_old_list=[]
    label_list=[]
    for i in range(0,len(df)):
        P_No_old_list.append(label_file['PatientID'][df['P_No'][i]])
        label_list.append(label_file['Result'][df['P_No'][i]])
    df=pd.concat([pd.DataFrame(P_No_old_list),df,pd.DataFrame(label_list)],axis=1)

    df.columns=['P_No_old','P_No','CV','PF','ST','PA','FV','CD','RR','PS','TM','Result','BRG'] #
    df=df.set_index('P_No')
    
    features=df[['CV','PF','ST','PA','FV','CD','RR','PS','TM','BRG']]#[['CV','PF','ST','PA','FV','CD','RR','PS','TM']] #
    label=df['Result']

    '''for i in range(0,len(features)):
        for j in range(0,len(features.iloc[i])):
            if features.iloc[i,j]=='Y':
                features.iloc[i,j]=np.float32(1)
            elif features.iloc[i,j]=='N':
                features.iloc[i,j]=np.float32(0)
            
    for i in range(0,len(label)):
        if label[i]=='FAIL' or label[i]=='Fail':
            label[i]=np.float32(1)
        elif label[i]=='SUCCESS' or label[i]=='Success':
            label[i]=np.float32(0)
        else:
            print(i,' Error!')'''
    label=label.astype(int)

    return features

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes,device, num_epochs, file_name):
    since = time.time()
    model_later=copy.deepcopy(model)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    train_loss=[]
    val_loss=[]

    train_acc=[]
    val_acc=[]
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_cm=0

            # Iterate over data.
            for inputs, labels, file_names in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs,file_names) #file_names
                    _, preds = torch.max(outputs, 1)
                    
                    #labels=labels.float()
                    #labels_loss=labels_loss.view(labels.size()[0],1)
                    #outputs=outputs.view(outputs.size()[0])

                    '''print('***********outputs************')
                    print(outputs)
                    print('***********preds************')
                    print(preds)
                    print('***********labels************')
                    print(labels)
                    print()'''

                    
                    #print(outputs.size(),labels_loss.size())
                    #print(outputs)
                    #print(F.sigmoid(outputs))

                    loss=criterion(outputs,labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += (loss.item()) * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_cm+=confusion_matrix(labels.cpu(), preds.cpu(),labels=[0,1])

            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_sen = (running_cm[1][1]/(running_cm[1][1]+running_cm[1][0]))
            
            if phase=='train':
                train_loss.append(epoch_loss)
                train_acc.append(np.float(epoch_acc))
            else:
                val_loss.append(epoch_loss)
                val_acc.append(np.float(epoch_acc))

            #print(phase,train_loss,val_loss)
            print('{} Loss: {:.4f} Acc: {:.4f} Sen+Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc,(epoch_acc*0.6+epoch_sen*0.4)))
            
            # deep copy the model
            if phase == 'val' and (epoch_acc*0.6+epoch_sen*0.4) > best_acc:
                best_acc = (epoch_acc*0.6+epoch_sen*0.4)
                best_model_wts = copy.deepcopy(model.state_dict())
            
            if phase == 'val' and (epoch_acc*0.6+epoch_sen*0.4) >= best_acc:
                #best_acc = epoch_acc
                best_model_wts_later = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc+Sen: {:4f}'.format(best_acc))

    plt.rcParams["figure.figsize"] = (8,8)
    fig, axs=plt.subplots(2)
    axs[0].set_title('model loss')
    axs[1].set_title('model accuracy')
    for ax in axs.flat:
        ax.set_ylim([0.0,1.0])
    axs[0].plot(train_loss,'r',val_loss,'g',)
    axs[1].plot(train_acc,'r',val_acc,'g')
    fig.tight_layout()
    for ax in axs.flat:
        leg=ax.legend(['train','val'])
    if file_name != None:
        plt.savefig('/home/yslee/CNN_Prediction_Endo/figs/'+file_name+datetime.today().strftime('_%m-%d-%H:%M')+'.png')
    '''
    #loss graph
    plt.figure()
    plt.rcParams["figure.figsize"] = (10,5)
    plt.title('model loss')
    plt.plot(train_loss,'r',val_loss,'g')
    plt.ylim([0.0, 1.0])
    plt.legend(['train','val'])
    plt.show()
    if file_name != None:
        #print(file_name+datetime.today().strftime('_LOSS_%m-%d-%H:%M')+'.png',' Saved!!')
        plt.savefig('/home/yslee/CNN_Prediction_Endo/figs/'+file_name+datetime.today().strftime('_LOSS_%m-%d-%H:%M')+'.png')

    plt.figure()
    plt.rcParams["figure.figsize"] = (10,5)
    plt.title('model accuracy')
    plt.plot(train_acc,'r',val_acc,'g')
    plt.legend(['train','val'])
    plt.show()
    if file_name != None:
        #print(file_name+datetime.today().strftime('_ACC_%m-%d-%H:%M')+'.png',' Saved!!')
        plt.savefig('/home/yslee/CNN_Prediction_Endo/figs/'+file_name+datetime.today().strftime('_ACC_%m-%d-%H:%M')+'.png')
    '''
    # load best model weights
    model.load_state_dict(best_model_wts)
    model_later.load_state_dict(best_model_wts_later)
    
    return model,model_later

def test_model(model,dataloaders,device):
    CM=0

    model.eval()
    with torch.no_grad():
        for data in dataloaders['test']:
            images, labels, file_name = data
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images,file_name) #file_name
            preds = torch.argmax(outputs.data, 1)

            CM+=confusion_matrix(labels.cpu(), preds.cpu(),labels=[0,1])
            
        tn=CM[0][0]
        tp=CM[1][1]
        fp=CM[0][1]
        fn=CM[1][0]
        acc=np.sum(np.diag(CM)/np.sum(CM))
        
        print('\nTestset Accuracy(mean): %f %%' % (100 * acc))
        print()
        print('Confusion Matirx : ')
        print(CM)
        print('- Sensitivity : ',(tp/(tp+fn))*100)
        print('- Specificity : ',(tn/(tn+fp))*100)
        print('- Precision: ',(tp/(tp+fp))*100)
        print('- NPV: ',(tn/(tn+fn))*100)
        print()

                
    return acc, CM

##Path##
split_path='/home/NAS_mount/yslee/dataset/augmented_split5'
train_path=os.path.join(split_path,'train')
val_path=os.path.join(split_path,'val')
test_path=os.path.join(split_path,'test')

##Define dataset, dataloaders##
org_dic = np.load('/home/NAS_mount/yslee/dataset/org_dic_0817.npy',allow_pickle='TRUE').item()

label_file=pd.read_csv('/home/NAS_mount/yslee/dataset/premolar_labels_0810.csv')
label_file=label_file.set_index('PatientID_new')
label_file=label_file['Result']
#label_file=label_file['tooth_class']

'''preprocessed_mean=0.61
preprocessed_std=0.25
org_mean=0.69
org_std=0.20'''

preprocessed_mean=0.59
preprocessed_std=0.24
org_mean=0.70
org_std=0.20

data_normalization={'preprocessed': transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((preprocessed_mean,), (preprocessed_std,))
]),
    'original': transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((org_mean,), (org_std,))
])          
}

dataset={'train': DentalDataset(preprocessed_dir=train_path,org_dic=org_dic,label_file=label_file,normalize=data_normalization,augmentation=True),
            'val': DentalDataset(preprocessed_dir=val_path,org_dic=org_dic,label_file=label_file,normalize=data_normalization,augmentation=False),
            'test': DentalDataset(preprocessed_dir=test_path,org_dic=org_dic,label_file=label_file,normalize=data_normalization,augmentation=False)}

dataloaders={'train': torch.utils.data.DataLoader(dataset['train'],batch_size=30,shuffle=True),
                'val':torch.utils.data.DataLoader(dataset['val'],batch_size=30,shuffle=True),
                'test':torch.utils.data.DataLoader(dataset['test'],batch_size=30,shuffle=False)}

dataset_sizes = {'train': dataset['train'].__len__(),
                    'val':dataset['val'].__len__(),
                    'test':dataset['test'].__len__()}

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device Name : ',torch.cuda.get_device_name(device))
print('The number of devices: ',torch.cuda.device_count())
print('Dataset Sizes : ',dataset_sizes)
print()

##Define model, loss, optimizer, scheduler##
'''model=torchvision.models.resnet18(pretrained=True)
model.conv1=nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)'''
#model.fc=nn.Linear(in_features=512, out_features=3, bias=True)

#model=Ensemble()
model=Net()
model=model.to(device)
clinical_features=get_clinical_features()
#print(clinical_features)
clinical_model=ClinicalModel(model,clinical_features,device)
clinical_model=nn.DataParallel(clinical_model)
clinical_model=clinical_model.to(device)

#model=Ensemble()
#model=Net2()
#model.load_state_dict(torch.load('/home/yslee/CNN_Prediction_Endo/models/keep/fails_net_08-06-15:16.pt', map_location={"cuda" : "cpu"}))
'''model=nn.DataParallel(model)
model=model.to(device)'''

learning_rate=0.00005#0.001 #0.00001
#criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss()#weight=torch.tensor([1.,1.5],device='cuda')
optimizer = optim.Adam(clinical_model.parameters(), lr=learning_rate)#, weight_decay=0.001
#exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[25,50,75,100,120,150,200], gamma=0.5) #[20,40,50,70,80,100,120,150,200]
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
#exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

clinical_model,clinical_model_later= train_model(clinical_model, criterion, optimizer, exp_lr_scheduler, dataloaders, dataset_sizes, device, num_epochs=40, file_name='/0819/net')

torch.save(clinical_model.module.state_dict(),'/home/yslee/CNN_Prediction_Endo/models/0819/'+'net_'+datetime.today().strftime('%m-%d-%H:%M')+'.pt')
torch.save(clinical_model_later.module.state_dict(),'/home/yslee/CNN_Prediction_Endo/models/0819/'+'net_later_'+datetime.today().strftime('%m-%d-%H:%M')+'.pt')

device_cpu=torch.device('cpu')
acc,confusion_matrix1=test_model(clinical_model,dataloaders,device_cpu)
acc2,confusion_matrix2=test_model(clinical_model_later,dataloaders,device_cpu)