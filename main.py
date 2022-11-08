import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from training_modules import train_model,test_model,test_model2,get_clinical_features,train_3model2
from dental_datasets import DentalDataset
from torch import optim
from torch.optim import lr_scheduler
from datetime import datetime
import pandas as pd
from Attention_ResNet import Ensemble, Net, Net2,ClinicalModel, ResNet18
import numpy as np
import pandas as pd

##Path##
split_path='/home/NAS_mount/yslee/dataset/crossval_dataset2-1103flip'
#split_path='/home/NAS_mount/yslee/dataset/augmented_split-flip2' #'/home/yslee/CNN_Prediction_Endo/dataset/pm_1-682_exp1424_split'
#split_path='/home/NAS_mount/yslee/dataset/crossval_dataset1-flip'
train_path=os.path.join(split_path,'train')
val_path=os.path.join(split_path,'val')
test_path=os.path.join(split_path,'test')

##Define dataset, dataloaders##
#org_dic = np.load('/home/NAS_mount/yslee/dataset/org_dic_0817.npy',allow_pickle='TRUE').item()
#org_dic = np.load('/home/NAS_mount/yslee/dataset/org_dic_flip2.npy',allow_pickle='TRUE').item()
org_dic = np.load('/home/NAS_mount/yslee/dataset/org_dic_1101.npy',allow_pickle='TRUE').item()
#org_dic = np.load('/home/NAS_mount/yslee/dataset/org_dic_1424.npy',allow_pickle='TRUE').item()
#grad_dic = np.load('/home/NAS_mount/yslee/dataset/PFgrad_dic_1019.npy',allow_pickle='TRUE').item()
#grad_dic = np.load('/home/NAS_mount/yslee/dataset/PFgrad_th_dic2.npy',allow_pickle='TRUE').item()
#grad_dic2 = np.load('/home/NAS_mount/yslee/dataset/STgrad_dic_1019.npy',allow_pickle='TRUE').item()

label_file=pd.read_csv('/home/NAS_mount/yslee/dataset/premolar_labels_0810.csv')
#label_file=pd.read_csv('/home/NAS_mount/yslee/dataset/labels_1424.csv')
#label_file=label_file.set_index('PatientID')
label_file=label_file.set_index('PatientID_new')
#label_file=get_clinical_features()
#label_file=label_file['PA']
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
'''grad_mean=0.36
grad_std=0.22'''
'''grad_mean=0.37
grad_std=0.48'''
'''grad_mean2=0.48
grad_std2=0.19'''
'''grad_mean2=0.37
grad_std2=0.48'''
#maskPF, maskST
grad_mean=0.35
grad_std=0.48
grad_mean2=0.54
grad_std2=0.50

data_normalization={'preprocessed': transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((preprocessed_mean,), (preprocessed_std,))
]),
    'original': transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((org_mean,), (org_std,))
]),
    'grad':transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((grad_mean,), (grad_std,))
]),
    'grad2':transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((grad_mean2,), (grad_std2,))
])          
}

dataset={'train': DentalDataset(preprocessed_dir=train_path,org_dic=org_dic,label_file=label_file,normalize=data_normalization,augmentation=True,grad_dic=None,grad_dic2=None),
            'val': DentalDataset(preprocessed_dir=val_path,org_dic=org_dic,label_file=label_file,normalize=data_normalization,augmentation=False,grad_dic=None,grad_dic2=None),
            'test': DentalDataset(preprocessed_dir=test_path,org_dic=org_dic,label_file=label_file,normalize=data_normalization,augmentation=False,grad_dic=None,grad_dic2=None)}

dataloaders={'train': torch.utils.data.DataLoader(dataset['train'],batch_size=30,shuffle=True),
                'val':torch.utils.data.DataLoader(dataset['val'],batch_size=30,shuffle=True),
                'test':torch.utils.data.DataLoader(dataset['test'],batch_size=30,shuffle=False)}

dataset_sizes = {'train': dataset['train'].__len__(),
                    'val':dataset['val'].__len__(),
                    'test':dataset['test'].__len__()}

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device Name : ',torch.cuda.get_device_name(device))
print('The number of devices: ',torch.cuda.device_count())
print('Dataset Sizes : ',dataset_sizes)
print()

##Define model, loss, optimizer, scheduler##
'''model=torchvision.models.resnet18(pretrained=False)
model.conv1=nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc=nn.Linear(in_features=512, out_features=2, bias=True)'''

model=Net2()
#model.load_state_dict(torch.load('/home/yslee/CNN_Prediction_Endo/models/0923/Net17_later_09-25-21:35.pt', map_location={"cuda" : "cpu"}))
#model.load_state_dict(torch.load('/home/yslee/CNN_Prediction_Endo/models/keep/net_08-27-22:31.pt', map_location={"cuda" : "cpu"}))
#model=nn.DataParallel(model)
model=model.to(device)

learning_rate=0.0001 #0.0001 #0.001 #0.00001 #feature classification: 0.00005
#criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss()#weight=torch.tensor([1.,1.8],device='cuda')
optimizer = optim.Adam(model.parameters(), lr=learning_rate) #,weight_decay=0.001
#exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[25,50,75,100,120,150,200], gamma=0.5) #[20,40,50,70,80,100,120,150,200]
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
#exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

accf1model, accf1model_later, accmodel, accmodel_later, f1model, f1model_later= train_3model2(model, criterion, optimizer, exp_lr_scheduler, dataloaders, dataset_sizes, device, num_epochs=50, file_name='1101/Atten2')

torch.save(accf1model.module.state_dict(),'/home/NAS_mount/yslee/dataset/models/1101/'+'accf1Atten2_'+datetime.today().strftime('%m-%d-%H:%M')+'.pt')
torch.save(accf1model_later.module.state_dict(),'/home/NAS_mount/yslee/dataset/models/1101/'+'accf1Atten2_later_'+datetime.today().strftime('%m-%d-%H:%M')+'.pt')

torch.save(accmodel.module.state_dict(),'/home/NAS_mount/yslee/dataset/models/1101/'+'accAtten2_'+datetime.today().strftime('%m-%d-%H:%M')+'.pt')
torch.save(accmodel_later.module.state_dict(),'/home/NAS_mount/yslee/dataset/models/1101/'+'accAtten2_later_'+datetime.today().strftime('%m-%d-%H:%M')+'.pt')

torch.save(f1model.module.state_dict(),'/home/NAS_mount/yslee/dataset/models/1101/'+'f1Atten2_'+datetime.today().strftime('%m-%d-%H:%M')+'.pt')
torch.save(f1model_later.module.state_dict(),'/home/NAS_mount/yslee/dataset/models/1101/'+'f1Atten2_later_'+datetime.today().strftime('%m-%d-%H:%M')+'.pt')

device_cpu=torch.device('cpu')

acc5,confusion_matrix5=test_model(accf1model,dataloaders,device_cpu)
acc6,confusion_matrix6=test_model(accf1model_later,dataloaders,device_cpu)

acc1,confusion_matrix1=test_model(accmodel,dataloaders,device_cpu)
acc2,confusion_matrix2=test_model(accmodel_later,dataloaders,device_cpu)

acc3,confusion_matrix3=test_model(f1model,dataloaders,device_cpu)
acc4,confusion_matrix4=test_model(f1model_later,dataloaders,device_cpu)



