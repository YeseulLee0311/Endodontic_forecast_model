
import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch import optim
from torch.optim import lr_scheduler
from BackBone import ShallowResNet
from training_modules import train_model,test_model
from dental_datasets import DentalDataset,get_pm_label
from datetime import datetime

##Path##
split_path='/home/yslee/CNN_Prediction_Endo/dataset/preprocessed_split_1-153_4'
train_path=os.path.join(split_path,'train')
val_path=os.path.join(split_path,'val')
test_path=os.path.join(split_path,'test')

##Define dataset, dataloaders##
label_file=get_pm_label()

preprocessed_mean=0.61
preprocessed_std=0.25
org_mean=0.69
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
dataset={'train': DentalDataset(preprocessed_dir=train_path,org_dir=None,label_file=label_file,normalize=data_normalization,org_dic=None, augmentation=True),
            'val': DentalDataset(preprocessed_dir=val_path,org_dir=None,label_file=label_file,normalize=data_normalization,org_dic=None,augmentation=False),
            'test': DentalDataset(preprocessed_dir=test_path,org_dir=None,label_file=label_file,normalize=data_normalization,org_dic=None,augmentation=False)}

dataloaders={'train': torch.utils.data.DataLoader(dataset['train'],batch_size=10,shuffle=True),
                'val':torch.utils.data.DataLoader(dataset['val'],batch_size=10,shuffle=True),
                'test':torch.utils.data.DataLoader(dataset['test'],batch_size=10,shuffle=True)}

dataset_sizes = {'train': dataset['train'].__len__(),
                    'val':dataset['val'].__len__(),
                    'test':dataset['test'].__len__()}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('\nDevice Name : ',torch.cuda.get_device_name(device))
print('Dataset Sizes : ',dataset_sizes)
print()


##Define model, loss, optimizer, scheduler##
model=ShallowResNet()
model= model.to(device)

learning_rate=0.02
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[25,50,100,150,200], gamma=0.5)

##Train, test model##
model,model_later= train_model(model, criterion, optimizer, exp_lr_scheduler, dataloaders, dataset_sizes, device, num_epochs=100, file_name='BackBone')
acc,confusion_matrix=test_model(model,dataloaders,device)
acc2,confusion_matrix2=test_model(model_later,dataloaders,device)

torch.save(model.state_dict(),'/home/yslee/CNN_Prediction_Endo/models/'+'BackBone_'+datetime.today().strftime('%m-%d-%H:%M')+'.pt')
torch.save(model_later.state_dict(),'/home/yslee/CNN_Prediction_Endo/models/'+'BackBone_later_'+datetime.today().strftime('%m-%d-%H:%M')+'.pt')