import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import os
from PIL import Image
import random
import pandas as pd
import numpy as np

class DentalDataset(Dataset):
    def __init__(self,preprocessed_dir,org_dic,label_file,normalize,augmentation=False,grad_dic=None,grad_dic2=None):
        self.preprocessed_dir=preprocessed_dir
        #self.org_dir=org_dir
        self.label_file=label_file
        self.augmentation=augmentation
        self.normalize=normalize
        self.org_dic=org_dic
        self.grad_dic=grad_dic
        self.grad_dic2=grad_dic2

        if self.preprocessed_dir is not None:
            self.preprocessed_file_list=[]
            for file in os.listdir(self.preprocessed_dir):
                if file.endswith('.bmp'):
                    self.preprocessed_file_list.append(file)
            self.preprocessed_file_list.sort()
            
        if self.org_dic is not None:
            self.org_file_list=list(org_dic.keys())
            '''
            self.org_file_list=[]
            for file in os.listdir(self.org_dir):
                if file.endswith('.dcm'):
                    self.org_file_list.append(file)
            self.org_file_list.sort()'''
            
    def __len__(self):
        if self.preprocessed_dir is not None:
            return len(self.preprocessed_file_list)
        else:
            return len(self.org_file_list)
    
    def __getitem__(self, idx):

        if self.preprocessed_dir is not None and self.org_dic is not None and self.grad_dic is not None and self.grad_dic2 is not None:
            #get preprocessed file name
            preprocessed_file=self.preprocessed_file_list[idx]
            patID=preprocessed_file.split('_')[0]
            file_name,ext=os.path.splitext(preprocessed_file)

            #get original file name which is same with preprocessed file name
            '''
            org_file=None
            for f in self.org_file_list:
                if f.startswith(patID):
                    org_file=f
            '''
            
            #raise exception if there's no matched original file
            '''if not org_file:
                raise FileNotFoundError('Original File Not Found')'''
                
            #get preprocessed image
            preprocessed_img=Image.open(os.path.join(self.preprocessed_dir,preprocessed_file))
            org_img=self.org_dic[file_name]
            grad_img=self.grad_dic[file_name]
            grad_img2=self.grad_dic[file_name]


            '''
            #get original image
            org_img=pydicom.read_file(os.path.join(self.org_dir,org_file))
            org_img=org_img.pixel_array
            #scale original image
            maxm=org_img.max()
            minm=org_img.min()
            for i in range(0,len(org_img)):
                for j in range(0,len(org_img[i])):
                    org_img[i][j]=(org_img[i][j]-minm)/(maxm-minm)*255
            org_img=org_img.astype(np.uint8)
            org_img=Image.fromarray(org_img)
            '''

            #image transformation   
            #data augmentation
            if self.augmentation:
                #random horizontal, vertical flip
                if random.random()>0.5:
                    preprocessed_img=TF.hflip(preprocessed_img)
                    org_img=TF.hflip(org_img)
                    grad_img=TF.hflip(grad_img)
                    grad_img2=TF.hflip(grad_img2)
                if random.random() > 0.5:
                    preprocessed_img = TF.vflip(preprocessed_img)
                    org_img = TF.vflip(org_img)
                    grad_img=TF.vflip(grad_img)
                    grad_img2=TF.vflip(grad_img2)

                '''
                if random.random() > 0.5:
                    factor=round(random.uniform(0.5,1.2),2)
                    preprocessed_img=TF.adjust_brightness(preprocessed_img,factor)
                    org_img=TF.adjust_brightness(org_img,factor)
                if random.random() > 0.5:
                    factor=round(random.uniform(1,1.7),2)
                    preprocessed_img=TF.adjust_contrast(preprocessed_img,factor)
                    org_img=TF.adjust_contrast(org_img,factor)
                '''
                '''
                if random.random() > 0.5:
                    angle=np.random.randint(-30,30)
                    preprocessed_img=TF.rotate(preprocessed_img,angle,expand=False)
                '''
            
            #resize
            resize=transforms.Resize((600,600))
            preprocessed_img=resize(preprocessed_img)
            org_img=resize(org_img)
            grad_img=resize(grad_img)
            grad_img2=resize(grad_img2)
            
            preprocessed_img=self.normalize['preprocessed'](preprocessed_img)
            org_img=self.normalize['original'](org_img)
            grad_img=self.normalize['grad'](grad_img)
            grad_img2=self.normalize['grad2'](grad_img2)

            
            '''
            if self.transform:
                preprocessed_img = self.transform(preprocessed_img)
                preprocessed_img=self.normalize['preprocessed'](preprocessed_img)
                org_img = self.transform(org_img)
                org_img=self.normalize['original'](org_img)
            else:
                preprocessed_img=self.normalize['preprocessed'](preprocessed_img)
                org_img=self.normalize['original'](org_img)
            '''

            #get label of the image
            label=self.label_file.loc[patID]
            
            #final sample to return 
            sample=(torch.cat((preprocessed_img,org_img,grad_img,grad_img2),0),int(label),file_name)

        ###Preprocessed Image + Original Image + Grad Image###
        elif self.preprocessed_dir is not None and self.org_dic is not None and self.grad_dic is not None:
            #get preprocessed file name
            preprocessed_file=self.preprocessed_file_list[idx]
            patID=preprocessed_file.split('_')[0]
            file_name,ext=os.path.splitext(preprocessed_file)

            #get original file name which is same with preprocessed file name
            '''
            org_file=None
            for f in self.org_file_list:
                if f.startswith(patID):
                    org_file=f
            '''
            
            #raise exception if there's no matched original file
            '''if not org_file:
                raise FileNotFoundError('Original File Not Found')'''
                
            #get preprocessed image
            preprocessed_img=Image.open(os.path.join(self.preprocessed_dir,preprocessed_file))
            org_img=self.org_dic[file_name]
            grad_img=self.grad_dic[file_name]


            '''
            #get original image
            org_img=pydicom.read_file(os.path.join(self.org_dir,org_file))
            org_img=org_img.pixel_array
            #scale original image
            maxm=org_img.max()
            minm=org_img.min()
            for i in range(0,len(org_img)):
                for j in range(0,len(org_img[i])):
                    org_img[i][j]=(org_img[i][j]-minm)/(maxm-minm)*255
            org_img=org_img.astype(np.uint8)
            org_img=Image.fromarray(org_img)
            '''

            #image transformation   
            #data augmentation
            if self.augmentation:
                #random horizontal, vertical flip
                if random.random()>0.5:
                    preprocessed_img=TF.hflip(preprocessed_img)
                    org_img=TF.hflip(org_img)
                    grad_img=TF.hflip(grad_img)
                if random.random() > 0.5:
                    preprocessed_img = TF.vflip(preprocessed_img)
                    org_img = TF.vflip(org_img)
                    grad_img=TF.vflip(grad_img)

                '''
                if random.random() > 0.5:
                    factor=round(random.uniform(0.5,1.2),2)
                    preprocessed_img=TF.adjust_brightness(preprocessed_img,factor)
                    org_img=TF.adjust_brightness(org_img,factor)
                if random.random() > 0.5:
                    factor=round(random.uniform(1,1.7),2)
                    preprocessed_img=TF.adjust_contrast(preprocessed_img,factor)
                    org_img=TF.adjust_contrast(org_img,factor)
                '''
                '''
                if random.random() > 0.5:
                    angle=np.random.randint(-30,30)
                    preprocessed_img=TF.rotate(preprocessed_img,angle,expand=False)
                '''
            
            #resize
            resize=transforms.Resize((600,600))
            preprocessed_img=resize(preprocessed_img)
            org_img=resize(org_img)
            grad_img=resize(grad_img)
            
            preprocessed_img=self.normalize['preprocessed'](preprocessed_img)
            org_img=self.normalize['original'](org_img)
            grad_img=self.normalize['grad'](grad_img)

            
            '''
            if self.transform:
                preprocessed_img = self.transform(preprocessed_img)
                preprocessed_img=self.normalize['preprocessed'](preprocessed_img)
                org_img = self.transform(org_img)
                org_img=self.normalize['original'](org_img)
            else:
                preprocessed_img=self.normalize['preprocessed'](preprocessed_img)
                org_img=self.normalize['original'](org_img)
            '''

            #get label of the image
            label=self.label_file.loc[patID]
            
            #final sample to return 
            sample=(torch.cat((preprocessed_img,org_img,grad_img),0),int(label),file_name)
        
        ###Preprocessed Image + Original Image###
        elif self.preprocessed_dir is not None and self.org_dic is not None:
            #get preprocessed file name
            preprocessed_file=self.preprocessed_file_list[idx]
            patID=preprocessed_file.split('_')[0]
            file_name,ext=os.path.splitext(preprocessed_file)

            #get original file name which is same with preprocessed file name
            '''
            org_file=None
            for f in self.org_file_list:
                if f.startswith(patID):
                    org_file=f
            '''
            
            #raise exception if there's no matched original file
            '''if not org_file:
                raise FileNotFoundError('Original File Not Found')'''
                
            #get preprocessed image
            preprocessed_img=Image.open(os.path.join(self.preprocessed_dir,preprocessed_file))
            org_img=self.org_dic[file_name]


            '''
            #get original image
            org_img=pydicom.read_file(os.path.join(self.org_dir,org_file))
            org_img=org_img.pixel_array
            #scale original image
            maxm=org_img.max()
            minm=org_img.min()
            for i in range(0,len(org_img)):
                for j in range(0,len(org_img[i])):
                    org_img[i][j]=(org_img[i][j]-minm)/(maxm-minm)*255
            org_img=org_img.astype(np.uint8)
            org_img=Image.fromarray(org_img)
            '''

            #image transformation   
            #data augmentation
            if self.augmentation:
                #random horizontal, vertical flip
                if random.random()>0.5:
                    preprocessed_img=TF.hflip(preprocessed_img)
                    org_img=TF.hflip(org_img)
                if random.random() > 0.5:
                    preprocessed_img = TF.vflip(preprocessed_img)
                    org_img = TF.vflip(org_img)

                '''
                if random.random() > 0.5:
                    factor=round(random.uniform(0.5,1.2),2)
                    preprocessed_img=TF.adjust_brightness(preprocessed_img,factor)
                    org_img=TF.adjust_brightness(org_img,factor)
                if random.random() > 0.5:
                    factor=round(random.uniform(1,1.7),2)
                    preprocessed_img=TF.adjust_contrast(preprocessed_img,factor)
                    org_img=TF.adjust_contrast(org_img,factor)
                '''
                '''
                if random.random() > 0.5:
                    angle=np.random.randint(-30,30)
                    preprocessed_img=TF.rotate(preprocessed_img,angle,expand=False)
                '''
            
            #resize
            resize=transforms.Resize((600,600))
            preprocessed_img=resize(preprocessed_img)
            org_img=resize(org_img)
            
            preprocessed_img=self.normalize['preprocessed'](preprocessed_img)
            org_img=self.normalize['original'](org_img)
            
            '''
            if self.transform:
                preprocessed_img = self.transform(preprocessed_img)
                preprocessed_img=self.normalize['preprocessed'](preprocessed_img)
                org_img = self.transform(org_img)
                org_img=self.normalize['original'](org_img)
            else:
                preprocessed_img=self.normalize['preprocessed'](preprocessed_img)
                org_img=self.normalize['original'](org_img)
            '''

            #get label of the image
            patID=patID.split('.')[0]
            patID=patID.split('(')[0]
            
            label=self.label_file.loc[patID]
            

            #final sample to return 
            sample=(torch.cat((preprocessed_img,org_img),0),int(label),file_name)
            
        ### Preprocessed Image Only ### 
        elif self.preprocessed_dir is not None and self.org_dic is None:
            #get preprocessed file and image
            preprocessed_file=self.preprocessed_file_list[idx]
            preprocessed_img=Image.open(os.path.join(self.preprocessed_dir,preprocessed_file))
      
            #image transformation
            #resize
            #data augmentation
            if self.augmentation:
                #random horizontal, vertical flip
                if random.random() > 0.5:
                    preprocessed_img=TF.hflip(preprocessed_img)
                if random.random() > 0.5:
                    preprocessed_img = TF.vflip(preprocessed_img)

                '''
                if random.random() > 0.5:
                    angle=np.random.randint(-30,30)
                    preprocessed_img=TF.rotate(preprocessed_img,angle,expand=False)
                '''
            
            resize=transforms.Resize((600,600))
            preprocessed_img=resize(preprocessed_img)
            
            preprocessed_img=self.normalize['preprocessed'](preprocessed_img)
            
            '''
            if self.transform:
                preprocessed_img = self.transform(preprocessed_img)
                preprocessed_img=self.normalize['preprocessed'](preprocessed_img)
            else:
                preprocessed_img=self.normalize['preprocessed'](preprocessed_img)
            
            '''
            #get label of the image
            patID=preprocessed_file.split('_')[0]
            patID=patID.split('.')[0]
            patID=patID.split('(')[0]
            
            label=self.label_file.loc[patID]

            #final sample to return 
            sample=(preprocessed_img,int(label),patID)
        
        ### Original Image Only ###
        elif self.preprocessed_dir is None and self.org_dic is not None:
            #get original file and image
            org_file=self.org_file_list[idx]
            org_img=self.org_dic[org_file]

            '''
            org_img=pydicom.read_file(os.path.join(self.org_dir,org_file))
            org_img=org_img.pixel_array
            
            maxm=org_img.max()
            minm=org_img.min()
            for i in range(0,len(org_img)):
                for j in range(0,len(org_img[i])):
                    org_img[i][j]=(org_img[i][j]-minm)/(maxm-minm)*255
            org_img=org_img.astype(np.uint8)
            org_img=Image.fromarray(org_img)
            '''
            
            #image transformation
            #resize
            resize=transforms.Resize((600,600))
            org_img=resize(org_img)
            
            #data augmentation
            if self.augmentation:
                #random horizontal, vertical flip
                if random.random()>0.5:
                    org_img=TF.hflip(org_img)
                if random.random() > 0.5:
                    org_img = TF.vflip(org_img)

            org_img=self.normalize['original'](org_img)
            
            '''
            if self.transform:
                org_img=self.transform(org_img)
                org_img=self.normalize['original'](org_img)
                
            else:
                org_img=self.normalize['original'](org_img)
            '''
            #get label of the image
            patID=org_file.split('_')[0]
            patID=patID.split('.')[0]
            patID=patID.split('(')[0]
            
            label=self.label_file.loc[patID]
            
            #final sample to return
            sample=(org_img,int(label.values),patID)
        
        #Raise Exception If Both Directories are None
        else:
            raise DirectoryNotFoundError('Directory Not Found')

        
        return sample

'''
def get_pm_label(csv_path='/home/yslee/CNN_Prediction_Endo/dataset/label_1-153.csv'):
    df=pd.read_csv(csv_path,sep=',',header=None,encoding = "CP949")
    df=df.iloc[:,:2]
    df.columns=['PatientID','Prognosis']
    df=df.set_index('PatientID')
    df=df['Prognosis']
    
    prognosis_dic={}
    for patID in df.index:
        if df.loc[patID]=='Success' or df.loc[patID]=='SUCCESS':
            prognosis_dic[patID]=0
        elif df.loc[patID]=='FAIL' or df.loc[patID]=='Fail':
            prognosis_dic[patID]=1
        else:
            print(patID, ' : labeling failed!!')
    label=pd.DataFrame.from_dict(prognosis_dic,orient='index',columns=['label'])
    
    return label

def get_inc_label(csv_path='/home/yslee/CNN_Prediction_Endo/dataset/inc_1-48.csv'):
    df=pd.read_csv(csv_path,sep=',',header=0,encoding = "CP949")
    for i in range(0,len(df.iloc[:,0])):
        df.iloc[i,0]=df.iloc[i,0].replace('_','-')
        
    df=df.set_index(df.iloc[:,0])
    df=df.iloc[:,11]

    prognosis_dic={}
    for patID in df.index:
        if df.loc[patID]=='Success' or df.loc[patID]=='SUCCESS':
            prognosis_dic[patID]=0
        elif df.loc[patID]=='FAIL' or df.loc[patID]=='Fail':
            prognosis_dic[patID]=1
        else:
            print(patID, ' : labeling failed!!')
    label=pd.DataFrame.from_dict(prognosis_dic,orient='index',columns=['label'])
    
    return label
'''