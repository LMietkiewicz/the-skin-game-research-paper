# dataloader.py

import os
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import AutoImageProcessor

class HAM10000(Dataset):
    # Class-level label encoder to ensure consistency
    label_encoder = None
    
    def __init__(self, model_repository, metadata_path, images_dir, train=True):
        self.metadata = pd.read_csv(metadata_path, usecols=['image_id','dx'])
        self.processor = AutoImageProcessor.from_pretrained(model_repository)
        self.__image_dir = images_dir
        
        # Initialize label encoder only once for the training set
        if train and HAM10000.label_encoder is None:
            HAM10000.label_encoder = LabelEncoder()
            HAM10000.label_encoder.fit(self.metadata.dx)
        
        # Use the same label encoder for both train and test
        if HAM10000.label_encoder is not None:
            self.metadata['dx'] = HAM10000.label_encoder.transform(self.metadata.dx)
        else:
            raise ValueError("Label encoder not initialized. Create training dataset first.")
    
    @property
    def num_classes(self):
        return len(HAM10000.label_encoder.classes_)
    
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_path = os.path.join(self.__image_dir, self.metadata.iloc[idx,0] + '.jpg')
        image = read_image(img_path)
        image = image.float()
        image = self.processor(image, return_tensors="pt")
        image = image['pixel_values'].squeeze(0)
        image_id = self.metadata.iloc[idx,0]
        label = torch.tensor(self.metadata.loc[self.metadata['image_id'] == image_id, 'dx'].values[0], dtype=torch.long)
        return {"pixel_values": image, "label": label}

class Dermnet(Dataset):
    label_encoder = None
    
    def __init__(self, model_repository, base_path, train=True):
        self.__path = base_path
        dirs = os.listdir(self.__path)
        self.processor = AutoImageProcessor.from_pretrained(model_repository)
        
        # Collect all image files and labels
        image_files = []
        labels = []
        for d in dirs:
            i = os.listdir(os.path.join(self.__path, d))
            image_files.extend(i)
            labels.extend([d] * len(i))
        
        self.__h_metadata = pd.DataFrame({'image_id': image_files, 'dx': labels})
        self.metadata = self.__h_metadata.copy()
        
        # Initialize label encoder only once for the training set
        if train and Dermnet.label_encoder is None:
            Dermnet.label_encoder = LabelEncoder()
            Dermnet.label_encoder.fit(self.metadata.dx)
        
        # Use the same label encoder for both train and test
        if Dermnet.label_encoder is not None:
            self.metadata['dx'] = Dermnet.label_encoder.transform(self.metadata.dx)
        else:
            raise ValueError("Label encoder not initialized. Create training dataset first.")
            
        self.metadata['image_id'] = self.metadata['image_id'].str.replace('.jpg', '', regex=False)
    
    @property
    def num_classes(self):
        return len(Dermnet.label_encoder.classes_)
       
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_path = os.path.join(self.__path, self.__h_metadata.iloc[idx,1], self.__h_metadata.iloc[idx,0])
        image = read_image(img_path)
        image = image.float()
        image = self.processor(image, return_tensors="pt")
        image = image['pixel_values'].squeeze(0)
        image_id = self.metadata.iloc[idx,0]
        label = torch.tensor(self.metadata.loc[self.metadata['image_id'] == image_id, 'dx'].values[0], dtype=torch.long)
        return {"pixel_values": image, "label": label}
    
class IsicAtlas(Dataset):
    # Class-level label encoder to ensure consistency
    label_encoder = None
    
    def __init__(self, model_repository, base_path, train=True):
        self.__path = base_path
        dirs = os.listdir(self.__path)
        self.processor = AutoImageProcessor.from_pretrained(model_repository)
        
        # Collect all image files and labels
        image_files = []
        labels = []
        for d in dirs:
            i = os.listdir(os.path.join(self.__path, d))
            image_files.extend(i)
            labels.extend([d] * len(i))
        
        self.__h_metadata = pd.DataFrame({'image_id': image_files, 'dx': labels})
        self.metadata = self.__h_metadata.copy()
        
        # Initialize label encoder only once for the training set
        if train and IsicAtlas.label_encoder is None:
            IsicAtlas.label_encoder = LabelEncoder()
            IsicAtlas.label_encoder.fit(self.metadata.dx)
        
        # Use the same label encoder for both train and test
        if IsicAtlas.label_encoder is not None:
            self.metadata['dx'] = IsicAtlas.label_encoder.transform(self.metadata.dx)
        else:
            raise ValueError("Label encoder not initialized. Create training dataset first.")
            
        self.metadata['image_id'] = self.metadata['image_id'].str.replace('.jpg', '', regex=False)
    
    @property
    def num_classes(self):
        """Return the number of unique classes in the dataset"""
        return len(IsicAtlas.label_encoder.classes_)
    
    @property
    def class_names(self):
        """Return the list of class names"""
        return list(IsicAtlas.label_encoder.classes_)
    
    def get_label_name(self, label_id):
        """Convert a numeric label back to its string name"""
        return IsicAtlas.label_encoder.inverse_transform([label_id])[0]
    
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_path = os.path.join(self.__path, 
                               self.__h_metadata.iloc[idx,1], 
                               self.__h_metadata.iloc[idx,0])
        image = read_image(img_path)
        image = image.float()
        image = self.processor(image, return_tensors="pt")
        image = image['pixel_values'].squeeze(0)
        image_id = self.metadata.iloc[idx,0]
        label = torch.tensor(self.metadata.loc[self.metadata['image_id'] == image_id, 'dx'].values[0], 
                           dtype=torch.long)
        return {"pixel_values": image, "label": label}
    
    def get_class_distribution(self):
        """Return the distribution of classes in the dataset"""
        return self.metadata['dx'].value_counts().sort_index()
    
class Isic2024(Dataset):
    def __init__(self, model_repository, metadata_path, images_dir):
        self.metadata = pd.read_csv(metadata_path, usecols=['isic_id','target'])
        self.processor = AutoImageProcessor.from_pretrained(model_repository)
        self.__image_dir = images_dir   
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_path = os.path.join(self.__image_dir, self.metadata.iloc[idx,0] + '.jpg')
        image = read_image(img_path)
        image = image.float()
        image = self.processor(image, return_tensors="pt")
        image = image['pixel_values'].squeeze(0)
        image_id = self.metadata.iloc[idx,0]
        label = torch.tensor(self.metadata.loc[self.metadata['isic_id'] == image_id, 'target'].values[0], dtype=torch.long)
        return {"pixel_values": image, "label": label}