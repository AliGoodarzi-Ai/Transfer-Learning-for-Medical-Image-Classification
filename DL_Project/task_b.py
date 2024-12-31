import copy
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

import torch.optim as optim
from torch.optim import lr_scheduler 

# For balancing the dataset
from collections import Counter
from sklearn.utils import resample

# Hyper Parameters
batch_size = 24
num_classes = 5  # 5 DR levels
learning_rate = 0.0001
num_epochs = 15

# Base directory - update this to your project root directory
BASE_DIR = r"C:\Users\Ali Goodarzi\Desktop\deep learning\deep project\aptos"

# Paths using os.path.join for better compatibility
PATHS = {
    'train_images': os.path.join(BASE_DIR, 'train_images'),
    'val_images': os.path.join(BASE_DIR, 'val_images'),
    'test_images': os.path.join(BASE_DIR, 'test_images'),
    'train_csv': os.path.join(BASE_DIR, 'train_1.csv'),
    'val_csv': os.path.join(BASE_DIR, 'valid.csv'),
    'test_csv': os.path.join(BASE_DIR, 'test.csv')
}


# Channel Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Average pooling pathway
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # Max pooling pathway
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # Combine and apply sigmoid
        out = avg_out + max_out
        return self.sigmoid(out)

# Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compute average and max projections
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate along the channel dimension
        x = torch.cat([avg_out, max_out], dim=1)
        # Convolution and sigmoid activation
        x = self.conv1(x)
        return self.sigmoid(x)
    

# Resnet34 Single
class MyModel_resnet34(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()

        self.backbone = models.resnet34(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove the original classification layer

                # Add Attention Modules
        self.channel_attention = ChannelAttention(in_planes=512)  # 512: feature map channels in ResNet34
        self.spatial_attention = SpatialAttention()

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

# Resnet34 Dual
class MyDualModel_resnet34(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()

        backbone = models.resnet34(pretrained=True)
        backbone.fc = nn.Identity()  # Remove the original classification layer

        # Two separate ResNet34 backbones with unshared weights
        self.backbone1 = copy.deepcopy(backbone)
        self.backbone2 = copy.deepcopy(backbone)

                # Attention Modules
        self.channel_attention1 = ChannelAttention(in_planes=512)
        self.spatial_attention1 = SpatialAttention()
        self.channel_attention2 = ChannelAttention(in_planes=512)
        self.spatial_attention2 = SpatialAttention()

        self.fc = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, images):
        image1, image2 = images

        x1 = self.backbone1(image1)
        x2 = self.backbone2(image2)

        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x

# EfficientNet-B0 Single
class MyModel_EfficientNet(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()

        self.backbone = models.efficientnet_b0(pretrained=True)
        self.backbone.classifier = nn.Identity()  # Remove the original classification layer

                # Add Attention Modules
        self.channel_attention = ChannelAttention(in_planes=1280)  # 1280: feature map channels in EfficientNet-B0
        self.spatial_attention = SpatialAttention()

        self.fc = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

# EfficientNet-B0 Dual
class MyDualModel_EfficientNet(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()

        backbone = models.efficientnet_b0(pretrained=True)
        backbone.classifier = nn.Identity()  # Remove the original classification layer

        self.backbone1 = copy.deepcopy(backbone)
        self.backbone2 = copy.deepcopy(backbone)

                # Attention Mechanisms for each branch
        self.channel_attention1 = ChannelAttention(1024)
        self.spatial_attention1 = SpatialAttention()

        self.channel_attention2 = ChannelAttention(1024)
        self.spatial_attention2 = SpatialAttention()

        self.fc = nn.Sequential(
            nn.Linear(1280 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, images):
        image1, image2 = images

        x1 = self.backbone1(image1)
        x2 = self.backbone2(image2)

        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x


# VGG16 Single
class MyModel_VGG(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()

        self.backbone = models.vgg16(pretrained=True)
        self.backbone.classifier = nn.Identity()  # Remove the original classifier

        #attention
        self.channel_attention = ChannelAttention(512)  # For VGG16, 512 is the number of channels
        self.spatial_attention = SpatialAttention()

        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# VGG16 Dual
class MyDualModel_VGG(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()

        backbone = models.vgg16(pretrained=True)
        backbone.classifier = nn.Identity()  # Remove the original classifier

        self.backbone1 = copy.deepcopy(backbone)
        self.backbone2 = copy.deepcopy(backbone)

        #attention
        self.channel_attention1 = ChannelAttention(512)  # For VGG16, 512 is the number of channels
        self.spatial_attention1 = SpatialAttention()

        self.channel_attention2 = ChannelAttention(512)
        self.spatial_attention2 = SpatialAttention()


        

        self.fc = nn.Sequential(
            nn.Linear((512 * 7 * 7) * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, images):
        image1, image2 = images

        x1 = self.backbone1.features(image1)
        x2 = self.backbone2.features(image2)

        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)

        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x
    

# DenseNet121 single
class MyModel_DenseNet121(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()

        self.backbone = models.densenet121(pretrained=True)
        self.backbone.classifier = nn.Identity()  # Remove the original classification layer


                # Attention Modules
        self.channel_attention = ChannelAttention(in_planes=1024)
        self.spatial_attention = SpatialAttention()

        self.fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

# DenseNet121 dual
class MyDualModel_DenseNet121(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()

        backbone = models.densenet121(pretrained=True)
        backbone.classifier = nn.Identity()  # Remove the original classification layer

        self.backbone1 = copy.deepcopy(backbone)
        self.backbone2 = copy.deepcopy(backbone)


        self.channel_attention1 = ChannelAttention(1024)
        self.spatial_attention1 = SpatialAttention()
        self.channel_attention2 = ChannelAttention(1024)
        self.spatial_attention2 = SpatialAttention()


        self.fc = nn.Sequential(
            nn.Linear(1024 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, images):
        image1, image2 = images

        x1 = self.backbone1(image1)
        x2 = self.backbone2(image2)

        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x
    

class MyModel_DenseNet161(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()

        self.backbone = models.densenet161(pretrained=True)
        self.backbone.classifier = nn.Identity()  # Remove the original classification layer

        self.channel_attention1 = ChannelAttention(2208)
        self.spatial_attention1 = SpatialAttention()


        self.fc = nn.Sequential(
            nn.Linear(2208, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x


class MyDualModel_DenseNet161(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()

        backbone = models.densenet161(pretrained=True)
        backbone.classifier = nn.Identity()  # Remove the original classification layer

        self.backbone1 = copy.deepcopy(backbone)
        self.backbone2 = copy.deepcopy(backbone)

        self.channel_attention1 = ChannelAttention(2208)
        self.spatial_attention1 = SpatialAttention()
        self.channel_attention2 = ChannelAttention(2208)
        self.spatial_attention2 = SpatialAttention()



        self.fc = nn.Sequential(
            nn.Linear(2208 * 2, 256),  # Concatenate features from both backbones
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, images):
        image1, image2 = images

        x1 = self.backbone1(image1)
        x2 = self.backbone2(image2)

        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x

# Data processing with Balancing
class RetinopathyDataset_Oversample(Dataset):
    def __init__(self, ann_file, image_dir, transform=None, mode='single', test=False, oversample=True):
        self.ann_file = ann_file
        self.image_dir = image_dir
        self.transform = transform
        self.test = test
        self.mode = mode
        self.oversample = oversample

        if self.mode == 'single':
            self.data = self.load_data()
        else:
            self.data = self.load_data_dual()

        if self.oversample and not self.test:
            self.data = self.oversample_data()

    def __getitem__(self, index):
        if self.mode == 'single':
            return self.get_item(index)
        else:
            return self.get_item_dual(index)

    def load_data(self):
        df = pd.read_csv(self.ann_file)
        df['patient_id'] = df['img_path'].str.split('_').str[0]
        
        data = []
        for _, row in df.iterrows():
            file_info = dict()
            file_info['img_path'] = os.path.join(self.image_dir, row['img_path'])
            file_info['patient_id'] = row['patient_id']
            if not self.test:
                file_info['dr_level'] = int(row['patient_DR_Level'])
            data.append(file_info)
        return data

    def oversample_data(self):
        # Group by patient_id and find the class distribution
        grouped = {}
        for item in self.data:
            patient_id = item['patient_id']
            if patient_id not in grouped:
                grouped[patient_id] = []
            grouped[patient_id].append(item)

        patient_classes = {k: v[0]['dr_level'] for k, v in grouped.items()}
        class_counts = Counter(patient_classes.values())
        max_count = max(class_counts.values())

        # Oversample patients
        oversampled_data = []
        for cls, count in class_counts.items():
            patients_in_class = [v for k, v in grouped.items() if patient_classes[k] == cls]
            oversampled_patients = resample(
                patients_in_class,
                replace=True,
                n_samples=max_count,
                random_state=42
            )
            oversampled_data.extend([item for sublist in oversampled_patients for item in sublist])

        return oversampled_data
    
    def __len__(self):
        return len(self.data)

    def get_item(self, index):
        data = self.data[index]
        img = Image.open(data['img_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)

        if not self.test:
            label = torch.tensor(data['dr_level'], dtype=torch.int64)
            return img, label
        else:
            return img

    def load_data_dual(self):
        df = pd.read_csv(self.ann_file)
        df['patient_id'] = df['image_id'].str.split('_').str[0]
        df['suffix'] = df['image_id'].str.split('_').str[1].str[0]

        grouped = df.groupby(['patient_id', 'suffix'])

        data = []
        for (patient_id, suffix), group in grouped:
            if len(group) == 2:
                file_info = dict()
                file_info['patient_id'] = patient_id
                file_info['img_path1'] = os.path.join(self.image_dir, group.iloc[0]['img_path'])
                file_info['img_path2'] = os.path.join(self.image_dir, group.iloc[1]['img_path'])
                if not self.test:
                    file_info['dr_level'] = int(group.iloc[0]['patient_DR_Level'])
                data.append(file_info)

        if self.oversample and not self.test:
            data = self.oversample_data_dual(data)

        return data

    def oversample_data_dual(self, data):
        # Group by patient_id and find the class distribution
        grouped = {}
        for item in data:
            patient_id = item['patient_id']
            if patient_id not in grouped:
                grouped[patient_id] = []
            grouped[patient_id].append(item)

        patient_classes = {k: v[0]['dr_level'] for k, v in grouped.items()}
        class_counts = Counter(patient_classes.values())
        max_count = max(class_counts.values())

        # Oversample patients
        oversampled_data = []
        for cls, count in class_counts.items():
            patients_in_class = [v for k, v in grouped.items() if patient_classes[k] == cls]
            oversampled_patients = resample(
                patients_in_class,
                replace=True,
                n_samples=max_count,
                random_state=42
            )
            oversampled_data.extend([item for sublist in oversampled_patients for item in sublist])

        return oversampled_data

    def get_item_dual(self, index):
        data = self.data[index]
        img1 = Image.open(data['img_path1']).convert('RGB')
        img2 = Image.open(data['img_path2']).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        if not self.test:
            label = torch.tensor(data['dr_level'], dtype=torch.int64)
            return [img1, img2], label
        else:
            return [img1, img2]

class RetinopathyDataset_APTOS(Dataset):
    def __init__(self, csv_path, image_dir, transform=None, test=False):
        self.image_dir = image_dir
        self.transform = transform
        self.test = test

        # Read CSV and handle missing values
        self.data = pd.read_csv(csv_path)
        print(f"Loading data from {csv_path}")
        print(f"Columns in CSV: {self.data.columns.tolist()}")

        if 'idid_code' in self.data.columns:
            self.data = self.data.rename(columns={'idid_code': 'id_code'})

        if not os.path.exists(image_dir):
            raise ValueError(f"Image directory not found: {image_dir}")

        available_images = set(os.listdir(image_dir))
        print(f"Found {len(available_images)} files in image directory")

        valid_data = []
        for idx, row in self.data.iterrows():
            img_id = row['id_code']
            img_filename = f"{img_id}.png"
            if img_filename in available_images:
                valid_data.append(row)

        self.data = pd.DataFrame(valid_data)
        print(f"Found {len(self.data)} valid images with matching CSV entries")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_id = self.data.iloc[idx]['id_code']
        img_path = os.path.join(self.image_dir, f"{img_id}.png")

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        if not self.test:
            label = int(self.data.iloc[idx]['diagnosis'])
            return image, label
        return image, img_id


class RetinopathyDataset(Dataset):
    def __init__(self, ann_file, image_dir, transform=None, mode='single', test=False):
        self.ann_file = ann_file
        self.image_dir = image_dir
        self.transform = transform

        self.test = test
        self.mode = mode

        if self.mode == 'single':
            self.data = self.load_data()
        else:
            self.data = self.load_data_dual()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.mode == 'single':
            return self.get_item(index)
        else:
            return self.get_item_dual(index)

    # 1. single image
    def load_data(self):
        df = pd.read_csv(self.ann_file)

        data = []
        for _, row in df.iterrows():
            file_info = dict()
            file_info['img_path'] = os.path.join(self.image_dir, row['img_path'])
            if not self.test:
                file_info['dr_level'] = int(row['patient_DR_Level'])
            data.append(file_info)
        return data

    def get_item(self, index):
        data = self.data[index]
        img = Image.open(data['img_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)

        if not self.test:
            label = torch.tensor(data['dr_level'], dtype=torch.int64)
            return img, label
        else:
            return img

    # 2. dual image
    def load_data_dual(self):
        df = pd.read_csv(self.ann_file)

        df['prefix'] = df['image_id'].str.split('_').str[0]  # The patient id of each image
        df['suffix'] = df['image_id'].str.split('_').str[1].str[0]  # The left or right eye
        grouped = df.groupby(['prefix', 'suffix'])

        data = []
        for (prefix, suffix), group in grouped:
            file_info = dict()
            file_info['img_path1'] = os.path.join(self.image_dir, group.iloc[0]['img_path'])
            file_info['img_path2'] = os.path.join(self.image_dir, group.iloc[1]['img_path'])
            if not self.test:
                file_info['dr_level'] = int(group.iloc[0]['patient_DR_Level'])
            data.append(file_info)
        return data

    def get_item_dual(self, index):
        data = self.data[index]
        img1 = Image.open(data['img_path1']).convert('RGB')
        img2 = Image.open(data['img_path2']).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        if not self.test:
            label = torch.tensor(data['dr_level'], dtype=torch.int64)
            return [img1, img2], label
        else:
            return [img1, img2]
        
class SLORandomPad:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        pad_width = max(0, self.size[0] - img.width)
        pad_height = max(0, self.size[1] - img.height)
        pad_left = random.randint(0, pad_width)
        pad_top = random.randint(0, pad_height)
        pad_right = pad_width - pad_left
        pad_bottom = pad_height - pad_top
        return transforms.functional.pad(img, (pad_left, pad_top, pad_right, pad_bottom))


class FundRandomRotate:
    def __init__(self, prob, degree):
        self.prob = prob
        self.degree = degree

    def __call__(self, img):
        if random.random() < self.prob:
            angle = random.uniform(-self.degree, self.degree)
            return transforms.functional.rotate(img, angle)
        return img

# Data augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Updated augmentation transform - 1
# train_transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.RandomCrop((224, 224)),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomVerticalFlip(p=0.5),
#     transforms.ColorJitter(brightness=(0.1, 0.9), contrast=0.5, saturation=0.5, hue=0.2),
#     FundRandomRotate(prob=0.5, degree=30),
#     SLORandomPad((224, 224)),
#     transforms.ToTensor(),
#     transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# Updated augmentation transform - 2
# train_transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.RandomCrop((224, 224)),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomVerticalFlip(p=0.5),
#     transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
#     transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
#     transforms.ColorJitter(brightness=(0.1, 0.9), contrast=0.5, saturation=0.5, hue=0.2),
#     transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
#     FundRandomRotate(prob=0.5, degree=30),
#     SLORandomPad((224, 224)),
#     transforms.ToTensor(),
#     transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CutOut(object):
    def __init__(self, mask_size, p=0.5):
        self.mask_size = mask_size
        self.p = p

    def __call__(self, img):
        if np.random.rand() > self.p:
            return img

        # Ensure the image is a tensor
        if not isinstance(img, torch.Tensor):
            raise TypeError('Input image must be a torch.Tensor')

        # Get height and width of the image
        h, w = img.shape[1], img.shape[2]
        mask_size_half = self.mask_size // 2
        offset = 1 if self.mask_size % 2 == 0 else 0

        cx = np.random.randint(mask_size_half, w + offset - mask_size_half)
        cy = np.random.randint(mask_size_half, h + offset - mask_size_half)

        xmin, xmax = cx - mask_size_half, cx + mask_size_half + offset
        ymin, ymax = cy - mask_size_half, cy + mask_size_half + offset
        xmin, xmax = max(0, xmin), min(w, xmax)
        ymin, ymax = max(0, ymin), min(h, ymax)

        img[:, ymin:ymax, xmin:xmax] = 0
        return img


class SLORandomPad:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        pad_width = max(0, self.size[0] - img.width)
        pad_height = max(0, self.size[1] - img.height)
        pad_left = random.randint(0, pad_width)
        pad_top = random.randint(0, pad_height)
        pad_right = pad_width - pad_left
        pad_bottom = pad_height - pad_top
        return transforms.functional.pad(img, (pad_left, pad_top, pad_right, pad_bottom))


class FundRandomRotate:
    def __init__(self, prob, degree):
        self.prob = prob
        self.degree = degree

    def __call__(self, img):
        if random.random() < self.prob:
            angle = random.uniform(-self.degree, self.degree)
            return transforms.functional.rotate(img, angle)
        return img


transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((210, 210)),
    SLORandomPad((224, 224)),
    FundRandomRotate(prob=0.5, degree=30),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=(0.1, 0.9)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def train_model(model, train_loader, val_loader, device, criterion, optimizer, lr_scheduler, num_epochs=25,
                checkpoint_path='model.pth'):
    best_model = model.state_dict()
    best_epoch = None
    best_val_kappa = -1.0  # Initialize the best kappa score

    for epoch in range(1, num_epochs + 1):
        print(f'\nEpoch {epoch}/{num_epochs}')
        running_loss = []
        all_preds = []
        all_labels = []

        model.train()

        with tqdm(total=len(train_loader), desc=f'Training', unit=' batch', file=sys.stdout) as pbar:
            for images, labels in train_loader:
                if not isinstance(images, list):
                    images = images.to(device)  # single image case
                else:
                    images = [x.to(device) for x in images]  # dual images case

                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, labels.long())

                loss.backward()
                optimizer.step()

                preds = torch.argmax(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                running_loss.append(loss.item())

                pbar.set_postfix({'lr': f'{optimizer.param_groups[0]["lr"]:.1e}', 'Loss': f'{loss.item():.4f}'})
                pbar.update(1)

        lr_scheduler.step()

        epoch_loss = sum(running_loss) / len(running_loss)

        train_metrics = compute_metrics(all_preds, all_labels, per_class=True)
        kappa, accuracy, precision, recall = train_metrics[:4]

        print(f'[Train] Kappa: {kappa:.4f} Accuracy: {accuracy:.4f} '
              f'Precision: {precision:.4f} Recall: {recall:.4f} Loss: {epoch_loss:.4f}')

        if len(train_metrics) > 4:
            precision_per_class, recall_per_class = train_metrics[4:]
            for i, (precision, recall) in enumerate(zip(precision_per_class, recall_per_class)):
                print(f'[Train] Class {i}: Precision: {precision:.4f}, Recall: {recall:.4f}')

        # Evaluation on the validation set at the end of each epoch
        val_metrics = evaluate_model(model, val_loader, device)
        val_kappa, val_accuracy, val_precision, val_recall = val_metrics[:4]
        print(f'[Val] Kappa: {val_kappa:.4f} Accuracy: {val_accuracy:.4f} '
              f'Precision: {val_precision:.4f} Recall: {val_recall:.4f}')

        if val_kappa > best_val_kappa:
            best_val_kappa = val_kappa
            best_epoch = epoch
            best_model = model.state_dict()
            torch.save(best_model, checkpoint_path)

    print(f'[Val] Best kappa: {best_val_kappa:.4f}, Epoch {best_epoch}')

    return model


def evaluate_model(model, test_loader, device, test_only=False, prediction_path='./test_predictions.csv'):
    model.eval()

    all_preds = []
    all_labels = []
    all_image_ids = []

    with tqdm(total=len(test_loader), desc=f'Evaluating', unit=' batch', file=sys.stdout) as pbar:
        for i, data in enumerate(test_loader):

            if test_only:
                images = data
            else:
                images, labels = data

            if not isinstance(images, list):
                images = images.to(device)  # single image case
            else:
                images = [x.to(device) for x in images]  # dual images case

            with torch.no_grad():
                outputs = model(images)
                preds = torch.argmax(outputs, 1)

            if not isinstance(images, list):
                # single image case
                all_preds.extend(preds.cpu().numpy())
                image_ids = [
                    os.path.basename(test_loader.dataset.data[idx]['img_path']) for idx in
                    range(i * test_loader.batch_size, i * test_loader.batch_size + len(images))
                ]
                all_image_ids.extend(image_ids)
                if not test_only:
                    all_labels.extend(labels.numpy())
            else:
                # dual images case
                for k in range(2):
                    all_preds.extend(preds.cpu().numpy())
                    image_ids = [
                        os.path.basename(test_loader.dataset.data[idx][f'img_path{k + 1}']) for idx in
                        range(i * test_loader.batch_size, i * test_loader.batch_size + len(images[k]))
                    ]
                    all_image_ids.extend(image_ids)
                    if not test_only:
                        all_labels.extend(labels.numpy())

            pbar.update(1)

    # Save predictions to csv file for Kaggle online evaluation
    if test_only:
        df = pd.DataFrame({
            'ID': all_image_ids,
            'TARGET': all_preds
        })
        df.to_csv(prediction_path, index=False)
        print(f'[Test] Save predictions to {os.path.abspath(prediction_path)}')
    else:
        metrics = compute_metrics(all_preds, all_labels)
        return metrics


def compute_metrics(preds, labels, per_class=False):
    kappa = cohen_kappa_score(labels, preds, weights='quadratic')
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted', zero_division=0)

    # Calculate and print precision and recall for each class
    if per_class:
        precision_per_class = precision_score(labels, preds, average=None, zero_division=0)
        recall_per_class = recall_score(labels, preds, average=None, zero_division=0)
        return kappa, accuracy, precision, recall, precision_per_class, recall_per_class

    return kappa, accuracy, precision, recall


class MyModel(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()

        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove the original classification layer

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x


class MyDualModel(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()

        backbone = models.resnet18(pretrained=True)
        backbone.fc = nn.Identity()

        # Here the two backbones will have the same structure but unshared weights
        self.backbone1 = copy.deepcopy(backbone)
        self.backbone2 = copy.deepcopy(backbone)

        self.fc = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, images):
        image1, image2 = images

        x1 = self.backbone1(image1)
        x2 = self.backbone2(image2)

        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x

# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS_APTOS = 1
NUM_EPOCHS_DEEPDRID = 1
LEARNING_RATE = 1e-4
NUM_CLASSES = 5

# Model selection function
def get_model(model_name='densenet161'):
    if model_name == 'resnet34':
        model = models.resnet34(weights='IMAGENET1K_V1')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    elif model_name == 'efficientnet_b1':
        model = models.efficientnet_b1(weights='IMAGENET1K_V1')
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    elif model_name == 'densenet161':
        model = models.densenet161(weights='IMAGENET1K_V1')
        model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    elif model_name == 'resnet34_attention':
        model = MyModel_resnet34(num_classes=NUM_CLASSES)
    elif model_name == 'densenet161_attention':
        model = MyModel_DenseNet161(num_classes=NUM_CLASSES)
    elif model_name == 'vgg16_attention':
        model = MyModel_VGG(num_classes=NUM_CLASSES)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    predictions = []
    labels = []
    
    for images, targets in tqdm(train_loader, desc='Training'):
        if isinstance(images, list):  # Handle dual image case
            images = [img.to(device) for img in images]
        else:
            images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        predictions.extend(outputs.argmax(1).cpu().numpy())
        labels.extend(targets.cpu().numpy())
    
    epoch_loss = running_loss / len(train_loader)
    epoch_kappa = cohen_kappa_score(labels, predictions, weights='quadratic')
    
    return epoch_loss, epoch_kappa

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    predictions = []
    labels = []
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc='Validating'):
            if isinstance(images, list):  # Handle dual image case
                images = [img.to(device) for img in images]
            else:
                images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            predictions.extend(outputs.argmax(1).cpu().numpy())
            labels.extend(targets.cpu().numpy())
    
    val_loss = running_loss / len(val_loader)
    val_kappa = cohen_kappa_score(labels, predictions, weights='quadratic')
    
    return val_loss, val_kappa

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs, device, checkpoint_path):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_kappa = 0.0
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        
        train_loss, train_kappa = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_kappa = validate(model, val_loader, criterion, device)
        
        print(f'Train Loss: {train_loss:.4f}, Train Kappa: {train_kappa:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Kappa: {val_kappa:.4f}')
        
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_kappa)
            else:
                scheduler.step()
        
        if val_kappa > best_kappa:
            best_kappa = val_kappa
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_kappa': best_kappa,
            }, checkpoint_path)
            print(f'New best model saved with kappa: {best_kappa:.4f}')
    
    model.load_state_dict(best_model_wts)
    return model, best_kappa
    
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Initialize model
    model_name = 'vgg16_attention'  # or 'resnet34' or 'densenet161' we have studied many different ones because we were thinking that we need a model diversity for task d ensemble!
    model = get_model(model_name)
    model = model.to(device)

    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True
    
    # Step 1: Train on APTOS dataset
    print("\nStep 1: Training on APTOS dataset...")
    
    # Create APTOS dataloaders
    train_dataset_aptos = RetinopathyDataset_APTOS(
        PATHS['train_csv'],
        PATHS['train_images'],
        train_transform
    )
    
    val_dataset_aptos = RetinopathyDataset_APTOS(
        PATHS['val_csv'],
        PATHS['val_images'],
        val_transform
    )
    
    train_loader_aptos = DataLoader(
        train_dataset_aptos,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader_aptos = DataLoader(
        val_dataset_aptos,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=2,
    verbose=True
    )
    
    # Train on APTOS
    model, aptos_kappa = train_model(
        model, train_loader_aptos, val_loader_aptos,
        criterion, optimizer, scheduler,
        NUM_EPOCHS_APTOS, device,
        'best_model_aptos.pth'
    )
    
    print(f"\nAPTOS Training Complete. Best Kappa: {aptos_kappa:.4f}")
    
    # Step 2: Fine-tune on DeepDRiD dataset
    print("\nStep 2: Fine-tuning on DeepDRiD dataset...")
    
    # Create DeepDRiD dataloaders
    train_dataset = RetinopathyDataset(
        './DeepDRiD/train.csv',
        './DeepDRiD/train/',
        transform_train,
        mode='single'
    )
    
    val_dataset = RetinopathyDataset(
        './DeepDRiD/val.csv',
        './DeepDRiD/val/',
        transform_test,
        mode='single'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Reset optimizer and scheduler for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE * 0.1)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Fine-tune on DeepDRiD
    model, deepdrid_kappa = train_model(
        model, train_loader, val_loader,
        criterion, optimizer, scheduler,
        NUM_EPOCHS_DEEPDRID, device,
        'best_model_final.pth'
    )
    
    print(f"\nTraining Complete!")
    print(f"APTOS Best Kappa: {aptos_kappa:.4f}")
    print(f"DeepDRiD Best Kappa: {deepdrid_kappa:.4f}")

if __name__ == '__main__':
    main()