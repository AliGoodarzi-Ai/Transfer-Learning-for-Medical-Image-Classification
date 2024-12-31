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

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# For balancing the dataset
from collections import Counter
from sklearn.utils import resample

# Hyper Parameters
batch_size = 24
num_classes = 5  # 5 DR levels
learning_rate = 0.0001
num_epochs = 20

import torch.nn.functional as F

from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from torchvision import models

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

# Trying out different models

################################################################################################

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


# # Resnet34 Single
# class MyModel_resnet34(nn.Module):
#     def __init__(self, num_classes=5, dropout_rate=0.5):
#         super().__init__()

#         self.backbone = models.resnet34(pretrained=True)
#         self.backbone.fc = nn.Identity()  # Remove the original classification layer

#         self.fc = nn.Sequential(
#             nn.Linear(512, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=dropout_rate),
#             nn.Linear(256, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=dropout_rate),
#             nn.Linear(128, num_classes)
#         )

#     def forward(self, x):
#         x = self.backbone(x)
#         x = self.fc(x)
#         return x

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


# EfficientNet-B4 Single
class MyModel_EfficientNetB4(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()

        self.backbone = models.efficientnet_b4(pretrained=True)
        self.backbone.classifier = nn.Identity()  # Remove the original classification layer

                # Add Attention Modules
        self.channel_attention = ChannelAttention(in_planes=1792)  # 1280: feature map channels in EfficientNet-B0
        self.spatial_attention = SpatialAttention()

        self.fc = nn.Sequential(
            nn.Linear(1792, 256),
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



def apply_gaussian_blur(image, radius=2):
    return image.filter(ImageFilter.GaussianBlur(radius))


def apply_sharpening(image):
    return image.filter(ImageFilter.SHARPEN)





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
        img = apply_gaussian_blur(img) 
        img = apply_sharpening(img)     
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

        img1 = apply_gaussian_blur(img1) 
        img1 = apply_sharpening(img1)    
        img2 = apply_gaussian_blur(img2)  
        img2 = apply_sharpening(img2)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        if not self.test:
            label = torch.tensor(data['dr_level'], dtype=torch.int64)
            return [img1, img2], label
        else:
            return [img1, img2]

# Ensemble methods

class EnsembleMethods:
    def __init__(self, models_dict, device):
        self.device = device
        self.models = {}
        
        # Load all models
        for model_name, model_path in models_dict.items():
            if 'resnet34' in model_name.lower():
                model = MyModel_resnet34()
            elif 'efficient' in model_name.lower():
                model = MyModel_EfficientNetB4()
            elif 'vgg' in model_name.lower():
                model = MyModel_VGG()
            elif 'dense' in model_name.lower():
                model = MyModel_DenseNet161()
            else:
                model = MyModel()
                
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model = model.to(device)
            model.eval()
            self.models[model_name] = model

    def get_predictions(self, dataloader):
        """Get predictions from all models"""
        predictions = {}
        all_labels = []
        first_pass = True
        
        for model_name, model in self.models.items():
            model_preds = []
            
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f'Getting predictions for {model_name}'):
                    images, labels = batch
                    
                    if first_pass:
                        all_labels.extend(labels.cpu().numpy())
                    
                    if not isinstance(images, list):
                        images = images.to(self.device)
                    else:
                        images = [x.to(self.device) for x in images]
                        
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)
                    model_preds.extend(probs.cpu().numpy())
            
            predictions[model_name] = np.array(model_preds)
            first_pass = False
            
        return predictions, np.array(all_labels)

    def weighted_average(self, dataloader):
        """Weighted average ensemble"""
        predictions, labels = self.get_predictions(dataloader)
        weights = {model_name: 1/len(self.models) for model_name in self.models.keys()}
            
        ensemble_preds = np.zeros_like(list(predictions.values())[0])
        for model_name, preds in predictions.items():
            ensemble_preds += weights[model_name] * preds
            
        final_preds = np.argmax(ensemble_preds, axis=1)
        return final_preds, labels

    def max_voting(self, dataloader):
        """Max voting ensemble"""
        predictions, labels = self.get_predictions(dataloader)
        
        model_preds = []
        for preds in predictions.values():
            model_preds.append(np.argmax(preds, axis=1))
            
        ensemble_preds = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=np.array(model_preds)
        )
        
        return ensemble_preds, labels

    def stacking(self, dataloader):
        """Stacking ensemble using validation set split"""
        predictions, labels = self.get_predictions(dataloader)
        
        # Split validation set into two parts
        n_samples = len(labels)
        n_train = n_samples // 2
        indices = np.random.permutation(n_samples)
        
        # Prepare data
        X = np.concatenate([preds for preds in predictions.values()], axis=1)
        X = X[indices]
        y = labels[indices]
        
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]
        
        meta_learner = LogisticRegression(multi_class='multinomial', max_iter=1000)
        meta_learner.fit(X_train, y_train)
        
        ensemble_preds = meta_learner.predict(X_val)
        return ensemble_preds, y_val

    def boosting(self, dataloader):
        """Boosting ensemble using validation set split"""
        predictions, labels = self.get_predictions(dataloader)
        
        # Split validation set into two parts
        n_samples = len(labels)
        n_train = n_samples // 2
        indices = np.random.permutation(n_samples)
        
        # Prepare data
        X = np.concatenate([preds for preds in predictions.values()], axis=1)
        X = X[indices]
        y = labels[indices]
        
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]
        
        boosting = GradientBoostingClassifier(n_estimators=100)
        boosting.fit(X_train, y_train)
        
        ensemble_preds = boosting.predict(X_val)
        return ensemble_preds, y_val

    def bagging(self, dataloader, n_bags=5):
        """Bagging ensemble"""
        predictions, labels = self.get_predictions(dataloader)
        
        all_preds = []
        for _ in range(n_bags):
            sampled_models = np.random.choice(
                list(predictions.keys()),
                size=len(predictions),
                replace=True
            )
            
            bag_preds = np.zeros_like(list(predictions.values())[0])
            for model_name in sampled_models:
                bag_preds += predictions[model_name]
            bag_preds /= len(sampled_models)
            
            all_preds.append(np.argmax(bag_preds, axis=1))
            
        ensemble_preds = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=np.array(all_preds)
        )
        
        return ensemble_preds, labels

def evaluate_ensembles(models_dict, val_loader, device):
    """Evaluate all ensemble methods using validation dataset"""
    ensemble = EnsembleMethods(models_dict, device)
    results = {}
    
    # Evaluate individual models first
    print("\nINDIVIDUAL MODEL PERFORMANCE:")
    predictions, labels = ensemble.get_predictions(val_loader)
    for model_name, preds in predictions.items():
        model_preds = np.argmax(preds, axis=1)
        metrics = compute_metrics(model_preds, labels)
        print(f'\n{model_name.upper()}:')
        print(f'Kappa: {metrics[0]:.4f}')
        print(f'Accuracy: {metrics[1]:.4f}')
        print(f'Precision: {metrics[2]:.4f}')
        print(f'Recall: {metrics[3]:.4f}')
    
    # Evaluate ensemble methods
    print("\nENSEMBLE METHODS PERFORMANCE:")
    
    # print("\nEvaluating Weighted Average...")
    # preds, labels = ensemble.weighted_average(val_loader)
    # results['weighted_average'] = compute_metrics(preds, labels)
    
    # print("\nEvaluating Max Voting...")
    # preds, labels = ensemble.max_voting(val_loader)
    # results['max_voting'] = compute_metrics(preds, labels)
    
    print("\nEvaluating Stacking...")
    preds, labels = ensemble.stacking(val_loader)
    results['stacking'] = compute_metrics(preds, labels)

    class_names = [f"Class {i}" for i in range(len(np.unique(labels)))]
    plot_confusion_matrix(labels, preds, class_names, normalize=True)
    
    # print("\nEvaluating Boosting...")
    # preds, labels = ensemble.boosting(val_loader)
    # results['boosting'] = compute_metrics(preds, labels)
    
    # print("\nEvaluating Bagging...")
    # preds, labels = ensemble.bagging(val_loader)
    # results['bagging'] = compute_metrics(preds, labels)
    
    return results

################################################################################################

# Data processing without balancing



#   Gaussian Blur - sharpening filter
def apply_gaussian_blur(image, radius=2):
    return image.filter(ImageFilter.GaussianBlur(radius))

def apply_sharpening(image):
    return image.filter(ImageFilter.SHARPEN)


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

        img = apply_gaussian_blur(img)  
        img = apply_sharpening(img)     
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

                # Gaussian blur and sharpening 
        img1 = apply_gaussian_blur(img1)
        img1 = apply_sharpening(img1) 
        img2 = apply_gaussian_blur(img2)  
        img2 = apply_sharpening(img2)    


        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        if not self.test:
            label = torch.tensor(data['dr_level'], dtype=torch.int64)
            return [img1, img2], label
        else:
            return [img1, img2]


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

# Original augmentation transform
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


# Updated augmentation transform - 1
# transform_train = transforms.Compose([
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
# transform_train = transforms.Compose([
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


transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def train_model(model, train_loader, val_loader, device, criterion, optimizer, lr_scheduler, num_epochs=25,
                checkpoint_path='model.pth'):
    best_model = model.state_dict()
    best_epoch = None
    best_val_kappa = -1.0  

    epoch_losses = []
    kappas = []

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

        epoch_losses.append(epoch_loss)

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
        
        kappas.append(val_kappa)

        if val_kappa > best_val_kappa:
            best_val_kappa = val_kappa
            best_epoch = epoch
            best_model = model.state_dict()
            torch.save(best_model, checkpoint_path)

    # Create the seaborn lineplot
    sns.lineplot(x=list(range(num_epochs)), y=epoch_losses, label="Loss")
    sns.lineplot(x=list(range(num_epochs)), y=kappas, label="Kappa")

    # Set plot labels and title
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Loss and Kappa Over Epochs")

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()

    print(f'[Val] Best kappa: {best_val_kappa:.4f}, Epoch {best_epoch}')

    return model


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=cmap)
    plt.title(title)
    plt.show()


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

        class_names = [f"Class {i}" for i in range(len(np.unique(all_labels)))]
        plot_confusion_matrix(all_labels, all_preds, class_names, normalize=True)
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

        #  two backbones will have the same structure
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
    
def save_ensemble_predictions(predictions, test_loader, prediction_path='./ensemble_predictions.csv'):
    all_image_ids = []

    # Extract image IDs
    for i in range(len(test_loader.dataset)):
        img_path = test_loader.dataset.data[i]['img_path']  #
        image_id = os.path.basename(img_path)
        all_image_ids.append(image_id)

    # Ensure predictions match the dataset length
    assert len(predictions) == len(all_image_ids), "Mismatch between predictions and dataset size"

    # Create DataFrame and save
    df = pd.DataFrame({
        'ID': all_image_ids,
        'TARGET': predictions
    })
    df.to_csv(prediction_path, index=False)
    print(f"Ensemble predictions saved to {os.path.abspath(prediction_path)}")
    

# Function to load models
def load_model(model_class, checkpoint_path, device):
    model = model_class()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


if __name__ == '__main__':

    models_dict = {
        'resnet34': './MyModel_resnet34.pth',
        'efficientnet': './MyModel_EfficientNetB4.pth',
        'vgg': './MyModel_VGG.pth',
        'densenet': './MyModel_DenseNet161.pth'
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create validation dataset and dataloader
    val_dataset = RetinopathyDataset('./DeepDRiD/val.csv', './DeepDRiD/val/', transform_test, 'single')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Evaluate all ensemble methods
    results = evaluate_ensembles(models_dict, val_loader, device)
    
    # Print ensemble results
    for method, metrics in results.items():
        kappa, accuracy, precision, recall = metrics
        print(f'\n{method.upper()} RESULTS:')
        print(f'Kappa: {kappa:.4f}')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')