import warnings
import sklearn
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix, f1_score
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import pytorch_lightning as pl
from pytorch_metric_learning import losses, miners
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import matplotlib.pyplot as plt
import numpy as np
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import seaborn as sns
import cv2
from PIL import Image
from typing import List

class RandomResolutionDrop(object):
    def __init__(self, min_scale=0.5, max_scale=1.0):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        if random.random() < 0.5:  
            scale = random.uniform(self.min_scale, self.max_scale)
            new_width = int(img.width * scale)
            new_height = int(img.height * scale)
            img = img.resize((new_width, new_height), Image.BICUBIC)
        return img
        
def calculate_class_weights(dataset):

    class_counts = {}
    for _, label in dataset:
        class_counts[label] = class_counts.get(label, 0) + 1
    total_samples = len(dataset)
    class_weights = {}
    for class_idx in class_counts:
        class_weights[class_idx] = total_samples / class_counts[class_idx]
    return class_weights
    
class PatchConcatenation:
    """
    0.5torchvision.transforms
    8patch
    """
    def __init__(self):
        self.max_num_patches = 8

    def __call__(self, img):
        # img.dataset
        dataset = img.dataset
        label = dataset.targets[dataset.imgs.index(img)]
        same_class_images = [img for img_, label_ in zip(dataset.imgs, dataset.targets) if label_ == label]

        # 0.5
        orientation = 'horizontal' if random.random() < 0.5 else 'vertical'

        # 18
        num_patches = random.randint(1, self.max_num_patches)
        selected_images = random.sample(same_class_images, num_patches)

        return self.patch_concatenation(selected_images, orientation)

    def patch_concatenation(self, images: List[Image.Image], orientation='horizontal') -> Image.Image:
  
        if orientation == 'horizontal':
            widths, heights = zip(*(img.size for img in images))
            total_width = sum(widths)
            max_height = max(heights)
            new_image = Image.new('RGB', (total_width, max_height))
            x_offset = 0
            for img in images:
                new_image.paste(img, (x_offset, 0))
                x_offset += img.width
        elif orientation == 'vertical':
            widths, heights = zip(*(img.size for img in images))
            max_width = max(widths)
            total_height = sum(heights)
            new_image = Image.new('RGB', (max_width, total_height))
            y_offset = 0
            for img in images:
                new_image.paste(img, (0, y_offset))
                y_offset += img.height
        else:
            raise ValueError("orientation 'horizontal'  'vertical'")

        # 
        width, height = new_image.size
        max_size = max(width, height)
        square_image = Image.new('RGB', (max_size, max_size), (0, 0, 0))  # 
        paste_x = (max_size - width) // 2 if width < max_size else 0
        paste_y = (max_size - height) // 2 if height < max_size else 0
        square_image.paste(new_image, (paste_x, paste_y))

        return square_image

class blurImg():
    """
    : 
    """
    def __init__(self, ksize, ratio=0.6):
        self.ksize = ksize
        self.ratio = ratio

    def __call__(self, image):
        if random.random() > self.ratio:
            size = random.choice(self.ksize)  # size>99
            kernel_size = (size, size)
            image = np.array(image, dtype=np.uint8)
            image = cv2.GaussianBlur(image, ksize=kernel_size, sigmaX=0, sigmaY=0)
            return Image.fromarray(image.astype('uint8')).convert('RGB')
        else:
            return image

class motionBlur(object):

    def __init__(self, degree_max=15, angle_max=30, ratio=0.6):
        self.degree_max = degree_max
        self.angle_max = angle_max
        self.ratio = ratio

    def __call__(self, image):
        if random.random() > 1 - self.ratio:
            blurred = self.motion_blur(image)
            return Image.fromarray(blurred.astype('uint8')).convert('RGB')
        else:
            return image


    def motion_blur(self, image):
        image = np.array(image)
        angle = random.randint(-self.angle_max, self.angle_max)
        degree = random.randint(1, self.degree_max)

        # kernel degree
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

        motion_blur_kernel = motion_blur_kernel / degree
        blurred = cv2.filter2D(image, -1, motion_blur_kernel)

        # convert to uint8
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        blurred = np.array(blurred, dtype=np.uint8)

        return blurred
        
def triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings, labels, margin=0.5):
    assert labels.dim() == 1 and labels.dtype == torch.long, "labels must be a 1D tensor of type torch.long"
    pos_distances = torch.sqrt(torch.sum((anchor_embeddings - positive_embeddings) ** 2, dim=1))
    neg_distances = torch.sqrt(torch.sum((anchor_embeddings - negative_embeddings) ** 2, dim=1))
    losses = torch.relu(pos_distances - neg_distances + margin)
    loss = losses.mean()
    return loss


class TripletDataset(Dataset):
    def __init__(self, dataset, num_classes):
        self.dataset = dataset
        self.num_classes = num_classes
        self.class_to_indices = {i: [] for i in range(num_classes)}
        for idx, (_, label) in enumerate(self.dataset):
            self.class_to_indices[label].append(idx)

        self.negative_class_weights = self.calculate_negative_class_weights()
    
    def calculate_negative_class_weights(self):
        class_sizes = np.array([len(self.class_to_indices[i]) for i in range(self.num_classes)])
        normal_index = 4  
        class_weights = np.full(self.num_classes, 0.6)  
        class_weights[normal_index] = 0.4  
        class_weights /= np.sum(class_weights)  
        return class_weights
        
    def __getitem__(self, index):
        anchor_image, anchor_label = self.dataset[index]
        # print(f"anchor label: {anchor_label}") 
        positive_class = anchor_label

        positive_indices = self.class_to_indices[positive_class]
        positive_index = random.choice(positive_indices)
        positive_image, _ = self.dataset[positive_index]
        # print(f"positive_image label: {_}") 

        negative_class = np.random.choice(
            range(self.num_classes),
            p=self.negative_class_weights
        )
        while negative_class == positive_class:
            negative_class = np.random.choice(
                range(self.num_classes),
                p=self.negative_class_weights
            )
        negative_indices = self.class_to_indices[negative_class]
        negative_index = random.choice(negative_indices)
        negative_image, _ = self.dataset[negative_index]
        # print(f"negative_image label: {_}") 

        return anchor_image, positive_image, negative_image, anchor_label

    def __len__(self):
        return len(self.dataset)
        
        
class TripletModel(pl.LightningModule):
    def __init__(self, num_classes):
        super(TripletModel, self).__init__()
        self.feature_extractor = EfficientNet.from_pretrained('efficientnet-b4')
        # 1024
        self.feature_extractor._fc = nn.Identity()
        self.triplet_embedding_size = 1024  # 
        self.fc_triplet = nn.Linear(1792, self.triplet_embedding_size)
        # self.fc_triplet = nn.Linear(1024, self.triplet_embedding_size)
        self.classification_head = nn.Linear(self.triplet_embedding_size, num_classes)
        self.triplet_loss = losses.TripletMarginLoss(margin=0.5)  # TripletLossmargin
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        #  epoch 
        self.preds_per_epoch_train = {i: [] for i in range(self.num_classes)}
        self.targets_per_epoch_train = {i: [] for i in range(self.num_classes)}
        #  epoch 
        self.preds_per_epoch_val = {i: [] for i in range(self.num_classes)}
        self.targets_per_epoch_val = {i: [] for i in range(self.num_classes)}
        # 
        self.best_accuracy = 0
        self.best_precision = 0
        self.best_recall = 0
        self.best_f1_score = 0
        self.best_model_state_dict = None
        
    def forward(self, anchor, positive, negative):
        # print(f"anchor size:  {anchor.size()},  positive size: {positive.size()}, negative size: {negative.size()}")
        anchor_features = self.feature_extractor(anchor)
        positive_features = self.feature_extractor(positive)
        negative_features = self.feature_extractor(negative)

        anchor_triplet_features = self.fc_triplet(anchor_features)
        positive_triplet_features = self.fc_triplet(positive_features)
        negative_triplet_features = self.fc_triplet(negative_features)

        return anchor_triplet_features, positive_triplet_features, negative_triplet_features
        
    def training_step(self, batch, batch_idx):
        anchor_images, positive_images, negative_images, labels = batch
        # print(f"labels: {labels}")
        anchor_triplet_features, positive_triplet_features, negative_triplet_features = self(anchor_images, positive_images, negative_images)
        # print(f"anchor_triplet_features: {anchor_triplet_features.size()}, labels: {labels.size()}")
        triplet_loss_value = triplet_loss(anchor_triplet_features, positive_triplet_features, negative_triplet_features, labels)

        anchor_class_logits = self.classification_head(anchor_triplet_features)

        cross_entropy_loss_value = self.cross_entropy_loss(anchor_class_logits, labels)

        total_loss = triplet_loss_value + cross_entropy_loss_value

        preds = torch.argmax(anchor_class_logits, dim=1)
        for c in range(self.num_classes):
            self.preds_per_epoch_train[c].append(preds[labels == c].cpu().numpy())
            self.targets_per_epoch_train[c].append(labels[labels == c].cpu().numpy())

        print('triplet_loss', triplet_loss_value)
        print('cross_entropy_loss', cross_entropy_loss_value)
        print('total_loss', total_loss)
       
        # all_losses = triplet_loss(anchor_triplet_features, positive_triplet_features,
        #                   negative_triplet_features, labels).detach().cpu().numpy()
        
        # print("all_losses value:", all_losses)
        # print("all_losses type:", type(all_losses))
        # hard_sample_indices = np.argsort(all_losses)[int(len(all_losses) * 0.7):]
   
        # hard_negative_indices = hard_sample_indices[::2]
        # hard_negative_images = [negative_images[i] for i in hard_negative_indices]
        # hard_negative_labels = [labels[i] for i in hard_negative_indices]

        # new_anchor_images = torch.cat([anchor_images, anchor_images[hard_negative_indices]], dim=0)
        # new_positive_images = torch.cat([positive_images, positive_images[hard_negative_indices]], dim=0)
        # new_negative_images = torch.cat([negative_images, torch.stack(hard_negative_images)], dim=0)
        # new_labels = torch.cat([labels, torch.tensor(hard_negative_labels)], dim=0)

        # new_anchor_triplet_features, new_positive_triplet_features, new_negative_triplet_features = self(
        #     new_anchor_images, new_positive_images, new_negative_images)
        # new_triplet_loss_value = triplet_loss(new_anchor_triplet_features, new_positive_triplet_features,
        #                                       new_negative_triplet_features, new_labels)

        # new_anchor_class_logits = self.classification_head(new_anchor_triplet_features)
        # new_cross_entropy_loss_value = self.cross_entropy_loss(new_anchor_class_logits, new_labels)

        # new_total_loss = new_triplet_loss_value + new_cross_entropy_loss_value

        return total_loss
        
    def validation_step(self, batch, batch_idx):
        anchor_images, positive_images, negative_images, labels = batch
      
        anchor_triplet_features, positive_triplet_features, negative_triplet_features = self(anchor_images, positive_images, negative_images)

        anchor_class_logits = self.classification_head(anchor_triplet_features)

        preds = torch.argmax(anchor_class_logits, dim=1)

        for c in range(self.num_classes):
            self.preds_per_epoch_val[c].append(preds[labels == c].cpu().numpy())
            self.targets_per_epoch_val[c].append(labels[labels == c].cpu().numpy())

        # print(f"Batch {batch_idx} : {preds}")  # batch
        # print(f"Batch {batch_idx} : {labels}")  # batch

    def on_validation_epoch_end(self):
        all_preds_val = []
        all_targets_val = []
        for c in range(self.num_classes):
            preds_c = np.concatenate(self.preds_per_epoch_val[c])
            targets_c = np.concatenate(self.targets_per_epoch_val[c])
            all_preds_val.append(preds_c)
            all_targets_val.append(targets_c)

            # print(f" {c}  epoch : {preds_c}")  # epoch
            # print(f" {c}  epoch : {targets_c}")  # epoch

            accuracy = accuracy_score(targets_c, preds_c)
            precision = precision_score(targets_c, preds_c, average='macro')  # 'micro'
            recall = recall_score(targets_c, preds_c, average='macro')
            f1 = f1_score(targets_c, preds_c, average='macro')
            print(f'val_accuracy_class_{c}', accuracy)
            print(f'val_precision_class_{c}', precision)
            print(f'val_recall_class_{c}', recall)
            print(f'val_f1_score_class_{c}', f1)

        all_preds_val = np.concatenate(all_preds_val)
        all_targets_val = np.concatenate(all_targets_val)
        overall_accuracy = accuracy_score(all_targets_val, all_preds_val)
        overall_precision = precision_score(all_targets_val, all_preds_val, average='macro')
        overall_recall = recall_score(all_targets_val, all_preds_val, average='macro')
        overall_f1_score = f1_score(all_targets_val, all_preds_val, average='macro')
        print('val_overall_accuracy', overall_accuracy)
        print('val_overall_precision', overall_precision)
        print('val_overall_recall', overall_recall)
        print('val_overall_f1_score', overall_f1_score)

        conf_matrix = confusion_matrix(all_targets_val, all_preds_val)
        conf_matrix = torch.from_numpy(conf_matrix)
        accuracy_from_conf_matrix = conf_matrix.diag().sum().float() / conf_matrix.sum().float()
        print('val_confusion_matrix_accuracy', accuracy_from_conf_matrix)

        if overall_f1_score > self.best_f1_score:
            self.best_f1_score = overall_f1_score
            self.best_model_state_dict = self.state_dict()
            torch.save(self.best_model_state_dict, 'best_model_6.pth')

        #  epoch  epoch 
        self.preds_per_epoch_val = {i: [] for i in range(self.num_classes)}
        self.targets_per_epoch_val = {i: [] for i in range(self.num_classes)}

    def on_train_epoch_end(self):
        all_preds_train = []
        all_targets_train = []
        for c in range(self.num_classes):
            preds_c = np.concatenate(self.preds_per_epoch_train[c])
            targets_c = np.concatenate(self.preds_per_epoch_train[c])
            all_preds_train.append(preds_c)
            all_targets_train.append(targets_c)

            accuracy = accuracy_score(targets_c, preds_c)
            precision = precision_score(targets_c, preds_c, average='macro')  # 'micro'
            recall = recall_score(targets_c, preds_c, average='macro')
        print(f'train_accuracy_class_{c}', accuracy)
        print(f'train_precision_class_{c}', precision)
        print(f'train_recall_class_{c}', recall)

        all_preds_train = np.concatenate(all_preds_train)
        all_targets_train = np.concatenate(all_targets_train)
        overall_accuracy = accuracy_score(all_targets_train, all_preds_train)
        overall_precision = precision_score(all_targets_train, all_preds_train, average='macro')
        overall_recall = recall_score(all_targets_train, all_preds_train, average='macro')
        print('train_overall_accuracy', overall_accuracy)
        print('train_overall_precision', overall_precision)
        print('train_overall_recall', overall_recall)

        #  epoch  epoch 
        self.preds_per_epoch_train = {i: [] for i in range(self.num_classes)}
        self.targets_per_epoch_train = {i: [] for i in range(self.num_classes)}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
        
if __name__ == "__main__":
    
    train_transform = transforms.Compose([
    	                                    RandomResolutionDrop(min_scale=0.5, max_scale=1.0),
                                            transforms.Resize((224, 224)),
                                            transforms.CenterCrop(224),
                                            transforms.ColorJitter(brightness=0.5),
                                            transforms.ColorJitter(saturation=0.3),
                                            transforms.ColorJitter(contrast=0.3),
                                            motionBlur(degree_max=10, angle_max=20, ratio=0.6),
                                            blurImg(ksize=(3, 5, 7), ratio=0.6),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                        ])
                                        
    val_transform = transforms.Compose([
                                   transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                 ])
                                     
    loaded_train_dataset = ImageFolder(root='/root/sharespace/wy/AbNormal_Classification/AbNormal_Classification_dataset/original_image',
                                    transform=train_transform)
    num_classes = len(loaded_train_dataset.classes)
    
    train_dataset = TripletDataset(dataset=loaded_train_dataset, num_classes=num_classes)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    loaded_val_dataset = ImageFolder(root='/root/sharespace/wy/AbNormal_Classification/AbNormal_Classification_dataset/split_new_new',
                                    transform=val_transform)
    
    val_dataset = TripletDataset(dataset=loaded_val_dataset, num_classes=num_classes)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    model = TripletModel(num_classes)
    trainer = pl.Trainer(max_epochs=50, accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                         devices=1 if torch.cuda.is_available() else None,
                         default_root_dir='logs')
    trainer.fit(model, train_loader, val_loader)