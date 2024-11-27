import torch
import torchvision
from torchvision import transforms
import os
from efficientnet_pytorch import EfficientNet
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from PIL import Image
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from train import TripletModel
import torch.nn.functional as F

class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []  
        self.load_image_paths() 

    def load_image_paths(self):
        import os
        for file_name in os.listdir(self.root_dir):
            file_path = os.path.join(self.root_dir, file_name)
            if file_path.endswith(('.jpg', '.png', '.jpeg')):  
                self.image_paths.append(file_path)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert('RGB')
        image = transforms.ToTensor()(image)
        if self.transform:
            image = self.transform(image)
        return img_path, image, 0

    def __len__(self):
        return len(self.image_paths)


def inference(model, num_classes, class_names, test_loader):
    model.eval()  
    all_preds = []
    all_targets = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():  
        for batch in test_loader:
            img_path, images, _ = batch
            images = images.to(device)
            anchor_triplet_features, _, _ = model(images, images, images)  
            class_logits = model.classification_head(anchor_triplet_features)
            probabilities = F.softmax(class_logits, dim=1)
            preds = torch.argmax(class_logits, dim=1)
            
            predicted_class_index = preds.cpu()  # 
            predicted_class_name = class_names[predicted_class_index]  # 

            print(f"Image path: {img_path}")
            print(f"Prediction: {predicted_class_name}")
            print(f"Probabilities: {probabilities}")
            print("-" * 50)


if __name__ == "__main__":
	

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
       
    folder_path = '/root/sharespace/wy/AbNormal_Classification/AbNormal_Classification_dataset/pipeline_test_imgs'
    test_dataset = TestDataset(root_dir=folder_path, transform=data_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    num_classes = 9  
    class_names = ['chuandi_keyiwupin', 'jian_keyiwupin', 'jiaotoujieer', 'jushou', 'normal', 'qili', 'shouzhuoxia_maitou', 'xianghoupiantou', 'zuoyoupiantou']
    
    model = TripletModel(num_classes=num_classes)  
    model.load_state_dict(torch.load('classbalance_best_model_2.pth'))
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')  

    inference(model, num_classes, class_names, test_loader)
    
    
    
    