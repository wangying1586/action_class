import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
from torchvision import datasets, transforms, models
import torch.nn as nn
# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 	
def infer(image_path):
    # 
    image = Image.open(image_path).convert('RGB')  #  RGB 
    image = transform(image).unsqueeze(0)  # 

    with torch.no_grad():  # 
        outputs = model(image.to(device))  # 

    probabilities = torch.softmax(outputs, dim=1)  # 
    predicted_class = torch.argmax(probabilities, dim=1).item()  # 
    
    predicted_class_name = class_names[predicted_class]  # 

    return predicted_class_name, probabilities[0].cpu().numpy()  # 

if __name__ == '__main__':
	# Load the best model
	model = models.resnet50(pretrained=False)  # Set pretrained=True if you want to load ImageNet weights
	
	# Modify the final fully connected layer to match your number of classes
	num_classes = 9  # Set this to the number of classes in your dataset
	model.fc = nn.Linear(model.fc.in_features, num_classes)
	
	# Load the model weights from the specified file
	model.load_state_dict(torch.load('9class_train_val_augmentation_best_model.pth', weights_only=False))
	

	model.to(device)
	model.eval()  # Set model to evaluation mode
	
	# 
	transform = transforms.Compose([
		transforms.Resize((224, 224)),  # 
		transforms.ToTensor(),  # 
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 
		])
		
	# 
	class_names = [
		"chuandi_keyiwupin", 
		"jian_keyiwupin",
		"jiaotoujieer",
		"jushou",
		"normal",
		"qili",
		"shouzhuoxia_maitou",
		"xianghoupiantou",
		"zuoyoupiantou"
	]
			
	# 
	image_path = './chuandi_keyiwupin_1_keyframe_30.jpg'  # 
	predicted_class_name, probabilities = infer(image_path)
	
	# 
	print(f'Predicted Class: {predicted_class_name}')
	print('Probabilities:', probabilities)

