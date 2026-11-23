# Building-an-AI-Classifier-Identifying-Cats-Dogs-Pandas-with-PyTorch-
# Cat Dog Panda Image Classification Using Transfer Learning

## AIM
To develop an image classification model using Transfer Learning in PyTorch to classify images into Cat, Dog, and Panda categories.

## Dataset
Kaggle Cats vs Dogs vs Pandas Dataset

Dataset Link:
https://www.kaggle.com/datasets/gpiosenka/cats-dogs-pandas-images

## Model
- Pre-trained Model: ResNet18
- Hidden Layer: 256 Neurons
- Activation: ReLU
- Dropout: 0.5
- Output Classes: 3
- Optimizer: Adam
- Learning Rate: 0.001
- Loss Function: CrossEntropyLoss
- Epochs: 12

## CUDA Verification
```python
import torch
print("CUDA Available:", torch.cuda.is_available())
print("Device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
```

## Installation
```bash
pip install -r requirements.txt
```

## Dataset Download
```bash
kaggle datasets download -d gpiosenka/cats-dogs-pandas-images -p ./data/raw --unzip
```

Configure your Kaggle API token before downloading.

## Program
```


!pip install -q huggingface_hub scikit-learn seaborn
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader,Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from huggingface_hub import snapshot_download
from google.colab import files
SEED=42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:",device)
print("Downloading dataset...")
root=Path(snapshot_download(repo_id="Melisa13/Animals_dataset",repo_type="dataset",local_dir="/content/hf_dataset"))
print("Dataset downloaded successfully!")
train_dirs=[p for p in root.rglob("train") if (p/"cats").exists() and (p/"dogs").exists() and (p/"panda").exists()]
test_dirs=[p for p in root.rglob("test") if (p/"cats").exists() and (p/"dogs").exists() and (p/"panda").exists()]
if not train_dirs or not test_dirs:
    raise FileNotFoundError("Train/Test folders with cats, dogs and panda were not found. Please check dataset folder structure.")
train_dir=train_dirs[0]
test_dir=test_dirs[0]
print("Train folder:",train_dir)
print("Test folder:",test_dir)
IMG_SIZE=224
BATCH_SIZE=16
train_transform=transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)),transforms.RandomHorizontalFlip(),transforms.RandomRotation(10),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
eval_transform=transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
full_train=datasets.ImageFolder(train_dir,transform=train_transform)
val_source=datasets.ImageFolder(train_dir,transform=eval_transform)
test_dataset=datasets.ImageFolder(test_dir,transform=eval_transform)
class_names=full_train.classes
print("Classes:",class_names)
print("Training images:",len(full_train))
print("Testing images:",len(test_dataset))
if len(class_names)!=3:
    raise ValueError("Dataset must contain exactly 3 classes.")
indices=np.arange(len(full_train))
targets=full_train.targets
train_indices,val_indices=train_test_split(indices,test_size=0.15,random_state=SEED,stratify=targets)
train_dataset=Subset(full_train,train_indices)
val_dataset=Subset(val_source,val_indices)
train_loader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=2,pin_memory=True)
val_loader=DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=2,pin_memory=True)
test_loader=DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=2,pin_memory=True)
print("Train split:",len(train_dataset))
print("Validation split:",len(val_dataset))
print("Test split:",len(test_dataset))
def show_images(dataset,class_names,count=6):
    plt.figure(figsize=(12,7))
    for i in range(min(count,len(dataset))):
        image,label=dataset[i]
        image=image.permute(1,2,0).numpy()
        mean=np.array([0.485,0.456,0.406])
        std=np.array([0.229,0.224,0.225])
        image=np.clip(image*std+mean,0,1)
        plt.subplot(2,3,i+1)
        plt.imshow(image)
        plt.title(class_names[label])
        plt.axis("off")
    plt.tight_layout()
    plt.show()
show_images(train_dataset,class_names)
weights=models.ResNet18_Weights.DEFAULT
model=models.resnet18(weights=weights)
for param in model.parameters():
    param.requires_grad=False
model.fc=nn.Linear(model.fc.in_features,3)
model=model.to(device)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.fc.parameters(),lr=0.001)
print("ResNet18 Transfer Learning Model Loaded")
EPOCHS=10
history={"train_loss":[],"val_loss":[],"train_acc":[],"val_acc":[]}
best_val_acc=0.0
for epoch in range(EPOCHS):
    model.train()
    running_loss=0.0
    correct=0
    total=0
    for images,labels in train_loader:
        images,labels=images.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()*images.size(0)
        _,predicted=torch.max(outputs,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
    train_loss=running_loss/total
    train_acc=correct/total
    model.eval()
    val_loss_sum=0.0
    val_correct=0
    val_total=0
    with torch.no_grad():
        for images,labels in val_loader:
            images,labels=images.to(device),labels.to(device)
            outputs=model(images)
            loss=criterion(outputs,labels)
            val_loss_sum+=loss.item()*images.size(0)
            _,predicted=torch.max(outputs,1)
            val_total+=labels.size(0)
            val_correct+=(predicted==labels).sum().item()
    val_loss=val_loss_sum/val_total
    val_acc=val_correct/val_total
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    if val_acc>best_val_acc:
        best_val_acc=val_acc
        torch.save(model.state_dict(),"/content/best_cat_dog_panda_model.pth")
    print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f} Train Accuracy: {train_acc*100:.2f}% Validation Loss: {val_loss:.4f} Validation Accuracy: {val_acc*100:.2f}%")
print("Training completed!")
plt.figure(figsize=(8,5))
plt.plot(history["train_acc"],label="Training Accuracy")
plt.plot(history["val_acc"],label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(8,5))
plt.plot(history["train_loss"],label="Training Loss")
plt.plot(history["val_loss"],label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
model.load_state_dict(torch.load("/content/best_cat_dog_panda_model.pth",map_location=device))
model.eval()
all_labels=[]
all_predictions=[]
with torch.no_grad():
    for images,labels in test_loader:
        images=images.to(device)
        outputs=model(images)
        _,predicted=torch.max(outputs,1)
        all_labels.extend(labels.numpy())
        all_predictions.extend(predicted.cpu().numpy())
test_accuracy=accuracy_score(all_labels,all_predictions)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print("\nClassification Report:")
print(classification_report(all_labels,all_predictions,target_names=class_names,zero_division=0))
cm=confusion_matrix(all_labels,all_predictions)
plt.figure(figsize=(7,5))
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=class_names,yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()
def predict_image(image_path):
    image=Image.open(image_path).convert("RGB")
    input_tensor=eval_transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output=model(input_tensor)
        probabilities=torch.softmax(output,dim=1)
        confidence,prediction=torch.max(probabilities,1)
    return class_names[prediction.item()],confidence.item()*100
plt.figure(figsize=(12,8))
for i in range(min(6,len(test_dataset))):
    image,label=test_dataset[i]
    image_path,_=test_dataset.samples[i]
    predicted,confidence=predict_image(image_path)
    image=image.permute(1,2,0).numpy()
    mean=np.array([0.485,0.456,0.406])
    std=np.array([0.229,0.224,0.225])
    image=np.clip(image*std+mean,0,1)
    plt.subplot(2,3,i+1)
    plt.imshow(image)
    plt.title(f"Actual: {class_names[label]}\nPredicted: {predicted} ({confidence:.1f}%)")
    plt.axis("off")
plt.tight_layout()
plt.show()
print("Upload one Cat, Dog or Panda image for prediction:")
uploaded=files.upload()
for filename in uploaded.keys():
    try:
        image=Image.open(filename).convert("RGB")
        predicted,confidence=predict_image(filename)
        plt.figure(figsize=(5,5))
        plt.imshow(image)
        plt.title(f"Prediction: {predicted}\nConfidence: {confidence:.2f}%")
        plt.axis("off")
        plt.show()
        print("Predicted Class:",predicted)
        print(f"Confidence: {confidence:.2f}%")
    except Exception:
        print("Please upload a valid JPG, JPEG or PNG image.")
model_path="/content/Cat_Dog_Panda_Classifier.pth"
torch.save({"model_state_dict":model.state_dict(),"class_names":class_names,"img_size":IMG_SIZE},model_path)
print("Model saved successfully!")
files.download(model_path)


```
## Output

<img width="765" height="563" alt="Screenshot 2026-11-23 103623" src="https://github.com/user-attachments/assets/e383cef5-8929-4914-95a4-3bfffc53e7e1" />

## Execution
Open the notebook:
notebooks/cat_dog_panda_transfer_learning.ipynb

Run all cells in order.

## Results
The model reports test loss, test accuracy, confusion matrix, and example predictions.
