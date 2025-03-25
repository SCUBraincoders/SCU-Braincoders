import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader,Dataset
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import numpy as np
import timm
import warnings

from PIL import Image
import os
import copy
import matplotlib.pyplot as plt
import gc

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### MODELLERİ YÜKLE
class KDResNet18(nn.Module):
    def __init__(self):
        super(KDResNet18, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.model(x)

class KDResNet50(nn.Module):
    def __init__(self):
        super(KDResNet50, self).__init__()
        self.model = models.resnet50(pretrained=False)
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.model(x)

class KDDenseNet121(nn.Module):
    def __init__(self):
        super(KDDenseNet121, self).__init__()
        self.model = models.densenet121(pretrained=False)
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.model(x)

class KDDenseNet201(nn.Module):
    def __init__(self):
        super(KDDenseNet201, self).__init__()
        self.model = models.densenet201(pretrained=False)
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.model(x)

class KDEfficientNetB0(nn.Module):
    def __init__(self):
        super(KDEfficientNetB0, self).__init__()
        self.model = models.efficientnet_b0(pretrained=False)
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier[1].in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.model(x)

class KDEfficientNetB3(nn.Module):
    def __init__(self):
        super(KDEfficientNetB3, self).__init__()
        self.model = models.efficientnet_b3(pretrained=False)
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier[1].in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.model(x)

# Modelleri yükle
kd_resnet18 = KDResNet18().to(device)
kd_resnet18.load_state_dict(torch.load(r"/content/drive/MyDrive/Teknofest_Braincoders/EnsembleModels/kd_resnet18_Fold1.pth"))
kd_resnet18.eval()

kd_resnet50 = KDResNet50().to(device)
kd_resnet50.load_state_dict(torch.load(r"/content/drive/MyDrive/Teknofest_Braincoders/EnsembleModels/kd_resnet50_Fold1.pth"))
kd_resnet50.eval()

kd_densenet121 = KDDenseNet121().to(device)
kd_densenet121.load_state_dict(torch.load(r"/content/drive/MyDrive/Teknofest_Braincoders/EnsembleModels/kd_densenet121_Fold1.pth"))
kd_densenet121.eval()

kd_densenet201= KDDenseNet201().to(device)
kd_densenet201.load_state_dict(torch.load(r"/content/drive/MyDrive/Teknofest_Braincoders/EnsembleModels/kd_densenet201_Fold1.pth"))
kd_densenet201.eval()

kd_efficientnetb0 = KDEfficientNetB0().to(device)
kd_efficientnetb0.load_state_dict(torch.load(r"/content/drive/MyDrive/Teknofest_Braincoders/EnsembleModels/kd_efficientnet_b0_Fold1.pth"))
kd_efficientnetb0.eval()

kd_efficientnetb3 = KDEfficientNetB3().to(device)
kd_efficientnetb3.load_state_dict(torch.load(r"/content/drive/MyDrive/Teknofest_Braincoders/EnsembleModels/kd_efficientnet_b3_Fold1.pth"))
kd_efficientnetb3.eval()

models = [kd_efficientnetb0, kd_efficientnetb3, kd_densenet201, kd_resnet18,kd_resnet50, kd_densenet121]

#models = [kd_efficientnetb0, kd_efficientnetb3]

### Veriyi hazırlıyoruz
class StrokeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = {'Inme-Yok': 0, 'Inme-Var': 1}
        self.image_paths = []
        self.labels = []
        valid_extensions = ['.jpg', '.jpeg', '.png']

        for class_name, class_idx in self.classes.items():
            class_folder = os.path.join(root_dir, class_name)
            if not os.path.exists(class_folder):
                continue
            for img_name in os.listdir(class_folder):
                if os.path.splitext(img_name)[1].lower() in valid_extensions:
                    img_path = os.path.join(class_folder, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Dönüşümler
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

base_path=r'/content/drive/MyDrive/Teknofest_Braincoders/Eksternal_Veriseti/head_ct'
test_dataset = StrokeDataset(base_path, transform=transform) 
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

## Modelleri Test Etme
logits_list = []
true_labels = []
predicted_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        true_labels.extend(labels.cpu().numpy())

        logits = []

        for model in models:
            logits.append(model(images))

        logits_avg = torch.mean(torch.stack(logits), dim=0)
        logits_list.append(logits_avg.cpu())

        predicted_labels.extend(torch.argmax(logits_avg, dim=1).cpu().numpy())

# with torch.no_grad():
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             true_labels.extend(labels.cpu().numpy())
#             outputs = kd_resnet18(images)
#             _, preds = torch.max(outputs, 1)
#             predicted_labels.extend(preds.cpu().numpy())

true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

conf_matrix = confusion_matrix(true_labels, predicted_labels)
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')

print("Confusion Matrix:\n", conf_matrix)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

file_path = r'/content/drive/MyDrive/Teknofest_Braincoders/Son_Kisim/kd_ensemble_6model.txt'
with open(file_path, "w") as file:
        file.write(f"Final Metrics for fold1 :\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall: {recall:.4f}\n")
        file.write(f"F1 Score: {f1:.4f}\n")
print(f"Metrikler kaydedildi")