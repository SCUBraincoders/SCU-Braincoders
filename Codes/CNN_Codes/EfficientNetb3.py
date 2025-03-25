import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image
import os
import copy
import matplotlib.pyplot as plt
import gc

# Veri kümesini hazırlama
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

# Dönüşümler
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Optimizasyon ve Tanımlamalar
base_path = r'/content/drive/MyDrive/Teknofest_Braincoders/Alt_Kumeler_Augmantasyon'
if not os.path.exists(base_path):
    raise FileNotFoundError(f"Veri seti dizini bulunamadı: {base_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Kullanılmayan GPU belleğini temizle (Bellek optimizasyonu için)
torch.cuda.empty_cache()
gc.collect()

folds = ['Fold1', 'Fold2', 'Fold3']
epoch = 50

# Model tanımlama
class StrokeModel(nn.Module):
    def __init__(self):
        super(StrokeModel, self).__init__()
        self.efficientnet = models.efficientnet_b3(pretrained=True)
        self.efficientnet.classifier = nn.Sequential(
            nn.Linear(self.efficientnet.classifier[1].in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # İkili sınıflandırma
        )

    def forward(self, x):
        return self.efficientnet(x)

# Eğitim fonksiyonu
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=epoch, fold_name="fold"):
    print(f"{fold_name} eğitimine {epochs} epoch ile başlanıyor")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(correct_train / total_train)

        # Test
        model.eval()
        correct_test, total_test, test_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_test += (preds == labels).sum().item()
                total_test += labels.size(0)

        test_losses.append(test_loss / len(test_loader))
        test_accuracy = correct_test / total_test
        test_accuracies.append(test_accuracy)

        if test_accuracy > best_acc:
            best_acc = test_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), f'/content/drive/MyDrive/Teknofest_Braincoders/efficientnet_best_model_{fold}.pth')

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accuracies[-1]:.4f}")

    model.load_state_dict(best_model_wts)
    return model, train_losses, test_losses, train_accuracies, test_accuracies

def plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies):
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_accuracies, label="Training Accuracy")
    plt.plot(epochs_range, test_accuracies, label="Testing Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Testing Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_losses, label="Training Loss")
    plt.plot(epochs_range, test_losses, label="Testing Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Testing Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate_model(model, test_loader):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    return all_labels, all_preds

def calculate_metrics(labels, preds):
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='binary')
    recall = recall_score(labels, preds, average='binary')
    f1 = f1_score(labels, preds, average='binary')
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

# Cross Validation
for fold in folds:
    print(f"{fold} eğitimi yapılıyor")

    train_dataset = StrokeDataset(os.path.join(base_path, fold, 'train'), transform=transform)
    test_dataset = StrokeDataset(os.path.join(base_path, fold, 'test'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = StrokeModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    model, train_losses, test_losses, train_accuracies, test_accuracies = train_model(
        model, train_loader, test_loader, criterion, optimizer, fold_name=fold
    )

    # En iyi modeli kaydet
    torch.save(model.state_dict(), f'/content/drive/MyDrive/Teknofest_Braincoders/efficientnet_best_model_{fold}.pth')

    print(f"{fold} değerlendirmesi yapılıyor")

    # Test için eğitilmiş modeli tekrar yükle
    model.load_state_dict(torch.load(f'/content/drive/MyDrive/Teknofest_Braincoders/efficientnet_best_model_{fold}.pth'))
    model.eval()  # Modeli test moduna al

    labels, preds = evaluate_model(model, test_loader)
    calculate_metrics(labels, preds)

    plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies)