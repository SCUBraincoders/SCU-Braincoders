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

# Veri kümesini hazırlama
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

model_name = "EfficientNetb0"

# Optimizasyon ve Tanımlamalar
dataset_path = r'/content/drive/MyDrive/Teknofest_Braincoders/Alt_Kumeler_Augmantasyon'
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Veri seti dizini bulunamadı: {dataset_path}")

result_path = r'/content/drive/MyDrive/Teknofest_Braincoders/Sonuclar'
if not os.path.exists(result_path):
    raise FileNotFoundError(f"Sonuç ekleme dizini bulunamadı: {result_path}")

trained_model_path = "/content/drive/MyDrive/Teknofest_Braincoders/Modeller"
if not os.path.exists(trained_model_path):
    raise FileNotFoundError(f"Modelin kaydedileceği dizin bulunamadı: {trained_model_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Kullanılmayan GPU belleğini temizle (Bellek optimizasyonu için)
torch.cuda.empty_cache()
gc.collect()

folds = ['Fold1', 'Fold2', 'Fold3']
epoch = 50

print(f"{model_name} modeli {epoch} epoch ile eğitilecek.")
print(f"Eğitilecek klasörler: {folds}")
print(f"Kullanılacak cihaz: {device}")

# Model tanımlama
class StrokeModel(nn.Module):
    def __init__(self):
        super(StrokeModel, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        self.efficientnet.classifier = nn.Sequential(
            nn.Linear(self.efficientnet.classifier[1].in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # 2 sınıf (Inme-Yok, Inme-Var)
        )

    def forward(self, x):
        return self.efficientnet(x)

# Eğitim fonksiyonu
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=epoch, fold_name="fold"):

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []
    model_path = os.path.join(trained_model_path, f"{model_name}_{fold_name}.pth")

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
            torch.save(model.state_dict(), model_path)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accuracies[-1]:.4f}")
        save_metrics_to_file_txt(fold_name, epoch, train_losses[-1], test_losses[-1], train_accuracies[-1], test_accuracies[-1])

    model.load_state_dict(best_model_wts)
    return model, train_losses, test_losses, train_accuracies, test_accuracies

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

def plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies, fold_name, save_dir=result_path):

    os.makedirs(save_dir, exist_ok=True)  # Klasör yoksa oluştur
    file_path = os.path.join(save_dir, f"{model_name}_plots_{fold_name}.png")

    plt.figure(figsize=(10, 6))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label="Training Accuracy")
    plt.plot(test_accuracies, label="Testing Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{fold_name} Training and Testing Accuracy")
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(test_losses, label="Testing Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{fold_name} Training and Testing Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(file_path)  # PNG olarak kaydet
    plt.show()
    print(f"Grafikler kaydedildi")

# Lossları kaydet
def save_metrics_to_file_txt(fold_name, epoch, train_loss, test_loss, train_acc, test_acc, save_dir=result_path):
    os.makedirs(save_dir, exist_ok=True)  # Klasör yoksa oluştur
    file_path = os.path.join(save_dir, f"{model_name}_losses_{fold_name}.txt")

    with open(file_path, "a") as file:  # Append mode: Önceki verileri silmeden ekleme yapar
        file.write(f"Epoch {epoch+1}:\n")
        file.write(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}\n")
        file.write(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}\n")
        file.write("="*50 + "\n")  # Ayrım çizgisi ekleyelim


# Metrikleri kaydet
def save_final_metrics_to_file(fold_name, labels, preds, save_dir=result_path):
    # Dosya yolunu oluştur
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{model_name}_metrics_{fold_name}.txt")

    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='binary')
    recall = recall_score(labels, preds, average='binary')
    f1 = f1_score(labels, preds, average='binary')

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    with open(file_path, "w") as file:
        file.write(f"Final Metrics for {fold_name}:\n")
        file.write(f"Accuracy: {acc:.4f}\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall: {recall:.4f}\n")
        file.write(f"F1 Score: {f1:.4f}\n")
    print(f"Metrikler kaydedildi")

print(f"Veriler yükleniyor")

# Cross Validation
for fold in folds:

    train_dataset = StrokeDataset(os.path.join(dataset_path, fold, 'train'), transform=transform)
    test_dataset = StrokeDataset(os.path.join(dataset_path, fold, 'test'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"{fold} eğitimi yapılıyor")

    model = StrokeModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    model, train_losses, test_losses, train_accuracies, test_accuracies = train_model(
        model, train_loader, test_loader, criterion, optimizer, fold_name=fold
    )

    # En iyi modeli kaydet
    model_path = os.path.join(trained_model_path, f"{model_name}_{fold}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"{fold} öğrenci modeli kaydedildi")

    print(f"{fold} değerlendirmesi yapılıyor")

    # Test için eğitilmiş modeli tekrar yükle
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Modeli test moduna al

    labels, preds = evaluate_model(model, test_loader)
    save_final_metrics_to_file(fold, labels, preds)
    plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies, fold)
    print(f"{fold} eğitimi tamamlandı")