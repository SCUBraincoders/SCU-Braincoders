import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import torchvision.models.inception
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

# Dönüşümler
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

base_path = r'D:\TEKNOFEST\KD_STROKE\Alt_Kumeler_Augmantasyon'
if not os.path.exists(base_path):
    raise FileNotFoundError(f"Veri seti dizini bulunamadı: {base_path}")

result_path = r'D:\TEKNOFEST\KD_STROKE\Sonuclar'
if not os.path.exists(result_path):
    raise FileNotFoundError(f"Sonuç ekleme dizini bulunamadı: {result_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Kullanılmayan GPU belleğini temizle (Bellek optimizasyonu için)
torch.cuda.empty_cache()
gc.collect()

folds = ['Fold1', 'Fold2', 'Fold3']
epochs = 50
batch_size=32

# Model Tanımı
class StrokeModel(nn.Module):
    def __init__(self):
        super(StrokeModel, self).__init__()
        self.inception = models.inception_v3(pretrained=True, aux_logits=True)
        self.inception.fc = nn.Sequential(
            nn.Linear(self.inception.fc.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # İkili sınıflandırma            
        )

    def forward(self, x):
        outputs = self.inception(x)  # InceptionOutputs dönecek
        # InceptionOutputs sınıfını doğru yerden kontrol ediyoruz
        if isinstance(outputs, torchvision.models.inception.InceptionOutputs):
            if self.inception.aux_logits:
                return outputs.logits, outputs.aux_logits  # Ana ve yardımcı çıkışlar
            return outputs.logits  # Sadece ana çıkış
        return outputs  # Yalnızca tensor döndürülürse, tek çıkış

# Öğrenci Modeli Tanımlama
class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.model = models.resnet18(pretrained=True)        
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
    def forward(self, x):
        return self.model(x)

# Model Yükleme Fonksiyonu
def load_teacher_model(model_path, device):
    model = StrokeModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def knowledge_distillation_loss(student_outputs, teacher_outputs, labels, temperature, alpha):
    kd_loss = nn.KLDivLoss(reduction='batchmean')(nn.functional.log_softmax(student_outputs / temperature, dim=1),
                                                  nn.functional.softmax(teacher_outputs / temperature, dim=1))
    ce_loss = nn.CrossEntropyLoss()(student_outputs, labels)
    return alpha * kd_loss + (1 - alpha) * ce_loss

def train_model_kd(student_model, teacher_model, train_loader, test_loader, scheduler, 
                   optimizer, epochs=epochs, fold_name="fold", temperature=4.0, alpha=0.5):
    
    best_model_wts = copy.deepcopy(student_model.state_dict())
    best_acc = 0.0
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    for epoch in range(epochs):
        student_model.train()  # Her epoch başında train moduna al!
        running_loss, correct_train, total_train = 0.0, 0, 0

        # Eğitim aşaması
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            student_outputs = student_model(images)

            with torch.no_grad():
                teacher_outputs = teacher_model(images)
                if isinstance(teacher_outputs, tuple):  # iki çıktı varsa
                    teacher_outputs = teacher_outputs[0]  # sadece logits'i al

            loss = knowledge_distillation_loss(student_outputs, teacher_outputs, labels, temperature, alpha)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(student_outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)

        # Değerlendirme aşaması
        student_model.eval()
        correct_test, total_test, test_loss = 0, 0, 0.0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = student_model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_test += (preds == labels).sum().item()
                total_test += labels.size(0)

        epoch_test_loss = test_loss / len(test_loader)
        epoch_test_acc = correct_test / total_test
        test_losses.append(epoch_test_loss)
        test_accuracies.append(epoch_test_acc)

        # LR güncellemesi test_loss'a göre yapılıyor!
        scheduler.step(epoch_test_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1} tamamlandı, güncel learning rate: {current_lr:.8f}")
        print(f"Epoch {epoch+1}: Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc:.4f}")

        # Metrikleri yazdır ve .txt dosyasına kaydet
        print(f"Epoch {epoch+1}: Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc:.4f}")
        save_metrics_to_file_txt(fold_name, epoch, epoch_train_loss, epoch_test_loss, epoch_train_acc, epoch_test_acc)
        
        # Accuracy önemliyse, test accuracy'ye göre en iyi modeli kaydet
        if epoch_test_acc > best_acc:
            best_acc = epoch_test_acc
            best_model_wts = copy.deepcopy(student_model.state_dict())
            
    student_model.load_state_dict(best_model_wts)
    return student_model, train_losses, test_losses, train_accuracies, test_accuracies

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
    file_path = os.path.join(save_dir, f"kd_student2_resnet18_{fold_name}.png")

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
    file_path = os.path.join(save_dir, f"kd_student2_resnet18_losses_{fold_name}.txt")

    with open(file_path, "a") as file:  # Append mode: Önceki verileri silmeden ekleme yapar
        file.write(f"Epoch {epoch+1}:\n")
        file.write(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}\n")
        file.write(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}\n")
        file.write("="*50 + "\n")  # Ayrım çizgisi ekleyelim


# Metrikleri kaydet
def save_final_metrics_to_file(fold_name, labels, preds, save_dir=result_path):
    # Dosya yolunu oluştur
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"kd_student2_resnet18_metrics_{fold_name}.txt")

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

    train_dataset = StrokeDataset(os.path.join(base_path, fold, 'train'), transform=transform)
    test_dataset = StrokeDataset(os.path.join(base_path, fold, 'test'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    teacher_model_path = "D:\TEKNOFEST\KD_STROKE\Modeller\inception_best_model_Fold1.pth"  # Model yolunu uygun şekilde değiştir
    teacher_model = load_teacher_model(teacher_model_path, device)
    teacher_model.eval()  # Modeli test moduna al

    student_model = StudentModel().to(device)
    optimizer = optim.Adam(student_model.parameters(), lr=0.0001)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    print(f"{fold} eğitimi yapılıyor")
    temperature = 5.0
    alpha = 0.7
    student_model, train_losses, test_losses, train_accuracies, test_accuracies = train_model_kd(
        student_model, teacher_model, train_loader,test_loader,scheduler, optimizer, fold_name=fold, temperature=temperature, alpha=alpha
    )

    # En iyi modeli kaydet
    model_path = f"D:\TEKNOFEST\KD_STROKE\kd_student2_resnet18_{fold}.pth"
    torch.save(student_model.state_dict(), model_path)
    print(f"{fold} öğrenci modeli kaydedildi")

    print(f"{fold} değerlendirmesi yapılıyor")

    # Test için eğitilmiş modeli tekrar yükle
    student_model.load_state_dict(torch.load(model_path))
    student_model.eval()  # Modeli test moduna al

    # Değerleri kaydet
    labels, preds = evaluate_model(student_model, test_loader)
    save_final_metrics_to_file(fold, labels, preds)
    plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies, fold)

    print(f"{fold} eğitimi tamamlandı")
