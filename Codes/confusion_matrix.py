import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
import gc

model_name = "KD_Efficientnet_b0"

# Yol tanımlamaları
dataset_path = r'/content/drive/MyDrive/Teknofest_Braincoders/Alt_Kumeler_Augmantasyon'
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Veri seti dizini bulunamadı: {dataset_path}")

output_path = '/content/drive/MyDrive/Teknofest_Braincoders/Results/Confusion_Matrix/KD_Efficientnet_b0'
if not os.path.exists(output_path):
    raise FileNotFoundError(f"Sonuç ekleme dizini bulunamadı: {output_path}")

model_path_template = '/content/drive/MyDrive/Teknofest_Braincoders/Models/KD_Models/KD_Efficientnet_b0/kd_efficientnet_b0_{fold}.pth'

folds = ['Fold1']

# Cihaz seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Kullanılmayan GPU belleğini temizle (Bellek optimizasyonu için)
torch.cuda.empty_cache()
gc.collect()

print(f"{model_name} modeli için confusion matrisi oluşturulacak.")
print(f"Değerlendirilecek modeller: {folds}")
print(f"Kullanılacak cihaz: {device}")

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
        return image, label, img_path  # img_path ekledik

# Görüntü dönüşümleri
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Modeli yükleme fonksiyonu
def load_model(model_path):
    model = models.efficientnet_b0(pretrained=False)

    # Modify classifier as needed
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 2)  # 2 classes: Inme-Yok, Inme-Var
    )

    # Load the saved state_dict
    state_dict = torch.load(model_path, map_location=device)

    # Remove the "model." prefix from state_dict keys (or any other prefix)
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

    # Filter out keys that don't match the current model architecture
    model_state_dict = model.state_dict()

    # Create a new state_dict that only contains the matching layers
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}

    # Load the matching state_dict into the model
    model_state_dict.update(filtered_state_dict)

    try:
        # Now load the filtered state_dict into the model
        model.load_state_dict(model_state_dict, strict=True)
    except RuntimeError as e:
        print(f"Error loading model state dict: {e}")
        # You can also print the keys to debug
        print("Mismatch in keys, here are the mismatched layers:")
        print(f"Loaded keys: {list(state_dict.keys())}")
        print(f"Expected keys: {list(model_state_dict.keys())}")

    model.to(device)
    model.eval()
    return model

# Confusion matrisi oluşturma ve kaydetme fonksiyonu
def evaluate_and_save_confusion_matrix(model, dataloader, fold_name, output_path):
    y_true, y_pred, file_names = [], [], []

    with torch.no_grad():
        for images, labels, paths in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            file_names.extend(paths)

    # Confusion Matrix hesapla
    cm = confusion_matrix(y_true, y_pred)
    class_names = ['Inme-Yok', 'Inme-Var']

    # Confusion Matrix görselleştir
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 35})
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')
    plt.title(f'Confusion Matrix - {fold_name}')

    # Kaydet
    os.makedirs(output_path, exist_ok=True)
    cm_img_path = os.path.join(output_path, f'{model_name}_confusion_matrix_{fold_name}.png')
    plt.savefig(cm_img_path)
    plt.show()

    # Confusion Matrix değerlerini txt olarak kaydet
    cm_txt_path = os.path.join(output_path, f'{model_name}_confusion_matrix_{fold_name}.txt')
    with open(cm_txt_path, 'w') as f:
        f.write(f'Confusion Matrix for {fold_name}\n')
        f.write(np.array2string(cm))

    print("Matris kaydedildi")

    # Yanlış tahmin edilen verileri txt'ye kaydet
    errors = [(file_names[i], y_true[i], y_pred[i]) for i in range(len(y_true)) if y_true[i] != y_pred[i]]
    errors_df = pd.DataFrame(errors, columns=['File', 'True Label', 'Predicted Label'])
    errors_df.to_csv(os.path.join(output_path, f'{model_name}_errors_{fold_name}.csv'), index=False)

    print("Tahmin değerleri kaydedildi")

# Her fold için işlemleri gerçekleştir
for fold in folds:

    print(f'{fold} işleniyor...')

    model_path = model_path_template.format(fold=fold)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

    print(f"{fold} için model yüklenecek: {model_path}")
    model = load_model(model_path)

    test_dataset = StrokeDataset(os.path.join(dataset_path, fold, 'test'), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    evaluate_and_save_confusion_matrix(model, test_loader, fold, output_path)
    print(f'{fold} tamamlandı')