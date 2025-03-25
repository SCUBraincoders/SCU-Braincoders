# SCU-Braincoders
Bu proje kapsamında, Teknofest 2021 tarafından sağlanan inme (stroke) veri seti kullanılarak iki sınıflı bir veri seti (inme var / inme yok) oluşturulacaktır. Projenin temel amacı, transfer öğrenme temelli farklı derin öğrenme modellerinin bu veri seti üzerinde performansını değerlendirmek ve en başarılı iki modelin çıktılarından yararlanarak topluluk öğrenme (ensemble learning) yaklaşımı ile daha güçlü ve kararlı bir sınıflandırma modeli geliştirmektir. Bu sayede inme tanısında yapay zekâ tabanlı karar destek sistemlerinin doğruluk ve güvenilirliğinin artırılması hedeflenmektedir.

##	 Veri Seti
Kaynak: T.C. Sağlık Bakanlığı (2021) inme verisi
(https://acikveri.saglik.gov.tr/Home/DataSetDetail/1)

Sınıflar: İnme Var / İnme Yok
Başlangıçta “İnme Yok”, “Kanama” ve “İskemi” olarak üç sınıf içeren veri seti, bu projedeki ikili sınıflandırma hedefi doğrultusunda “Kanama” ve “İskemi” sınıfları birleştirilerek “İnme Var” şeklinde yeniden düzenlenmiştir.

Fold Kullanımı: Çapraz validasyon (CV) ile 3 alt küme oluşturulmuş; en iyi model ağırlıkları CV3 üzerinden elde edilmiştir. CV3 ile topluluk öğrenme modeli (VGG16 + MobileNetV3_Large) %99.67 F1 skoru ile en yüksek performansı göstermiştir.

**Veri Artırma:**  
Görüntülerin çeşitlendirilmesi amacıyla veri artırma (augmentation) yöntemleri kullanılmıştır.  
Uygulanan işlemler:

- Rastgele -10 ile +10 derece döndürme  
- 1.0 ile 1.2 oranında yakınlaştırma  
- -10 ile +10 piksel arasında yatay ve dikey kaydırma  
- Yatay çevirme

Eğitim ve Test Ayrımı: Veriler %80 eğitim ve %20 test olacak şekilde bölünmüştür.

Ekstra Test Seti: Modelin genellenebilirliğini değerlendirmek amacıyla Kaggle üzerinden alınan harici bir test seti
(https://www.kaggle.com/datasets/felipekitamura/head-ct-hemorrhage) ile ek test gerçekleştirilmiştir.

## Kullanılan Modeller 
Bu çalışmada, ResNet18, ResNet50, DenseNet121, DenseNet201, InceptionV3, EfficientNetB0 ve EfficientNetB3 modelleri kullanılmıştır. Tüm modellerin öznitelik çıkarım katmanları sabit tutularak inme sınıflandırma amacına uyarlanmış, bilgi damıtma yöntemiyle öğrenci modellerin eğitilmesinde rehber olarak kullanılmıştır.

Modellerin sonuna, düzleştirilmiş öznitelikler üzerine 256 nöronlu iki tam bağlantılı katman eklenmiş ve softmax aktivasyon fonksiyonu ile ikili sınıflandırma gerçekleştirilmiştir.

## 🔧Kurulum ve Kullanım Kılavuzu
### 1. Ortamı Hazırlama
Gerekli tüm Python kütüphanelerini aşağıdaki komutla kurabilirsiniz:

```bash
pip install -r requirements.txt
```


### 2. Model Eğitimi
Bu projede, ResNet18, ResNet50, DenseNet121, DenseNet201, InceptionV3, EfficientNetB0 ve EfficientNetB3 modelleri kullanılmıştır.
Tüm modellerde, öznitelik katmanları sabit tutulmuş; ardından düzleştirme işlemi uygulanarak 256x256 boyutunda iki tam bağlantılı katman eklenmiş ve softmax aktivasyon fonksiyonu ile ikili sınıflandırma gerçekleştirilmiştir. Ayrıca, bilgi damıtma (knowledge distillation) yöntemiyle bazı modeller öğretici (teacher) olarak kullanılmıştır.

Aşağıdaki komutlarla modelleri ayrı ayrı eğitebilirsiniz:
```bash
python ResNet18_Train.py
python ResNet50_Train.py
python DenseNet121_Train.py
python DenseNet201_Train.py
python InceptionV3_Train.py
python EfficientNetB0_Train.py
python EfficientNetB3_Train.py
```
  
### 3. Topluluk Öğrenme Modelini Oluşturma
En yüksek ortalama F1 skoru ve doğruluk değerlerine sahip modeller InceptionV3, EfficientNetB3 ve EfficientNetB0 olmuştur.
Özellikle InceptionV3, tüm metriklerde en yüksek sonuçları vererek en güçlü genel performansı göstermiştir.
Bu bağlamda, topluluk öğrenme modeli en başarılı iki model olan EfficientNetB3 + InceptionV3 kullanılarak oluşturulmuştur:

```bash
python topluluk_ogrenme_inception_efficientnetb3.py
```

### 4. KD (Knowledge Distillation) ile Eğitilen Modeller
**KD Yöntemi ile En Başarılı Model:**  
KD (Knowledge Distillation) yöntemiyle eğitilen modeller arasında **EfficientNetB0**, aşağıdaki değerlerle en yüksek genel başarıyı göstermiştir:

- Ortalama F1 skoru: 0.9797  
- Precision: 0.9954  
- Recall: 0.9644  
- Doğruluk: 0.9800  

Bu nedenle, sınıflandırma görevleri için **KD ile eğitilen EfficientNetB0 modeli** önerilmektedir.

  
### 5. Harici Veri Seti ile Test
Kaggle üzerinden elde edilen harici veri seti ile modelin genel performansını test etmek için:

```bash
python external/external_test.py
```

### 6. Örnek Tahmin
Bir "inme var" ve bir "inme yok" görüntüsü üzerinden örnek tahmin almak için:
****
veri seti(melis)
****

## Sonuçlar
(melis)

## 🤝 Katkıda Bulunma
Projeye katkıda bulunmak için fork alarak değişiklik yapabilir ve pull request gönderebilirsiniz.
Hata bildirimleri ve öneriler için ise issue oluşturmanız yeterlidir.

Her türlü katkı ve geri bildirim memnuniyetle karşılanır. 🙌









