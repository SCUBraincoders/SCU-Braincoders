# SCU-Braincoders
Bu proje kapsamında, Teknofest 2021 tarafından sağlanan inme (stroke) veri seti kullanılarak iki sınıflı bir veri seti (inme var / inme yok) oluşturulacaktır. Projenin temel amacı, transfer öğrenme temelli farklı derin öğrenme modellerinin bu veri seti üzerinde performansını değerlendirmek ve en başarılı iki modelin çıktılarından yararlanarak topluluk öğrenme (ensemble learning) yaklaşımı ile daha güçlü ve kararlı bir sınıflandırma modeli geliştirmektir. Bu sayede inme tanısında yapay zekâ tabanlı karar destek sistemlerinin doğruluk ve güvenilirliğinin artırılması hedeflenmektedir.

# Veri Seti
Kaynak: T.C. Sağlık Bakanlığı (2021) inme verisi
(https://acikveri.saglik.gov.tr/Home/DataSetDetail/1)

Sınıflar: İnme Var / İnme Yok
Başlangıçta “İnme Yok”, “Kanama” ve “İskemi” olarak üç sınıf içeren veri seti, bu projedeki ikili sınıflandırma hedefi doğrultusunda “Kanama” ve “İskemi” sınıfları birleştirilerek “İnme Var” şeklinde yeniden düzenlenmiştir.

Fold Kullanımı: Çapraz validasyon (CV) ile 3 alt küme oluşturulmuş; en iyi model ağırlıkları CV3 üzerinden elde edilmiştir. CV3 ile topluluk öğrenme modeli (VGG16 + MobileNetV3_Large) %99.67 F1 skoru ile en yüksek performansı göstermiştir.

Veri Artırma: Görüntülerin çeşitlendirilmesi amacıyla veri artırma (augmentation) yöntemleri kullanılmıştır. Uygulanan işlemler:
– Rastgele -10 ile +10 derece döndürme
– 1.0 ile 1.2 oranında yakınlaştırma
– -10 ile +10 piksel arasında yatay ve dikey kaydırma
– Yatay çevirme

Eğitim ve Test Ayrımı: Veriler %80 eğitim ve %20 test olacak şekilde bölünmüştür.

Ekstra Test Seti: Modelin genellenebilirliğini değerlendirmek amacıyla Kaggle üzerinden alınan harici bir test seti
(https://www.kaggle.com/datasets/felipekitamura/head-ct-hemorrhage) ile ek test gerçekleştirilmiştir.

# Veri Seti
Bu çalışmada, ResNet18, ResNet50, DenseNet121, DenseNet201, InceptionV3, EfficientNetB0 ve EfficientNetB3 modelleri kullanılmıştır. Tüm modellerin öznitelik çıkarım katmanları sabit tutularak inme sınıflandırma amacına uyarlanmış, bilgi damıtma yöntemiyle öğrenci modellerin eğitilmesinde rehber olarak kullanılmıştır.

Modellerin sonuna, düzleştirilmiş öznitelikler üzerine 256 nöronlu iki tam bağlantılı katman eklenmiş ve softmax aktivasyon fonksiyonu ile ikili sınıflandırma gerçekleştirilmiştir.

#Kullanım Talimatları
1. Ortamı Hazırlama
Gerekli tüm Python kütüphanelerini aşağıdaki komutla kurabilirsiniz:

```bash
pip install -r requirements.txt
```


2. Model Eğitimi

  
3. Topluluk Öğrenme Modelini Oluşturma


4. Harici Veri Seti ile Test Etme

  
5. Örnek Çalıştırma







