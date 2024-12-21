
![proje](https://github.com/user-attachments/assets/ab58b797-327f-4e93-96fa-3bef9ec15f78)

# Kömür Sınıflandırma Uygulaması - Proje Detayları
## Proje Özellikleri

Bu proje, farklı kömür türlerini sınıflandırmak için **EfficientNetB0** tabanlı bir derin öğrenme modeli sunar. Model, kullanıcıdan alınan bir kömür görüntüsünü analiz ederek şu sınıflardan birini tahmin eder:

- **Anthracite**
- **Bituminous**
- **Lignite**
- **Peat**

Proje aynı zamanda tahmin sonuçlarını çubuk grafik ve pasta grafiklerle görselleştirir ve web arayüzü üzerinden kolay kullanım sağlar.

## Kullanılan Teknolojiler

- **TensorFlow/Keras**: EfficientNetB0 modeli ve derin öğrenme altyapısı için.
- **Streamlit**: Kullanıcı dostu web arayüzü oluşturmak için.
- **Playwright**: Web arayüz testlerini gerçekleştirmek için.
- **Pytest**: Model ve fonksiyonlar için birim testler.
- **Plotly**: Grafiksel görselleştirme için.
- **Python**: Projenin temel programlama dili.
- **NumPy**: Veri işlemleri ve matematiksel hesaplamalar için.
- **Matplotlib**: Görsel doğrulama ve veri kümesi görselleştirme.
- **Seaborn**: Görselleştirmeler için estetik veri grafikleri.
- **OpenCV**: Görüntü işleme araçları için.

## Kütüphaneler

Proje aşağıdaki Python kütüphanelerini kullanmaktadır:

```bash
tensorflow
keras
numpy
streamlit
plotly
pytest
playwright
opencv-python
pillow
matplotlib
seaborn
```

## Kurulum

Bu projeyi kullanmak için aşağıdaki adımları takip edin:

### 1. Gerekli Bağımlılıkları Kurun

Proje dizininde aşağıdaki komutları çalıştırarak gerekli kütüphaneleri yükleyin:

```bash
pip install tensorflow keras numpy streamlit plotly pytest playwright opencv-python pillow matplotlib seaborn
```

### 2. Playwright Testleri İçin Kurulum

Playwright ile testleri çalıştırmak için:

```bash
playwright install
```

## Kullanım

### 1. Uygulamayı Çalıştırma

Streamlit uygulamasını başlatmak için proje dizininde şu komutu çalıştırın:

```bash
streamlit run app.py
```

### 2. Web Arayüzü Üzerinden Kullanım

- Açılan tarayıcıda bir kömür görüntüsü yükleyin.
- Model, görüntüyü analiz ederek tahmin sonuçlarını ve olasılıkları grafiksel olarak gösterecektir.

### 3. Testleri Çalıştırma

Projedeki testleri çalıştırmak için şu komutları kullanın:

```bash
pytest
```

### 4. Kodun Test Edilmesi (Playwright ile)

Playwright testlerini çalıştırmak için şu komutları çalıştırın:

```bash
pytest tests/test_playwright.py
```

## Proje Yapısı

```plaintext
project/
├── app.py                   # Ana Streamlit uygulaması
├── model_training.py        # Model eğitimi için kod
├── test_model.py            # Pytest ile model testi
├── playwright_tests/        # Playwright ile web arayüz testleri
├── requirements.txt         # Gerekli kütüphaneler
└── README.md                # Proje açıklaması
```

Bu rehber ile projeyi kolayca kurabilir, çalıştırabilir ve test edebilirsiniz! 😊
```

