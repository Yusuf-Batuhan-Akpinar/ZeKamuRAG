# TÜBİTAK RAG Projesi 🔬

Google Gemini API kullanarak geliştirilmiş, belge tabanlı Soru-Cevap (RAG) sistemi.

## 🎯 Özellikler

- ✅ PDF belgelerinden otomatik bilgi çıkarma
- ✅ FAISS vektör veritabanı ile hızlı arama
- ✅ Google Gemini 2.5 Flash ile doğal dil işleme
- ✅ Streamlit ile modern web arayüzü
- ✅ Halüsinasyon önleme mekanizması
- ✅ Kaynak belge referansları

## 📋 Gereksinimler

- Python 3.10 veya üzeri
- Google Gemini API Key
- İnternet bağlantısı (API çağrıları için)

## 🚀 Kurulum

### 1. Projeyi İndirin
```bash
cd rag-tubitak-project
```

### 2. Sanal Ortam Oluşturun (Opsiyonel ama Önerilen)
```bash
python -m venv venv

# Windows için:
venv\Scripts\activate

# Linux/Mac için:
source venv/bin/activate
```

### 3. Gereksinimleri Yükleyin
```bash
pip install -r requirements.txt
```

### 4. Google API Key Ayarlayın

#### Yöntem 1: Environment Variable (Önerilen)
```bash
# Windows PowerShell:
$env:GOOGLE_API_KEY="your_api_key_here"

# Windows CMD:
set GOOGLE_API_KEY=your_api_key_here

# Linux/Mac:
export GOOGLE_API_KEY="your_api_key_here"
```

#### Yöntem 2: .env Dosyası
```bash
# .env.example dosyasını kopyalayın
copy .env.example .env

# .env dosyasını düzenleyin ve API key'inizi ekleyin
```

**API Key Nasıl Alınır?**
1. https://makersuite.google.com/app/apikey adresine gidin
2. Google hesabınızla giriş yapın
3. "Create API Key" butonuna tıklayın
4. Oluşturulan key'i kopyalayın

### 5. PDF Belgelerinizi Ekleyin
```bash
# data/ klasörü otomatik oluşturulacaktır
# PDF dosyalarınızı data/ klasörüne kopyalayın
```

## 💻 Kullanım

### Uygulamayı Başlatın
```bash
streamlit run app.py
```

Tarayıcınızda otomatik olarak `http://localhost:8501` açılacaktır.

### Adım Adım Kullanım

1. **PDF Belgelerini Ekleyin**: `data/` klasörüne PDF dosyalarınızı kopyalayın

2. **Sistemi Başlatın**: Sol sidebar'daki "Sistemi Başlat" butonuna tıklayın

3. **Soru Sorun**: Ana ekrandaki chat kutusuna sorunuzu yazın

4. **Cevabı Alın**: Sistem sadece yüklediğiniz belgelerden cevap üretecektir

5. **Kaynakları İnceleyin**: Cevabın altındaki "Kaynak Belgeler" bölümünü açarak referansları görebilirsiniz

## 📁 Proje Yapısı

```
rag-tubitak-project/
│
├── app.py                  # Ana uygulama dosyası
├── requirements.txt        # Python bağımlılıkları
├── .env.example           # API key şablon dosyası
├── README.md              # Bu dosya
│
├── data/                  # PDF belgeleriniz (siz oluşturacaksınız)
│   ├── belge1.pdf
│   ├── belge2.pdf
│   └── ...
│
└── vectorstore/           # FAISS vektör veritabanı (otomatik oluşur)
    ├── index.faiss
    └── index.pkl
```

## 🔧 Yapılandırma

### Model Ayarları

`app.py` dosyasında aşağıdaki parametreleri değiştirebilirsiniz:

```python
# LLM Modeli
self.llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # veya "gemini-1.5-pro"
    temperature=0.3,            # 0.0-1.0 arası (düşük = daha tutarlı)
)

# Chunk Ayarları
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Her parça boyutu
    chunk_overlap=200,    # Parçalar arası örtüşme
)

# Retrieval Ayarları
retriever=self.vectorstore.as_retriever(
    search_kwargs={"k": 3}  # Kaç belge parçası getirileceği
)
```

## ⚠️ Sorun Giderme

### "GOOGLE_API_KEY bulunamadı" Hatası
- API key'in doğru ayarlandığından emin olun
- Terminal/PowerShell'i yeniden başlatın

### "PDF dosyası bulunamadı" Uyarısı
- `data/` klasörünü oluşturun
- PDF dosyalarını bu klasöre kopyalayın
- "Sistemi Başlat" butonuna tekrar tıklayın

### Yavaş Yanıt Süresi
- `gemini-1.5-flash` yerine daha hızlı bir model kullanın
- `chunk_size` değerini düşürün
- İnternet bağlantınızı kontrol edin

### Hatalı Cevaplar
- `temperature` değerini düşürün (0.1-0.2)
- Prompt şablonunu daha spesifik hale getirin
- Daha kaliteli/yapılandırılmış PDF'ler kullanın

## 📝 Notlar

- **Veri Gizliliği**: Belgeler sadece yerel vektör veritabanında saklanır
- **API Limitleri**: Google Gemini API'nin ücretsiz kotası vardır
- **Performans**: İlk çalıştırmada vektör oluşturma biraz zaman alabilir
- **Güncelleme**: Yeni PDF eklerseniz `vectorstore/` klasörünü silin ve sistemi yeniden başlatın

## 🤝 Katkıda Bulunma

Bu bir TÜBİTAK projesi olduğu için kod temiz ve yorumlu tutulmuştur. 
Geliştirme önerileri için lütfen proje sorumlusuyla iletişime geçin.

## 📄 Lisans

TÜBİTAK Projesi - Sadece eğitim ve araştırma amaçlıdır.

## 🆘 Destek

Sorun yaşarsanız:
1. Bu README dosyasını dikkatlice okuyun
2. `app.py` içindeki Türkçe yorumları inceleyin
3. Hata mesajlarını not alın
4. Proje ekibiyle iletişime geçin

---

**Geliştirici Notu**: Bu proje LangChain, FAISS ve Google Gemini API'lerini kullanır. 
Her bileşen modüler yapıda olup, ihtiyaca göre özelleştirilebilir.
