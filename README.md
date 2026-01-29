# 🖼️ Object Detector - Resim Obje Tanıma Programı

Bu program, verdiğiniz resimlerdeki objeleri otomatik olarak tanır ve listeler.
YOLOv8 yapay zeka modelini kullanır ve 80 farklı obje kategorisini tanıyabilir.

## 🚀 Kurulum

```bash
# Gerekli kütüphaneleri yükleyin
pip install -r requirements.txt
```

## 📖 Kullanım

### Temel Kullanım
```bash
python object_detector.py resim.jpg
```

### Tespit Edilen Objeleri Kaydet
```bash
python object_detector.py resim.jpg --save
```
Bu komut, objelerin üzerine kutu çizerek yeni bir resim oluşturur.

### Güven Eşiğini Ayarla
```bash
python object_detector.py resim.jpg --conf=0.3
```
Daha düşük değer = daha fazla obje tespit edilir (ama daha az güvenilir)

## 🎯 Tanınabilen Objeler (80 kategori)

- **İnsanlar**: person
- **Araçlar**: bicycle, car, motorcycle, airplane, bus, train, truck, boat
- **Hayvanlar**: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
- **Yiyecekler**: banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake
- **Elektronik**: tv, laptop, mouse, remote, keyboard, cell phone
- **Mobilya**: chair, couch, bed, dining table
- **Ve daha fazlası...**

## 📋 Örnek Çıktı

```
🔍 Resim analiz ediliyor: test.jpg
--------------------------------------------------

✅ Toplam 5 obje tespit edildi:

📋 Tespit Edilen Objeler:
========================================
  • car: 2 adet
  • person: 3 adet

📊 Detaylı Liste:
========================================
  1. person (Güven: %92.5)
  2. person (Güven: %88.3)
  3. car (Güven: %85.1)
  4. person (Güven: %78.9)
  5. car (Güven: %71.2)
```

## 💡 İpuçları

1. **İlk çalıştırma** biraz uzun sürebilir çünkü YOLOv8 modeli indirilecek (~6MB)
2. Daha iyi sonuçlar için **net ve iyi aydınlatılmış** resimler kullanın
3. Küçük objeler için **--conf=0.3** gibi düşük güven eşiği deneyin

## 🛠️ Gereksinimler

- Python 3.8+
- ultralytics
- pillow
