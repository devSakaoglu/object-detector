"""
Object Detector - Resimdeki Objeleri Tanıma Programı
=====================================================
Bu program, verilen resimdeki objeleri tanır ve listeler.
YOLOv8 modelini kullanır.
"""

import sys
import os
from pathlib import Path

def check_dependencies():
    """Gerekli kütüphaneleri kontrol et ve yükle"""
    try:
        from ultralytics import YOLO
        from PIL import Image
        return True
    except ImportError:
        print("Gerekli kütüphaneler yükleniyor...")
        os.system("pip install ultralytics pillow")
        return False

def detect_objects(image_path: str, confidence_threshold: float = 0.5):
    """
    Resimden objeleri tespit et ve listele
    
    Args:
        image_path: Resim dosyasının yolu
        confidence_threshold: Minimum güven eşiği (0-1 arası)
    
    Returns:
        Tespit edilen objelerin listesi
    """
    from ultralytics import YOLO
    from PIL import Image
    
    # Dosya kontrolü
    if not os.path.exists(image_path):
        print(f"❌ Hata: '{image_path}' dosyası bulunamadı!")
        return []
    
    print(f"\n🔍 Resim analiz ediliyor: {image_path}")
    print("-" * 50)
    
    # YOLOv8 modelini yükle (ilk çalıştırmada indirilecek)
    model = YOLO('yolov8n.pt')  # nano model - hızlı ve hafif
    
    # Resmi analiz et
    results = model(image_path, verbose=False)
    
    # Tespit edilen objeleri topla
    detected_objects = []
    object_counts = {}
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            confidence = float(box.conf[0])
            if confidence >= confidence_threshold:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                # Obje sayısını takip et
                if class_name in object_counts:
                    object_counts[class_name] += 1
                else:
                    object_counts[class_name] = 1
                
                detected_objects.append({
                    'name': class_name,
                    'confidence': confidence,
                    'bbox': box.xyxy[0].tolist()
                })
    
    # Sonuçları yazdır
    if detected_objects:
        print(f"\n✅ Toplam {len(detected_objects)} obje tespit edildi:\n")
        
        # Obje sayılarını göster
        print("📋 Tespit Edilen Objeler:")
        print("=" * 40)
        for obj_name, count in sorted(object_counts.items()):
            print(f"  • {obj_name}: {count} adet")
        
        print("\n📊 Detaylı Liste:")
        print("=" * 40)
        for i, obj in enumerate(detected_objects, 1):
            confidence_percent = obj['confidence'] * 100
            print(f"  {i}. {obj['name']} (Güven: %{confidence_percent:.1f})")
    else:
        print("\n⚠️ Hiçbir obje tespit edilemedi.")
        print("   Farklı bir resim deneyin veya güven eşiğini düşürün.")
    
    return detected_objects


def save_annotated_image(image_path: str, output_path: str = None):
    """
    Tespit edilen objeleri işaretleyerek yeni bir resim kaydet
    """
    from ultralytics import YOLO
    
    if not os.path.exists(image_path):
        print(f"❌ Hata: '{image_path}' dosyası bulunamadı!")
        return None
    
    model = YOLO('yolov8n.pt')
    results = model(image_path)
    
    # Çıktı yolunu belirle
    if output_path is None:
        path = Path(image_path)
        output_path = str(path.parent / f"{path.stem}_detected{path.suffix}")
    
    # İşaretli resmi kaydet
    for result in results:
        result.save(output_path)
    
    print(f"\n💾 İşaretli resim kaydedildi: {output_path}")
    return output_path


def main():
    """Ana program"""
    print("=" * 60)
    print("🖼️  OBJECT DETECTOR - Resim Obje Tanıma Programı")
    print("=" * 60)
    
    # Bağımlılıkları kontrol et
    if not check_dependencies():
        print("\n⚠️ Kütüphaneler yüklendi. Programı tekrar çalıştırın.")
        return
    
    # Komut satırı argümanlarını kontrol et
    if len(sys.argv) < 2:
        print("\n📖 Kullanım:")
        print("   python object_detector.py <resim_yolu> [--save]")
        print("\n📌 Örnekler:")
        print("   python object_detector.py foto.jpg")
        print("   python object_detector.py C:\\Resimler\\test.png --save")
        print("\n🔧 Seçenekler:")
        print("   --save    : Tespit edilen objeleri resim üzerinde işaretleyip kaydet")
        print("   --conf=X  : Güven eşiği (varsayılan: 0.5, örnek: --conf=0.3)")
        return
    
    image_path = sys.argv[1]
    save_output = "--save" in sys.argv
    
    # Güven eşiğini al
    confidence = 0.5
    for arg in sys.argv:
        if arg.startswith("--conf="):
            try:
                confidence = float(arg.split("=")[1])
            except ValueError:
                print("⚠️ Geçersiz güven değeri, varsayılan (0.5) kullanılıyor.")
    
    # Objeleri tespit et
    detected = detect_objects(image_path, confidence)
    
    # İsteğe bağlı olarak işaretli resmi kaydet
    if save_output and detected:
        save_annotated_image(image_path)
    
    print("\n" + "=" * 60)
    print("✨ İşlem tamamlandı!")
    print("=" * 60)


if __name__ == "__main__":
    main()
