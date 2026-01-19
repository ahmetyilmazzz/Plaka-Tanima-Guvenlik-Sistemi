"""
YOLOv8 Araç Tipi Sınıflandırma Eğitim Scripti
Araç tiplerini (otomobil, otobüs, kamyon) sınıflandırmak için YOLOv8 modelini eğitir
Veri setini temizler (tekilleştirir) ve eğitim/doğrulama olarak böler
"""

import os
import shutil
import logging
from pathlib import Path
from collections import defaultdict
import yaml
from ultralytics import YOLO

# Proje kök dizini
PROJECT_ROOT = Path(__file__).parent
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Loglama
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def veri_setini_temizle(dataset_path):
    """
    Veri setindeki kopya görüntüleri temizler (tekilleştirme)
    Aynı içeriğe sahip dosyaları tespit eder ve siler
    """
    logger.info(f"Veri seti temizleniyor: {dataset_path}")

    if not dataset_path.exists():
        logger.error(f"Veri seti klasörü bulunamadı: {dataset_path}")
        return

    # Her sınıf için işlem yap
    siniflar = ['car', 'bus', 'truck']

    # Hash tabanlı tekilleştirme
    hash_map = defaultdict(list)
    import hashlib

    for sinif in siniflar:
        sinif_path = dataset_path / sinif
        if not sinif_path.exists():
            logger.warning(f"Sınıf klasörü bulunamadı: {sinif_path}")
            continue

        dosyalar = list(sinif_path.glob("*.jpg")) + list(sinif_path.glob("*.png"))

        for dosya in dosyalar:
            # Dosya hash'ini hesapla
            with open(dosya, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()

            hash_map[file_hash].append(dosya)

    # Kopyaları sil
    silinen_sayi = 0
    for file_hash, dosyalar in hash_map.items():
        if len(dosyalar) > 1:
            # İlk dosya hariç diğerlerini sil
            for dosya in dosyalar[1:]:
                try:
                    dosya.unlink()
                    silinen_sayi += 1
                    logger.info(f"Kopya silindi: {dosya.name}")
                except Exception as e:
                    logger.error(f"Silme hatası: {e}")

    logger.info(f"Toplam {silinen_sayi} kopya dosya silindi")
    return silinen_sayi


def veri_setini_bol(dataset_path, train_ratio=0.8):
    """
    Veri setini eğitim ve doğrulama setleri olarak böler
    YOLO formatında veri seti yapısı oluşturur
    """
    logger.info("Veri seti bölünüyor (eğitim/doğrulama)")

    # YOLO veri seti yapısını oluştur
    yolo_dataset_path = PROJECT_ROOT / "yolo_dataset"
    train_images = yolo_dataset_path / "images" / "train"
    val_images = yolo_dataset_path / "images" / "val"
    train_labels = yolo_dataset_path / "labels" / "train"
    val_labels = yolo_dataset_path / "labels" / "val"

    for klasor in [train_images, val_images, train_labels, val_labels]:
        klasor.mkdir(parents=True, exist_ok=True)

    # Sınıf eşleşmeleri
    sinif_id = {'car': 0, 'bus': 1, 'truck': 2}
    siniflar = ['car', 'bus', 'truck']

    import random
    random.seed(42)

    for sinif in siniflar:
        sinif_path = dataset_path / sinif
        if not sinif_path.exists():
            logger.warning(f"Sınıf klasörü bulunamadı: {sinif_path}")
            continue

        dosyalar = list(sinif_path.glob("*.jpg")) + list(sinif_path.glob("*.png"))
        random.shuffle(dosyalar)

        # Böl
        split_idx = int(len(dosyalar) * train_ratio)
        train_dosyalar = dosyalar[:split_idx]
        val_dosyalar = dosyalar[split_idx:]

        logger.info(f"{sinif.upper()}: {len(train_dosyalar)} eğitim, {len(val_dosyalar)} doğrulama")

        # Eğitim dosyalarını kopyala
        for dosya in train_dosyalar:
            # Görüntüyü kopyala
            hedef_img = train_images / dosya.name
            shutil.copy2(dosya, hedef_img)

            # Etiket dosyası oluştur (tüm görüntü sınıf için)
            # YOLO formatı: class_id x_center y_center width height (normalize)
            # Basitlik için tüm görüntüyü kapsayan bbox kullanıyoruz
            label_name = dosya.stem + ".txt"
            label_path = train_labels / label_name

            # Görüntü boyutlarını al
            from PIL import Image
            try:
                with Image.open(dosya) as img:
                    w, h = img.size
            except:
                w, h = 640, 640  # Varsayılan

            # YOLO formatında etiket (tüm görüntüyü kapsayan bbox)
            # 0.5 0.5 1.0 1.0 = merkez noktası ve tam genişlik/yükseklik
            with open(label_path, 'w') as f:
                f.write(f"{sinif_id[sinif]} 0.5 0.5 1.0 1.0\n")

        # Doğrulama dosyalarını kopyala
        for dosya in val_dosyalar:
            hedef_img = val_images / dosya.name
            shutil.copy2(dosya, hedef_img)

            label_name = dosya.stem + ".txt"
            label_path = val_labels / label_name

            from PIL import Image
            try:
                with Image.open(dosya) as img:
                    w, h = img.size
            except:
                w, h = 640, 640

            with open(label_path, 'w') as f:
                f.write(f"{sinif_id[sinif]} 0.5 0.5 1.0 1.0\n")

    # data.yaml dosyası oluştur
    data_yaml = {
        'path': str(yolo_dataset_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 3,  # Number of classes
        'names': ['car', 'bus', 'truck']
    }

    yaml_path = yolo_dataset_path / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    logger.info(f"YOLO veri seti oluşturuldu: {yolo_dataset_path}")
    logger.info(f"data.yaml oluşturuldu: {yaml_path}")

    return yolo_dataset_path, yaml_path


def yolo_egit(yaml_path, epochs=50, img_size=640):
    """
    YOLOv8 modelini eğitir
    """
    logger.info("YOLOv8 eğitimi başlatılıyor...")
    logger.info(f"Epochs: {epochs}, Image Size: {img_size}")

    # YOLOv8n modelini yükle (nano - en hızlı)
    model = YOLO('yolov8n.pt')

    # Eğit
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=img_size,
        batch=16,
        device='cpu',  # GPU yoksa CPU kullan
        project=str(PROJECT_ROOT / "models"),
        name='vehicle_classifier',
        exist_ok=True,
        verbose=True
    )

    logger.info("Eğitim tamamlandı!")
    logger.info(f"Model kaydedildi: {PROJECT_ROOT / 'models' / 'vehicle_classifier'}")

    return results


def main():
    """
    Ana eğitim fonksiyonu
    """
    logger.info("=" * 60)
    logger.info("YOLOV8 ARAÇ TİPİ SINIFLANDIRMA EĞİTİMİ")
    logger.info("=" * 60)

    # 1. Veri seti yolu
    dataset_path = PROJECT_ROOT / "dataset" / "train"

    if not dataset_path.exists():
        logger.error(f"Veri seti bulunamadı: {dataset_path}")
        logger.info("Lütfen dataset/train/ klasörüne araç resimlerini ekleyin")
        return

    # 2. Veri setini temizle
    veri_setini_temizle(dataset_path)

    # 3. Veri setini böl
    yolo_dataset_path, yaml_path = veri_setini_bol(dataset_path, train_ratio=0.8)

    # 4. Modeli eğit
    results = yolo_egit(yaml_path, epochs=50, img_size=640)

    logger.info("=" * 60)
    logger.info("EĞİTİM TAMAMLANDI")
    logger.info("=" * 60)
    logger.info(f"Eğitilmiş model: models/vehicle_classifier/weights/best.pt")
    logger.info("Bu modeli main.py'de kullanmak için yolov8n.pt yerine bu dosyayı kullanın")


if __name__ == "__main__":
    main()
