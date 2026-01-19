"""
Eksik Model Verilerini Tamamlama Scripti
Plaka tespit modeli için gerekli dosyaları kontrol eder ve eksikleri oluşturur
"""

import os
import logging
from pathlib import Path
from ultralytics import YOLO
import shutil

# Proje kök dizini
PROJECT_ROOT = Path(__file__).parent.parent
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def model_kontrol_ve_yukle():
    """
    Eksik modelleri kontrol eder ve indirir/yükler
    """
    logger.info("Modeller kontrol ediliyor...")

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    # 1. YOLOv8n (ana model)
    yolov8_path = PROJECT_ROOT / "yolov8n.pt"
    if not yolov8_path.exists():
        logger.info("YOLOv8n modeli indiriliyor...")
        try:
            model = YOLO('yolov8n.pt')
            logger.info("YOLOv8n modeli indirildi")
        except Exception as e:
            logger.error(f"YOLOv8n indirme hatası: {e}")
            logger.info("İnternet bağlantınızı kontrol edin veya manuel olarak indirin")
    else:
        logger.info("YOLOv8n modeli mevcut")

    # 2. Plaka tespit modeli (veya placeholder)
    plaka_model_path = models_dir / "license_plate_detector.pt"
    if not plaka_model_path.exists():
        logger.warning("Plaka tespit modeli bulunamadı")
        logger.info("Yedek olarak YOLOv8n kullanılacak")

        # Placeholder oluştur - aslında yolov8n'ın bir kopyası
        # Gerçek plaka modeli için ayrı eğitim gerekir
        try:
            if yolov8_path.exists():
                shutil.copy(yolov8_path, plaka_model_path)
                logger.info(f"Placeholder plaka modeli oluşturuldu: {plaka_model_path}")
        except Exception as e:
            logger.error(f"Placeholder oluşturma hatası: {e}")
    else:
        logger.info("Plaka tespit modeli mevcut")

    # 3. Eğitilmiş araç sınıflandırıcısı (varsa)
    classifier_path = models_dir / "vehicle_classifier" / "weights" / "best.pt"
    if classifier_path.exists():
        logger.info(f"Eğitilmiş araç sınıflandırıcısı mevcut: {classifier_path}")
    else:
        logger.info("Eğitilmiş araç sınıflandırıcısı bulunamadı")
        logger.info("Eğitim için: python yolo_train.py komutunu çalıştırın")

    logger.info("Model kontrolü tamamlandı")


def plaka_modeli_ici_n_asamalar():
    """
    Plaka tespit modeli için eğitim verilerini hazırlar
    Kullanıcı için talimatlar verir
    """
    logger.info("=" * 60)
    logger.info("PLAKA TESPİT MODELİ EĞİTİMİ TALIMATLARI")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Plaka tespit modeli için özel bir eğitim veri setine ihtiyacınız var.")
    logger.info("")
    logger.info("Seçenekler:")
    logger.info("1. Kaggle'dan Türk plaka veri setini indirin:")
    logger.info("   https://www.kaggle.com/datasets/smaildurcan/turkish-license-plate-dataset")
    logger.info("")
    logger.info("2. Veri setini data/raw/turkish_plates/ klasörüne çıkarın")
    logger.info("")
    logger.info("3. Aşağıdaki Python kodunu çalıştırarak modeli eğitin:")
    logger.info("")
    logger.info("```python")
    logger.info("from ultralytics import YOLO")
    logger.info("")
    logger.info("# Plaka veri seti YAML dosyası")
    logger.info("data_yaml = 'data/raw/turkish_plates/data.yaml'")
    logger.info("")
    logger.info("# Modeli yükle ve eğit")
    logger.info("model = YOLO('yolov8n.pt')")
    logger.info("model.train(data=data_yaml, epochs=50, imgsz=640)")
    logger.info("")
    logger.info("# Modeli kaydet")
    logger.info("model.save('models/license_plate_detector.pt')")
    logger.info("```")
    logger.info("")
    logger.info("Not: Şu anda main.py, plaka modeli yoksa COCO modelini kullanır")
    logger.info("=" * 60)


def main():
    """
    Ana fonksiyon
    """
    logger.info("=" * 60)
    logger.info("EKSİK MODEL VERİLERİNİ TAMAMLAMA")
    logger.info("=" * 60)

    model_kontrol_ve_yukle()
    plaka_modeli_ici_n_asamalar()

    logger.info("")
    logger.info("İşlem tamamlandı!")


if __name__ == "__main__":
    main()
