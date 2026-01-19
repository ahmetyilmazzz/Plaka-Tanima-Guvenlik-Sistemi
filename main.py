"""
Araç Plaka Tanıma ve Güvenlik Sistemi
Proje Ana Dosyası - Görüntüleri işler, araçları tespit eder, plakaları okur ve yetkilendirme yapar
"""

import os
import sys
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import easyocr
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# UTF-8 encoding için Windows düzeltmesi
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Proje kök dizini
PROJECT_ROOT = Path(__file__).parent
# OpenMP çakışması için Windows düzeltmesi
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# GPU kullanımı için ayar
USE_GPU = torch.cuda.is_available()
DEVICE = 'cuda' if USE_GPU else 'cpu'

# Loglama yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'sistem.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


# ==================== VERİTABANI FONKSİYONLARI ====================

import database as db


# ==================== PLAKA OKUMA FONKSİYONLARI ====================

def format_plate_text(text):
    """
    Validates and formats a license plate string to a generic format.
    A valid plate is an alphanumeric string between 4 and 10 characters.
    """
    if not text:
        return None

    clean_text = ''.join(filter(str.isalnum, text)).upper()

    if 4 <= len(clean_text) <= 10:
        return clean_text

    return None


def plaka_oku_coklu_deneme(plate_img):
    """
    Plaka görüntüsünü farklı ön işleme yöntemleri ile okur
    En iyi sonucu döndürür - GPU destekli
    """
    global ocr_reader
    if 'ocr_reader' not in globals():
        logger.info(f"EasyOCR yükleniyor (GPU: {USE_GPU})...")
        ocr_reader = easyocr.Reader(['en', 'tr'], gpu=USE_GPU)
        logger.info("EasyOCR yüklendi")

    adaylar = []

    h, w = plate_img.shape[:2]
    if h < 50 or w < 100:
        scale = max(3.0, 150 / w)
        plate_img = cv2.resize(plate_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    if len(plate_img.shape) == 3:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_img

    islenmis_gorseller = [gray]

    # 1. Adaptive Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    islenmis_gorseller.append(adaptive_thresh)
    islenmis_gorseller.append(cv2.bitwise_not(adaptive_thresh))

    # 2. Dilation and Erosion
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(gray, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    islenmis_gorseller.append(dilated)
    islenmis_gorseller.append(eroded)

    for img in islenmis_gorseller:
        try:
            sonuc = ocr_reader.readtext(
                img,
                allowlist='0123456789ABCDEFGHJKLMNPRSTUVWXYZ',
                detail=1,
                paragraph=False,
                width_ths=0.8,
                height_ths=0.8,
                decoder='beamsearch',
                contrast_ths=0.4,
                adjust_contrast=0.5
            )

            for bbox, text, prob in sonuc:
                if prob > 0.4:
                    text = text.upper().replace('O', '0').replace('I', '1').replace('S', '5').replace('L', '1')
                    temiz_text = ''.join(c for c in text if c.isalnum())
                    formatted_plate = format_plate_text(temiz_text)
                    if formatted_plate:
                        adaylar.append((formatted_plate, prob))
        except Exception as e:
            continue

    if adaylar:
        from collections import defaultdict
        oy_plakalar = defaultdict(list)
        for plaka, prob in adaylar:
            oy_plakalar[plaka].append(prob)

        en_iyi_plaka = max(oy_plakalar.items(), key=lambda x: (len(x[1]), sum(x[1])))
        ortalama_prob = sum(en_iyi_plaka[1]) / len(en_iyi_plaka[1])

        logger.info(f"Plaka okundu: {en_iyi_plaka[0]} (güven: {ortalama_prob:.2f})")
        return en_iyi_plaka[0]

    logger.warning("Plaka okunamadı")
    return "OKUNAMADI"


# ==================== VLM FONKSİYONLARI ====================

def vlm_ile_arac_analizi(image_path, arac_tipi):
    """
    BLIP modeli kullanarak araç hakkında görsel analiz yapar
    Rengi, markası ve durumu hakkında bilgi döndürür - GPU destekli
    """
    global vlm_processor, vlm_model

    try:
        if 'vlm_processor' not in globals():
            logger.info(f"VLM modeli yükleniyor (GPU: {USE_GPU})...")
            vlm_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            vlm_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            vlm_model.to(DEVICE)
            vlm_model.eval()
            logger.info(f"VLM modeli yüklendi (device: {DEVICE})")

        image = Image.open(image_path).convert('RGB')
        inputs = vlm_processor(image, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            out = vlm_model.generate(**inputs, max_length=50)

        caption = vlm_processor.decode(out[0], skip_special_tokens=True)
        logger.info(f"VLM Caption: {caption}")

        # Caption'dan araç bilgilerini çıkar
        yorum = caption

        # Renk tespiti
        renkler = ['white', 'black', 'red', 'blue', 'green', 'yellow', 'gray', 'silver']
        renk = "UNKNOWN"
        for r in renkler:
            if r.lower() in caption.lower():
                renk = r.upper()
                break

        # Marka tespiti (basit)
        markalar = ['toyota', 'bmw', 'mercedes', 'audi', 'volkswagen', 'fiat', 'renault',
                    'honda', 'hyundai', 'ford', 'peugeot', 'citroen']
        marka = "UNKNOWN"
        for m in markalar:
            if m in caption.lower():
                marka = m.upper()
                break

        # Durum tespiti
        durum = "PARKED" if 'parked' in caption.lower() else "MOVING"

        yorum = f"{renk} {marka} {arac_tipi} ({durum})"
        logger.info(f"VLM Comment: {yorum}")

        return yorum

    except Exception as e:
        logger.error(f"VLM hatası: {e}")
        return "VLM ANALYSIS FAILED"


# ==================== GÖRSELLEŞTİRME ====================

def sonucları_gorsellestir(image_path, arac_tipi, plaka, vlm_yorumu, karar):
    """
    Analiz sonuçlarını görüntü üzerine yazdırıp gösterir
    """
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Görüntü okunamadı: {image_path}")
        return

    h, w = img.shape[:2]

    # Görüntü boyutuna göre yazı boyutlarını ayarla
    scale = max(1.0, min(w, h) / 1000.0)  # 1000px'de scale=1

    # Renkler
    yesil = (0, 200, 0)
    kirmizi = (0, 0, 220)
    beyaz = (240, 240, 240)
    sari = (0, 200, 200)
    mavi = (200, 100, 0)
    siyah = (10, 10, 10)

    renk = yesil if karar == "ALLOWED" else kirmizi

    # Bilgi paneli - görüntü boyutuna göre ayarla
    panel_h = int(100 * scale)  # Increased panel height
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    # Çizgi kalınlığı ve font boyutları
    line_thick = max(1, int(2 * scale))
    font_scale_large = 0.7 * scale
    font_scale_medium = 0.6 * scale
    font_scale_small = 0.5 * scale

    y_offset = int(30 * scale)
    line_spacing = int(32 * scale)

    # Satır 1: Karar (en büyük, renkli arka plan)
    karar_text = f"STATUS: {karar}"
    (tw, th), _ = cv2.getTextSize(karar_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale_large, line_thick + 1)
    cv2.rectangle(img, (10, y_offset - th - 10), (20 + tw, y_offset + 5), renk, -1)
    cv2.putText(img, karar_text, (15, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_large, siyah, line_thick + 1)

    # Satır 2: Araç tipi ve Plaka (yan yana)
    y_offset += line_spacing
    tip_text = f"{arac_tipi}"
    plaka_text = f"PLATE: {plaka}"

    (tw1, th1), _ = cv2.getTextSize(tip_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale_medium, line_thick)
    (tw2, th2), _ = cv2.getTextSize(plaka_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale_medium, line_thick)

    x_pos = 15
    cv2.putText(img, tip_text, (x_pos, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_medium, beyaz, line_thick)

    x_pos += tw1 + int(50 * scale)
    cv2.putText(img, plaka_text, (x_pos, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_medium, sari, line_thick)

    # Satır 3: VLM yorumu
    y_offset += line_spacing
    cv2.putText(img, f"VLM: {vlm_yorumu}", (15, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, mavi, max(1, line_thick - 1))

    # Sağ alt köşe zaman damgası (küçük)
    zaman = datetime.now().strftime('%H:%M:%S')
    cv2.putText(img, zaman, (w - 100, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

    # Görüntüyü göster (daha küçük pencere)
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Kaydet
    output_path = PROJECT_ROOT / "outputs" / f"sonuc_{Path(image_path).stem}.jpg"
    output_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(output_path), img)
    logger.info(f"Sonuç kaydedildi: {output_path}")


# ==================== ANA İŞ AKIŞI ====================

def main():
    """
    Ana iş akışı - tüm sistemi yönetir
    """
    logger.info("=" * 60)
    logger.info("ARAÇ PLAKA TANİMA VE GÜVENLİK SİSTEMİ")
    logger.info("=" * 60)

    # 1. Veritabanını hazırla
    db.init_database()

    # 2. Modelleri yükle
    logger.info("Modeller yükleniyor...")

    # COCO modeli (araç tespiti için)
    coco_model = YOLO('yolov8n.pt')
    logger.info("COCO modeli yüklendi")

    # Plaka tespit modeli (varsa)
    plaka_model_path = PROJECT_ROOT / "models" / "license_plate_detector.pt"
    plaka_model = None
    if plaka_model_path.exists():
        plaka_model = YOLO(str(plaka_model_path))
        logger.info("Plaka tespit modeli yüklendi")
    else:
        logger.warning("Plaka tespit modeli bulunamadı, COCO modeli kullanılacak")

    # 3. Görüntüleri işle
    images_dir = PROJECT_ROOT / "images"

    if not images_dir.exists():
        logger.error(f"Görüntü klasörü bulunamadı: {images_dir}")
        logger.info("Lütfen 'images' klasörüne araç resimleri ekleyin")
        return

    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))

    if not image_files:
        logger.error("Görüntü bulunamadı")
        return

    logger.info(f"{len(image_files)} görüntü işlenecek")

    # Her görüntüyü işle
    for image_path in image_files:
        logger.info("-" * 40)
        logger.info(f"İşleniyor: {image_path.name}")

        # 1. Araç tespiti
        results = coco_model(image_path, verbose=False)
        arac_bulundu = False
        arac_tipi = "BİLİNMEYEN"
        arac_bbox = None

        COCO_CLASSES = {
            2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'
        }

        VEHICLE_NAMES = {
            'car': 'CAR',
            'motorcycle': 'MOTORCYCLE',
            'bus': 'BUS',
            'truck': 'TRUCK'
        }

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id in COCO_CLASSES:
                    arac_tipi = COCO_CLASSES[cls_id]
                    arac_bbox = box.xyxy[0].tolist()
                    arac_bulundu = True
                    break
            if arac_bulundu:
                break

        if not arac_bulundu:
            logger.warning("Araç tespit edilemedi")
            continue

        arac_tipi_en = VEHICLE_NAMES.get(arac_tipi, arac_tipi.upper())
        logger.info(f"Araç tespit edildi: {arac_tipi_en}")

        # Araç görüntüsünü kes
        img = cv2.imread(str(image_path))
        x1, y1, x2, y2 = map(int, arac_bbox)
        arac_img = img[y1:y2, x1:x2]

        # 2. Plaka tespiti ve okuma
        plaka_bulundu = False
        plaka_img = None

        if plaka_model:
            # Özel plaka modeli ile tespit dene
            plaka_results = plaka_model(arac_img, verbose=False)
            for pr in plaka_results:
                for pbox in pr.boxes:
                    px1, py1, px2, py2 = map(int, pbox.xyxy[0].tolist())
                    plaka_img = arac_img[py1:py2, px1:px2]
                    plaka_bulundu = True
                    break
                if plaka_bulundu:
                    break

        # Plaka bulunamadıysa araç görüntüsünün alt yarısını kullan
        if not plaka_bulundu:
            h, w = arac_img.shape[:2]
            plaka_img = arac_img[int(h/2):h, :]
            logger.info("Plaka bölgesi tespit edilemedi, alt bölüm kullanılıyor")

        # Plakayı oku
        plaka_text = plaka_oku_coklu_deneme(plaka_img)

        # 3. VLM ile araç analizi
        vlm_yorumu = vlm_ile_arac_analizi(image_path, arac_tipi_en)

        # 4. Yetkilendirme kontrolü
        if arac_tipi_en != 'CAR':
            karar = "DENIED"
            sebep = "Vehicle type not authorized"
        elif plaka_text == "OKUNAMADI":
            karar = "DENIED"
            sebep = "Plate not readable"
        elif db.plaka_izinli_mi(plaka_text):
            karar = "ALLOWED"
            sebep = "Authorized plate"
        else:
            karar = "DENIED"
            sebep = "Unauthorized plate"

        logger.info(f"Karar: {karar} ({sebep})")

        # 5. Log kaydı
        db.log_kaydet(plaka_text, arac_tipi_en, vlm_yorumu, karar)

        # 6. Görselleştirme
        sonucları_gorsellestir(str(image_path), arac_tipi_en, plaka_text, vlm_yorumu, karar)

    logger.info("=" * 60)
    logger.info("İŞLEM TAMAMLANDI")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
