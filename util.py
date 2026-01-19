"""
Yardımcı Fonksiyonlar Modülü
Proje genelinde kullanılan yardımcı fonksiyonlar
"""

import os
import cv2
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


def plaka_format_kontrol(plaka):
    """
    Türk plaka formatı kontrolü
    Format: XX ABCXXX (XX: şehir kodu, ABC: harfler, XXX: rakamlar)

    Args:
        plaka: Kontrol edilecek plaka metni

    Returns:
        Format uygunsa True, değilse False
    """
    if not plaka or not isinstance(plaka, str):
        return False

    # Boşlukları ve tireleri kaldır
    temiz = plaka.replace(' ', '').replace('-', '').upper()

    # Türk plaka formatı: 2 rakam + 1-3 harf + 2-4 rakam
    import re
    pattern = r'^\d{2}[A-Z]{1,4}\d{2,5}$'

    return bool(re.match(pattern, temiz))


def plaka_normalize(plaka):
    """
    Plaka metnini normalize eder (boşlukları kaldırır, büyük harfe çevirir)

    Args:
        plaka: Normalize edilecek plaka metni

    Returns:
        Normalize edilmiş plaka metni
    """
    if not plaka:
        return ""

    return plaka.replace(' ', '').replace('-', '').upper()


def plaka_bicimlendir(plaka):
    """
    Plakayı Türk plaka formatında biçimlendirir (XX ABC 123)

    Args:
        plaka: Biçimlendirilecek plaka metni

    Returns:
        Biçimlendirilmiş plaka metni
    """
    temiz = plaka_normalize(plaka)

    if len(temiz) < 5:
        return temiz

    try:
        # XX ABC 123 formatı
        sehir = temiz[:2]
        harfler = ""
        rakamlar = ""

        i = 2
        while i < len(temiz) and temiz[i].isalpha():
            harfler += temiz[i]
            i += 1

        while i < len(temiz) and temiz[i].isdigit():
            rakamlar += temiz[i]
            i += 1

        if harfler and rakamlar:
            return f"{sehir} {harfler} {rakamlar}"
        return temiz
    except:
        return temiz


def goruntu_on_isle(img):
    """
    Görüntüyü OCR için ön işlemden geçirir

    Args:
        img: Görüntü (numpy array)

    Returns:
        İşlenmiş görüntü listesi
    """
    islenmis = []

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Farklı ölçeklerde yeniden boyutlandır
    for scale in [2.0, 3.0, 4.0, 5.0]:
        h, w = gray.shape[:2]
        resized = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        islenmis.append(resized)

        # CLAHE ile kontrast artır
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(2, 2))
        enhanced = clahe.apply(resized)
        islenmis.append(enhanced)

        # Otsu eşikleme
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        islenmis.append(binary)

        # Tersini de dene
        islenmis.append(cv2.bitwise_not(binary))

    return islenmis


def plaka_bolgesi_tespit(img):
    """
    Görüntüdeki olası plaka bölgelerini tespit eder

    Args:
        img: Görüntü (numpy array)

    Returns:
    Tespit edilen plaka bölgelerinin listesi [x, y, w, h]
    """
    plakalar = []

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Kenar tespiti
    for canny_low in [30, 50, 70]:
        for canny_high in [100, 150, 200]:
            filtered = cv2.bilateralFilter(gray, 11, 17, 17)
            edged = cv2.Canny(filtered, canny_low, canny_high)
            contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if h > 0:
                    aspect_ratio = float(w) / h
                    # Plaka aspect ratio: genellikle 2:1 ile 5:1 arası
                    if 1.5 <= aspect_ratio <= 6.0 and w > 40 and h > 10:
                        plakalar.append([x, y, w, h])

    # Renk tabanlı tespit
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Sarı plakalar (Türkiye)
    yellow_mask = cv2.inRange(hsv, np.array([15, 50, 50]), np.array([40, 255, 255]))
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 0:
            aspect_ratio = float(w) / h
            if 1.5 <= aspect_ratio <= 6.0 and w > 40 and h > 10:
                plakalar.append([x, y, w, h])

    # Beyaz plakalar
    white_mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 40, 255]))
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 0:
            aspect_ratio = float(w) / h
            if 1.5 <= aspect_ratio <= 6.0 and w > 40 and h > 10:
                plakalar.append([x, y, w, h])

    # Overlapping box'ları birleştir (NMS)
    if plakalar:
        plakalar = nms_boxes(plakalar, overlap_threshold=0.3)

    return plakalar


def nms_boxes(boxes, overlap_threshold=0.3):
    """
    Overlapping bounding box'ları birleştirir (Non-Maximum Suppression)

    Args:
        boxes: Box listesi [[x, y, w, h], ...]
        overlap_threshold: Overlap eşik değeri

    Returns:
        Filtrelenmiş box listesi
    """
    if not boxes:
        return []

    boxes = np.array(boxes)
    picked = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    area = boxes[:, 2] * boxes[:, 3]

    idxs = np.argsort(y2)

    while len(idxs) > 0:
        i = idxs[0]
        pick = boxes[i]
        picked.append(pick.tolist())

        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        overlap = (w * h) / area[idxs[1:]]

        idxs = np.delete(idxs, np.concatenate(([0], np.where(overlap > overlap_threshold)[0] + 1)))

    return picked


def log_yaz(dosya_adi, mesaj):
    """
    Log dosyasına mesaj yazar

    Args:
        dosya_adi: Log dosyası adı
        mesaj: Yazılacak mesaj
    """
    log_path = Path(__file__).parent / "logs" / dosya_adi
    log_path.parent.mkdir(exist_ok=True)

    zaman = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"[{zaman}] {mesaj}\n")


def cikti_kaydet(img, dosya_adi):
    """
    Çıktı görüntüsünü kaydeder

    Args:
        img: Kaydedilecek görüntü
        dosya_adi: Dosya adı

    Returns:
        Kaydedilen dosyanın yolu
    """
    cikti_path = Path(__file__).parent / "outputs" / dosya_adi
    cikti_path.parent.mkdir(exist_ok=True)

    cv2.imwrite(str(cikti_path), img)
    logger.info(f"Çıktı kaydedildi: {cikti_path}")

    return str(cikti_path)


def get_zaman_damgasi():
    """
    Geçerli zaman damgasını döndürür

    Returns:
        Zaman damgası string'i
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')
