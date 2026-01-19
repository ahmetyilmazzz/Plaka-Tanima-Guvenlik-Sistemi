"""
Database management functions for the Security System project.
"""

import sqlite3
from pathlib import Path
from datetime import datetime
import logging

PROJECT_ROOT = Path(__file__).parent
DB_PATH = PROJECT_ROOT / "guvenlik_sistemi.db"
logger = logging.getLogger(__name__)

def init_database():
    """
    Initializes the database and creates tables if they don't exist.
    Adds default plates if the table is empty.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Create izinli_plakalar table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS izinli_plakalar (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plaka TEXT UNIQUE NOT NULL,
                eklenme_tarihi TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create gecis_loglari table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS gecis_loglari (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plaka TEXT,
                arac_tipi TEXT,
                vlm_yorumu TEXT,
                gecis_durumu TEXT,
                islem_zamani TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

        # Add default plates if the table is empty
        cursor.execute('SELECT COUNT(*) FROM izinli_plakalar')
        if cursor.fetchone()[0] == 0:
            default_plates = ['KGT671', 'MCL498E', 'MCI498E', 'MC1498E', '34ABC123', '06XYZ999', '35DEF456']
            for plate in default_plates:
                cursor.execute('INSERT OR IGNORE INTO izinli_plakalar (plaka) VALUES (?)', (plate,))
            conn.commit()
            logger.info(f"Added {len(default_plates)} default allowed plates.")

        conn.close()
        logger.info("Database initialized.")
        return DB_PATH

    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        return None

def get_allowed_plates():
    """Retrieves the list of allowed plates from the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT plaka FROM izinli_plakalar')
        plates = [row[0] for row in cursor.fetchall()]
        conn.close()
        return plates
    except sqlite3.Error as e:
        logger.error(f"Database error while fetching plates: {e}")
        return []

def add_allowed_plate(plate):
    """Adds a new allowed plate to the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('INSERT OR IGNORE INTO izinli_plakalar (plaka) VALUES (?)', (plate,))
        conn.commit()
        conn.close()
        logger.info(f"Added plate: {plate}")
        return True
    except sqlite3.Error as e:
        logger.error(f"Database error while adding plate: {e}")
        return False

def get_recent_logs(limit=10):
    """Retrieves the most recent access logs."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT plaka, arac_tipi, vlm_yorumu, gecis_durumu, islem_zamani
            FROM gecis_loglari
            ORDER BY id DESC
            LIMIT ?
        ''', (limit,))
        logs = cursor.fetchall()
        conn.close()
        return logs
    except sqlite3.Error as e:
        logger.error(f"Database error while fetching logs: {e}")
        return []

def log_kaydet(plaka, arac_tipi, vlm_yorumu, gecis_durumu):
    """Saves an access log to the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        islem_zamani = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute('''
            INSERT INTO gecis_loglari (plaka, arac_tipi, vlm_yorumu, gecis_durumu, islem_zamani)
            VALUES (?, ?, ?, ?, ?)
        ''', (plaka, arac_tipi, vlm_yorumu, gecis_durumu, islem_zamani))
        conn.commit()
        conn.close()
        logger.info(f"Log saved: {plaka} - {gecis_durumu}")
    except sqlite3.Error as e:
        logger.error(f"Database error while saving log: {e}")

def plaka_izinli_mi(plaka):
    """Checks if a license plate is in the allowed list."""
    return plaka in get_allowed_plates()
