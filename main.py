# Manga OCR & Typeset Tool v16.2.0
# ==============================
# 📦 Import modul bawaan Python
# ==============================
import os
import sys
import time
import json
import re
import hashlib
import pickle
import configparser
import lama_cleaner
from datetime import date
from openai import OpenAI

# ==============================
# 📦 Library pihak ketiga
# ==============================
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*resume_download.*")
import numpy as np
import cv2
import pytesseract
import requests
import easyocr
import fitz  # from PyMuPDF
import google.generativeai as genai
from PIL import Image
from PIL.ImageQt import ImageQt
import io
from PIL import ImageFile
import math

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==============================
# 📦 PyQt5 (dibagi per kategori)
# ==============================
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QFileDialog, QTextEdit, QScrollArea, QComboBox, QMessageBox,
    QProgressBar, QShortcut, QListWidget, QListWidgetItem, QColorDialog, QFontDialog,
    QLineEdit, QAction, QDialog, QDialogButtonBox, QCheckBox, QStatusBar, QAbstractItemView, QSpinBox,
    QTabWidget, QGroupBox, QGridLayout, QFrame, QSplitter, QRadioButton, QToolButton,
    QFontComboBox, QDoubleSpinBox, QMenu
)
from PyQt5.QtGui import (
    QPixmap, QPainter, QPen, QColor, QFont, QKeySequence, QPolygon,
    QPainterPath, QPolygonF, QImage, QIcon, QWheelEvent, QTextDocument,
    QTextCharFormat, QTextCursor, QBrush, QFontMetrics, QTransform, QTextBlockFormat
)
from PyQt5.QtCore import (
    Qt, QRect, QPoint, pyqtSignal, QTimer, QThread, QObject,
    QFileSystemWatcher, QRectF, QMutex, QPointF, QSignalBlocker
)

# ==============================
# 📦 Dependensi opsional (DL & OCR)
# ==============================
def check_dependency(import_name, pip_name=None, alias=None):
    """
    Coba impor modul. Jika gagal, kembalikan None dan tampilkan peringatan.
    """
    try:
        module = __import__(import_name)
        return module
    except ImportError:
        pip_msg = f" (pip install {pip_name})" if pip_name else ""
        print(f"Peringatan: Pustaka '{import_name}' tidak ditemukan. Fitur terkait akan dinonaktifkan.{pip_msg}", file=sys.stderr)
        return None if alias is None else alias

# --- Core Deep Learning ---
torch = check_dependency("torch")
is_gpu_available = torch.cuda.is_available() if torch else False

# --- Model Deteksi ---
onnxruntime = check_dependency("onnxruntime", "onnxruntime atau onnxruntime-gpu")
YOLO = None
try:
    from ultralytics import YOLO as YOLO_cls
    YOLO = YOLO_cls
except ImportError:
    print("Peringatan: Pustaka 'ultralytics' tidak ditemukan. (pip install ultralytics)", file=sys.stderr)

# --- Engine OCR ---
paddleocr = check_dependency("paddleocr", "paddleocr paddlepaddle")
doctr = check_dependency("doctr", "python-doctr[torch]")
RapidOCR = None
try:
    from rapidocr_onnxruntime import RapidOCR as RapidOCR_cls
    RapidOCR = RapidOCR_cls
except ImportError:
    print("Peringatan: Pustaka 'rapidocr_onnxruntime' tidak ditemukan. (pip install rapidocr_onnxruntime)", file=sys.stderr)

# --- [BARU] Engine Inpainting ---
lama_cleaner = check_dependency("lama_cleaner", "lama-cleaner")

# --- [BARU] API Provider ---
openai = check_dependency("openai", "openai")


# --- Fungsi untuk menangani config.ini ---
def create_default_config(config_path: str = "config.ini"):
    """
    Membuat file config.ini default jika belum ada.
    """
    config = configparser.ConfigParser()
    config['API'] = {
        'DEEPL_KEY': 'your_deepl_key_here',
        'GEMINI_KEY': 'your_gemini_key_here',
        'OPENAI_KEY': 'your_openai_key_here'  # [BARU] Kunci OpenAI
    }

    # Path default Tesseract sesuai OS
    if sys.platform.startswith("win"):
        tess_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    elif sys.platform.startswith("darwin"):  # macOS
        tess_path = "/usr/local/bin/tesseract"
    else:  # Linux / Unix
        tess_path = "/usr/bin/tesseract"

    config['PATHS'] = {
        'TESSERACT_PATH': tess_path
    }

    with open(config_path, 'w') as configfile:
        config.write(configfile)

    return config


def load_or_create_config(config_path: str = "config.ini"):
    """
    Memuat file config.ini. Jika belum ada, buat default lalu minta user edit.
    """
    config = configparser.ConfigParser()

    if not os.path.exists(config_path):
        print(f"'{config_path}' tidak ditemukan. Membuat file default.")
        config = create_default_config(config_path)

        # Tampilkan pesan ke user (jika aplikasi Qt sudah jalan)
        try:
            QMessageBox.information(
                None,
                "Configuration File Created",
                f"'{config_path}' tidak ditemukan.\n"
                "File default telah dibuat.\n\n"
                "Silakan buka file tersebut, masukkan API keys Anda, "
                "periksa path Tesseract, lalu restart aplikasi."
            )
        except RuntimeError:
            # Jika Qt belum siap
            print(f"Pesan: '{config_path}' dibuat. Edit file lalu restart aplikasi.")

        # Return None agar aplikasi utama bisa handle exit
        return None

    config.read(config_path)
    return config


# --- Muat Konfigurasi Saat Startup ---
try:
    config = load_or_create_config()
    if config is None:  # Jika config belum ada, hentikan aplikasi
        sys.exit()

    DEEPL_API_KEY = config.get('API', 'DEEPL_KEY', fallback="")
    GEMINI_API_KEY = config.get('API', 'GEMINI_KEY', fallback="")
    OPENAI_API_KEY = config.get('API', 'OPENAI_KEY', fallback="") # [BARU]
    TESSERACT_PATH = config.get('PATHS', 'TESSERACT_PATH', fallback="")

except (configparser.Error, KeyError) as e:
    try:
        QMessageBox.critical(
            None,
            "Configuration Error",
            f"Gagal membaca 'config.ini'. Pastikan memiliki [API] dan [PATHS].\nError: {e}"
        )
    except RuntimeError:
        print(f"Configuration Error: {e}", file=sys.stderr)
    sys.exit()


# --- Konfigurasi Gemini API ---
if GEMINI_API_KEY and "your_gemini_key_here" not in GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"Gagal mengkonfigurasi Gemini API: {e}", file=sys.stderr)
else:
    print("Peringatan: GEMINI_API_KEY belum diset di config.ini", file=sys.stderr)

# --- [BARU] Konfigurasi OpenAI API ---
openai_client = None
if openai and OPENAI_API_KEY and "your_openai_key_here" not in OPENAI_API_KEY:
    try:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        print(f"Gagal mengkonfigurasi OpenAI API: {e}", file=sys.stderr)
else:
    print("Peringatan: OPENAI_API_KEY belum diset di config.ini atau library openai tidak terinstal.", file=sys.stderr)

# --- Konfigurasi DeepL API ---
if not DEEPL_API_KEY or "your_deepl_key_here" in DEEPL_API_KEY:
    print("Peringatan: DEEPL_API_KEY belum diset di config.ini", file=sys.stderr)


# --- Konfigurasi Manga-OCR ---
try:
    from manga_ocr import MangaOcr
except ImportError:
    MangaOcr = None
    print("Peringatan: Pustaka 'manga-ocr' tidak ditemukan. Engine Manga-OCR dinonaktifkan.", file=sys.stderr)


# --- Konfigurasi Tesseract ---
try:
    if TESSERACT_PATH and os.path.exists(TESSERACT_PATH):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    else:
        print("Peringatan: Tesseract path dari config.ini tidak valid atau kosong.", file=sys.stderr)
except Exception as e:
    print(f"Gagal mengatur Tesseract path: {e}", file=sys.stderr)

# --------------------------------------------------------------------

# Kelas dasar sinyal untuk worker (pekerja latar belakang)
class WorkerSignals(QObject):
    finished = pyqtSignal()      # Sinyal jika proses selesai
    error = pyqtSignal(str)      # Sinyal jika terjadi error, membawa pesan error
    progress = pyqtSignal(int, str) # Sinyal progress dengan persentase dan pesan status


# Sinyal untuk detektor otomatis (Bubble atau Teks)
class AutoDetectorSignals(WorkerSignals):
    detection_complete = pyqtSignal(str, list) # Sinyal jika deteksi selesai: image_path, list of dicts {'polygon': QPolygon, 'text': str|None}
    overall_progress = pyqtSignal(int, str)      # Sinyal progress keseluruhan (persentase & status)


# Sinyal khusus untuk pemrosesan antrian pekerjaan
class QueueProcessorSignals(WorkerSignals):
    job_complete = pyqtSignal(str, object, str, str)  # image_path, new_area, original_text, translated_text
    queue_status = pyqtSignal(int)          # Sinyal jumlah item dalam antrian
    worker_finished = pyqtSignal(int)           # Sinyal saat 1 worker selesai (dengan ID worker)
    status_update = pyqtSignal(str)     # Sinyal update status bar (aman dari thread)


# Sinyal khusus untuk pemrosesan batch (sekumpulan pekerjaan)
class BatchProcessorSignals(WorkerSignals):
    batch_job_complete = pyqtSignal(str, object)  # Sinyal jika 1 job batch selesai
    batch_finished = pyqtSignal()           # Sinyal jika semua batch selesai


# Sinyal khusus untuk penyimpanan hasil batch
class BatchSaveSignals(WorkerSignals):
    file_saved = pyqtSignal(str)            # Sinyal jika file berhasil disimpan


# [BARU] ComboBox kustom yang dapat digulir dengan roda mouse
class ScrollableComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)  # Pastikan bisa menerima event scroll

    def wheelEvent(self, event: QWheelEvent):
        """Menangani event scroll mouse untuk mengganti item."""
        delta = event.angleDelta().y()
        current_index = self.currentIndex()
        count = self.count()

        if count == 0:
            return

        if delta > 0:  # Scroll ke atas
            next_index = (current_index - 1 + count) % count
            self.setCurrentIndex(next_index)
        elif delta < 0:  # Scroll ke bawah
            next_index = (current_index + 1) % count
            self.setCurrentIndex(next_index)

        event.accept()


# Kelas untuk menyimpan hasil teks dengan informasi apakah ada isinya atau tidak
class EnhancedResult:
    def __init__(self, text: str):
        self.text = text
        self.parts = bool(text)                          # True jika ada teks, False jika kosong

# Worker untuk memproses job dalam antrian secara paralel
class QueueProcessorWorker(QObject):
    def __init__(self, main_app, worker_id: int):
        super().__init__()
        self.main_app = main_app
        self.worker_id = worker_id
        self.is_running = True
        self.signals = QueueProcessorSignals()

    # Fungsi utama yang dijalankan oleh thread worker
    def run(self):
        print(f"Worker {self.worker_id} started.")
        while self.is_running:
            try:
                job = self.main_app.get_job_from_queue()
                if not job:
                    break  # hentikan worker jika tidak ada pekerjaan

                self.signals.queue_status.emit(len(self.main_app.processing_queue))

                image_path = job['image_path']
                cropped_cv_img = job['cropped_cv_img']
                settings = job['settings']
                # Teks dari fase deteksi (opsional, untuk menghemat panggilan OCR)
                pre_detected_text = job.get('text')

                original_text, translated_text = self.process_job(cropped_cv_img, settings, pre_detected_text)

                if translated_text:
                    new_area = TypesetArea(
                        job['rect'], translated_text,
                        settings['font'], settings['color'],
                        job.get('polygon'),
                        orientation=settings.get('orientation_mode', 'horizontal'),
                        effect=settings.get('text_effect', 'none'),
                        effect_intensity=settings.get('effect_intensity', 20.0),
                        bezier_points=settings.get('bezier_points'),
                        bubble_enabled=settings.get('create_bubble', False),
                        alignment=settings.get('alignment', 'center'),
                        line_spacing=settings.get('line_spacing', 1.1),
                        char_spacing=settings.get('char_spacing', 100.0),
                        margins=settings.get('margins', {'top': 12, 'right': 12, 'bottom': 12, 'left': 12})
                    )
                    self.signals.job_complete.emit(image_path, new_area, original_text, translated_text)

            except Exception as e:
                print(f"Error in Worker {self.worker_id}: {e}")
                self.signals.error.emit(str(e))
                continue

        print(f"Worker {self.worker_id} finished.")
        self.signals.worker_finished.emit(self.worker_id)

    def apply_safe_mode(self, text: str) -> str:
        """Menerapkan filter Safe Mode pada teks terjemahan."""
        if not text:
            return text
        # Gunakan re.sub dengan flag IGNORECASE untuk penggantian yang tidak sensitif huruf besar/kecil
        text = re.sub(r'vagina', 'meong', text, flags=re.IGNORECASE)
        text = re.sub(r'penis', 'burung', text, flags=re.IGNORECASE)
        # Tambahkan kata lain di sini jika diperlukan
        return text

    # Menentukan pipeline mana yang akan dipakai (standar / enhanced)
    def process_job(self, cropped_cv_img, settings: dict, pre_detected_text: str = None):
        original_text, translated_text = (
            self.run_enhanced_pipeline(cropped_cv_img, settings)
            if settings.get('enhanced_pipeline')
            else self.run_standard_pipeline(cropped_cv_img, settings, pre_detected_text)
        )

        # Terapkan Safe Mode setelah semua proses translasi dan naturalisasi selesai
        if settings.get('safe_mode') and translated_text:
            translated_text = self.apply_safe_mode(translated_text)

        return original_text, translated_text


    # Melakukan OCR sesuai engine yang dipilih
    def perform_ocr(self, image_to_process, settings: dict) -> str:
        # Panggil metode OCR dari main app yang sudah terpusat
        return self.main_app.perform_ocr(image_to_process, settings)


    # Pipeline standar: OCR → Cleaning → Translate → Naturalize (opsional)
    def run_standard_pipeline(self, cropped_cv_img, settings: dict, pre_detected_text: str = None):
        # Jika teks sudah dideteksi sebelumnya (mode Text Detect), lewati OCR
        if pre_detected_text:
            raw_text = pre_detected_text
        else:
            preprocessed_image, _ = self.main_app.preprocess_for_ocr(cropped_cv_img, settings['orientation'])
            raw_text = self.perform_ocr(preprocessed_image, settings)
        
        processed_text = self.main_app.clean_and_join_text(raw_text)

        if not processed_text or "[ERROR:" in raw_text or "[TESSERACT ERROR:" in raw_text:
            return raw_text, "" # Kembalikan pesan error jika ada
        
        # --- [DIUBAH] Logika terjemahan yang lebih fleksibel ---
        provider, model_name = settings['ai_model']

        # Opsi 1: AI-Only Translate
        if settings.get('use_ai_only_translate'):
            if not self.wait_for_api_slot(provider, model_name):
                return processed_text, None
            
            # Dapatkan hasil terjemahan dari AI yang dipilih (Gemini atau OpenAI)
            translated_text = self.main_app.translate_with_ai(processed_text, settings['target_lang'], provider, model_name, settings)
            return processed_text, translated_text

        # Opsi 2: DeepL-Only Translate
        if settings.get('use_deepl_only_translate'):
            translated_text = self.main_app.translate_text(processed_text, settings['target_lang'])
            return processed_text, translated_text

        # Opsi 3: Alur Standar (DeepL sebagai fallback/penerjemah utama non-AI)
        # Fitur koreksi dan naturalisasi AI dinonaktifkan sementara untuk alur standar yang lebih cepat
        translated_text = self.main_app.translate_text(processed_text, settings['target_lang'])

        return processed_text, translated_text

    # Pipeline enhanced: gabungkan hasil Manga-OCR + Tesseract → AI Pilihan
    def run_enhanced_pipeline(self, cropped_cv_img, settings: dict):
        preprocessed_image, _ = self.main_app.preprocess_for_ocr(cropped_cv_img, "Auto-Detect")

        # Paksa penggunaan engine yang sesuai untuk pipeline ini
        manga_ocr_settings = {**settings, 'ocr_engine': 'Manga-OCR', 'ocr_lang': 'ja'}
        tesseract_settings = {**settings, 'ocr_engine': 'Tesseract', 'ocr_lang': 'jpn'}

        manga_ocr_text = self.perform_ocr(preprocessed_image, manga_ocr_settings)
        tesseract_text = self.perform_ocr(preprocessed_image, tesseract_settings)

        original_text = manga_ocr_text if len(manga_ocr_text) > len(tesseract_text) else tesseract_text

        provider, model_name = settings['ai_model']
        if not self.wait_for_api_slot(provider, model_name):
            return original_text, None

        # [DIUBAH] Gunakan fungsi abstrak translate_with_ai
        translated_text = self.main_app.translate_with_ai(
            original_text, 
            settings['target_lang'], 
            provider, 
            model_name, 
            settings,
            is_enhanced=True, 
            ocr_results={'manga_ocr': manga_ocr_text, 'tesseract': tesseract_text}
        )
        return original_text, translated_text


    # Mekanisme tunggu jika API slot penuh (rate limit)
    def wait_for_api_slot(self, provider: str, model_name: str) -> bool:
        while self.is_running:
            if self.main_app.check_and_increment_usage(provider, model_name):
                return True
            now = time.time()
            wait_sec = 61 - int(time.strftime('%S', time.localtime(now)))
            self.signals.status_update.emit(f"API limit {model_name} tercapai. Tunggu {wait_sec}s...")
            time.sleep(wait_sec)
        return False

    # Hentikan worker
    def stop(self):
        self.is_running = False

class AutoDetectorWorker(QObject):
    def __init__(self, main_app, file_paths, settings, detection_mode):
        super().__init__()
        self.main_app = main_app
        self.file_paths = file_paths
        self.settings = settings
        self.detection_mode = detection_mode # "Bubble" atau "Text"
        self.signals = AutoDetectorSignals()
        self.is_cancelled = False

    def run(self):
        total_files = len(self.file_paths)
        for i, file_path in enumerate(self.file_paths):
            if self.is_cancelled:
                break

            self.signals.overall_progress.emit(int((i / total_files) * 100), f"Detecting in {os.path.basename(file_path)}...")

            try:
                image_pil = Image.open(file_path).convert('RGB')
                cv_image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

                detections = [] # List of dicts {'polygon': QPolygon, 'text': str|None}
                
                if self.detection_mode == "Bubble":
                    combined_mask = self.main_app.detect_bubble_with_dl_model(cv_image, self.settings)
                    if combined_mask is not None:
                        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in contours:
                            polygon = QPolygon([QPoint(p[0][0], p[0][1]) for p in cnt])
                            detections.append({'polygon': polygon, 'text': None})

                elif self.detection_mode == "Text":
                    # Panggil fungsi deteksi teks baru
                    text_results = self.main_app.detect_text_with_ocr_engine(cv_image, self.settings)
                    for text, polygon in text_results:
                        detections.append({'polygon': polygon, 'text': text})

                self.signals.detection_complete.emit(file_path, detections)

            except Exception as e:
                self.signals.error.emit(f"Error during {self.detection_mode} detection in {os.path.basename(file_path)}: {e}")
                continue

        self.signals.finished.emit()

    def cancel(self):
        self.is_cancelled = True

class BatchProcessorWorker(QObject):
    def __init__(self, main_app, batch_queue, settings):
        super().__init__()
        self.main_app = main_app
        self.batch_queue = batch_queue
        self.settings = settings
        self.signals = BatchProcessorSignals()

    # Fungsi utama untuk memproses batch
    def run(self):
        try:
            # Kelompokkan pekerjaan berdasarkan path gambar agar OCR dan translasi lebih efisien
            jobs_by_image = {}
            for job in self.batch_queue:
                jobs_by_image.setdefault(job['image_path'], []).append(job)

            # Proses setiap batch per gambar
            for image_path, jobs in jobs_by_image.items():
                self.process_image_batch(image_path, jobs)
        except Exception as e:
            self.signals.error.emit(f"Error in batch processor: {e}")
        finally:
            # Emit sinyal selesai agar UI bisa update status
            self.signals.batch_finished.emit()

    # Fungsi untuk memproses semua job dari satu gambar
    def process_image_batch(self, image_path, jobs):
        provider, model_name = self.settings['ai_model']

        # 1. Lakukan OCR untuk semua pekerjaan
        ocr_texts = []
        for job in jobs:
            try:
                # Jika teks sudah ada dari fase deteksi, gunakan itu
                if job.get('text'):
                    cleaned_text = self.main_app.clean_and_join_text(job['text'])
                else:
                    preprocessed, _ = self.main_app.preprocess_for_ocr(
                        job['cropped_cv_img'], self.settings['orientation']
                    )
                    raw_text = self.main_app.perform_ocr(preprocessed, self.settings)
                    cleaned_text = self.main_app.clean_and_join_text(raw_text)
                
                ocr_texts.append(cleaned_text)
            except Exception as e:
                ocr_texts.append("")  # agar tetap sinkron dengan urutan job
                self.signals.error.emit(f"OCR failed on {image_path}: {e}")

        # Filter teks kosong
        prompt_lines = [f"{i+1}. {text}" for i, text in enumerate(ocr_texts) if text and "[ERROR:" not in text]
        if not prompt_lines:
            return

        # 2. Buat prompt batch
        numbered_ocr_text = "\n".join(prompt_lines)
        target_lang = self.settings['target_lang']
        prompt_enhancements = self.main_app._build_prompt_enhancements(self.settings)

        prompt = f"""
As an expert manga translator, your task is to translate a batch of numbered text snippets from a single manga page.
1. Translate each numbered snippet into natural, colloquial {target_lang}.
2. Maintain the original numbering in your response. Each translation must start with its corresponding number (e.g., "1. ", "2. ").
3. If a snippet is untranslatable or nonsensical, return the original number followed by "[N/A]".

{prompt_enhancements}

Snippets to Translate:
{numbered_ocr_text}

Your final output must ONLY be the translated {target_lang} text, with each translation on a new line and correctly numbered.
"""

        # 3. Panggil API sekali untuk batch ini
        if not self.main_app.wait_for_api_slot(provider, model_name):
            return
        
        # [DIUBAH] Gunakan fungsi abstrak untuk memanggil API
        response_text = self.main_app.call_ai_for_batch(prompt, provider, model_name)
        
        if not response_text or "[ERROR]" in response_text or "[FAILED]" in response_text:
            self.signals.error.emit(
                f"Failed to process batch for {os.path.basename(image_path)}: API call failed."
            )
            return

        try:
            # 4. Parsing respons dan petakan kembali ke pekerjaan
            translated_lines = response_text.strip().splitlines()
            translation_map = {}
            for line in translated_lines:
                match = re.match(r"^\s*(\d+)\.\s*(.*)", line)
                if match:
                    translation_map[int(match.group(1))] = match.group(2).strip()

            for i, job in enumerate(jobs):
                original_text = ocr_texts[i]
                if not original_text:
                    continue

                translated_text = translation_map.get(i + 1)
                
                # Terapkan Safe Mode
                if self.settings.get('safe_mode') and translated_text:
                    translated_text = self.main_app.apply_safe_mode(translated_text)

                if translated_text and "[N/A]" not in translated_text:
                    new_area = TypesetArea(
                        job['rect'], translated_text,
                        self.settings['font'], self.settings['color'],
                        job.get('polygon'),
                        orientation=self.settings.get('orientation_mode', 'horizontal'),
                        effect=self.settings.get('text_effect', 'none'),
                        effect_intensity=self.settings.get('effect_intensity', 20.0),
                        bezier_points=self.settings.get('bezier_points'),
                        bubble_enabled=self.settings.get('create_bubble', False),
                        alignment=self.settings.get('alignment', 'center'),
                        line_spacing=self.settings.get('line_spacing', 1.1),
                        char_spacing=self.settings.get('char_spacing', 100.0),
                        margins=self.settings.get('margins', {'top': 12, 'right': 12, 'bottom': 12, 'left': 12})
                    )
                    self.signals.batch_job_complete.emit(image_path, new_area)

        except Exception as e:
            self.signals.error.emit(
                f"Failed to parse batch response for {os.path.basename(image_path)}: {e}"
            )

# --- Baru: Worker untuk Batch Save ---
class BatchSaveWorker(QObject):
    def __init__(self, main_app, files_to_save):
        super().__init__()
        self.main_app = main_app
        self.files_to_save = files_to_save
        self.signals = BatchSaveSignals()
        self.is_cancelled = False

    def run(self):
        total_files = len(self.files_to_save)
        for i, file_path in enumerate(self.files_to_save):
            if self.is_cancelled:
                break

            self.signals.progress.emit(int(((i + 1) / total_files) * 100), f"Saving {os.path.basename(file_path)}...")

            try:
                # Tentukan nama file output
                path_part, ext = os.path.splitext(file_path)
                save_path = f"{path_part}_typeset.png"

                # Muat gambar asli
                pil_image = Image.open(file_path).convert('RGB')

                # Konversi PIL.Image ke QPixmap dengan cara yang lebih andal
                # untuk menghindari TypeError di dalam thread.
                data = pil_image.tobytes('raw', 'RGB')
                qimage = QImage(data, pil_image.width, pil_image.height, pil_image.width * 3, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimage)

                # Dapatkan data typeset untuk gambar ini
                data_key = self.main_app.get_current_data_key(path=file_path)
                typeset_data = self.main_app.all_typeset_data.get(data_key, {'areas': []})
                areas = typeset_data['areas']

                if not areas:
                    continue # Lewati jika tidak ada yang perlu di-typeset

                # Gambar ulang semua area ke pixmap
                painter = QPainter(pixmap)
                for area in areas:
                    # Panggil draw_single_area dengan flag for_saving=True untuk mencegah pembaruan UI
                    self.main_app.draw_single_area(painter, area, pil_image, for_saving=True)
                painter.end()

                # Simpan gambar
                if not pixmap.save(save_path, "PNG"):
                    raise Exception(f"Failed to save image to {save_path}")

                self.signals.file_saved.emit(file_path)

            except Exception as e:
                self.signals.error.emit(f"Error saving {os.path.basename(file_path)}: {e}")
                continue

        self.signals.finished.emit()

    def cancel(self):
        self.is_cancelled = True


class ReviewDialog(QDialog):
    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Review OCR Text")
        self.setGeometry(200, 200, 400, 300)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Please review and edit the OCR text before translating:"))
        self.text_edit = QTextEdit()
        self.text_edit.setText(text)
        layout.addWidget(self.text_edit)
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("Continue")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.ok_button)
        layout.addLayout(button_layout)

    def get_text(self):
        return self.text_edit.toPlainText()


class AdvancedTextEditDialog(QDialog):
    EFFECT_OPTIONS = [
        ("None", "none"),
        ("Curved", "curved"),
        ("Wavy", "wavy"),
        ("Jagged Shout", "jagged"),
    ]
    ALIGN_OPTIONS = [
        ("Center", Qt.AlignHCenter),
        ("Left", Qt.AlignLeft),
        ("Right", Qt.AlignRight),
        ("Justify", Qt.AlignJustify),
    ]
    EMOJI_PRESETS = [
        ("Heart", "❤"),
        ("Sparkle", "✨"),
        ("Star", "★"),
        ("Music", "♪"),
        ("Shock", "!?"),
        ("Sweat", "💦"),
        ("Smile", "😊"),
        ("Angry", "😠"),
    ]

    def __init__(self, area, parent=None):
        super().__init__(parent)
        self.area = area
        self.result = None
        self.setWindowTitle("Advanced Text Editor")
        self.setModal(True)
        self.resize(820, 600)

        main_layout = QVBoxLayout(self)
        header = QLabel("Fine-tune text formatting. Select a range to style only that portion or press Ctrl+A to target the whole bubble.")
        header.setWordWrap(True)
        main_layout.addWidget(header)

        toolbar_layout = QHBoxLayout()
        self.font_combo = QFontComboBox(); self.font_combo.setMaximumWidth(240)
        toolbar_layout.addWidget(self.font_combo)

        self.font_size_spin = QDoubleSpinBox(); self.font_size_spin.setRange(4.0, 220.0); self.font_size_spin.setDecimals(1); self.font_size_spin.setSingleStep(1.0); self.font_size_spin.setSuffix(" pt")
        toolbar_layout.addWidget(self.font_size_spin)

        self.bold_button = QToolButton(); self.bold_button.setText("B"); self.bold_button.setCheckable(True); self.bold_button.setToolTip("Toggle bold")
        toolbar_layout.addWidget(self.bold_button)

        self.italic_button = QToolButton(); self.italic_button.setText("I"); self.italic_button.setCheckable(True); self.italic_button.setToolTip("Toggle italic")
        toolbar_layout.addWidget(self.italic_button)

        self.underline_button = QToolButton(); self.underline_button.setText("U"); self.underline_button.setCheckable(True); self.underline_button.setToolTip("Toggle underline")
        toolbar_layout.addWidget(self.underline_button)

        self.color_button = QToolButton(); self.color_button.setText("Color"); self.color_button.setToolTip("Change text color")
        toolbar_layout.addWidget(self.color_button)

        self.emoji_button = QToolButton(); self.emoji_button.setText("Emotes"); self.emoji_button.setToolTip("Insert emoticons or symbols")
        self.emoji_menu = QMenu(self)
        for label, symbol in self.EMOJI_PRESETS:
            action = self.emoji_menu.addAction(f"{label} {symbol}")
            action.triggered.connect(lambda checked=False, s=symbol: self._insert_emoji(s))
        self.emoji_button.setMenu(self.emoji_menu); self.emoji_button.setPopupMode(QToolButton.InstantPopup)
        toolbar_layout.addWidget(self.emoji_button)

        toolbar_layout.addStretch()
        main_layout.addLayout(toolbar_layout)

        self.text_edit = QTextEdit(); self.text_edit.setAcceptRichText(True); self.text_edit.setMinimumHeight(260)
        main_layout.addWidget(self.text_edit, 1)

        layout_group = QGroupBox("Layout & Effects")
        layout_grid = QGridLayout(layout_group)

        self.orientation_combo = QComboBox(); self.orientation_combo.addItems(["Horizontal", "Vertical"])
        layout_grid.addWidget(QLabel("Orientation:"), 0, 0); layout_grid.addWidget(self.orientation_combo, 0, 1)

        self.effect_combo = QComboBox()
        for label, value in self.EFFECT_OPTIONS:
            self.effect_combo.addItem(label, value)
        layout_grid.addWidget(QLabel("Effect:"), 0, 2); layout_grid.addWidget(self.effect_combo, 0, 3)

        self.effect_intensity_spin = QDoubleSpinBox(); self.effect_intensity_spin.setRange(0.0, 300.0); self.effect_intensity_spin.setDecimals(1); self.effect_intensity_spin.setSingleStep(5.0)
        layout_grid.addWidget(QLabel("Effect Strength:"), 1, 0); layout_grid.addWidget(self.effect_intensity_spin, 1, 1)

        self.alignment_combo = QComboBox()
        for label, _ in self.ALIGN_OPTIONS:
            self.alignment_combo.addItem(label)
        layout_grid.addWidget(QLabel("Alignment:"), 1, 2); layout_grid.addWidget(self.alignment_combo, 1, 3)

        self.line_spacing_spin = QDoubleSpinBox(); self.line_spacing_spin.setRange(0.6, 3.0); self.line_spacing_spin.setSingleStep(0.1); self.line_spacing_spin.setValue(1.0)
        layout_grid.addWidget(QLabel("Line Spacing:"), 2, 0); layout_grid.addWidget(self.line_spacing_spin, 2, 1)

        self.char_spacing_spin = QDoubleSpinBox(); self.char_spacing_spin.setRange(50.0, 400.0); self.char_spacing_spin.setSingleStep(5.0); self.char_spacing_spin.setSuffix(" %")
        layout_grid.addWidget(QLabel("Character Spacing:"), 2, 2); layout_grid.addWidget(self.char_spacing_spin, 2, 3)

        self.bubble_checkbox = QCheckBox("Render bubble (white fill, black outline)")
        layout_grid.addWidget(self.bubble_checkbox, 3, 0, 1, 4)

        main_layout.addWidget(layout_group)

        margin_group = QGroupBox("Inner Margins (px)")
        margin_grid = QGridLayout(margin_group)
        self.margin_top_spin = QSpinBox(); self.margin_top_spin.setRange(0, 400)
        self.margin_right_spin = QSpinBox(); self.margin_right_spin.setRange(0, 400)
        self.margin_bottom_spin = QSpinBox(); self.margin_bottom_spin.setRange(0, 400)
        self.margin_left_spin = QSpinBox(); self.margin_left_spin.setRange(0, 400)
        margin_grid.addWidget(QLabel("Top:"), 0, 0); margin_grid.addWidget(self.margin_top_spin, 0, 1)
        margin_grid.addWidget(QLabel("Right:"), 0, 2); margin_grid.addWidget(self.margin_right_spin, 0, 3)
        margin_grid.addWidget(QLabel("Bottom:"), 1, 0); margin_grid.addWidget(self.margin_bottom_spin, 1, 1)
        margin_grid.addWidget(QLabel("Left:"), 1, 2); margin_grid.addWidget(self.margin_left_spin, 1, 3)
        main_layout.addWidget(margin_group)

        bezier_group = QGroupBox("Bezier Control Points (0.0 - 1.0)")
        bezier_layout = QGridLayout(bezier_group)
        self.cp1x_spin = self._create_bezier_spin(); self.cp1y_spin = self._create_bezier_spin()
        self.cp2x_spin = self._create_bezier_spin(); self.cp2y_spin = self._create_bezier_spin()
        bezier_layout.addWidget(QLabel("Control 1 X:"), 0, 0); bezier_layout.addWidget(self.cp1x_spin, 0, 1)
        bezier_layout.addWidget(QLabel("Control 1 Y:"), 0, 2); bezier_layout.addWidget(self.cp1y_spin, 0, 3)
        bezier_layout.addWidget(QLabel("Control 2 X:"), 1, 0); bezier_layout.addWidget(self.cp2x_spin, 1, 1)
        bezier_layout.addWidget(QLabel("Control 2 Y:"), 1, 2); bezier_layout.addWidget(self.cp2y_spin, 1, 3)
        main_layout.addWidget(bezier_group)

        button_box = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Save)
        button_box.button(QDialogButtonBox.Save).setText("Apply")
        button_box.accepted.connect(self._handle_accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

        self.font_combo.currentFontChanged.connect(self._change_font_family)
        self.font_size_spin.valueChanged.connect(self._change_font_size)
        self.bold_button.toggled.connect(self._toggle_bold)
        self.italic_button.toggled.connect(self._toggle_italic)
        self.underline_button.toggled.connect(self._toggle_underline)
        self.color_button.clicked.connect(self._choose_color)
        self.text_edit.cursorPositionChanged.connect(self._sync_toolbar_from_cursor)
        self.alignment_combo.currentIndexChanged.connect(self._apply_alignment)
        self.line_spacing_spin.valueChanged.connect(self._apply_line_spacing)
        self.char_spacing_spin.valueChanged.connect(self._apply_char_spacing)

        self._load_area_into_editor()
        self._sync_toolbar_from_cursor()

    def _create_bezier_spin(self):
        spin = QDoubleSpinBox()
        spin.setRange(-1.0, 2.0)
        spin.setDecimals(3)
        spin.setSingleStep(0.05)
        return spin

    def _load_area_into_editor(self):
        self.text_edit.clear()
        cursor = QTextCursor(self.text_edit.document())
        cursor.movePosition(QTextCursor.Start)

        segments = self.area.get_segments()
        if not segments:
            segments = [{'text': self.area.text or '', 'font': self.area.font_to_dict(self.area.get_font()), 'color': self.area.get_color().name(), 'underline': False}]

        for segment in segments:
            text_value = segment.get('text', '')
            if not text_value:
                continue
            fmt = QTextCharFormat()
            seg_font = self.area.segment_to_qfont(segment)
            fmt.setFont(seg_font)
            fmt.setForeground(QBrush(self.area.segment_to_color(segment)))
            if segment.get('underline', seg_font.underline()):
                fmt.setFontUnderline(True)

            parts = text_value.split('\n')
            for idx, part in enumerate(parts):
                cursor.insertText(part, fmt)
                if idx < len(parts) - 1:
                    cursor.insertBlock()

        orientation = self.area.get_orientation()
        with QSignalBlocker(self.orientation_combo):
            self.orientation_combo.setCurrentIndex(1 if orientation == 'vertical' else 0)

        current_effect = self.area.get_effect()
        effect_idx = next((i for i, (_, val) in enumerate(self.EFFECT_OPTIONS) if val == current_effect), 0)
        with QSignalBlocker(self.effect_combo):
            self.effect_combo.setCurrentIndex(effect_idx)

        self.effect_intensity_spin.setValue(self.area.get_effect_intensity())
        self.bubble_checkbox.setChecked(bool(getattr(self.area, 'bubble_enabled', False)))

        bezier = self.area.get_bezier_points()
        if len(bezier) >= 2:
            self.cp1x_spin.setValue(bezier[0].get('x', 0.25)); self.cp1y_spin.setValue(bezier[0].get('y', 0.2))
            self.cp2x_spin.setValue(bezier[1].get('x', 0.75)); self.cp2y_spin.setValue(bezier[1].get('y', 0.2))

        margins = self.area.get_margins()
        self.margin_top_spin.setValue(int(margins.get('top', 12)))
        self.margin_right_spin.setValue(int(margins.get('right', 12)))
        self.margin_bottom_spin.setValue(int(margins.get('bottom', 12)))
        self.margin_left_spin.setValue(int(margins.get('left', 12)))

        align_value = self.area.get_alignment()
        align_idx = next((i for i, (label, _) in enumerate(self.ALIGN_OPTIONS) if label.lower().startswith(align_value)), 0)
        with QSignalBlocker(self.alignment_combo):
            self.alignment_combo.setCurrentIndex(align_idx)
        self._apply_alignment()

        self.line_spacing_spin.setValue(self.area.get_line_spacing())
        self.char_spacing_spin.setValue(self.area.get_char_spacing())
        self._apply_line_spacing()
        self._apply_char_spacing()

        base_font = self.area.get_font()
        with QSignalBlocker(self.font_combo):
            self.font_combo.setCurrentFont(base_font)
        with QSignalBlocker(self.font_size_spin):
            self.font_size_spin.setValue(base_font.pointSizeF() or base_font.pointSize())

    def _insert_emoji(self, text):
        cursor = self.text_edit.textCursor()
        cursor.insertText(text)
        self.text_edit.setTextCursor(cursor)

    def _merge_char_format(self, fmt):
        cursor = self.text_edit.textCursor()
        if not cursor.hasSelection():
            cursor.select(QTextCursor.WordUnderCursor)
        cursor.mergeCharFormat(fmt)
        self.text_edit.mergeCurrentCharFormat(fmt)

    def _change_font_family(self, font):
        fmt = QTextCharFormat(); fmt.setFontFamily(font.family())
        self._merge_char_format(fmt)

    def _change_font_size(self, value):
        fmt = QTextCharFormat(); fmt.setFontPointSize(float(value))
        self._merge_char_format(fmt)

    def _toggle_bold(self, checked):
        fmt = QTextCharFormat(); fmt.setFontWeight(QFont.Bold if checked else QFont.Normal)
        self._merge_char_format(fmt)

    def _toggle_italic(self, checked):
        fmt = QTextCharFormat(); fmt.setFontItalic(checked)
        self._merge_char_format(fmt)

    def _toggle_underline(self, checked):
        fmt = QTextCharFormat(); fmt.setFontUnderline(checked)
        self._merge_char_format(fmt)

    def _choose_color(self):
        color = QColorDialog.getColor(self._current_color_from_cursor(), self, "Select Text Color")
        if color.isValid():
            fmt = QTextCharFormat(); fmt.setForeground(QBrush(color))
            self._merge_char_format(fmt)
            self._update_color_button(color)

    def _current_color_from_cursor(self):
        fmt = self.text_edit.currentCharFormat()
        brush = fmt.foreground()
        return brush.color() if brush.style() != Qt.NoBrush else self.area.get_color()

    def _update_color_button(self, color):
        if color.isValid():
            self.color_button.setStyleSheet(f"background-color: {color.name()}; border: 1px solid #222; color: white;")
        else:
            self.color_button.setStyleSheet("")

    def _apply_alignment(self):
        doc = self.text_edit.document()
        original = self.text_edit.textCursor()
        cursor = QTextCursor(doc)
        cursor.select(QTextCursor.Document)
        block_format = QTextBlockFormat()
        align_label, align_flag = self.ALIGN_OPTIONS[self.alignment_combo.currentIndex()]
        block_format.setAlignment(align_flag)
        block_format.setLineHeight(int(self.line_spacing_spin.value() * 100), QTextBlockFormat.ProportionalHeight)
        cursor.setBlockFormat(block_format)
        self.text_edit.setTextCursor(original)

    def _apply_line_spacing(self):
        doc = self.text_edit.document()
        original = self.text_edit.textCursor()
        cursor = QTextCursor(doc)
        cursor.select(QTextCursor.Document)
        block_format = QTextBlockFormat()
        block_format.setLineHeight(int(self.line_spacing_spin.value() * 100), QTextBlockFormat.ProportionalHeight)
        _, align_flag = self.ALIGN_OPTIONS[self.alignment_combo.currentIndex()]
        block_format.setAlignment(align_flag)
        cursor.setBlockFormat(block_format)
        self.text_edit.setTextCursor(original)

    def _apply_char_spacing(self):
        spacing_value = self.char_spacing_spin.value()
        original = self.text_edit.textCursor()
        cursor = QTextCursor(self.text_edit.document())
        cursor.beginEditBlock()
        cursor.select(QTextCursor.Document)
        fmt = QTextCharFormat()
        base_font = self.text_edit.currentCharFormat().font()
        if not base_font.family():
            base_font = self.area.get_font()
        font = QFont(base_font)
        font.setLetterSpacing(QFont.PercentageSpacing, spacing_value)
        fmt.setFont(font)
        cursor.mergeCharFormat(fmt)
        cursor.endEditBlock()
        self.text_edit.setTextCursor(original)

    def _sync_toolbar_from_cursor(self):
        fmt = self.text_edit.currentCharFormat()
        fam = fmt.fontFamily() or self.area.get_font().family()
        with QSignalBlocker(self.font_combo): self.font_combo.setCurrentFont(QFont(fam))
        point = fmt.fontPointSize() or self.area.get_font().pointSizeF() or self.area.get_font().pointSize()
        with QSignalBlocker(self.font_size_spin): self.font_size_spin.setValue(point)
        with QSignalBlocker(self.bold_button): self.bold_button.setChecked(fmt.fontWeight() >= QFont.Bold)
        with QSignalBlocker(self.italic_button): self.italic_button.setChecked(fmt.fontItalic())
        with QSignalBlocker(self.underline_button): self.underline_button.setChecked(fmt.fontUnderline())
        self._update_color_button(self._current_color_from_cursor())

    def _extract_segments(self):
        doc = self.text_edit.document()
        segments = []
        block = doc.begin()
        while block != doc.end():
            it = block.begin()
            while not it.atEnd():
                fragment = it.fragment()
                if fragment.isValid():
                    text = fragment.text()
                    if text:
                        fmt = fragment.charFormat()
                        font = fmt.font()
                        segments.append({
                            'text': text,
                            'font': TypesetArea.font_to_dict(font),
                            'color': fmt.foreground().color().name() if fmt.foreground().color().isValid() else self.area.get_color().name(),
                            'underline': fmt.fontUnderline(),
                        })
                it += 1
            block = block.next()
            if block != doc.end():
                segments.append({'text': '\n', 'font': TypesetArea.font_to_dict(self.area.get_font()), 'color': self.area.get_color().name(), 'underline': False})
        return segments

    def _handle_accept(self):
        segments = self._extract_segments()
        plain_text = ''.join(seg.get('text', '') for seg in segments)
        orientation = 'vertical' if self.orientation_combo.currentIndex() == 1 else 'horizontal'
        effect = self.effect_combo.currentData()
        bezier_data = [
            {'x': self.cp1x_spin.value(), 'y': self.cp1y_spin.value()},
            {'x': self.cp2x_spin.value(), 'y': self.cp2y_spin.value()},
        ]
        margins = {
            'top': self.margin_top_spin.value(),
            'right': self.margin_right_spin.value(),
            'bottom': self.margin_bottom_spin.value(),
            'left': self.margin_left_spin.value(),
        }
        align_label, _ = self.ALIGN_OPTIONS[self.alignment_combo.currentIndex()]
        self.result = {
            'segments': segments,
            'plain_text': plain_text,
            'orientation': orientation,
            'effect': effect,
            'effect_intensity': self.effect_intensity_spin.value(),
            'bezier_points': bezier_data,
            'bubble_enabled': self.bubble_checkbox.isChecked(),
            'alignment': align_label.lower(),
            'line_spacing': self.line_spacing_spin.value(),
            'char_spacing': self.char_spacing_spin.value(),
            'margins': margins,
        }
        self.accept()

    def get_result(self):
        return self.result


class TypesetArea:
    def __init__(
        self,
        rect,
        text,
        font,
        color,
        polygon=None,
        orientation='horizontal',
        effect='none',
        effect_intensity=20.0,
        bezier_points=None,
        bubble_enabled=False,
        segments=None,
        bubble_fill='#ffffff',
        bubble_outline='#000000',
        bubble_outline_width=3.0,
        alignment='center',
        line_spacing=1.1,
        char_spacing=100.0,
        margins=None,
    ):
        self.rect = rect
        self.text = text or ""
        self.font_info = self.font_to_dict(font)
        self.color_info = color.name()
        self.polygon = polygon
        self.orientation = orientation
        self.effect = effect
        self.effect_intensity = effect_intensity
        self.bezier_points = bezier_points or [
            {'x': 0.25, 'y': 0.2},
            {'x': 0.75, 'y': 0.2},
        ]
        self.bubble_enabled = bubble_enabled
        self.bubble_fill = bubble_fill
        self.bubble_outline = bubble_outline
        self.bubble_outline_width = bubble_outline_width
        self.alignment = alignment
        self.line_spacing = line_spacing
        self.char_spacing = char_spacing
        self.margins = margins or {'top': 12, 'right': 12, 'bottom': 12, 'left': 12}
        self.text_segments = segments if segments is not None else self._build_segments_from_plain(self.text, font, color)
        self.ensure_defaults()

    def ensure_defaults(self):
        if not hasattr(self, 'orientation'): self.orientation = 'horizontal'
        if not hasattr(self, 'effect'): self.effect = 'none'
        if not hasattr(self, 'effect_intensity'): self.effect_intensity = 20.0
        if not getattr(self, 'bezier_points', None):
            self.bezier_points = [{'x': 0.25, 'y': 0.2}, {'x': 0.75, 'y': 0.2}]
        if not hasattr(self, 'bubble_enabled'): self.bubble_enabled = False
        if not getattr(self, 'bubble_fill', None): self.bubble_fill = '#ffffff'
        if not getattr(self, 'bubble_outline', None): self.bubble_outline = '#000000'
        if not hasattr(self, 'bubble_outline_width'): self.bubble_outline_width = 3.0
        if not hasattr(self, 'alignment'): self.alignment = 'center'
        if not hasattr(self, 'line_spacing') or self.line_spacing is None: self.line_spacing = 1.1
        if not hasattr(self, 'char_spacing') or self.char_spacing is None: self.char_spacing = 100.0
        if 'letterSpacing' not in self.font_info:
            self.font_info['letterSpacing'] = self.char_spacing
            self.font_info['letterSpacingType'] = QFont.PercentageSpacing
        if not getattr(self, 'margins', None): self.margins = {'top': 12, 'right': 12, 'bottom': 12, 'left': 12}
        if not getattr(self, 'text_segments', None):
            self.text_segments = self._build_segments_from_plain(self.text, self.get_font(), self.get_color())
        if not getattr(self, 'text', None):
            self.text = self._segments_to_plain_text(self.text_segments)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.ensure_defaults()

    @staticmethod
    def font_to_dict(font):
        return {
            'family': font.family(),
            'pointSize': font.pointSizeF() if hasattr(font, 'pointSizeF') else font.pointSize(),
            'weight': font.weight(),
            'italic': font.italic(),
            'underline': font.underline(),
            'letterSpacing': font.letterSpacing(),
            'letterSpacingType': font.letterSpacingType(),
        }

    def get_font(self):
        font = QFont()
        info = self.font_info
        font.setFamily(info.get('family', 'Arial'))
        point_size = info.get('pointSize', 14)
        font.setPointSizeF(point_size)
        font.setWeight(info.get('weight', QFont.Normal))
        font.setItalic(info.get('italic', False))
        font.setUnderline(info.get('underline', False))
        font.setLetterSpacing(QFont.PercentageSpacing, self.char_spacing or 100.0)
        return font

    def get_color(self):
        return QColor(self.color_info)

    def segment_to_qfont(self, segment):
        info = segment.get('font', self.font_info)
        font = QFont()
        font.setFamily(info.get('family', self.font_info.get('family', 'Arial')))
        font.setPointSizeF(info.get('pointSize', self.font_info.get('pointSize', 14)))
        font.setWeight(info.get('weight', self.font_info.get('weight', QFont.Normal)))
        font.setItalic(info.get('italic', self.font_info.get('italic', False)))
        font.setUnderline(segment.get('underline', info.get('underline', False)))
        spacing = info.get('letterSpacing', self.char_spacing or 100.0)
        font.setLetterSpacing(info.get('letterSpacingType', QFont.PercentageSpacing), spacing)
        return font

    def segment_to_color(self, segment):
        return QColor(segment.get('color', self.color_info))

    def get_segments(self):
        self.ensure_defaults()
        return self.text_segments

    def set_segments(self, segments):
        self.text_segments = segments or []
        self.text = self._segments_to_plain_text(self.text_segments)

    def get_orientation(self):
        return getattr(self, 'orientation', 'horizontal') or 'horizontal'

    def get_effect(self):
        return getattr(self, 'effect', 'none') or 'none'

    def get_effect_intensity(self):
        try:
            return float(getattr(self, 'effect_intensity', 20.0) or 20.0)
        except (TypeError, ValueError):
            return 20.0

    def get_bezier_points(self):
        self.ensure_defaults()
        return self.bezier_points

    def get_bubble_fill_color(self):
        return QColor(getattr(self, 'bubble_fill', '#ffffff'))

    def get_bubble_outline_color(self):
        return QColor(getattr(self, 'bubble_outline', '#000000'))

    def get_alignment(self):
        return getattr(self, 'alignment', 'center') or 'center'

    def get_line_spacing(self):
        try:
            val = float(getattr(self, 'line_spacing', 1.1) or 1.1)
            return max(0.6, min(val, 5.0))
        except (TypeError, ValueError):
            return 1.1

    def get_char_spacing(self):
        try:
            val = float(getattr(self, 'char_spacing', 100.0) or 100.0)
            return max(10.0, min(val, 500.0))
        except (TypeError, ValueError):
            return 100.0

    def get_margins(self):
        margins = getattr(self, 'margins', {'top': 12, 'right': 12, 'bottom': 12, 'left': 12})
        for key in ('top', 'right', 'bottom', 'left'):
            if key not in margins:
                margins[key] = 12
        return margins

    def _build_segments_from_plain(self, text, font, color):
        segment = {
            'text': text or '',
            'font': self.font_to_dict(font),
            'color': color.name(),
            'underline': font.underline(),
        }
        return [segment]

    def _segments_to_plain_text(self, segments):
        if not segments:
            return ''
        return ''.join(seg.get('text', '') for seg in segments)

class SelectableImageLabel(QLabel):
    areaDoubleClicked = pyqtSignal(TypesetArea)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.setMouseTracking(True)
        self.selection_start = None
        self.selection_end = None
        self.selection_rect = QRect()
        self.dragging = False
        self.polygon_points = []
        self.current_mouse_pos = None
        self.setCursor(Qt.CrossCursor)
        self.hovered_area = None
        self.trash_icon_rect = QRect()
        self.edit_icon_rect = QRect()

        # --- Variabel baru untuk deteksi interaktif ---
        self.pending_bubble_polygon = None
        self.pending_bubble_rect = QRect()
        self.pending_trash_icon_rect = QRect()
        self.hovering_pending_trash = False

        self.detected_items = [] # List of dicts {'polygon': QPolygon, 'text': str}
        self.hovered_item_index = -1

    def get_selection_mode(self):
        return self.main_window.selection_mode_combo.currentText()

    def get_polygon_points(self):
        return self.polygon_points
    
    # --- Metode baru untuk bubble/text interaktif ---
    def set_pending_item(self, polygon):
        """Menetapkan item yang terdeteksi untuk konfirmasi pengguna."""
        self.clear_selection()
        if polygon and not polygon.isEmpty():
            self.pending_bubble_polygon = polygon
            self.pending_bubble_rect = polygon.boundingRect()
        else:
            self.pending_bubble_polygon = None
            self.pending_bubble_rect = QRect()
        self.update()

    def cancel_pending_item(self):
        """Membatalkan item yang menunggu konfirmasi."""
        self.pending_bubble_polygon = None
        self.pending_bubble_rect = QRect()
        self.main_window.statusBar().showMessage("Deteksi dibatalkan.", 2000)
        self.update()

    def set_detected_items(self, items):
        self.detected_items = items
        self.hovered_item_index = -1
        self.update()

    def clear_detected_items(self):
        self.detected_items = []
        self.update()

    def mouseDoubleClickEvent(self, event):
        if self.main_window.is_in_confirmation_mode: return
        if not self.main_window.original_pixmap: return
        unzoomed_pos = self.main_window.unzoom_coords(event.pos(), as_point=True)
        if unzoomed_pos:
            for area in reversed(self.main_window.typeset_areas):
                if area.rect.contains(unzoomed_pos):
                    self.areaDoubleClicked.emit(area)
                    return

    def mousePressEvent(self, event):
        if not self.main_window.original_pixmap: return
        
        # --- Logika baru untuk menangani item yang menunggu konfirmasi ---
        if self.pending_bubble_polygon:
            unzoomed_pos = self.main_window.unzoom_coords(event.pos(), as_point=True)
            if self.pending_trash_icon_rect.contains(event.pos()):
                self.cancel_pending_item()
                return

            if unzoomed_pos and self.pending_bubble_polygon.containsPoint(unzoomed_pos, Qt.OddEvenFill):
                if event.button() == Qt.RightButton:
                    self.main_window.confirm_pending_item(self.pending_bubble_polygon)
                    self.pending_bubble_polygon = None # Hapus setelah dikonfirmasi
                    self.update()
                    return
                if event.button() == Qt.MiddleButton:
                    self.cancel_pending_item()
                    return

        if self.main_window.is_in_confirmation_mode:
            if event.button() == Qt.LeftButton and self.hovered_item_index != -1:
                self.main_window.remove_detected_item(self.hovered_item_index)
                self.hovered_item_index = -1
                return

        if self.hovered_area:
            if self.trash_icon_rect.contains(event.pos()):
                if self.main_window.is_in_confirmation_mode: return
                self.main_window.delete_typeset_area(self.hovered_area)
                self.hovered_area = None
                self.update()
                return
            if self.edit_icon_rect.contains(event.pos()):
                if self.main_window.is_in_confirmation_mode: return
                self.main_window.start_inline_edit(self.hovered_area)
                self.hovered_area = None
                self.update()
                return

        if event.button() == Qt.LeftButton:
            mode = self.get_selection_mode()
            if "Bubble Finder" in mode or "Direct OCR" in mode:
                self.clear_selection()
                self.selection_start = event.pos()
                self.selection_end = event.pos()
                self.dragging = True
            elif mode == "Pen Tool":
                if not self.polygon_points:
                    self.clear_selection()
                self.polygon_points.append(event.pos())
                self.main_window.update_pen_tool_buttons_visibility(True)
            self.update()

    def mouseMoveEvent(self, event):
        self.current_mouse_pos = event.pos()

        # Cek hover di atas ikon tong sampah untuk item yang menunggu
        if self.pending_bubble_polygon:
            new_hover_state = self.pending_trash_icon_rect.contains(self.current_mouse_pos)
            if self.hovering_pending_trash != new_hover_state:
                self.hovering_pending_trash = new_hover_state
                self.update() # Perbarui untuk mengubah warna ikon

        if self.main_window.is_in_confirmation_mode:
            unzoomed_pos = self.main_window.unzoom_coords(self.current_mouse_pos, as_point=True)
            new_hover_index = -1
            if unzoomed_pos:
                # Iterasi terbalik agar item di atas terdeteksi dulu
                for i in range(len(self.detected_items) - 1, -1, -1):
                    item = self.detected_items[i]
                    if item['polygon'].containsPoint(unzoomed_pos, Qt.OddEvenFill):
                        new_hover_index = i
                        break
            if self.hovered_item_index != new_hover_index:
                self.hovered_item_index = new_hover_index
                self.update()
        else:
            unzoomed_pos = self.main_window.unzoom_coords(self.current_mouse_pos, as_point=True)
            new_hover_area = None
            if unzoomed_pos:
                for area in reversed(self.main_window.typeset_areas):
                    if area.rect.contains(unzoomed_pos):
                        new_hover_area = area
                        break

            if self.hovered_area != new_hover_area:
                self.hovered_area = new_hover_area
                self.update()

        mode = self.get_selection_mode()
        if "Bubble Finder" in mode or "Direct OCR" in mode:
            if self.dragging:
                self.selection_end = self.current_mouse_pos
                self.update()
        elif mode == "Pen Tool":
            if self.polygon_points:
                self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton and self.get_selection_mode() == "Pen Tool":
            if len(self.polygon_points) >= 3:
                self.main_window.confirm_pen_selection()
            else:
                self.main_window.cancel_pen_selection()
            return

        if event.button() == Qt.LeftButton:
            mode = self.get_selection_mode()
            if "Bubble Finder" in mode or "Direct OCR" in mode:
                if self.dragging:
                    self.dragging = False
                    self.selection_rect = QRect(self.selection_start, self.selection_end).normalized()
                    
                    if self.selection_rect.width() > 10 and self.selection_rect.height() > 10:
                        unzoomed_rect = self.main_window.unzoom_coords(self.selection_rect)
                        if unzoomed_rect:
                            if "Bubble Finder" in mode:
                                self.main_window.find_bubble_in_rect(unzoomed_rect)
                            elif mode == "Direct OCR (Rect)":
                                self.main_window.process_rect_area(self.selection_rect)
                            elif mode == "Direct OCR (Oval)":
                                # Buat poligon dari elips dan proses
                                path = QPainterPath()
                                path.addEllipse(QRectF(self.selection_rect))
                                polygon = path.toFillPolygon().toPolygon()
                                self.main_window.process_polygon_area(list(polygon))
                        else:
                            self.clear_selection()
                    
                    # Jangan hapus selection_rect agar bisa ditampilkan sampai bubble ditemukan/dibatalkan
                    if "Bubble Finder" not in mode:
                        self.clear_selection()

                    self.update()


    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.main_window.original_pixmap: return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self.main_window.is_in_confirmation_mode and self.detected_items:
            painter.save()
            scale = self.main_window.zoom_factor
            pixmap = self.pixmap()
            if pixmap and not pixmap.isNull():
                label_size = self.size()
                pixmap_size = pixmap.size()
                offset_x = max(0, (label_size.width() - pixmap_size.width()) // 2)
                offset_y = max(0, (label_size.height() - pixmap_size.height()) // 2)
                painter.translate(offset_x, offset_y)
                painter.scale(scale, scale)

            for i, item in enumerate(self.detected_items):
                path = QPainterPath()
                path.addPolygon(QPolygonF(item['polygon']))

                if i == self.hovered_item_index:
                    painter.fillPath(path, QColor(255, 80, 80, 150))
                    painter.setPen(QPen(QColor(255, 100, 100), 3 / scale))
                else:
                    # Biru untuk bubble, Hijau untuk teks
                    fill_color = QColor(0, 120, 215, 100) if item['text'] is None else QColor(0, 180, 100, 100)
                    pen_color = QColor(90, 180, 255) if item['text'] is None else QColor(90, 220, 150)
                    painter.fillPath(path, fill_color)
                    painter.setPen(QPen(pen_color, 2 / scale))

                painter.drawPath(path)
            painter.restore()

        mode = self.get_selection_mode()
        if "Bubble Finder" in mode or "Direct OCR" in mode:
            if self.dragging:
                rect = QRect(self.selection_start, self.selection_end).normalized()
                painter.setPen(QPen(QColor(0, 120, 215), 2, Qt.DashLine))
                if "Oval" in mode:
                    painter.drawEllipse(rect)
                else:
                    painter.drawRect(rect)
                painter.fillRect(rect, QColor(0, 120, 215, 50))
        elif mode == "Pen Tool" and self.polygon_points:
            painter.setPen(QPen(QColor(0, 255, 0), 2, Qt.SolidLine))
            for point in self.polygon_points:
                painter.drawEllipse(point, 3, 3)
            if len(self.polygon_points) > 1:
                painter.drawPolyline(QPolygon(self.polygon_points))
            if self.current_mouse_pos:
                painter.setPen(QPen(QColor(0, 255, 0), 2, Qt.DashLine))
                painter.drawLine(self.polygon_points[-1], self.current_mouse_pos)
        
        # --- Gambar item yang menunggu konfirmasi ---
        if self.pending_bubble_polygon:
            painter.save()
            scale = self.main_window.zoom_factor
            pixmap = self.pixmap()
            if pixmap and not pixmap.isNull():
                label_size = self.size()
                pixmap_size = pixmap.size()
                offset_x = max(0, (label_size.width() - pixmap_size.width()) // 2)
                offset_y = max(0, (label_size.height() - pixmap_size.height()) // 2)
                painter.translate(offset_x, offset_y)
                painter.scale(scale, scale)

            path = QPainterPath()
            path.addPolygon(QPolygonF(self.pending_bubble_polygon))
            
            # Gambar outline
            painter.setPen(QPen(QColor(255, 200, 0), 3 / scale, Qt.DashLine))
            painter.fillPath(path, QColor(255, 200, 0, 80))
            painter.drawPath(path)
            painter.restore()
            
            # Gambar ikon di atasnya
            zoomed_rect = self.main_window.zoom_coords(self.pending_bubble_rect)
            icon_size = 24; margin = 5
            self.pending_trash_icon_rect = QRect(zoomed_rect.topRight().x() - icon_size - margin, zoomed_rect.topRight().y() + margin, icon_size, icon_size)
            
            # Ikon Tong Sampah
            trash_color = QColor(255, 100, 100, 220) if self.hovering_pending_trash else QColor(255, 80, 80, 200)
            painter.setBrush(trash_color); painter.setPen(Qt.NoPen)
            painter.drawEllipse(self.pending_trash_icon_rect)
            pen = QPen(Qt.white, 2); painter.setPen(pen)
            painter.drawLine(self.pending_trash_icon_rect.topLeft() + QPoint(6,6), self.pending_trash_icon_rect.bottomRight() - QPoint(6,6))
            painter.drawLine(self.pending_trash_icon_rect.topRight() - QPoint(6,-6), self.pending_trash_icon_rect.bottomLeft() + QPoint(6,-6))


        if self.hovered_area and not self.main_window.is_in_confirmation_mode:
            zoomed_rect = self.main_window.zoom_coords(self.hovered_area.rect)
            icon_size = 24
            margin = 5

            # Trash Icon
            self.trash_icon_rect = QRect(zoomed_rect.topRight().x() - icon_size - margin, zoomed_rect.topRight().y() + margin, icon_size, icon_size)
            painter.setBrush(QColor(255, 80, 80, 200)); painter.setPen(Qt.NoPen)
            painter.drawEllipse(self.trash_icon_rect)
            pen = QPen(Qt.white, 2); painter.setPen(pen)
            painter.drawLine(self.trash_icon_rect.topLeft() + QPoint(6,6), self.trash_icon_rect.bottomRight() - QPoint(6,6))
            painter.drawLine(self.trash_icon_rect.topRight() - QPoint(6,-6), self.trash_icon_rect.bottomLeft() + QPoint(6,-6))

            # Edit Icon
            self.edit_icon_rect = QRect(self.trash_icon_rect.left() - icon_size - margin, self.trash_icon_rect.top(), icon_size, icon_size)
            painter.setBrush(QColor(80, 150, 255, 200)); painter.setPen(Qt.NoPen)
            painter.drawEllipse(self.edit_icon_rect)
            painter.setPen(pen)
            # Draw a simple pencil
            poly = QPolygon([QPoint(6, 18), QPoint(6, 14), QPoint(14, 6), QPoint(18, 10), QPoint(10, 18), QPoint(6, 18)])
            painter.drawPolyline(poly.translated(self.edit_icon_rect.topLeft()))
            painter.drawLine(QPoint(15,7)+self.edit_icon_rect.topLeft(), QPoint(17,9)+self.edit_icon_rect.topLeft())


    def clear_selection(self):
        self.selection_start = None
        self.selection_end = None
        self.selection_rect = QRect()
        self.dragging = False
        self.polygon_points = []
        self.current_mouse_pos = None
        # Jangan hapus item yang menunggu konfirmasi saat seleksi dibersihkan
        if self.main_window:
            self.main_window.update_pen_tool_buttons_visibility(False)
        self.update()

class MangaOCRApp(QMainWindow):
    DARK_THEME_STYLESHEET = """
        QMainWindow, QDialog {
            background-color: #0f141b;
            color: #f3f6fb;
        }
        QWidget {
            background-color: #121a24;
            color: #f3f6fb;
            font-size: 10pt;
            font-family: 'Segoe UI', 'Open Sans', sans-serif;
        }
        QLabel {
            padding: 2px;
            background-color: transparent;
        }
        QLabel#h3 {
            color: #7faeff;
            font-size: 12pt;
            font-weight: 600;
            margin-top: 10px;
            border-bottom: 1px solid #1f2b3b;
            padding-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }
        QMenuBar {
            background-color: #141e2a;
            color: #f3f6fb;
            border: none;
        }
        QMenuBar::item {
            padding: 6px 12px;
            margin: 0 2px;
            border-radius: 4px;
        }
        QMenuBar::item:selected {
            background-color: #1f88ff;
        }
        QMenu {
            background-color: #152231;
            color: #f3f6fb;
            border: 1px solid #223347;
            padding: 6px;
        }
        QMenu::item {
            border-radius: 4px;
            padding: 6px 12px;
        }
        QMenu::item:selected {
            background-color: #1f88ff;
        }
        QPushButton {
            background-color: #1f88ff;
            color: #ffffff;
            padding: 8px 14px;
            border-radius: 6px;
            border: none;
            margin: 2px;
            font-weight: 600;
        }
        QPushButton:hover {
            background-color: #3a9bff;
        }
        QPushButton:pressed {
            background-color: #186cd6;
        }
        QPushButton:disabled {
            background-color: #253043;
            color: #7c8a9d;
        }
        QTextEdit, QComboBox, QListWidget, QLineEdit, QSpinBox, QCheckBox, QRadioButton {
            background-color: #172330;
            border: 1px solid #1f2b3b;
            border-radius: 6px;
            padding: 6px 8px;
            color: #f3f6fb;
            selection-background-color: #1f88ff;
        }
        QComboBox::drop-down {
            border: none;
            width: 26px;
        }
        QListWidget::item {
            padding: 8px;
            border-radius: 4px;
        }
        QListWidget::item:selected {
            background-color: #1f88ff;
            color: #ffffff;
        }
        QScrollArea, QScrollArea QWidget {
            background: transparent;
            border: none;
        }
        QGroupBox {
            background-color: #141e2a;
            border: 1px solid #1f2b3b;
            border-radius: 10px;
            margin-top: 14px;
            padding: 14px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 16px;
            padding: 0 8px;
            color: #7faeff;
            font-weight: 600;
        }
        QTabWidget::pane {
            border: 1px solid #1f2b3b;
            border-radius: 10px;
            background-color: #121a24;
            margin-top: 10px;
        }
        QTabBar::tab {
            background: #141e2a;
            color: #adbcd3;
            padding: 10px 20px;
            border: 1px solid transparent;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            margin-right: 4px;
        }
        QTabBar::tab:selected {
            background: #1f88ff;
            color: #ffffff;
        }
        QTabBar::tab:hover:!selected {
            background: #1b2738;
            color: #d4e2ff;
        }
        QProgressBar {
            background-color: #1a2634;
            border: 1px solid #1f2b3b;
            border-radius: 6px;
            height: 18px;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #1f88ff;
            border-radius: 6px;
        }
        QStatusBar {
            background-color: #101824;
            color: #9cb4d0;
            border-top: 1px solid #1f2b3b;
        }
        QSplitter::handle {
            background-color: #1f2b3b;
            margin: 0 6px;
        }
        QFrame[frameShape="5"] {
            color: #1f2b3b;
        }
        QToolTip {
            background-color: #1b2738;
            color: #f3f6fb;
            border: 1px solid #1f88ff;
            padding: 6px 8px;
        }
    """
    LIGHT_THEME_STYLESHEET = """
        /* TODO: Implement a light theme if needed */
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Manga OCR & Typeset Tool v16.1.0")
        self.image_files = []
        self.current_image_path = None
        self.current_image_pil = None
        self.original_pixmap = None
        self.typeset_pixmap = None
        self.zoom_factor = 1.0

        self.all_typeset_data = {}
        self.typeset_areas = []
        self.redo_stack = []
        
        # --- OCR Engine Instances ---
        self.manga_ocr_reader = None
        self.easyocr_reader = None
        self.easyocr_lang = None
        self.paddle_ocr_reader = None
        self.paddle_lang = None
        self.doctr_predictor = None
        self.rapid_ocr_reader = None
        self.rapid_lang = None

        # --- [BARU] Inpainting Engine Instances ---
        self.inpaint_model = None
        self.current_inpaint_model_key = None # Untuk melacak model mana yang sedang dimuat

        self.current_project_path = None
        self.current_theme = 'dark'
        self.typeset_font = QFont("Arial", 12, QFont.Bold)
        self.typeset_color = QColor(Qt.black)

        self.processing_queue = []
        self.queue_mutex = QMutex()

        self.batch_save_worker = None
        self.batch_save_thread = None

        self.project_dir = None
        self.cache_dir = None
        self.file_watcher = QFileSystemWatcher(self)
        self.file_watcher.directoryChanged.connect(self.on_directory_changed)

        self.pdf_document = None
        self.current_pdf_page = -1

        self.usage_file_path = os.path.join(os.path.expanduser("~"), "manga_ocr_usage_v16.dat")
        self.usage_data = {}
        self.api_limit_timer = QTimer(self)
        self.api_limit_timer.setInterval(1000)
        self.api_limit_timer.timeout.connect(self.periodic_limit_check)
        self.autosave_timer = QTimer(self)
        self.autosave_timer.setInterval(300000)
        self.autosave_timer.timeout.connect(self.auto_save_project)

        self.dl_models = {
            'kitsumed_onnx': {'path': 'models/model_dynamic.onnx', 'instance': None, 'type': 'onnx'},
            'kitsumed_pt':   {'path': 'models/model.pt', 'instance': None, 'type': 'yolo'},
            'ogkalu_pt':     {'path': 'models/comic-speech-bubble-detector.pt', 'instance': None, 'type': 'yolo'},
            # [BARU] Inpainting Models
            'big_lama':      {'path': 'big-lama/models/best.ckpt', 'instance': None, 'type': 'inpaint'},
            'anime_inpaint': {'path': 'models/lama_large_512px.ckpt', 'instance': None, 'type': 'inpaint'},
        }
        
        # [DIUBAH] Status ketersediaan library dan hardware
        self.is_gpu_available = is_gpu_available
        self.is_yolo_available = YOLO is not None
        self.is_onnx_available = onnxruntime is not None
        self.is_paddle_available = paddleocr is not None
        self.is_doctr_available = doctr is not None
        self.is_rapidocr_available = RapidOCR is not None
        self.is_lama_available = lama_cleaner is not None
        self.is_openai_available = openai is not None and openai_client is not None

        self.total_cost = 0.0
        self.usd_to_idr_rate = 16200.0

        self.exchange_rate_thread = None
        self.exchange_rate_worker = None
        
        # [DIUBAH] Struktur data model AI untuk mendukung beberapa provider
        # Harga OpenAI per karakter diperkirakan dari harga per token (asumsi 1 token ~ 4 karakter)
        self.AI_PROVIDERS = {
            'Gemini': {
                'gemini-2.5-flash-lite': {
                    'display': 'Gemini 2.5 Flash Lite (Utama - Cepat & Murah)',
                    'pricing': {'input': 0.0001 / 1000, 'output': 0.0002 / 1000}, 'limits': {'rpm': 4000, 'rpd': 10000000}
                },
                'gemini-2.5-flash': {
                    'display': 'Gemini 2.5 Flash (Akurasi Lebih Tinggi)',
                    'pricing': {'input': 0.000125 / 1000, 'output': 0.00025 / 1000}, 'limits': {'rpm': 1000, 'rpd': 10000}
                },
                'gemini-2.5-pro': {
                    'display': 'Gemini 2.5 Pro (Teks Rumit & Penting)',
                    'pricing': {'input': 0.0025 / 1000, 'output': 0.0025 / 1000}, 'limits': {'rpm': 150, 'rpd': 10000}
                }
            },
            'OpenAI': {
                'gpt-4o-mini': {
                    'display': 'GPT-4o Mini (Alternatif Cepat)',
                    'pricing': {'input': (0.15 / 1000000) / 4, 'output': (0.6 / 1000000) / 4}, 'limits': {'rpm': 10000, 'rpd': 1000000}
                },
                'gpt-5-nano': {
                    'display': 'GPT-5 Nano (Hipotetikal)',
                    'pricing': {'input': (0.15 / 1000000) / 4, 'output': (0.6 / 1000000) / 4}, 'limits': {'rpm': 10000, 'rpd': 1000000}
                },
                 'gpt-5-mini': {
                    'display': 'GPT-5 Mini (Hipotetikal)',
                    'pricing': {'input': (0.15 / 1000000) / 4, 'output': (0.6 / 1000000) / 4}, 'limits': {'rpm': 10000, 'rpd': 1000000}
                }
            }
        }
        
        self.OCR_LANGS = {} # Akan diisi saat inisialisasi

        self.batch_processing_queue = []
        self.batch_processor_thread = None
        self.batch_processor_worker = None
        self.BATCH_SIZE_LIMIT = 20

        self.worker_pool = {}
        self.next_worker_id = 0
        self.MAX_WORKERS = 15
        self.WORKER_SPAWN_THRESHOLD = 1

        self.is_processing_selection = False

        self.ui_update_queue = []
        self.ui_update_mutex = QMutex()
        self.ui_update_timer = QTimer(self)
        self.ui_update_timer.setSingleShot(True)
        self.ui_update_timer.timeout.connect(self.process_ui_updates)
        self.is_processing_ui_updates = False

        self.is_in_confirmation_mode = False
        self.detection_thread = None
        self.detection_worker = None
        self.detected_items_map = {} # path -> list of dicts

        self.init_ui()
        self.setup_styles()
        self.setup_shortcuts()
        self.initialize_core_engines()
        self.load_usage_data()
        self.check_limits_and_update_ui()
        self.fetch_exchange_rate()


    def setup_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('&File')
        save_project_action = QAction('Save Project', self); save_project_action.setShortcut('Ctrl+S'); save_project_action.triggered.connect(self.save_project); file_menu.addAction(save_project_action)
        load_project_action = QAction('Load Project', self); load_project_action.setShortcut('Ctrl+O'); load_project_action.triggered.connect(self.load_project); file_menu.addAction(load_project_action)

        file_menu.addSeparator()
        batch_save_action = QAction('Batch Save...', self)
        batch_save_action.triggered.connect(self.open_batch_save_dialog)
        file_menu.addAction(batch_save_action)
        export_pdf_action = QAction('Export Typeset to PDF...', self)
        export_pdf_action.triggered.connect(self.export_to_pdf)
        file_menu.addAction(export_pdf_action)

        view_menu = menu_bar.addMenu('&View')
        toggle_theme_action = QAction('Toggle Light/Dark Mode', self); toggle_theme_action.triggered.connect(self.toggle_theme); view_menu.addAction(toggle_theme_action)
        help_menu = menu_bar.addMenu('&Help / Usage')
        about_action = QAction('About & API Usage', self); about_action.triggered.connect(self.show_about_dialog); help_menu.addAction(about_action)

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.setup_menu_bar()
        self.setStatusBar(QStatusBar(self))
        self.statusBar().addPermanentWidget(QProgressBar())
        self.overall_progress_bar = self.statusBar().findChild(QProgressBar)
        self.overall_progress_bar.setVisible(False)
        self.overall_progress_bar.setMaximumWidth(200)

        # Main splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)

        # Left Panel (File List)
        left_panel_widget = QWidget()
        left_panel_widget.setMinimumWidth(240)
        left_panel_layout = QVBoxLayout(left_panel_widget)
        left_panel_layout.setContentsMargins(10, 10, 10, 10)
        left_panel_layout.addWidget(QLabel("<h3>Image Files</h3>", objectName="h3"))
        self.file_list_widget = QListWidget()
        self.file_list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.file_list_widget.currentItemChanged.connect(self.on_file_selected)
        left_panel_layout.addWidget(self.file_list_widget)
        load_folder_button = QPushButton("Load Folder")
        load_folder_button.clicked.connect(self.load_folder)
        left_panel_layout.addWidget(load_folder_button)
        splitter.addWidget(left_panel_widget)

        # Center Panel (Image Viewer)
        center_panel_widget = QWidget()
        center_layout = QVBoxLayout(center_panel_widget)
        center_layout.setContentsMargins(0,10,0,0)
        center_layout.setSpacing(5)
        self.image_label = SelectableImageLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.areaDoubleClicked.connect(self.start_inline_edit)
        image_scroll = QScrollArea()
        image_scroll.setWidget(self.image_label)
        image_scroll.setWidgetResizable(True)
        center_layout.addWidget(image_scroll)

        # Navigation and Zoom Controls
        nav_zoom_widget = QWidget()
        nav_zoom_layout = QHBoxLayout(nav_zoom_widget)
        nav_zoom_layout.setContentsMargins(10, 5, 10, 5)
        self.prev_button = QPushButton("<< Prev")
        self.prev_button.clicked.connect(self.load_prev_image)
        nav_zoom_layout.addWidget(self.prev_button)
        nav_zoom_layout.addStretch()
        self.zoom_out_button = QPushButton("Zoom Out (-)"); self.zoom_out_button.clicked.connect(self.zoom_out)
        self.zoom_label = QLabel(f" Zoom: {self.zoom_factor:.1f}x ")
        self.zoom_in_button = QPushButton("Zoom In (+)"); self.zoom_in_button.clicked.connect(self.zoom_in)
        nav_zoom_layout.addWidget(self.zoom_out_button); nav_zoom_layout.addWidget(self.zoom_label); nav_zoom_layout.addWidget(self.zoom_in_button)
        nav_zoom_layout.addStretch()
        self.next_button = QPushButton("Next >>")
        self.next_button.clicked.connect(self.load_next_image)
        nav_zoom_layout.addWidget(self.next_button)
        center_layout.addWidget(nav_zoom_widget)
        splitter.addWidget(center_panel_widget)

        # Right Panel (Controls)
        right_panel_layout = self.setup_right_panel()
        right_panel_content = QWidget()
        right_panel_content.setObjectName("right-panel")
        right_panel_content.setLayout(right_panel_layout)

        self.right_panel_scroll = QScrollArea()
        self.right_panel_scroll.setWidgetResizable(True)
        self.right_panel_scroll.setFrameShape(QFrame.NoFrame)
        self.right_panel_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.right_panel_scroll.setMinimumWidth(360)
        self.right_panel_scroll.setWidget(right_panel_content)
        splitter.addWidget(self.right_panel_scroll)

        # Make splitter adaptive across screen sizes
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([260, 960, 420])

        main_layout.addWidget(splitter)

    def setup_right_panel(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10,10,10,10)

        # Tabs for main functions
        self.tabs = QTabWidget()
        self.tabs.setObjectName("main-tabs")
        self.tabs.setDocumentMode(True)
        self.tabs.setMovable(False)
        self.tabs.addTab(self._create_translate_tab(), "Translate")
        self.tabs.addTab(self._create_cleanup_tab(), "Cleanup")
        self.tabs.addTab(self._create_typeset_tab(), "Typeset")
        self.tabs.addTab(self._create_ai_hardware_tab(), "AI Hardware")
        main_layout.addWidget(self.tabs)
        main_layout.addStretch()

        # --- Bottom section for persistent controls ---
        bottom_frame = QFrame()
        bottom_frame.setFrameShape(QFrame.NoFrame)
        bottom_layout = QVBoxLayout(bottom_frame)
        bottom_layout.setContentsMargins(0,10,0,0)

        # Action Buttons
        h_line = QFrame(); h_line.setFrameShape(QFrame.HLine); h_line.setFrameShadow(QFrame.Sunken)
        bottom_layout.addWidget(h_line)
        actions_label = QLabel("<h3>Actions</h3>"); actions_label.setObjectName("h3")
        bottom_layout.addWidget(actions_label)

        self.process_batch_button = QPushButton("Process Batch Now (0 items)")
        self.process_batch_button.clicked.connect(self.start_batch_processing)
        bottom_layout.addWidget(self.process_batch_button)
        self.on_batch_mode_changed(False) # Initially hidden

        self.batch_process_button = QPushButton("Detect All Files")
        self.batch_process_button.setToolTip("Detects all bubbles/text in every file in the folder, lets you confirm, then processes them.")
        self.batch_process_button.clicked.connect(self.start_interactive_batch_detection)
        bottom_layout.addWidget(self.batch_process_button)

        self.confirm_items_button = QPushButton("Confirm Items (0)")
        self.confirm_items_button.clicked.connect(self.process_confirmed_detections); self.confirm_items_button.setVisible(False)
        bottom_layout.addWidget(self.confirm_items_button)

        self.cancel_detection_button = QPushButton("Cancel Detection")
        self.cancel_detection_button.clicked.connect(self.cancel_interactive_batch); self.cancel_detection_button.setVisible(False)
        bottom_layout.addWidget(self.cancel_detection_button)

        undo_redo_layout = QHBoxLayout()
        self.undo_button = QPushButton("Undo"); self.undo_button.clicked.connect(self.undo_last_action); self.undo_button.setEnabled(False)
        self.redo_button = QPushButton("Redo"); self.redo_button.clicked.connect(self.redo_last_action); self.redo_button.setEnabled(False)
        undo_redo_layout.addWidget(self.undo_button); undo_redo_layout.addWidget(self.redo_button)
        bottom_layout.addLayout(undo_redo_layout)

        save_reset_layout = QHBoxLayout()
        self.reset_button = QPushButton("Reset Image"); self.reset_button.clicked.connect(self.reset_view_to_original)
        self.save_button = QPushButton("Save Image"); self.save_button.clicked.connect(self.save_image)
        save_reset_layout.addWidget(self.reset_button); save_reset_layout.addWidget(self.save_button)
        bottom_layout.addLayout(save_reset_layout)

        # API Status
        h_line2 = QFrame(); h_line2.setFrameShape(QFrame.HLine); h_line2.setFrameShadow(QFrame.Sunken)
        bottom_layout.addWidget(h_line2)
        api_label = QLabel("<h3>API Status</h3>"); api_label.setObjectName("h3")
        bottom_layout.addWidget(api_label)

        self.active_workers_label = QLabel("Active Workers: 0")
        bottom_layout.addWidget(self.active_workers_label)

        api_status_layout1 = QGridLayout()
        self.rpm_label = QLabel("RPM: 0 / 0")
        self.rpd_label = QLabel("RPD: 0 / 0")
        api_status_layout1.addWidget(self.rpm_label, 0, 0); api_status_layout1.addWidget(self.rpd_label, 0, 1)
        bottom_layout.addLayout(api_status_layout1)

        api_status_layout2 = QGridLayout()
        self.cost_label = QLabel("Cost (USD): $0.0000")
        self.cost_idr_label = QLabel("Cost (IDR): Rp 0")
        api_status_layout2.addWidget(self.cost_label, 0, 0); api_status_layout2.addWidget(self.cost_idr_label, 0, 1)
        bottom_layout.addLayout(api_status_layout2)

        self.countdown_label = QLabel("Cooldown: 60s")
        self.countdown_label.setStyleSheet("color: #ffc107;"); self.countdown_label.setVisible(False)
        bottom_layout.addWidget(self.countdown_label)

        main_layout.addWidget(bottom_frame)
        return main_layout

    def _create_translate_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 15, 10, 10)

        # OCR Group
        ocr_group = QGroupBox("OCR & Language")
        ocr_layout = QGridLayout(ocr_group)
        
        # Language Input
        ocr_layout.addWidget(QLabel("OCR Language:"), 1, 0)
        self.ocr_lang_combo = QComboBox()
        self.ocr_lang_combo.currentIndexChanged.connect(self.on_ocr_lang_changed)
        ocr_layout.addWidget(self.ocr_lang_combo, 1, 1)
        
        self.ocr_engine_info_label = QLabel("Engine akan dipilih otomatis.")
        self.ocr_engine_info_label.setWordWrap(True)
        self.ocr_engine_info_label.setStyleSheet("font-size: 8pt; color: #aaa;")
        ocr_layout.addWidget(self.ocr_engine_info_label, 2, 0, 1, 2)
        
        self.translate_combo = self._create_combo_box(ocr_layout, "Translate to:", ["Indonesian", "English"], 3, 0, 1, 2, default="Indonesian")
        self.orientation_combo = self._create_combo_box(ocr_layout, "Orientation:", ["Auto-Detect", "Horizontal", "Vertical"], 4, 0, 1, 2)

        layout.addWidget(ocr_group)

        layout.addStretch()
        return tab

    def _create_cleanup_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 15, 10, 10)

        # Auto-Detection Mode Group (NEW)
        detection_mode_group = QGroupBox("Auto-Detection Mode")
        detection_mode_layout = QHBoxLayout(detection_mode_group)
        self.bubble_detect_radio = QRadioButton("Bubble Detection")
        self.text_detect_radio = QRadioButton("Text Detection")
        self.bubble_detect_radio.setChecked(True) # Default
        detection_mode_layout.addWidget(self.bubble_detect_radio)
        detection_mode_layout.addWidget(self.text_detect_radio)
        layout.addWidget(detection_mode_group)

        selection_group = QGroupBox("Manual Selection Tool")
        selection_layout = QGridLayout(selection_group)
        
        selection_modes = [
            "Bubble Finder (Rect)", "Bubble Finder (Oval)",
            "Direct OCR (Rect)", "Direct OCR (Oval)",
            "Pen Tool"
        ]
        
        selection_layout.addWidget(QLabel("Mode:"), 0, 0)
        self.selection_mode_combo = ScrollableComboBox(self)
        self.selection_mode_combo.addItems(selection_modes)
        selection_layout.addWidget(self.selection_mode_combo, 0, 1, 1, 1)

        self.selection_mode_combo.currentTextChanged.connect(self.selection_mode_changed)
        pen_buttons_layout = QHBoxLayout()
        self.confirm_button = QPushButton("Confirm"); self.confirm_button.clicked.connect(self.confirm_pen_selection); self.confirm_button.setVisible(False)
        self.cancel_button = QPushButton("Cancel"); self.cancel_button.clicked.connect(self.cancel_pen_selection); self.cancel_button.setVisible(False)
        pen_buttons_layout.addWidget(self.confirm_button); pen_buttons_layout.addWidget(self.cancel_button)
        selection_layout.addLayout(pen_buttons_layout, 1, 0, 1, 2)

        self.create_bubble_checkbox = QCheckBox("Create white bubble with black outline")
        self.create_bubble_checkbox.setToolTip("When enabled, confirmed selections will render a bubble background behind the text.")
        selection_layout.addWidget(self.create_bubble_checkbox, 2, 0, 1, 2)
        layout.addWidget(selection_group)
        
        # [DIUBAH] Inpainting Group dengan model baru
        inpaint_group = QGroupBox("Inpainting (Text Removal)")
        inpaint_layout = QGridLayout(inpaint_group)
        self.inpaint_checkbox = QCheckBox("Gunakan Inpainting"); self.inpaint_checkbox.setChecked(True)
        inpaint_layout.addWidget(self.inpaint_checkbox, 0, 0, 1, 2)

        inpaint_models = ["OpenCV-NS", "OpenCV-Telea"]
        if self.is_lama_available:
            if os.path.exists(self.dl_models['big_lama']['path']):
                inpaint_models.append("Big-LaMa")
            if os.path.exists(self.dl_models['anime_inpaint']['path']):
                inpaint_models.append("Anime-Inpainting")
        
        self.inpaint_model_combo = self._create_combo_box(inpaint_layout, "Model:", inpaint_models, 1, 0)
        self.inpaint_padding_spinbox = self._create_spin_box(inpaint_layout, "Padding (px):", 1, 25, 5, 2, 0)
        layout.addWidget(inpaint_group)

        dl_detect_group = QGroupBox("Bubble Detector Model (Advanced)")
        dl_layout = QGridLayout(dl_detect_group)
        self.dl_bubble_detector_checkbox = QCheckBox("Gunakan DL Model untuk Bubble"); dl_layout.addWidget(self.dl_bubble_detector_checkbox, 0, 0, 1, 2)
        self.dl_model_provider_combo = self._create_combo_box(dl_layout, "Provider:", ["Kitsumed", "Ogkalu"], 1, 0)
        self.dl_model_file_combo = self._create_combo_box(dl_layout, "Model:", [], 2, 0)
        self.split_bubbles_checkbox = QCheckBox("Otomatis Pisahkan Bubble Panjang")
        dl_layout.addWidget(self.split_bubbles_checkbox, 3, 0, 1, 2)
        self.dl_bubble_detector_checkbox.stateChanged.connect(self.on_dl_detector_state_changed)
        self.dl_model_provider_combo.currentTextChanged.connect(self.on_dl_provider_changed)
        layout.addWidget(dl_detect_group)
        self.on_dl_provider_changed(self.dl_model_provider_combo.currentText())

        layout.addStretch()
        return tab

    def _create_typeset_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 15, 10, 10)

        font_style_group = QGroupBox("Font & Style")
        font_style_layout = QVBoxLayout(font_style_group)
        typeset_options_layout = QHBoxLayout()
        self.font_button = QPushButton("Choose Font"); self.font_button.clicked.connect(self.choose_font)
        self.color_button = QPushButton("Choose Color"); self.color_button.clicked.connect(self.choose_color)
        typeset_options_layout.addWidget(self.font_button); typeset_options_layout.addWidget(self.color_button)
        font_style_layout.addLayout(typeset_options_layout)
        self.vertical_typeset_checkbox = QCheckBox("Typeset Vertically")
        font_style_layout.addWidget(self.vertical_typeset_checkbox)
        layout.addWidget(font_style_group)

        layout.addStretch()
        return tab

    def _create_ai_hardware_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 15, 10, 10)
        layout.setSpacing(15)

        # AI Model Configuration
        ai_group = QGroupBox("AI Models & Translation")
        ai_layout = QGridLayout(ai_group)
        ai_layout.setHorizontalSpacing(12)
        ai_layout.setVerticalSpacing(10)

        self.ai_model_combo = self._create_combo_box(ai_layout, "AI Model:", [], 0, 0, 1, 2)
        self.ai_model_combo.currentTextChanged.connect(self.on_ai_model_changed)

        styles = [
            "Santai (Default)", "Formal (Ke Atasan)", "Akrab (Ke Teman/Pacar)",
            "Vulgar/Dewasa (Adegan Seks)", "Sesuai Konteks Manga"
        ]
        self.style_combo = self._create_combo_box(ai_layout, "Translation Style:", styles, 1, 0, 1, 2)
        layout.addWidget(ai_group)

        # Processing & Safety Modes
        mode_group = QGroupBox("Processing Modes")
        mode_layout = QVBoxLayout(mode_group)
        mode_layout.setSpacing(8)
        self.enhanced_pipeline_checkbox = QCheckBox("Enhanced Pipeline (JP Only, More API)")
        self.enhanced_pipeline_checkbox.stateChanged.connect(self.on_pipeline_mode_changed)
        mode_layout.addWidget(self.enhanced_pipeline_checkbox)

        self.ai_only_translate_checkbox = QCheckBox("AI-Only Translate")
        self.deepl_only_checkbox = QCheckBox("DeepL-Only Translate")
        self.ai_only_translate_checkbox.stateChanged.connect(self.on_translation_mode_changed)
        self.deepl_only_checkbox.stateChanged.connect(self.on_translation_mode_changed)
        mode_layout.addWidget(self.ai_only_translate_checkbox)
        mode_layout.addWidget(self.deepl_only_checkbox)

        self.safe_mode_checkbox = QCheckBox("Enable Safe Mode (Filter Konten Dewasa)")
        mode_layout.addWidget(self.safe_mode_checkbox)

        self.batch_mode_checkbox = QCheckBox("Enable Batch Processing")
        self.batch_mode_checkbox.stateChanged.connect(self.on_batch_mode_changed)
        mode_layout.addWidget(self.batch_mode_checkbox)
        layout.addWidget(mode_group)

        # Hardware Controls
        hardware_group = QGroupBox("Hardware & Performance")
        hardware_layout = QGridLayout(hardware_group)
        hardware_layout.setHorizontalSpacing(12)
        hardware_layout.setVerticalSpacing(10)

        self.use_gpu_checkbox = QCheckBox("Enable GPU Acceleration")
        self.use_gpu_checkbox.setChecked(self.is_gpu_available)
        self.use_gpu_checkbox.setEnabled(self.is_gpu_available)
        if not self.is_gpu_available:
            self.use_gpu_checkbox.setToolTip("Tidak ada GPU NVIDIA yang terdeteksi atau PyTorch tidak terinstal.")
        self.use_gpu_checkbox.stateChanged.connect(self.update_gpu_status_label)
        hardware_layout.addWidget(self.use_gpu_checkbox, 0, 0, 1, 2)

        self.gpu_status_label = QLabel("GPU Detected" if self.is_gpu_available else "GPU Not Detected")
        self.gpu_status_label.setObjectName("gpu-status")
        hardware_layout.addWidget(self.gpu_status_label, 0, 2, 1, 1, alignment=Qt.AlignRight)

        hardware_layout.addWidget(QLabel("Max Workers:"), 1, 0)
        self.max_workers_spinbox = QSpinBox()
        self.max_workers_spinbox.setRange(1, 50)
        self.max_workers_spinbox.setValue(self.MAX_WORKERS)
        self.max_workers_spinbox.valueChanged.connect(self.on_max_workers_changed)
        hardware_layout.addWidget(self.max_workers_spinbox, 1, 1)

        hardware_layout.addWidget(QLabel("Spawn Threshold:"), 2, 0)
        self.spawn_threshold_spinbox = QSpinBox()
        self.spawn_threshold_spinbox.setRange(1, 10)
        self.spawn_threshold_spinbox.setValue(self.WORKER_SPAWN_THRESHOLD)
        self.spawn_threshold_spinbox.valueChanged.connect(self.on_spawn_threshold_changed)
        hardware_layout.addWidget(self.spawn_threshold_spinbox, 2, 1)

        layout.addWidget(hardware_group)

        self.update_gpu_status_label()
        layout.addStretch()
        return tab

    def _create_combo_box(self, parent_layout, label_text, items, row, col, row_span=1, col_span=2, default=None):
        label = QLabel(label_text); parent_layout.addWidget(label, row, col)
        combo = QComboBox(); combo.addItems(items)
        if default: combo.setCurrentText(default)
        parent_layout.addWidget(combo, row, col + 1, row_span, col_span -1)
        return combo

    def _create_spin_box(self, parent_layout, label_text, min_val, max_val, default_val, row, col):
        label = QLabel(label_text); parent_layout.addWidget(label, row, col)
        spin_box = QSpinBox(); spin_box.setRange(min_val, max_val); spin_box.setValue(default_val)
        parent_layout.addWidget(spin_box, row, col + 1)
        return spin_box

    def setup_styles(self):
        self.setStyleSheet(self.DARK_THEME_STYLESHEET)

    def setup_shortcuts(self):
        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self); self.undo_shortcut.activated.connect(self.undo_last_action)
        self.redo_shortcut = QShortcut(QKeySequence("Ctrl+Y"), self); self.redo_shortcut.activated.connect(self.redo_last_action)

        # [BARU] Shortcut untuk mengganti mode seleksi
        QShortcut(QKeySequence("1"), self).activated.connect(lambda: self.set_selection_mode_by_index(0))
        QShortcut(QKeySequence("2"), self).activated.connect(lambda: self.set_selection_mode_by_index(1))
        QShortcut(QKeySequence("3"), self).activated.connect(lambda: self.set_selection_mode_by_index(2))
        QShortcut(QKeySequence("4"), self).activated.connect(lambda: self.set_selection_mode_by_index(3))
        QShortcut(QKeySequence("5"), self).activated.connect(lambda: self.set_selection_mode_by_index(4))

    # [BARU] Mengubah mode seleksi via shortcut keyboard
    def set_selection_mode_by_index(self, index):
        """Mengatur mode seleksi berdasarkan indeks dari shortcut."""
        if 0 <= index < self.selection_mode_combo.count():
            self.selection_mode_combo.setCurrentIndex(index)
            mode_text = self.selection_mode_combo.currentText()
            self.statusBar().showMessage(f"Mode Seleksi: {mode_text}", 2000)

    def on_max_workers_changed(self, value):
        self.MAX_WORKERS = value
        self.statusBar().showMessage(f"Max workers set to {value}", 2000)

    def on_spawn_threshold_changed(self, value):
        self.WORKER_SPAWN_THRESHOLD = value
        self.statusBar().showMessage(f"Worker spawn threshold set to {value}", 2000)

    def populate_ocr_languages(self):
        """Mengisi daftar bahasa dari semua engine yang tersedia."""
        self.OCR_LANGS.clear()

        # Manga-OCR (Hardcoded)
        if MangaOcr:
            self.OCR_LANGS['Japanese (Manga-OCR)'] = {'code': 'ja', 'engine': 'Manga-OCR'}

        # DocTR
        if self.is_doctr_available:
            doctr_langs = {'English': 'en', 'French': 'fr', 'German': 'de', 'Dutch': 'nl', 'Spanish': 'es', 'Italian': 'it'}
            for name, code in doctr_langs.items():
                self.OCR_LANGS[f'{name} (DocTR)'] = {'code': code, 'engine': 'DocTR'}
        
        # RapidOCR
        if self.is_rapidocr_available:
            rapid_langs = {'Chinese Simplified': 'ch_sim', 'Russian': 'ru'}
            for name, code in rapid_langs.items():
                self.OCR_LANGS[f'{name} (RapidOCR)'] = {'code': code, 'engine': 'RapidOCR'}

        # PaddleOCR
        if self.is_paddle_available:
            paddle_langs = {'English': 'en', 'Chinese Simplified': 'ch', 'German': 'german', 'French': 'french', 'Japanese': 'japan', 'Korean': 'korean', 'Russian': 'ru'}
            for name, code in paddle_langs.items():
                key = f'{name} (PaddleOCR)'
                if key not in self.OCR_LANGS: # Hindari duplikat jika sudah ada dari engine lain
                    self.OCR_LANGS[key] = {'code': code, 'engine': 'PaddleOCR'}
        
        # EasyOCR
        easyocr_langs = {'Afrikaans': 'af', 'Arabic': 'ar', 'Azerbaijani': 'az', 'Belarusian': 'be', 'Bulgarian': 'bg', 'Bengali': 'bn', 'Bosnian': 'bs', 'Czech': 'cs', 'Chinese (Simplified)': 'ch_sim', 'Chinese (Traditional)': 'ch_tra', 'German': 'de', 'English': 'en', 'Spanish': 'es', 'Estonian': 'et', 'French': 'fr', 'Hindi': 'hi', 'Croatian': 'hr', 'Hungarian': 'hu', 'Indonesian': 'id', 'Italian': 'it', 'Japanese': 'ja', 'Korean': 'ko', 'Lithuanian': 'lt', 'Latvian': 'lv', 'Malay': 'ms', 'Dutch': 'nl', 'Polish': 'pl', 'Portuguese': 'pt', 'Romanian': 'ro', 'Russian': 'ru', 'Slovak': 'sk', 'Slovenian': 'sl', 'Albanian': 'sq', 'Swedish': 'sv', 'Thai': 'th', 'Turkish': 'tr', 'Ukrainian': 'uk', 'Urdu': 'ur', 'Uzbek': 'uz', 'Vietnamese': 'vi'}
        for name, code in easyocr_langs.items():
            key = f'{name} (EasyOCR)'
            if key not in self.OCR_LANGS:
                self.OCR_LANGS[key] = {'code': code, 'engine': 'EasyOCR'}

        # Tesseract
        try:
            langs = [lang for lang in pytesseract.get_languages(config='--oem 1') if len(lang) == 3 and lang != 'osd']
            tess_langs = {lang.capitalize(): lang for lang in sorted(langs)}
            for name, code in tess_langs.items():
                key = f'{name} (Tesseract)'
                if key not in self.OCR_LANGS:
                    self.OCR_LANGS[key] = {'code': code, 'engine': 'Tesseract'}
        except Exception as e:
            print(f"Could not get Tesseract languages: {e}")
            tess_fallback = {'English (Tesseract)': {'code': 'eng', 'engine': 'Tesseract'}, 'Japanese (Tesseract)': {'code': 'jpn', 'engine': 'Tesseract'}}
            for k,v in tess_fallback.items():
                if k not in self.OCR_LANGS: self.OCR_LANGS[k] = v
        
        # Populate ComboBox
        self.ocr_lang_combo.blockSignals(True)
        self.ocr_lang_combo.clear()
        for display_name, data in sorted(self.OCR_LANGS.items()):
            self.ocr_lang_combo.addItem(display_name, data)
        self.ocr_lang_combo.blockSignals(False)

        # Set default to Japanese
        jp_index = self.ocr_lang_combo.findText("Japanese (Manga-OCR)")
        if jp_index != -1:
            self.ocr_lang_combo.setCurrentIndex(jp_index)
        
        self.on_ocr_lang_changed(self.ocr_lang_combo.currentIndex())

    def initialize_core_engines(self):
        """Initializes engines that don't depend on user input, like Manga-OCR and language list."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage("Initializing core engines...")

        self.populate_ocr_languages()
        self.populate_ai_models() # [BARU]

        # Manga-OCR (selalu diinisialisasi jika ada)
        if MangaOcr:
            try:
                self.manga_ocr_reader = MangaOcr()
                self.statusBar().showMessage("Manga-OCR initialized.", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Manga-OCR Error", f"Could not initialize Manga-OCR.\nError: {e}")
                self.manga_ocr_reader = None
        
        # Engine lain diinisialisasi on-demand
        
        QApplication.restoreOverrideCursor()
        self.statusBar().showMessage("Ready", 3000)

    def populate_ai_models(self):
        """Mengisi daftar model AI yang tersedia dari semua provider."""
        self.ai_model_combo.blockSignals(True)
        self.ai_model_combo.clear()
        
        for provider, models in self.AI_PROVIDERS.items():
            for model_name, model_info in models.items():
                display_text = f"[{provider}] {model_info['display']}"
                self.ai_model_combo.addItem(display_text)
        
        self.ai_model_combo.blockSignals(False)
        
        # Set default model
        default_index = self.ai_model_combo.findText("[Gemini] Gemini 2.5 Flash Lite (Utama - Cepat & Murah)")
        if default_index != -1:
            self.ai_model_combo.setCurrentIndex(default_index)

    def initialize_ocr_engine(self, lang_data):
        """Inisialisasi engine OCR yang dibutuhkan secara on-demand."""
        engine = lang_data['engine']
        lang_code = lang_data['code']
        
        # Gunakan setting dari checkbox
        use_gpu = self.use_gpu_checkbox.isChecked() and self.is_gpu_available

        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage(f"Initializing {engine} for '{lang_code}'...")
        QApplication.processEvents()
        
        try:
            if engine == 'EasyOCR' and (self.easyocr_reader is None or self.easyocr_lang != lang_code):
                lang_list = sorted(list(set(['en', lang_code])))
                self.easyocr_reader = easyocr.Reader(lang_list, gpu=use_gpu)
                self.easyocr_lang = lang_code
            
            # PERBAIKAN: Inisialisasi PaddleOCR dengan parameter yang benar
            elif engine == 'PaddleOCR' and (self.paddle_ocr_reader is None or self.paddle_lang != lang_code):
                try:
                    from paddleocr import PaddleOCR
                    
                    # Parameter untuk PaddleOCR versi terbaru
                    # Gunakan device parameter yang benar
                    device = 'gpu' if use_gpu and self.is_gpu_available else 'cpu'
                    
                    self.paddle_ocr_reader = PaddleOCR(
                        # Parameter dasar
                        lang=lang_code, 
                        device=device,  # Gunakan 'device' bukan 'use_gpu'
                        
                        # Parameter opsional
                        show_log=False,
                        use_textline_orientation=True,
                    )
                    self.paddle_lang = lang_code
                    print(f"PaddleOCR initialized for {lang_code} on {device.upper()}")
                    
                except Exception as e:
                    print(f"Error initializing PaddleOCR: {e}")
                    # Coba konfigurasi minimal
                    try:
                        self.paddle_ocr_reader = PaddleOCR(lang=lang_code)
                        print(f"PaddleOCR initialized with minimal config for {lang_code}")
                    except Exception as e2:
                        print(f"Minimal config also failed: {e2}")
                        self.paddle_ocr_reader = None

            elif engine == 'DocTR' and self.doctr_predictor is None:
                from doctr.models import ocr_predictor
                device = torch.device("cuda" if use_gpu else "cpu")
                self.doctr_predictor = ocr_predictor(pretrained=True).to(device)

            elif engine == 'RapidOCR' and (self.rapid_ocr_reader is None or self.rapid_lang != lang_code):
                self.rapid_ocr_reader = RapidOCR()
                self.rapid_lang = lang_code

            self.statusBar().showMessage(f"{engine} initialized.", 3000)
        except Exception as e:
            QMessageBox.critical(self, f"{engine} Error", f"Could not initialize {engine}.\nError: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    # [BARU] Inisialisasi on-demand untuk model inpainting
    def initialize_inpaint_engine(self):
        """Menginisialisasi engine inpainting LaMa yang dipilih."""
        settings = self.get_current_settings()
        model_key = settings['inpaint_model_key']
        
        # Jika tidak ada model key yang dipilih, return
        if not model_key:
            print("No inpainting model selected in settings")
            return

        if model_key == self.current_inpaint_model_key:
            return

        if not self.is_lama_available:
            print("Lama Cleaner not available")
            return

        model_info = self.dl_models.get(model_key)
        if not model_info or not os.path.exists(model_info['path']):
            print(f"Model file not found: {model_info['path'] if model_info else 'None'}")
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage(f"Initializing inpainting model: {model_key}...")
        
        try:
            # Tentukan device (CPU/GPU)
            use_gpu = self.use_gpu_checkbox.isChecked() and self.is_gpu_available
            device = "cuda" if use_gpu else "cpu"
            
            # Inisialisasi model manager
            from lama_cleaner.model_manager import ModelManager
            model_manager = ModelManager()
            
            # Tentukan jenis model
            model_type = "lama"  # Kedua model menggunakan arsitektur LaMa
            
            # Load model (try/catch karena API lama/baru bisa berbeda)
            loaded_model = None
            try:
                loaded_model = model_manager.init_model(device, model_info['path'], model_type=model_type)
            except Exception:
                # fallback jika api berbeda
                try:
                    loaded_model = model_manager.load_model(model_info['path'], device=device)
                except Exception as e:
                    print(f"Could not load model via ModelManager: {e}")
                    loaded_model = None

            if loaded_model is None:
                raise RuntimeError("Failed to initialize inpainting model instance.")

            # Bungkus model menjadi callable yang selalu mengembalikan PIL.Image
            self.inpaint_model = lambda pil_img, pil_mask: self._run_lama_inpaint(loaded_model, pil_img, pil_mask)
            self.current_inpaint_model_key = model_key
            self.statusBar().showMessage(f"Inpainting model {model_key} initialized on {device.upper()}.", 3000)
            
        except Exception as e:
            print(f"Error initializing inpainting model {model_key}: {e}")
            self.inpaint_model = None
            self.current_inpaint_model_key = None
            
        finally:
            QApplication.restoreOverrideCursor()

    def _run_lama_inpaint(self, model, pil_image, pil_mask):
        """
        Helper untuk memanggil model lama/baru dari lama_cleaner dan
        mengembalikan hasil sebagai numpy array (RGB).
        Menangani beberapa varian API yang mungkin tersedia.
        """
        try:
            # Pastikan mask ukuran sama dengan image
            if pil_mask.size != pil_image.size:
                pil_mask = pil_mask.resize(pil_image.size)

            # Coba beberapa cara pemanggilan model yang umum
            result = None
            try:
                # model bisa callable
                result = model(pil_image, pil_mask)
            except Exception:
                pass

            if result is None and hasattr(model, "process"):
                try:
                    result = model.process(pil_image, pil_mask)
                except Exception:
                    pass

            if result is None and hasattr(model, "inpaint"):
                try:
                    result = model.inpaint(pil_image, pil_mask)
                except Exception:
                    pass

            if result is None and hasattr(model, "run"):
                try:
                    # some apis expect keyword args
                    try:
                        result = model.run(image=pil_image, mask=pil_mask)
                    except TypeError:
                        result = model.run(pil_image, pil_mask)
                except Exception:
                    pass

            if result is None:
                raise RuntimeError("Inpainting model did not return a result (unsupported API).")

            # Normalisasi hasil menjadi PIL.Image atau numpy array (RGB)
            if isinstance(result, tuple) or isinstance(result, list):
                # kadang model mengembalikan (image, ...)
                candidate = result[0]
            else:
                candidate = result

            if hasattr(candidate, "convert") and hasattr(candidate, "size"):
                # PIL Image
                pil_out = candidate.convert("RGB")
                return np.array(pil_out)[:, :, ::-1]  # convert RGB->BGR for OpenCV path if necessary later
            elif isinstance(candidate, np.ndarray):
                # Pastikan format RGB
                arr = candidate
                if arr.ndim == 3 and arr.shape[2] == 3:
                    # as-is, convert to RGB ordering expected later
                    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if arr.dtype == np.uint8 else arr
                return arr
            elif isinstance(candidate, dict):
                # coba beberapa key umum
                for k in ("result", "image", "output", "pred"):
                    if k in candidate:
                        v = candidate[k]
                        if hasattr(v, "convert"):
                            return np.array(v.convert("RGB"))[:, :, ::-1]
                        if isinstance(v, np.ndarray):
                            return v
                raise RuntimeError("Unsupported dict result from inpaint model.")
            else:
                raise RuntimeError("Unsupported result type from inpaint model.")

        except Exception as e:
            print(f"Error running inpaint model: {e}")
            return None

    def add_api_cost(self, input_chars, output_chars, provider, model_name):
        model_info = self.AI_PROVIDERS.get(provider, {}).get(model_name)
        if not model_info: return

        pricing = model_info['pricing']
        cost = (input_chars * pricing['input']) + (output_chars * pricing['output'])
        self.total_cost += cost
        self.update_cost_display()
        self.save_usage_data()

    def update_cost_display(self):
        self.cost_label.setText(f"Cost (USD): ${self.total_cost:.4f}")
        cost_idr = self.total_cost * self.usd_to_idr_rate
        self.cost_idr_label.setText(f"Cost (IDR): Rp {cost_idr:,.0f}")

    def fetch_exchange_rate(self):
        if self.exchange_rate_thread and self.exchange_rate_thread.isRunning():
            return

        def fetch_and_finish():
            try:
                url = "https://api.exchangerate-api.com/v4/latest/USD"
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()
                rate = data.get('rates', {}).get('IDR')
                if rate:
                    self.usd_to_idr_rate = float(rate)
                    print(f"Successfully fetched USD to IDR rate: {self.usd_to_idr_rate}")
            except requests.exceptions.RequestException as e:
                print(f"Failed to fetch exchange rate: {e}. Using default.")
            finally:
                QTimer.singleShot(0, self.update_cost_display)
                if self.exchange_rate_thread: self.exchange_rate_thread.quit()

        self.exchange_rate_thread = QThread()
        self.exchange_rate_worker = QObject()
        self.exchange_rate_worker.moveToThread(self.exchange_rate_thread)
        self.exchange_rate_thread.started.connect(fetch_and_finish)
        self.exchange_rate_thread.finished.connect(self.exchange_rate_thread.deleteLater)
        self.exchange_rate_thread.finished.connect(self.exchange_rate_worker.deleteLater)
        self.exchange_rate_thread.start()

    def on_pipeline_mode_changed(self, state):
        is_enhanced = (state == Qt.Checked)
        if is_enhanced and self.ai_only_translate_checkbox.isChecked():
            self.ai_only_translate_checkbox.setChecked(False)

        self.ocr_lang_combo.setEnabled(not is_enhanced)
        if is_enhanced:
            self.ocr_lang_combo.setToolTip("Disabled in Enhanced Pipeline mode (uses Manga-OCR + Tesseract).")
        else:
            self.on_translation_mode_changed()

    def on_translation_mode_changed(self):
        # [BARU] Mengelola checkbox yang saling eksklusif
        sender = self.sender()
        if sender == self.ai_only_translate_checkbox and self.ai_only_translate_checkbox.isChecked():
            self.deepl_only_checkbox.setChecked(False)
        elif sender == self.deepl_only_checkbox and self.deepl_only_checkbox.isChecked():
            self.ai_only_translate_checkbox.setChecked(False)

        is_ai_only = self.ai_only_translate_checkbox.isChecked()
        is_deepl_only = self.deepl_only_checkbox.isChecked()
        
        # Nonaktifkan opsi yang tidak relevan
        self.translate_combo.setEnabled(not is_ai_only and not is_deepl_only)
        self.style_combo.setEnabled(is_ai_only)
        self.ai_model_combo.setEnabled(is_ai_only or self.enhanced_pipeline_checkbox.isChecked())


    def on_ocr_lang_changed(self, index):
        """Dipanggil saat pengguna memilih bahasa baru dari dropdown."""
        if index < 0: return
        lang_data = self.ocr_lang_combo.itemData(index)
        if lang_data:
            self.ocr_engine_info_label.setText(f"Engine: {lang_data['engine']}")
            self.initialize_ocr_engine(lang_data)

    def clean_and_join_text(self, raw_text):
        return ' '.join(raw_text.split())

    def _build_prompt_enhancements(self, settings):
        """Membangun string tambahan untuk prompt AI berdasarkan pengaturan."""
        enhancements = ""

        style_map = {
            "Santai (Default)": (
                "Your tone MUST be casual, relaxed, and colloquial, like everyday conversation between normal people. "
                "Use natural phrasing, contractions, and avoid stiff or textbook-like wording. "
                "Keep it light and friendly, suitable for general manga dialogue."
            ),
            "Formal (Ke Atasan)": (
                "Your tone MUST be formal, polite, and respectful, as if addressing a superior, elder, or teacher. "
                "Avoid slang or overly casual phrasing. "
                "Honorifics, polite endings, and respectful language should be preserved where appropriate."
            ),
            "Akrab (Ke Teman/Pacar)": (
                "Your tone MUST be intimate, playful, and very casual, suitable for close friends or romantic partners. "
                "Use warm, affectionate, or teasing expressions where natural. "
                "Convey emotional closeness, and allow a bit of informality, slang, or cuteness if it fits the context."
            ),
            "Vulgar/Dewasa (Adegan Seks)": (
                "Your tone MUST be explicit, vulgar, and direct, suitable for an adult or sexual scene. "
                "Do not soften or censor unless absolutely required by the target language. "
                "Use raw, straightforward words for sexual acts or body parts, while keeping the flow natural. "
                "The style should feel raw and intense, not clinical or overly polite."
            ),
            "Sesuai Konteks Manga": (
                "Analyze the text carefully and adjust your tone to best match the likely context of the manga scene. "
                "- For comedy: be witty, light, and playful. "
                "- For drama: be serious, emotional, and impactful. "
                "- For action: be sharp, concise, and energetic. "
                "- For horror: be tense, eerie, and unsettling. "
                "Always aim for immersion: the translation should feel like it belongs naturally in the scene."
            )
        }

        style = settings.get('translation_style', 'Santai (Default)')
        style_instruction = style_map.get(style, style_map["Santai (Default)"])
        enhancements += f"\n- Translation Style: {style_instruction}"

        return enhancements
    
    # [DIUBAH] Fungsi abstrak untuk memanggil AI
    def translate_with_ai(self, text_to_translate, target_lang, provider, model_name, settings, is_enhanced=False, ocr_results=None):
        if provider == 'Gemini':
            return self.translate_with_gemini(text_to_translate, target_lang, model_name, settings, is_enhanced, ocr_results)
        elif provider == 'OpenAI':
            return self.translate_with_openai(text_to_translate, target_lang, model_name, settings, is_enhanced, ocr_results)
        return f"[ERROR: Unknown AI provider '{provider}']"
    
    # [DIUBAH] Fungsi terjemahan Gemini yang dimodifikasi
    def translate_with_gemini(
        self,
        text_to_translate,
        target_lang,
        model_name,
        settings,
        is_enhanced=False,
        ocr_results=None,
        selected_style="Santai (Default)"
    ):
        if not text_to_translate.strip():
            return ""
        if not GEMINI_API_KEY or "your_gemini_key_here" in GEMINI_API_KEY:
            return "[GEMINI API KEY NOT CONFIGURED]"
        try:
            model = genai.GenerativeModel(model_name)
            prompt_enhancements = self._build_prompt_enhancements(settings)

            base_rule = (
                f"Your response must ONLY contain the final translation in {target_lang}, as RAW plain text.\n"
                f"- Do NOT wrap output in quotes, brackets, parentheses, or code fences.\n"
                f"- Do NOT include explanations, notes, the original text, markdown, or labels.\n"
                f"- Preserve line breaks if the input has multiple lines.\n"
            )

            if is_enhanced and ocr_results:
                prompt = f"""
    You are an expert manga translator.

    1. Automatically detect the language of the OCR text.
    2. If the text is Japanese:
    - Merge the following two OCR results into the most accurate Japanese text.
    - Silently correct any OCR mistakes.
    - Translate into natural, colloquial {target_lang}.
    3. If the text is already {target_lang}, return it exactly as-is.
    4. If the text is another language (not Japanese and not {target_lang}), translate it into {target_lang}.
    {prompt_enhancements}
    {base_rule}

    OCR Results:
    - Manga-OCR: {ocr_results.get('manga_ocr', '')}
    - Tesseract: {ocr_results.get('tesseract', '')}
    """
            else:
                prompt = f"""
    You are an expert manga translator.

    1. Automatically detect the language of the input text.
    2. If the text is Japanese:
    - Silently correct OCR mistakes.
    - Translate into natural, colloquial {target_lang}.
    3. If the text is already {target_lang}, return it exactly as-is.
    4. If the text is another language (not Japanese and not {target_lang}), translate it into {target_lang}.
    {prompt_enhancements}
    {base_rule}

    Raw OCR Text:
    {text_to_translate}
    """

            response = model.generate_content(prompt)
            if response.parts:
                self.add_api_cost(len(prompt), len(response.text), 'Gemini', model_name)
                return response.text.strip()
            return "[GEMINI FAILED]"
        except Exception as e:
            print(f"Error calling Gemini API for full translation: {e}")
            return "[GEMINI ERROR]"


    # [BARU] Fungsi terjemahan OpenAI
    def translate_with_openai(
        self,
        text_to_translate: str,
        target_lang: str,
        model_name: str,
        settings: dict,
        is_enhanced: bool = False,
        ocr_results: dict | None = None,
    ):
        """
        Terjemahkan teks manga via OpenAI Chat Completions.
        - Prompt diperketat agar output RAW tanpa tanda kutip/markdown.
        - Auto-sanitizer menghapus pembungkus seperti "…", “…”, 「…」, 『…』, «…», serta code fence ```…```.
        - Tidak mengirim 'temperature' untuk model yang tak mendukung (gpt-5-mini/nano).
        - Mode 'enhanced' menggabungkan hasil OCR (manga-ocr & tesseract).
        """

        # ---------- Helper: sanitizer output ----------
        def _sanitize_output(s: str) -> str:
            if not s:
                return s

            s = s.strip()

            # 1) Hilangkan code fence penuh (```...```), opsional dengan bahasa
            import re
            fence_match = re.fullmatch(r"```[a-zA-Z0-9_-]*\n([\s\S]*?)\n```", s)
            if fence_match:
                s = fence_match.group(1).strip()

            # 2) Hilangkan satu pasang pembungkus penuh jika SELURUH teks terbungkus
            #    (konservatif: hanya kalau jumlah tanda kutipnya tepat 2, agar tidak merusak kutip internal)
            pairs = [
                ("\"", "\""), ("“", "”"), ("‘", "’"),
                ("'", "'"),
                ("「", "」"), ("『", "』"),
                ("«", "»"),
            ]

            def _strip_pair(text, left, right):
                if text.startswith(left) and text.endswith(right):
                    # hitung kemunculan agar tidak mengupas jika ada kutip internal yang sama
                    if left == right:
                        if text.count(left) == 2:
                            return text[len(left):-len(right)].strip()
                    else:
                        # untuk pasangan berbeda (mis. “ ”, 「 」), cek hanya sekali di ujung
                        inner = text[len(left):-len(right)]
                        # jangan kupas kalau di dalam masih ada pasangan lengkap yang sama secara seimbang
                        return inner.strip()
                return None

            # coba kupas untuk semua pair di atas
            for l, r in pairs:
                # khusus pasangan berbeda, cukup jika terbungkus; tidak perlu hitung jumlah persis
                if s.startswith(l) and s.endswith(r):
                    inner = s[len(l):-len(r)].strip()
                    # Pastikan inner tidak kosong setelah kupas
                    if inner:
                        s = inner
                    break

            # 3) Hilangkan pembungkus yang kadang muncul seperti ( … ) atau [ … ] bila jelas bungkusan penuh
            bracket_pairs = [("(", ")"), ("[", "]"), ("（", "）")]
            for l, r in bracket_pairs:
                if s.startswith(l) and s.endswith(r):
                    inner = s[len(l):-len(r)].strip()
                    # Kupas hanya jika tidak ada baris baru dan tidak ada pasangan penutup lain di tengah
                    if inner and ("\n" not in inner):
                        s = inner
                    break

            # 4) Bersihkan backticks tunggal penuh
            if s.startswith("`") and s.endswith("`") and s.count("`") == 2:
                s = s[1:-1].strip()

            return s

        # ---------- Guard ----------
        if not text_to_translate or not text_to_translate.strip():
            return ""
        if not getattr(self, "is_openai_available", False):
            return "[OPENAI NOT CONFIGURED]"

        try:
            # --- Build prompts (lebih ketat soal output) ---
            # Aturan utama:
            # - Output HARUS raw text dalam {target_lang}
            # - Jangan bungkus dengan tanda kutip, kode, atau markdown apa pun
            # - Jangan sertakan teks asli, penjelasan, catatan, atau label
            prompt_enhancements = self._build_prompt_enhancements(settings) if hasattr(self, "_build_prompt_enhancements") else ""
            target_lang = (target_lang or "Indonesian").strip()

            base_rule = (
                f"Output ONLY the final translation in {target_lang}, as RAW plain text. "
                f"No quotes, no code fences, no markdown, no labels, no explanations, "
                f"no original text, no notes, no extra commentary. "
                f"Preserve line breaks if the OCR text is multi-line dialogue."
            )

            style_rules = (
                "Translation style rules:\n"
                "- Dialogue should sound natural and colloquial, like authentic manga speech.\n"
                "- Adapt tone: casual for friends, polite for formal situations, exaggerated for comedic or dramatic scenes.\n"
                "- Keep character-specific quirks (stuttering, slang, verbal tics) if detectable.\n"
                "- Keep consistency of names, nicknames, and terms across translations.\n"
                "- If OCR contains sound effects (e.g., 'ドキドキ', 'ガーン'), translate to natural equivalents or expressive onomatopoeia in target_lang.\n"
                "- Do NOT add translator notes.\n"
            )

            if is_enhanced and ocr_results:
                system_prompt = (
                    f"You are an expert manga translator.\n"
                    f"1. Automatically detect the language of the text.\n"
                    f"2. If Japanese → merge and correct the following OCR outputs, then translate into natural {target_lang}.\n"
                    f"3. If already in {target_lang} → return as-is with no changes.\n"
                    f"4. If in another language → translate into {target_lang}.\n"
                    f"{style_rules} {prompt_enhancements} {base_rule}"
                )
                user_prompt = (
                    "OCR Results:\n"
                    f"1. Manga-OCR: {ocr_results.get('manga_ocr', '')}\n"
                    f"2. Tesseract: {ocr_results.get('tesseract', '')}"
                )
            else:
                system_prompt = (
                    f"You are an expert manga translator.\n"
                    f"1. Automatically detect the language of the input text.\n"
                    f"2. If Japanese → silently correct OCR mistakes, then translate into natural {target_lang}.\n"
                    f"3. If already in {target_lang} → return as-is with no changes.\n"
                    f"4. If in another language → translate into {target_lang}.\n"
                    f"{style_rules} {prompt_enhancements} {base_rule}"
                )
                user_prompt = f"Raw OCR Text:\n{text_to_translate}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ]

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ]

            # --- Temperature handling ---
            model_lower = (model_name or "").lower()
            supports_temperature = not (
                model_lower.startswith("gpt-5-mini") or model_lower.startswith("gpt-5-nano")
            )
            desired_temp = settings.get("temperature", 0.5) if isinstance(settings, dict) else 0.5

            req_kwargs = {
                "model": model_name,
                "messages": messages,
            }
            if supports_temperature and desired_temp is not None:
                req_kwargs["temperature"] = float(desired_temp)

            # --- Call API ---
            client = getattr(self, "openai_client", None)
            if client is None:
                # fallback ke global jika ada
                client = openai_client  # pastikan objek ini ada di konteksmu

            response = client.chat.completions.create(**req_kwargs)
            output_text = (response.choices[0].message.content or "").strip()

            # --- Sanitizer utk jaga-jaga kalau model masih membungkus output ---
            output_text = _sanitize_output(output_text)

            # --- Hitung "biaya" sederhana ---
            prompt_chars = len(system_prompt) + len(user_prompt)
            output_chars = len(output_text)
            if hasattr(self, "add_api_cost"):
                self.add_api_cost(prompt_chars, output_chars, "OpenAI", model_name)

            return output_text or ""

        except Exception as e:
            err_msg = str(e)
            if "Unsupported value" in err_msg and "temperature" in err_msg:
                return "[OPENAI ERROR] Model ini tidak mendukung parameter temperature. Abaikan 'temperature' untuk model ini."
            if "invalid_request_error" in err_msg or "Error code: 400" in err_msg:
                return "[OPENAI ERROR] Permintaan tidak valid. Periksa parameter (model, messages, dsb)."
            if "rate_limit" in err_msg or "Rate limit" in err_msg:
                return "[OPENAI ERROR] Kena rate limit. Coba lagi beberapa saat."
            print(f"Error calling OpenAI API: {e}")
            return "[OPENAI ERROR]"

    def apply_safe_mode(self, text: str) -> str:
        """Applies the Safe Mode filter to the translated text."""
        if not text:
            return text
        # Use re.sub with IGNORECASE flag for case-insensitive replacement
        text = re.sub(r'vagina', 'meong', text, flags=re.IGNORECASE)
        text = re.sub(r'penis', 'burung', text, flags=re.IGNORECASE)
        # Add other words here if needed
        return text

    def preprocess_for_ocr(self, cv_image, orientation_hint="Auto-Detect"):
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if h == 0 or w == 0: return cv_image, 0
        angle = 0
        if orientation_hint == "Auto-Detect":
            try:
                coords = cv2.findNonZero(cv2.bitwise_not(gray))
                if coords is not None:
                    rect = cv2.minAreaRect(coords)
                    angle = rect[-1]
                    if w < h and angle < -45: angle = -(90 + angle)
                    elif w > h and angle > 45: angle = 90 - angle
                    else: angle = -angle
            except cv2.error: angle = 0
        elif orientation_hint == "Vertical":
            if w > h: angle = 90
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        thresh = cv2.adaptiveThreshold(rotated, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), angle
    
    # Safe image opener with several fallbacks for truncated/corrupt JPEGs
    def safe_open_image(self, file_path):
        """
        Try to open an image robustly:
         1) normal Image.open().convert('RGB')
         2) read raw bytes and open via BytesIO (calls load())
         3) incremental parse via ImageFile.Parser
        Raises original exception if all fail.
        """
        try:
            return Image.open(file_path).convert('RGB')
        except Exception as e1:
            # Try BytesIO
            try:
                with open(file_path, 'rb') as f:
                    data = f.read()
                im = Image.open(io.BytesIO(data))
                im.load()  # force load (may raise)
                return im.convert('RGB')
            except Exception:
                pass

            # Try incremental parser (useful for truncated files)
            try:
                parser = ImageFile.Parser()
                with open(file_path, 'rb') as f:
                    while True:
                        chunk = f.read(16384)
                        if not chunk:
                            break
                        parser.feed(chunk)
                im = parser.close()
                return im.convert('RGB')
            except Exception:
                pass

            # If all fallbacks fail, re-raise first error
            raise e1

    def start_worker(self):
        worker_id = self.next_worker_id
        self.next_worker_id += 1

        thread = QThread()
        worker = QueueProcessorWorker(self, worker_id)

        worker.moveToThread(thread)

        worker.signals.job_complete.connect(self.on_queue_job_complete)
        worker.signals.queue_status.connect(self.update_queue_status)
        worker.signals.error.connect(self.on_worker_error)
        worker.signals.worker_finished.connect(self.on_worker_finished)
        worker.signals.status_update.connect(self.update_status_bar)

        thread.started.connect(worker.run)
        thread.start()

        self.worker_pool[worker_id] = (thread, worker)
        self.update_active_workers_label()

    def update_status_bar(self, message):
        self.statusBar().showMessage(message)

    def on_worker_finished(self, worker_id):
        if worker_id in self.worker_pool:
            thread, worker = self.worker_pool.pop(worker_id)
            thread.quit()
            thread.wait()
            self.update_active_workers_label()

    def manage_worker_pool(self):
        self.queue_mutex.lock()
        queue_size = len(self.processing_queue)
        self.queue_mutex.unlock()

        active_workers = len(self.worker_pool)

        if queue_size > 0 and active_workers == 0:
            self.start_worker()
        elif queue_size > (active_workers * self.WORKER_SPAWN_THRESHOLD) and active_workers < self.MAX_WORKERS:
            self.start_worker()

    def get_job_from_queue(self):
        self.queue_mutex.lock()
        job = None
        if self.processing_queue:
            job = self.processing_queue.pop(0)
        self.queue_mutex.unlock()
        return job

    def on_queue_job_complete(self, image_path, new_area, original_text, translated_text):
        self.ui_update_mutex.lock()
        self.ui_update_queue.append((image_path, new_area))
        self.ui_update_mutex.unlock()

        # Hanya update UI jika ini adalah halaman yang sedang aktif
        current_key = self.get_current_data_key()
        if image_path == current_key:
            if not self.ui_update_timer.isActive():
                self.ui_update_timer.start(100) # Coalesce updates within 100ms

    def process_ui_updates(self):
        if self.is_processing_ui_updates:
            return

        self.is_processing_ui_updates = True
        try:
            self.ui_update_mutex.lock()
            if not self.ui_update_queue:
                self.ui_update_mutex.unlock()
                return

            # Filter updates hanya untuk halaman yang sedang aktif
            current_key = self.get_current_data_key()
            relevant_updates = [(path, area) for path, area in self.ui_update_queue if path == current_key]
            
            # Simpan semua updates untuk pemrosesan data
            updates_by_image = {}
            for image_path, new_area in self.ui_update_queue:
                if image_path not in updates_by_image:
                    updates_by_image[image_path] = []
                updates_by_image[image_path].append(new_area)

            self.ui_update_queue.clear()
            self.ui_update_mutex.unlock()

            # Update data untuk semua gambar
            for image_path, new_areas in updates_by_image.items():
                if image_path not in self.all_typeset_data:
                    self.all_typeset_data[image_path] = {'areas': [], 'redo': []}

                self.all_typeset_data[image_path]['areas'].extend(new_areas)
                self.all_typeset_data[image_path]['redo'].clear()

            # Hanya update UI untuk halaman yang sedang aktif
            if relevant_updates:
                self.typeset_areas = self.all_typeset_data.get(current_key, {'areas': []})['areas']
                self.redo_stack = self.all_typeset_data.get(current_key, {'redo': []})['redo']
                self.redraw_all_typeset_areas()
                self.update_undo_redo_buttons_state()
                
        finally:
            self.is_processing_ui_updates = False
            self.ui_update_mutex.lock()
            needs_another_run = bool(self.ui_update_queue)
            self.ui_update_mutex.unlock()
            if needs_another_run:
                self.ui_update_timer.start(100)

    def update_queue_status(self, count):
        if count > 0:
            self.statusBar().showMessage(f"{count} tasks remaining in queue...")
        else:
            self.statusBar().showMessage("Processing queue is empty.", 3000)
        self.manage_worker_pool()

    def update_active_workers_label(self):
        self.active_workers_label.setText(f"Active Workers: {len(self.worker_pool)}")

    def on_worker_error(self, error_msg):
        QTimer.singleShot(0, lambda: self.handle_worker_error_ui(error_msg))

    def handle_worker_error_ui(self, error_msg):
        QMessageBox.critical(self, "Processing Error", f"An error occurred in a worker thread:\n{error_msg}")
        self.overall_progress_bar.setVisible(False)
        self.batch_process_button.setEnabled(True)

    def update_overall_progress(self, value, text):
        self.overall_progress_bar.setVisible(True)
        self.overall_progress_bar.setValue(value)
        self.overall_progress_bar.setFormat(text)

    def get_current_settings(self):
        lang_data = self.ocr_lang_combo.currentData()
    
        # Pastikan inpaint_model_key selalu memiliki nilai
        inpaint_model_text = self.inpaint_model_combo.currentText()
        inpaint_model_key = None
        
        if "Big-LaMa" in inpaint_model_text:
            inpaint_model_key = 'big_lama'
        elif "Anime" in inpaint_model_text:
            inpaint_model_key = 'anime_inpaint'
        elif "OpenCV" in inpaint_model_text:
            inpaint_model_key = 'opencv'  # Untuk OpenCV fallback
            
        return {
            'ocr_engine': lang_data['engine'] if lang_data else None,
            'ocr_lang': lang_data['code'] if lang_data else None,
            'orientation': self.orientation_combo.currentText(),
            'target_lang': self.translate_combo.currentText(),
            'use_ai': True,
            'font': self.typeset_font,
            'color': self.typeset_color,
            'enhanced_pipeline': self.enhanced_pipeline_checkbox.isChecked(),
            'use_ai_only_translate': self.ai_only_translate_checkbox.isChecked(),
            'use_deepl_only_translate': self.deepl_only_checkbox.isChecked(),
            'use_dl_detector': self.dl_bubble_detector_checkbox.isChecked(),
            'dl_provider': self.dl_model_provider_combo.currentText(),
            'dl_model_file': self.dl_model_file_combo.currentText(),
            'ai_model': self.get_selected_model_name(),
            'translation_style': self.style_combo.currentText(),
            'auto_split_bubbles': self.split_bubbles_checkbox.isChecked(),
            'safe_mode': self.safe_mode_checkbox.isChecked(),
            'use_gpu': self.use_gpu_checkbox.isChecked(),
            # Pastikan ini sesuai dengan hardware Anda
            'use_inpaint': self.inpaint_checkbox.isChecked(),
            'inpaint_model_name': inpaint_model_text,
            'inpaint_model_key': inpaint_model_key,  # Pastikan ini tidak None
            'inpaint_padding': self.inpaint_padding_spinbox.value(),
            # Optimasi CPU
            'cpu_threads': 4,  # Sesuaikan dengan jumlah core CPU Anda
            'enable_mkldnn': True,  # Optimasi untuk CPU Intel
            'orientation_mode': 'vertical' if self.vertical_typeset_checkbox.isChecked() else 'horizontal',
            'create_bubble': getattr(self, 'create_bubble_checkbox', None) and self.create_bubble_checkbox.isChecked(),
            'text_effect': 'none',
            'effect_intensity': 20.0,
            'bezier_points': None,
            'alignment': 'center',
            'line_spacing': 1.1,
            'char_spacing': 100.0,
            'margins': {'top': 0, 'right': 0, 'bottom': 0, 'left': 0},
        }

    def update_gpu_status_label(self):
        if not hasattr(self, 'gpu_status_label') or not self.gpu_status_label:
            return

        if not self.is_gpu_available:
            self.gpu_status_label.setText("GPU Not Detected")
            self.gpu_status_label.setStyleSheet("color: #ff7b72;")
            return

        if self.use_gpu_checkbox.isChecked():
            self.gpu_status_label.setText("GPU Acceleration Active")
            self.gpu_status_label.setStyleSheet("color: #5de6c1;")
        else:
            self.gpu_status_label.setText("GPU Detected (Disabled)")
            self.gpu_status_label.setStyleSheet("color: #ffc857;")

    def translate_text(self, text, target_lang):
        # Fallback to DeepL if Gemini-only is not used
        if not text.strip(): return ""
        if not DEEPL_API_KEY or "your_deepl_key_here" in DEEPL_API_KEY: return "[DEEPL API KEY NOT CONFIGURED]"
        try:
            lang_map = {"Indonesian": "ID", "English": "EN-US", "Japanese": "JA", "Chinese": "ZH", "Korean": "KO"}
            url = "https://api-free.deepl.com/v2/translate"
            params = {"auth_key": DEEPL_API_KEY, "text": text, "target_lang": lang_map.get(target_lang, "ID")}
            response = requests.post(url, data=params, timeout=20); response.raise_for_status()
            return response.json()["translations"][0]["text"]
        except Exception as e: return f"[Translation Error: {e}]"

    def load_usage_data(self):
        try:
            if os.path.exists(self.usage_file_path):
                with open(self.usage_file_path, 'rb') as f:
                    self.usage_data = pickle.load(f)
            else:
                self.usage_data = {}

            if 'provider_usage' not in self.usage_data:
                self.usage_data['provider_usage'] = {}

            for provider, models in self.AI_PROVIDERS.items():
                if provider not in self.usage_data['provider_usage']:
                    self.usage_data['provider_usage'][provider] = {}
                for model_name in models:
                    if model_name not in self.usage_data['provider_usage'][provider]:
                        self.usage_data['provider_usage'][provider][model_name] = {'daily_count': 0, 'minute_count': 0, 'current_minute': ''}

            if 'date' not in self.usage_data or self.usage_data.get('date') != str(date.today()):
                self.usage_data['date'] = str(date.today())
                for provider, models in self.AI_PROVIDERS.items():
                    for model_name in models:
                        self.usage_data['provider_usage'][provider][model_name]['daily_count'] = 0
                        self.usage_data['provider_usage'][provider][model_name]['minute_count'] = 0

            self.total_cost = self.usage_data.get('total_cost', 0.0)
            self.update_cost_display()
            self.save_usage_data()
        except Exception as e:
            print(f"Could not load or create usage data file: {e}")
            self.usage_data = {'date': str(date.today()), 'total_cost': 0.0, 'provider_usage': {}}
            for provider, models in self.AI_PROVIDERS.items():
                self.usage_data['provider_usage'][provider] = {}
                for model_name in models:
                    self.usage_data['provider_usage'][provider][model_name] = {'daily_count': 0, 'minute_count': 0, 'current_minute': ''}

    def save_usage_data(self):
        try:
            self.usage_data['total_cost'] = self.total_cost
            with open(self.usage_file_path, 'wb') as f: pickle.dump(self.usage_data, f)
        except Exception as e: print(f"Could not save usage data: {e}")

    def check_and_increment_usage(self, provider, model_name):
        now = time.time()
        current_minute_str = time.strftime('%Y-%m-%d %H:%M', time.localtime(now))

        model_usage = self.usage_data['provider_usage'][provider][model_name]
        model_limits = self.AI_PROVIDERS[provider][model_name]['limits']

        if self.usage_data.get('date') != str(date.today()):
            self.usage_data['date'] = str(date.today())
            for p, models in self.usage_data['provider_usage'].items():
                for m in models:
                    self.usage_data['provider_usage'][p][m]['daily_count'] = 0
                    self.usage_data['provider_usage'][p][m]['minute_count'] = 0

        if model_usage.get('current_minute') != current_minute_str:
            model_usage['current_minute'] = current_minute_str
            model_usage['minute_count'] = 0

        if model_usage.get('daily_count', 0) >= model_limits['rpd']:
            QTimer.singleShot(0, self.check_limits_and_update_ui)
            return False
        if model_usage.get('minute_count', 0) >= model_limits['rpm']:
            QTimer.singleShot(0, self.check_limits_and_update_ui)
            return False

        model_usage['daily_count'] += 1
        model_usage['minute_count'] += 1

        self.save_usage_data()
        QTimer.singleShot(0, self.update_usage_display)
        return True

    def update_usage_display(self):
        provider, model_name = self.get_selected_model_name()
        if not model_name: return

        model_usage = self.usage_data['provider_usage'][provider][model_name]
        model_limits = self.AI_PROVIDERS[provider][model_name]['limits']

        rpm = model_usage.get('minute_count', 0)
        rpd = model_usage.get('daily_count', 0)

        self.rpm_label.setText(f"RPM: {rpm} / {model_limits['rpm']}")
        self.rpd_label.setText(f"RPD: {rpd} / {model_limits['rpd']}")

    def check_limits_and_update_ui(self):
        self.load_usage_data()

        provider, model_name = self.get_selected_model_name()
        if not model_name: return

        model_usage = self.usage_data['provider_usage'][provider][model_name]
        model_limits = self.AI_PROVIDERS[provider][model_name]['limits']

        now = time.time()
        current_minute_str = time.strftime('%Y-%m-%d %H:%M', time.localtime(now))

        if model_usage.get('current_minute') != current_minute_str:
            model_usage['current_minute'] = current_minute_str
            model_usage['minute_count'] = 0
            self.save_usage_data()

        self.update_usage_display()
        daily_limit_reached = model_usage.get('daily_count', 0) >= model_limits['rpd']
        minute_limit_reached = model_usage.get('minute_count', 0) >= model_limits['rpm']

        ai_disabled = daily_limit_reached or minute_limit_reached
        tooltip_message = ""

        if daily_limit_reached:
            tooltip_message = f"AI features disabled: Daily API limit reached for {model_name}."
            self.countdown_label.setVisible(False)
        elif minute_limit_reached:
            seconds_until_next_minute = 60 - int(time.strftime('%S', time.localtime(now)))
            tooltip_message = f"AI features disabled: Per-minute limit reached for {model_name}."
            self.countdown_label.setText(f"Cooldown: {seconds_until_next_minute}s")
            self.countdown_label.setVisible(True)
        else:
            self.countdown_label.setVisible(False)

        is_worker_running = (self.batch_save_thread and self.batch_save_thread.isRunning()) or \
                                 (self.detection_thread and self.detection_thread.isRunning()) or \
                                 (self.batch_processor_thread and self.batch_processor_thread.isRunning())

        enabled_state = not ai_disabled and not is_worker_running

        self.enhanced_pipeline_checkbox.setEnabled(enabled_state)
        self.ai_only_translate_checkbox.setEnabled(enabled_state)

        if ai_disabled:
            self.enhanced_pipeline_checkbox.setChecked(False)
            self.ai_only_translate_checkbox.setChecked(False)
            self.enhanced_pipeline_checkbox.setToolTip(tooltip_message)
            self.ai_only_translate_checkbox.setToolTip(tooltip_message)
            if not self.api_limit_timer.isActive(): self.api_limit_timer.start()
        else:
            self.on_pipeline_mode_changed(self.enhanced_pipeline_checkbox.checkState())
            if self.api_limit_timer.isActive(): self.api_limit_timer.stop()

    def periodic_limit_check(self):
        self.check_limits_and_update_ui()

    def load_folder(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Manga Folder")
        if dir_path:
            if self.project_dir and self.project_dir in self.file_watcher.directories():
                self.file_watcher.removePath(self.project_dir)

            if self.pdf_document:
                self.pdf_document.close()
                self.pdf_document = None
            self.current_pdf_page = -1

            self.project_dir = dir_path
            self.cache_dir = os.path.join(self.project_dir, ".cache")
            os.makedirs(self.cache_dir, exist_ok=True)
            self.all_typeset_data.clear()

            self.update_file_list()

            self.file_watcher.addPath(self.project_dir)

    def on_directory_changed(self, path):
        self.statusBar().showMessage(f"Folder changed, updating list...", 2000)
        self.update_file_list()

    def update_file_list(self):
        if not self.project_dir: return

        supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.pdf')
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

        try:
            dir_files = sorted([f for f in os.listdir(self.project_dir) if f.lower().endswith(supported_formats)], key=natural_sort_key)
            new_file_paths = [os.path.join(self.project_dir, fname) for fname in dir_files]
        except FileNotFoundError:
            self.statusBar().showMessage(f"Folder not found: {self.project_dir}", 5000)
            return

        current_selection_path = self.current_image_path

        self.image_files = new_file_paths
        self.file_list_widget.clear()
        self.file_list_widget.addItems([os.path.basename(p) for p in self.image_files])

        if current_selection_path and current_selection_path in self.image_files:
            try:
                row_to_select = self.image_files.index(current_selection_path)
                self.file_list_widget.setCurrentRow(row_to_select)
            except ValueError:
                if self.image_files:
                    self.file_list_widget.setCurrentRow(0)
        elif self.image_files:
            self.file_list_widget.setCurrentRow(0)

    def on_file_selected(self, current_item, previous_item):
        if not current_item:
            self.clear_view()
            return

        if previous_item and self.current_image_path:
            key = self.get_current_data_key(path=self.current_image_path, page=self.current_pdf_page if self.pdf_document else -1)
            self.all_typeset_data[key] = {'areas': self.typeset_areas, 'redo': self.redo_stack}

        row = self.file_list_widget.row(current_item)
        if 0 <= row < len(self.image_files):
            new_path = self.image_files[row]
            if new_path != self.current_image_path:
                self.load_item(new_path)
            else: # If it's the same item, could be a PDF page change
                self.load_item(new_path)


    def load_item(self, file_path):
        # Clear old confirmation state
        self.image_label.clear_detected_items()
        self.image_label.cancel_pending_item() # Hapus juga item yang menunggu

        self.current_image_path = file_path

        if not file_path.lower().endswith('.pdf') and self.pdf_document:
            self.pdf_document.close()
            self.pdf_document = None
            self.current_pdf_page = -1

        if file_path.lower().endswith('.pdf'):
            self.load_pdf(file_path)
        else:
            self.load_image(file_path)

        self.update_nav_buttons()

    def load_image(self, file_path):
        try:
            self.current_image_pil = Image.open(file_path).convert('RGB')
            self.original_pixmap = QPixmap(file_path)
            # Use robust opener to handle truncated/corrupt JPEGs
            self.current_image_pil = self.safe_open_image(file_path)

            # Create QPixmap from PIL image (safer than loading directly from a possibly corrupted file)
            pil_img = self.current_image_pil
            data = pil_img.tobytes('raw', 'RGB')
            qimage = QImage(data, pil_img.width, pil_img.height, pil_img.width * 3, QImage.Format_RGB888)
            self.original_pixmap = QPixmap.fromImage(qimage)

            key = self.get_current_data_key()
            img_data = self.all_typeset_data.get(key, {'areas': [], 'redo': []})
            self.typeset_areas = img_data['areas']
            self.redo_stack = img_data['redo']

            self.redraw_all_typeset_areas()
            self.update_undo_redo_buttons_state()
            self._refresh_detection_overlay()
        except Exception as e:
            QMessageBox.critical(self, "Error Loading Image", f"Could not load image: {file_path}\nError: {e}")
            self.clear_view()

    def load_pdf(self, file_path):
        try:
            if not self.pdf_document or self.pdf_document.name != file_path:
                if self.pdf_document: self.pdf_document.close()
                self.pdf_document = fitz.open(file_path)
                self.current_pdf_page = 0
            self.load_pdf_page(self.current_pdf_page)
        except Exception as e:
            QMessageBox.critical(self, "Error Loading PDF", f"Could not load PDF file: {file_path}\nError: {e}")
            self.pdf_document = None
            self.current_pdf_page = -1

    def load_pdf_page(self, page_number):
        if not self.pdf_document or not (0 <= page_number < self.pdf_document.page_count):
            return

        if self.current_pdf_page != -1 and self.current_pdf_page != page_number:
            key = self.get_current_data_key(page=self.current_pdf_page)
            self.all_typeset_data[key] = {'areas': self.typeset_areas, 'redo': self.redo_stack}

        self.current_pdf_page = page_number
        page = self.pdf_document.load_page(page_number)
        pix = page.get_pixmap(dpi=150)
        q_image = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888)
        self.original_pixmap = QPixmap.fromImage(q_image)
        self.current_image_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        key = self.get_current_data_key()
        img_data = self.all_typeset_data.get(key, {'areas': [], 'redo': []})
        self.typeset_areas = img_data['areas']
        self.redo_stack = img_data['redo']

        self.redraw_all_typeset_areas()
        self.update_undo_redo_buttons_state()
        self._refresh_detection_overlay()
        self.update_nav_buttons()

    def get_current_data_key(self, path=None, page=-1):
        path_to_use = path if path is not None else self.current_image_path
        page_to_use = page if page != -1 else self.current_pdf_page

        if path_to_use and path_to_use.lower().endswith('.pdf') and page_to_use != -1:
            return f"{path_to_use}::page::{page_to_use}"
        return path_to_use

    def clear_view(self):
        self.original_pixmap = None
        self.typeset_pixmap = None
        self.current_image_path = None
        self.current_image_pil = None
        self.typeset_areas.clear()
        self.redo_stack.clear()
        self.update_display()

    def load_next_image(self):
        if self.is_in_confirmation_mode: return
        if self.pdf_document:
            if self.current_pdf_page < self.pdf_document.page_count - 1:
                self.load_pdf_page(self.current_pdf_page + 1)
        else:
            current_row = self.file_list_widget.currentRow()
            if current_row < self.file_list_widget.count() - 1:
                self.file_list_widget.setCurrentRow(current_row + 1)

    def load_prev_image(self):
        if self.is_in_confirmation_mode: return
        if self.pdf_document:
            if self.current_pdf_page > 0:
                self.load_pdf_page(self.current_pdf_page - 1)
        else:
            current_row = self.file_list_widget.currentRow()
            if current_row > 0:
                self.file_list_widget.setCurrentRow(current_row - 1)

    def update_nav_buttons(self):
        if self.pdf_document:
            self.prev_button.setEnabled(self.current_pdf_page > 0)
            self.next_button.setEnabled(self.current_pdf_page < self.pdf_document.page_count - 1)
            self.statusBar().showMessage(f"PDF Page {self.current_pdf_page + 1} / {self.pdf_document.page_count}")
        else:
            current_row = self.file_list_widget.currentRow()
            self.prev_button.setEnabled(current_row > 0)
            self.next_button.setEnabled(current_row < self.file_list_widget.count() - 1)
            if self.current_image_path:
                self.statusBar().showMessage(f"Image {current_row + 1} / {self.file_list_widget.count()}")

    def update_display(self):
        if not self.typeset_pixmap:
            self.image_label.setPixmap(QPixmap())
            return
        self.zoom_label.setText(f" Zoom: {self.zoom_factor:.1f}x ")
        scaled_size = self.typeset_pixmap.size() * self.zoom_factor
        scaled_pixmap = self.typeset_pixmap.scaled(scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap); self.image_label.adjustSize()

    def zoom_in(self):
        self.zoom_factor = min(self.zoom_factor + 0.2, 8.0); self.update_display()

    def zoom_out(self):
        self.zoom_factor = max(self.zoom_factor - 0.2, 0.1); self.update_display()

    def reset_view_to_original(self):
        if self.original_pixmap:
            self.typeset_areas.clear(); self.redo_stack.clear()
            self.redraw_all_typeset_areas()
            self.update_undo_redo_buttons_state()

    def choose_font(self):
        font, ok = QFontDialog.getFont(self.typeset_font, self)
        if ok: self.typeset_font = font

    def choose_color(self):
        color = QColorDialog.getColor(self.typeset_color, self)
        if color.isValid(): self.typeset_color = color

    def unzoom_coords(self, selection_obj, as_point=False):
        pixmap = self.image_label.pixmap()
        if not pixmap or pixmap.isNull(): return None
        label_size = self.image_label.size(); pixmap_size = pixmap.size()
        offset_x = max(0, (label_size.width() - pixmap_size.width()) // 2)
        offset_y = max(0, (label_size.height() - pixmap_size.height()) // 2)
        if as_point and isinstance(selection_obj, QPoint):
            unzoomed_x = int((selection_obj.x() - offset_x) / self.zoom_factor)
            unzoomed_y = int((selection_obj.y() - offset_y) / self.zoom_factor)
            return QPoint(unzoomed_x, unzoomed_y)
        if isinstance(selection_obj, QRect):
            unzoomed_x = int((selection_obj.x() - offset_x) / self.zoom_factor); unzoomed_y = int((selection_obj.y() - offset_y) / self.zoom_factor)
            unzoomed_width = int(selection_obj.width() / self.zoom_factor); unzoomed_height = int(selection_obj.height() / self.zoom_factor)
            return QRect(unzoomed_x, unzoomed_y, unzoomed_width, unzoomed_height)
        elif isinstance(selection_obj, list):
            unzoomed_points = []
            for p in selection_obj:
                unzoomed_x = int((p.x() - offset_x) / self.zoom_factor); unzoomed_y = int((p.y() - offset_y) / self.zoom_factor)
                unzoomed_points.append(QPoint(unzoomed_x, unzoomed_y))
            polygon = QPolygon(unzoomed_points)
            return polygon, polygon.boundingRect()
        return None

    def process_rect_area(self, selection_rect):
        if self.is_in_confirmation_mode:
            unzoomed_rect = self.unzoom_coords(selection_rect)
            if not unzoomed_rect: return
            poly = QPolygon(unzoomed_rect)
            resolved_key = self._resolve_detection_key(self.get_current_data_key()) or self.get_current_data_key()
            if resolved_key:
                if resolved_key not in self.detected_items_map:
                    self.detected_items_map[resolved_key] = []
                # Menambahkan sebagai item baru yang terdeteksi secara manual
                new_item = {'polygon': poly, 'text': None} # Teks akan di-OCR nanti
                self.detected_items_map[resolved_key].append(new_item)
                self.image_label.set_detected_items(self.detected_items_map[resolved_key])
                self.update_confirmation_button_text()
            self.image_label.clear_selection()
            return

        if self.is_processing_selection: return

        self.is_processing_selection = True
        try:
            if not self.current_image_pil: return
            unzoomed_rect = self.unzoom_coords(selection_rect)
            if not unzoomed_rect or unzoomed_rect.width() <= 0 or unzoomed_rect.height() <= 0: return
            cropped_img = self.current_image_pil.crop((unzoomed_rect.x(), unzoomed_rect.y(), unzoomed_rect.right(), unzoomed_rect.bottom()))
            cropped_cv_img = cv2.cvtColor(np.array(cropped_img), cv2.COLOR_RGB2BGR)

            job = {
                'image_path': self.get_current_data_key(),
                'rect': unzoomed_rect,
                'polygon': None,
                'cropped_cv_img': cropped_cv_img,
                'settings': self.get_current_settings()
            }

            if self.batch_mode_checkbox.isChecked():
                self.add_to_batch_queue(job)
            else:
                self.processing_queue.append(job)
                self.update_queue_status(len(self.processing_queue))
                self.manage_worker_pool()
        finally:
            self.is_processing_selection = False

    def process_polygon_area(self, scaled_points):
        if self.is_in_confirmation_mode:
            result = self.unzoom_coords(scaled_points)
            if not result: return
            unzoomed_polygon, _ = result
            resolved_key = self._resolve_detection_key(self.get_current_data_key()) or self.get_current_data_key()
            if resolved_key:
                if resolved_key not in self.detected_items_map:
                    self.detected_items_map[resolved_key] = []
                new_item = {'polygon': unzoomed_polygon, 'text': None}
                self.detected_items_map[resolved_key].append(new_item)
                self.image_label.set_detected_items(self.detected_items_map[resolved_key])
                self.update_confirmation_button_text()
            self.image_label.clear_selection()
            return

        if self.is_processing_selection: return
        
        result = self.unzoom_coords(scaled_points)
        if not result: return
        unzoomed_polygon, unzoomed_bbox = result
        
        self.process_confirmed_polygon(unzoomed_polygon, unzoomed_bbox)


    def process_confirmed_polygon(self, unzoomed_polygon, unzoomed_bbox=None, pre_detected_text=None):
        """
        Memproses poligon yang koordinatnya sudah dalam sistem gambar penuh (unzoomed).
        """
        if self.is_processing_selection: return
        self.is_processing_selection = True

        try:
            if not self.current_image_pil: return

            if not unzoomed_bbox:
                unzoomed_bbox = unzoomed_polygon.boundingRect()

            if not unzoomed_bbox or unzoomed_bbox.width() <= 0 or unzoomed_bbox.height() <= 0:
                return

            cropped_pil_img = self.current_image_pil.crop((unzoomed_bbox.x(), unzoomed_bbox.y(), unzoomed_bbox.right(), unzoomed_bbox.bottom()))
            cropped_cv_img = cv2.cvtColor(np.array(cropped_pil_img), cv2.COLOR_RGB2BGR)
            mask = np.zeros(cropped_cv_img.shape[:2], dtype=np.uint8)
            relative_poly_points = [QPoint(p.x() - unzoomed_bbox.x(), p.y() - unzoomed_bbox.y()) for p in unzoomed_polygon]
            cv_poly_points = np.array([[p.x(), p.y()] for p in relative_poly_points], dtype=np.int32)
            cv2.fillPoly(mask, [cv_poly_points], 255)
            white_bg = np.full(cropped_cv_img.shape, 255, dtype=np.uint8)
            fg = cv2.bitwise_and(cropped_cv_img, cropped_cv_img, mask=mask)
            bg = cv2.bitwise_and(white_bg, white_bg, mask=cv2.bitwise_not(mask))
            img_for_ocr = cv2.add(fg, bg)

            # Gunakan current data key yang benar (bisa berbeda dari yang sedang ditampilkan)
            current_data_key = self.get_current_data_key()

            job = {
                'image_path': current_data_key,  # Pastikan ini path yang benar
                'rect': unzoomed_bbox,
                'polygon': unzoomed_polygon,
                'cropped_cv_img': img_for_ocr,
                'settings': self.get_current_settings(),
                'text': pre_detected_text # Tambahkan teks yang sudah di-OCR jika ada
            }

            if self.batch_mode_checkbox.isChecked():
                self.add_to_batch_queue(job)
            else:
                self.processing_queue.append(job)
                self.update_queue_status(len(self.processing_queue))
                self.manage_worker_pool()
        finally:
            self.is_processing_selection = False
        
    def redraw_all_typeset_areas(self):
        if not self.original_pixmap: return
        self.typeset_pixmap = self.original_pixmap.copy()
        painter = QPainter(self.typeset_pixmap)
        for area in self.typeset_areas:
            self.draw_single_area(painter, area, self.current_image_pil)
        painter.end()
        self.update_display()

    def get_background_color(self, full_cv_image, rect):
        if rect.width() <= 0 or rect.height() <= 0:
            return QColor(Qt.white)

        bubble_content = full_cv_image[rect.top():rect.bottom(), rect.left():rect.right()]

        h, w, _ = bubble_content.shape
        if h == 0 or w == 0:
            return QColor(Qt.white)

        gray_content = cv2.cvtColor(bubble_content, cv2.COLOR_BGR2GRAY)
        gray_content = cv2.GaussianBlur(gray_content, (5, 5), 0)

        try:
            _, mask = cv2.threshold(gray_content, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        except cv2.error:
            mean_val = cv2.mean(bubble_content)
            return QColor(int(mean_val[2]), int(mean_val[1]), int(mean_val[0]))

        if cv2.countNonZero(mask) < mask.size / 2:
            mask = cv2.bitwise_not(mask)

        mean_color_bgr = cv2.mean(bubble_content, mask=mask)
        return QColor(int(mean_color_bgr[2]), int(mean_color_bgr[1]), int(mean_color_bgr[0]))

    def _find_speech_bubble_mask_contour(self, full_cv_image, text_rect):
        padding = 25
        search_qt_rect = text_rect.adjusted(-padding, -padding, padding, padding)
        h, w, _ = full_cv_image.shape
        search_qt_rect.setLeft(max(0, search_qt_rect.left()))
        search_qt_rect.setTop(max(0, search_qt_rect.top()))
        search_qt_rect.setRight(min(w - 1, search_qt_rect.right()))
        search_qt_rect.setBottom(min(h - 1, search_qt_rect.bottom()))
        if search_qt_rect.width() <= 0 or search_qt_rect.height() <= 0: return None
        search_area_cv = full_cv_image[search_qt_rect.top():search_qt_rect.bottom(), search_qt_rect.left():search_qt_rect.right()]
        gray = cv2.cvtColor(search_area_cv, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 5)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        text_center_relative = QPoint(text_rect.center().x() - search_qt_rect.left(), text_rect.center().y() - search_qt_rect.top())
        candidate_contours = [cnt for cnt in contours if cv2.pointPolygonTest(cnt, (text_center_relative.x(), text_center_relative.y()), False) >= 0 and cv2.contourArea(cnt) > text_rect.width() * text_rect.height() * 0.5]
        if not candidate_contours: return None
        best_contour = max(candidate_contours, key=cv2.contourArea)
        final_mask = np.zeros(full_cv_image.shape[:2], dtype=np.uint8)
        shifted_contour = best_contour + np.array([search_qt_rect.left(), search_qt_rect.top()])
        cv2.drawContours(final_mask, [shifted_contour], -1, 255, thickness=cv2.FILLED)
        return final_mask

    def _run_onnx_inference(self, model_key, full_cv_image):
        if not self.is_onnx_available: return None
        model_info = self.dl_models[model_key]
        if model_info['instance'] is None:
            if not os.path.exists(model_info['path']): return None
            try: 
                # [BARU] Pilih provider CPU/GPU
                providers = ['CPUExecutionProvider']
                if self.use_gpu_checkbox.isChecked() and self.is_gpu_available:
                    providers.insert(0, 'CUDAExecutionProvider')
                model_info['instance'] = onnxruntime.InferenceSession(model_info['path'], providers=providers)
            except Exception as e: print(f"Error loading ONNX model: {e}"); return None

        session = model_info['instance']
        try:
            input_name = session.get_inputs()[0].name
            input_shape = session.get_inputs()[0].shape

            try: model_h, model_w = int(input_shape[2]), int(input_shape[3])
            except (TypeError, ValueError): model_h, model_w = 512, 512

            original_h, original_w, _ = full_cv_image.shape
            img_rgb = cv2.cvtColor(full_cv_image, cv2.COLOR_BGR2RGB)
            resized_img = cv2.resize(img_rgb, (model_w, model_h))

            input_tensor = (resized_img / 255.0).astype(np.float32).transpose(2, 0, 1)
            input_tensor = np.expand_dims(input_tensor, axis=0)

            ort_inputs = {input_name: input_tensor}
            ort_outs = session.run(None, ort_inputs)

            output_array = ort_outs[0]
            if output_array.ndim == 4: mask = output_array[0, 0, :, :]
            elif output_array.ndim == 3: mask = output_array[0, :, :]
            else: raise ValueError(f"Unexpected model output dimension: {output_array.ndim}")

            mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
            return (mask > 0.5).astype(np.uint8) * 255
        except Exception as e:
            print(f"Error during ONNX inference: {e}"); return None

    def _run_yolov8_inference(self, model_key, full_cv_image):
        if not self.is_yolo_available: return None
        model_info = self.dl_models[model_key]
        if model_info['instance'] is None:
            if not os.path.exists(model_info['path']): return None
            try: model_info['instance'] = YOLO(model_info['path'])
            except Exception as e: print(f"Error loading YOLO model: {e}"); return None

        model = model_info['instance']
        try:
            # [BARU] Pilih device CPU/GPU
            device = "cuda" if self.use_gpu_checkbox.isChecked() and self.is_gpu_available else "cpu"
            results = model(full_cv_image, verbose=False, device=device)
            if not results or not results[0].masks: return None

            final_mask = np.zeros((full_cv_image.shape[0], full_cv_image.shape[1]), dtype=np.uint8)
            for mask_tensor in results[0].masks.data:
                mask_np = mask_tensor.cpu().numpy().astype(np.uint8) * 255
                if mask_np.shape != final_mask.shape:
                    mask_np = cv2.resize(mask_np, (final_mask.shape[1], final_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                final_mask = cv2.bitwise_or(final_mask, mask_np)

            return final_mask
        except Exception as e:
            print(f"Error during YOLO inference: {e}"); return None

    def detect_bubble_with_dl_model(self, full_cv_image, settings):
        provider = settings['dl_provider']
        model_file = settings['dl_model_file']
        model_key = ""

        if provider == "Kitsumed": model_key = 'kitsumed_onnx' if model_file == 'model_dynamic.onnx' else 'kitsumed_pt'
        elif provider == "Ogkalu": model_key = 'ogkalu_pt'

        if not model_key: return None
        model_type = self.dl_models[model_key]['type']

        if model_type == 'onnx': return self._run_onnx_inference(model_key, full_cv_image)
        elif model_type == 'yolo': return self._run_yolov8_inference(model_key, full_cv_image)
        return None

    def find_speech_bubble_mask(self, full_cv_image, text_rect, settings, for_saving=False):
        if settings['use_dl_detector']:
            if not for_saving:
                self.statusBar().showMessage(f"Detecting bubble with {settings['dl_provider']} model...", 2000)
                QApplication.processEvents()

            combined_dl_mask = self.detect_bubble_with_dl_model(full_cv_image, settings)

            if combined_dl_mask is not None:
                contours, _ = cv2.findContours(combined_dl_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                text_center = (text_rect.center().x(), text_rect.center().y())

                for cnt in contours:
                    if cv2.pointPolygonTest(cnt, text_center, False) >= 0:
                        single_bubble_mask = np.zeros_like(combined_dl_mask)
                        cv2.drawContours(single_bubble_mask, [cnt], -1, 255, thickness=cv2.FILLED)
                        return single_bubble_mask

        if not for_saving:
            self.statusBar().showMessage("Detecting bubble with contour method...", 2000)
            QApplication.processEvents()
        return self._find_speech_bubble_mask_contour(full_cv_image, text_rect)

    # [DIUBAH] Logika inpainting yang diperbarui
    def draw_single_area(self, painter, area, source_pil_image, for_saving=False):
        settings = self.get_current_settings()
        cv_original = cv2.cvtColor(np.array(source_pil_image), cv2.COLOR_RGB2BGR)
        
        # 1. Buat mask dari bentuk area (polygon atau rectangle)
        base_mask = np.zeros(cv_original.shape[:2], dtype=np.uint8)
        if area.polygon:
            cv_poly_points = np.array([[p.x(), p.y()] for p in area.polygon], dtype=np.int32)
            cv2.fillPoly(base_mask, [cv_poly_points], 255)
        else:
            cv2.rectangle(base_mask, (area.rect.x(), area.rect.y()), 
                          (area.rect.right(), area.rect.bottom()), 255, -1)
        
        # 2. Gunakan bubble detector untuk mendapatkan mask bubble jika memungkinkan
        bubble_mask = self.find_speech_bubble_mask(cv_original, area.rect, settings, for_saving=for_saving)
        
        # 3. Gabungkan mask area dengan mask bubble
        if bubble_mask is not None:
            combined_mask = cv2.bitwise_and(base_mask, bubble_mask)
        else:
            combined_mask = base_mask
        
        # 4. Tambahkan padding untuk hasil yang lebih baik
        padding = settings['inpaint_padding']
        kernel = np.ones((max(1,padding), max(1,padding)), np.uint8)
        final_inpaint_mask = cv2.dilate(combined_mask, kernel, iterations=1)
        
        # 5. PROSES INPAINTING YANG DIPERBAIKI
        if settings['use_inpaint']:
            inpainted_cv = None
            
            # Coba advanced inpainting dengan LaMa models
            if settings['inpaint_model_key'] and self.inpaint_model:
                try:
                    # Inisialisasi model HANYA jika belum dilakukan atau model berbeda
                    if (self.inpaint_model is None or 
                        self.current_inpaint_model_key != settings['inpaint_model_key']):
                        self.initialize_inpaint_engine()
                    
                    if self.inpaint_model:
                        # Konversi ke format PIL untuk model
                        pil_original = Image.fromarray(cv2.cvtColor(cv_original, cv2.COLOR_BGR2RGB))
                        pil_mask = Image.fromarray((final_inpaint_mask > 0).astype(np.uint8) * 255).convert("L")
                        
                        # PROSES DENGAN LAMA-CLEANER melalui wrapper
                        out_arr = self.inpaint_model(pil_original, pil_mask)
                        
                        if out_arr is None:
                            raise RuntimeError("LaMa inpaint returned None")
                        
                        # out_arr diharapkan BGR numpy atau RGB numpy sesuai wrapper; kita normalisasi ke RGB numpy
                        if isinstance(out_arr, np.ndarray):
                            # Jika warna BGR (biasa dari OpenCV), konversi ke RGB
                            if out_arr.shape[2] == 3:
                                inpainted_cv = cv2.cvtColor(out_arr, cv2.COLOR_BGR2RGB)
                                inpainted_cv = cv2.cvtColor(inpainted_cv, cv2.COLOR_RGB2BGR)
                            else:
                                inpainted_cv = out_arr
                        else:
                            # Jika wrapper mengembalikan PIL Image (tidak mungkin karena wrapper mengubah ke np), convert
                            try:
                                inpainted_cv = cv2.cvtColor(np.array(out_arr.convert("RGB")), cv2.COLOR_RGB2BGR)
                            except Exception:
                                inpainted_cv = None
                        
                except Exception as e:
                    print(f"Advanced inpainting failed: {e}")
                    inpainted_cv = None
            
            # Fallback ke OpenCV jika advanced inpainting gagal atau tidak dipilih
            if inpainted_cv is None:
                try:
                    algo_map = {"OpenCV-NS": cv2.INPAINT_NS, "OpenCV-Telea": cv2.INPAINT_TELEA}
                    algo = algo_map.get(settings.get('inpaint_model_name', 'OpenCV-NS'), cv2.INPAINT_NS)
                    # inpaint expects 8-bit single channel mask
                    inpaint_mask_for_cv = (final_inpaint_mask > 0).astype(np.uint8) * 255
                    inpainted_cv = cv2.inpaint(cv_original, inpaint_mask_for_cv, 3, algo)
                except Exception as e:
                    print(f"OpenCV inpainting also failed: {e}")
                    # Fallback terakhir: background color filling
                    background_color = self.get_background_color(cv_original, area.rect)
                    painter.save()
                    path = QPainterPath()
                    contours, _ = cv2.findContours(final_inpaint_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        polygon = QPolygonF([QPoint(p[0][0], p[0][1]) for p in cnt])
                        path.addPolygon(polygon)
                    painter.setClipPath(path)
                    painter.fillRect(painter.window(), background_color)
                    painter.restore()
                    return  # Keluar early karena sudah digambar
            
            # Gambar hasil inpainting (baik LaMa maupun OpenCV)
            if inpainted_cv is not None:
                # Pastikan inpainted_cv dalam format BGR uint8
                if inpainted_cv.dtype != np.uint8:
                    inpainted_cv = (np.clip(inpainted_cv, 0, 255)).astype(np.uint8)
                # Convert ke RGB untuk QImage Format_RGB888
                rgb_img = cv2.cvtColor(inpainted_cv, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_img.shape
                bytes_per_line = ch * w
                inpainted_qimage = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
                
                painter.save()
                clip_path = QPainterPath()
                contours, _ = cv2.findContours(final_inpaint_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    polygon = QPolygonF([QPoint(p[0][0], p[0][1]) for p in cnt])
                    clip_path.addPolygon(polygon)
                painter.setClipPath(clip_path)
                painter.drawImage(0, 0, inpainted_qimage)
                painter.restore()

        else: 
            # CLEANUP MANUAL (tanpa inpainting)
            background_color = self.get_background_color(cv_original, area.rect)
            painter.save()
            path = QPainterPath()
            contours, _ = cv2.findContours(final_inpaint_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                polygon = QPolygonF([QPoint(p[0][0], p[0][1]) for p in cnt])
                path.addPolygon(polygon)
            painter.setClipPath(path)
            painter.fillRect(painter.window(), background_color)
            painter.restore()
        
        # 6. Render optional bubble and stylised text
        if getattr(area, 'bubble_enabled', False):
            painter.save()
            self.draw_area_bubble(painter, area)
            painter.restore()

        painter.save()
        self.draw_area_text(painter, area)
        painter.restore()

    def draw_area_bubble(self, painter, area):
        path = QPainterPath()
        if getattr(area, 'polygon', None):
            path.addPolygon(QPolygonF(area.polygon))
        else:
            rect = QRectF(area.rect)
            radius = max(8.0, min(rect.width(), rect.height()) * 0.18)
            path.addRoundedRect(rect, radius, radius)

        painter.setBrush(QBrush(area.get_bubble_fill_color()))
        outline_width = max(1.0, float(getattr(area, 'bubble_outline_width', 3.0) or 3.0))
        painter.setPen(QPen(area.get_bubble_outline_color(), outline_width))
        painter.drawPath(path)

    def draw_area_text(self, painter, area):
        rect = QRectF(area.rect)
        margins = area.get_margins()
        rect = rect.adjusted(margins['left'], margins['top'], -margins['right'], -margins['bottom'])
        if rect.width() <= 0 or rect.height() <= 0:
            return

        effect = area.get_effect().lower()
        orientation = area.get_orientation()

        if orientation == 'vertical' and effect != 'none':
            effect = 'none'

        if effect == 'none':
            self._draw_rich_text_document(painter, rect, area, orientation)
        else:
            self._draw_effect_text(painter, rect, area, effect)

    def _create_document_from_segments(self, area):
        doc = QTextDocument()
        doc.setDocumentMargin(0)
        option = doc.defaultTextOption()
        align_map = {
            'center': Qt.AlignHCenter,
            'left': Qt.AlignLeft,
            'right': Qt.AlignRight,
            'justify': Qt.AlignJustify,
        }
        option.setAlignment(align_map.get(area.get_alignment(), Qt.AlignHCenter))
        doc.setDefaultTextOption(option)

        cursor = QTextCursor(doc)
        cursor.movePosition(QTextCursor.Start)
        line_spacing = max(0.6, area.get_line_spacing())
        block_format = QTextBlockFormat()
        block_format.setLineHeight(int(line_spacing * 100), QTextBlockFormat.ProportionalHeight)
        block_format.setAlignment(option.alignment())
        cursor.setBlockFormat(block_format)

        first_segment = True
        for segment in area.get_segments():
            text_value = segment.get('text', '')
            if not text_value:
                continue
            fmt = QTextCharFormat()
            seg_font = area.segment_to_qfont(segment)
            fmt.setFont(seg_font)
            fmt.setForeground(QBrush(area.segment_to_color(segment)))
            if segment.get('underline', False):
                fmt.setFontUnderline(True)

            parts = text_value.split('\n')
            for idx, part in enumerate(parts):
                if not first_segment:
                    cursor.mergeBlockFormat(block_format)
                cursor.insertText(part, fmt)
                first_segment = False
                if idx < len(parts) - 1:
                    cursor.insertBlock(block_format)
                    cursor.setBlockFormat(block_format)

        return doc

    def _draw_rich_text_document(self, painter, rect, area, orientation):
        doc = self._create_document_from_segments(area)
        if doc.isEmpty():
            return

        target_width = rect.height() if orientation == 'vertical' else rect.width()
        doc.setTextWidth(max(1.0, target_width))
        doc_size = doc.size()
        if doc_size.isEmpty():
            return

        image_width = max(1, int(math.ceil(doc_size.width())))
        image_height = max(1, int(math.ceil(doc_size.height())))
        image = QImage(image_width, image_height, QImage.Format_ARGB32_Premultiplied)
        image.fill(Qt.transparent)
        doc_painter = QPainter(image)
        doc.drawContents(doc_painter)
        doc_painter.end()

        if orientation == 'vertical':
            scale_w = rect.width() / image_height if image_height else 1.0
            scale_h = rect.height() / image_width if image_width else 1.0
        else:
            scale_w = rect.width() / image_width if image_width else 1.0
            scale_h = rect.height() / image_height if image_height else 1.0
        scale = min(scale_w, scale_h)

        painter.translate(rect.center())
        if orientation == 'vertical':
            painter.rotate(90)
        painter.scale(scale, scale)
        painter.translate(-image_width / 2, -image_height / 2)
        painter.drawImage(0, 0, image)

    def _flatten_segments_to_lines(self, area):
        lines = []
        current_line = []
        for segment in area.get_segments():
            text = segment.get('text', '')
            if text is None:
                continue
            base_font = area.segment_to_qfont(segment)
            base_color = area.segment_to_color(segment)
            underline = segment.get('underline', base_font.underline())
            for char in text:
                if char == '\n':
                    lines.append(current_line)
                    current_line = []
                    continue
                glyph_font = QFont(base_font)
                glyph_font.setUnderline(underline)
                glyph_font.setLetterSpacing(QFont.PercentageSpacing, area.get_char_spacing())
                current_line.append({'char': char, 'font': glyph_font, 'color': QColor(base_color)})
        lines.append(current_line)
        return [line for line in lines if line]

    def _compute_line_metrics(self, lines, area):
        metrics = []
        for glyphs in lines:
            if not glyphs:
                base_font = area.get_font()
                fm = QFontMetrics(base_font)
                metrics.append({'ascent': fm.ascent(), 'descent': fm.descent(), 'height': (fm.ascent() + fm.descent()) * area.get_line_spacing()})
                continue
            ascents = [QFontMetrics(g['font']).ascent() for g in glyphs]
            descents = [QFontMetrics(g['font']).descent() for g in glyphs]
            ascent = max(ascents) if ascents else 0.0
            descent = max(descents) if descents else 0.0
            metrics.append({'ascent': ascent, 'descent': descent, 'height': (ascent + descent) * area.get_line_spacing()})
        return metrics

    def _draw_effect_text(self, painter, rect, area, effect):
        lines = self._flatten_segments_to_lines(area)
        if not lines:
            return

        metrics = self._compute_line_metrics(lines, area)
        total_height = sum(m['height'] for m in metrics)
        y_offset = rect.top() + max(0.0, (rect.height() - total_height) / 2.0)
        alignment = area.get_alignment()
        baseline = y_offset + (metrics[0]['ascent'] if metrics else 0.0)
        intensity = area.get_effect_intensity()

        for index, glyphs in enumerate(lines):
            if not glyphs:
                baseline += metrics[index]['height']
                continue
            if effect == 'curved':
                self._draw_curved_line(painter, rect, glyphs, area, index, len(lines), intensity)
            elif effect == 'wavy':
                self._draw_wavy_line(painter, rect, glyphs, baseline, intensity, alignment)
            elif effect == 'jagged':
                self._draw_jagged_line(painter, rect, glyphs, baseline, intensity, alignment)
            baseline += metrics[index]['height']

    def _draw_curved_line(self, painter, rect, glyphs, area, line_index, total_lines, intensity):
        total_width = sum(QFontMetrics(g['font']).horizontalAdvance(g['char']) for g in glyphs)
        if total_width <= 0:
            return

        offset_ratio = 0.0
        if total_lines > 1:
            offset_ratio = (line_index - (total_lines - 1) / 2.0) / max(1, total_lines - 1)
        center_y = rect.center().y() + offset_ratio * rect.height() * 0.2
        intensity_factor = max(0.0, min(intensity / 50.0, 5.0))
        bezier_points = area.get_bezier_points()

        def scale_point(point):
            px = rect.left() + rect.width() * point.get('x', 0.5)
            base_y = rect.top() + rect.height() * point.get('y', 0.5)
            blended_y = center_y + (base_y - center_y) * intensity_factor
            return QPointF(px, blended_y)

        p0 = QPointF(rect.left(), center_y)
        p3 = QPointF(rect.right(), center_y)
        cp1 = scale_point(bezier_points[0]) if len(bezier_points) > 0 else QPointF(rect.left() + rect.width() * 0.3, center_y - rect.height() * 0.2)
        cp2 = scale_point(bezier_points[1]) if len(bezier_points) > 1 else QPointF(rect.left() + rect.width() * 0.7, center_y - rect.height() * 0.2)

        progress = 0.0
        for glyph in glyphs:
            fm = QFontMetrics(glyph['font'])
            advance = fm.horizontalAdvance(glyph['char'])
            if advance <= 0:
                continue
            t_mid = min(1.0, max(0.0, (progress + advance / 2.0) / total_width))
            point = self._evaluate_cubic_bezier(t_mid, p0, cp1, cp2, p3)
            tangent = self._bezier_tangent(t_mid, p0, cp1, cp2, p3)
            angle = math.degrees(math.atan2(tangent.y(), tangent.x())) if (tangent.x() or tangent.y()) else 0.0

            painter.save()
            painter.translate(point)
            painter.rotate(angle)
            painter.setFont(glyph['font'])
            painter.setPen(QPen(glyph['color']))
            painter.drawText(QPointF(-advance / 2.0, 0), glyph['char'])
            painter.restore()

            progress += advance

    def _draw_wavy_line(self, painter, rect, glyphs, baseline, intensity, alignment):
        total_width = sum(QFontMetrics(g['font']).horizontalAdvance(g['char']) for g in glyphs)
        if total_width <= 0:
            return

        if alignment == 'left':
            start_x = rect.left()
        elif alignment == 'right':
            start_x = rect.right() - total_width
        else:
            start_x = rect.left() + (rect.width() - total_width) / 2.0

        amplitude = min(rect.height() * 0.3, max(2.0, intensity))
        frequency = (2.0 * math.pi) / max(total_width, 1.0)

        current_x = start_x
        for glyph in glyphs:
            fm = QFontMetrics(glyph['font'])
            advance = fm.horizontalAdvance(glyph['char'])
            if advance <= 0:
                continue
            mid_x = current_x + advance / 2.0
            wave_offset = math.sin(mid_x * frequency) * amplitude
            painter.save()
            painter.setFont(glyph['font'])
            painter.setPen(QPen(glyph['color']))
            painter.drawText(QPointF(current_x, baseline + wave_offset), glyph['char'])
            painter.restore()
            current_x += advance

    def _draw_jagged_line(self, painter, rect, glyphs, baseline, intensity, alignment):
        total_width = sum(QFontMetrics(g['font']).horizontalAdvance(g['char']) for g in glyphs)
        if total_width <= 0:
            return

        if alignment == 'left':
            start_x = rect.left()
        elif alignment == 'right':
            start_x = rect.right() - total_width
        else:
            start_x = rect.left() + (rect.width() - total_width) / 2.0

        amplitude = min(rect.height() * 0.4, max(4.0, intensity * 1.2))
        current_x = start_x
        for idx, glyph in enumerate(glyphs):
            fm = QFontMetrics(glyph['font'])
            advance = fm.horizontalAdvance(glyph['char'])
            if advance <= 0:
                continue
            offset = amplitude if idx % 2 == 0 else -amplitude
            painter.save()
            painter.translate(current_x, baseline + offset)
            painter.rotate(10 if idx % 2 == 0 else -10)
            bold_font = QFont(glyph['font'])
            bold_font.setWeight(max(bold_font.weight(), QFont.Black))
            painter.setFont(bold_font)
            painter.setPen(QPen(glyph['color']))
            painter.drawText(QPointF(0, 0), glyph['char'])
            painter.restore()
            current_x += advance

    def _evaluate_cubic_bezier(self, t, p0, p1, p2, p3):
        s = 1.0 - t
        x = (s ** 3) * p0.x() + 3 * (s ** 2) * t * p1.x() + 3 * s * (t ** 2) * p2.x() + (t ** 3) * p3.x()
        y = (s ** 3) * p0.y() + 3 * (s ** 2) * t * p1.y() + 3 * s * (t ** 2) * p2.y() + (t ** 3) * p3.y()
        return QPointF(x, y)

    def _bezier_tangent(self, t, p0, p1, p2, p3):
        s = 1.0 - t
        dx = 3 * (s ** 2) * (p1.x() - p0.x()) + 6 * s * t * (p2.x() - p1.x()) + 3 * (t ** 2) * (p3.x() - p2.x())
        dy = 3 * (s ** 2) * (p1.y() - p0.y()) + 6 * s * t * (p2.y() - p1.y()) + 3 * (t ** 2) * (p3.y() - p2.y())
        return QPointF(dx, dy)

    def selection_mode_changed(self, mode):
        self.image_label.clear_selection()
        # Batalkan item yang menunggu jika mode diubah
        self.image_label.cancel_pending_item()
        self.image_label.setCursor(Qt.CrossCursor if "Rect" in mode or "Oval" in mode else Qt.PointingHandCursor)

    def update_pen_tool_buttons_visibility(self, visible):
        self.confirm_button.setVisible(visible); self.cancel_button.setVisible(visible)

    def confirm_pen_selection(self):
        points = self.image_label.get_polygon_points()
        if len(points) < 3: QMessageBox.warning(self, "Invalid Shape", "Please select at least 3 points."); return
        self.process_polygon_area(points); self.image_label.clear_selection()

    def cancel_pen_selection(self):
        self.image_label.clear_selection()

    def save_image(self):
        if not self.typeset_pixmap: QMessageBox.warning(self, "No Image", "There is no image to save."); return

        if self.pdf_document:
            original_filename = os.path.basename(self.current_image_path)
            name, _ = os.path.splitext(original_filename)
            save_suggestion = os.path.join(os.path.dirname(self.current_image_path), f"{name}_page_{self.current_pdf_page + 1}_typeset.png")
        else:
            original_filename = os.path.basename(self.current_image_path)
            name, _ = os.path.splitext(original_filename)
            save_suggestion = os.path.join(os.path.dirname(self.current_image_path), f"{name}_typeset.png")

        filePath, _ = QFileDialog.getSaveFileName(self, "Save Typeset Image", save_suggestion, "PNG Image (*.png);;JPEG Image (*.jpg)")
        if filePath:
            if not self.typeset_pixmap.save(filePath): QMessageBox.critical(self, "Error", "Failed to save the image.")
            else: QMessageBox.information(self, "Success", f"Image saved to:\n{filePath}")

    def delete_typeset_area(self, area_to_delete):
        if area_to_delete in self.typeset_areas:
            self.typeset_areas.remove(area_to_delete)
            self.redo_stack.clear()
            self.redo_stack.append(area_to_delete)
            self.redraw_all_typeset_areas()
            self.update_undo_redo_buttons_state()

    def undo_last_action(self):
        if self.typeset_areas:
            undone_area = self.typeset_areas.pop(); self.redo_stack.append(undone_area)
            self.redraw_all_typeset_areas(); self.update_undo_redo_buttons_state(); self.image_label.clear_selection()

    def redo_last_action(self):
        if self.redo_stack:
            redone_area = self.redo_stack.pop(); self.typeset_areas.append(redone_area)
            self.redraw_all_typeset_areas(); self.update_undo_redo_buttons_state(); self.image_label.clear_selection()

    def update_undo_redo_buttons_state(self):
        self.undo_button.setEnabled(len(self.typeset_areas) > 0)
        self.redo_button.setEnabled(len(self.redo_stack) > 0)

    def save_project(self, is_auto=False):
        if not self.image_files: return False
    
        if self.current_image_path:
            current_key = self.get_current_data_key()
            self.all_typeset_data[current_key] = {
                'areas': self.typeset_areas[:], 
                'redo': self.redo_stack[:]
            }
    
        if not self.current_project_path:
            if is_auto: return False # Jangan munculkan dialog saat auto-save
            filePath, _ = QFileDialog.getSaveFileName(self, "Save Project", "", "Manga Project (*.manga_proj)")
            if not filePath: return False
            self.current_project_path = filePath
    
        project_data = {
            'project_dir': self.project_dir, 'current_path': self.current_image_path,
            'all_data': self.all_typeset_data, 'font': TypesetArea.font_to_dict(self.typeset_font),
            'color': self.typeset_color.name()
        }
        try:
            with open(self.current_project_path, 'wb') as f: pickle.dump(project_data, f)
            if not is_auto:
                self.setWindowTitle(f"Manga OCR & Typeset Tool v16.1.0 - {os.path.basename(self.current_project_path)}"); 
            self.autosave_timer.start(); 
            return True
        except Exception as e:
            if not is_auto:
                QMessageBox.critical(self, "Error", f"Failed to save project: {e}"); 
            return False

    def auto_save_project(self):
        if QApplication.activeModalWidget() is not None:
            return 
    
        if self.current_project_path and os.path.exists(os.path.dirname(self.current_project_path)):
            if self.save_project(is_auto=True): 
                self.statusBar().showMessage(f"Project auto-saved at {time.strftime('%H:%M:%S')}", 3000)

    def load_project(self):
        filePath, _ = QFileDialog.getOpenFileName(self, "Load Project", "", "Manga Project (*.manga_proj)")
        if not filePath: return
        try:
            with open(filePath, 'rb') as f: project_data = pickle.load(f)

            self.project_dir = project_data.get('project_dir')
            if not self.project_dir or not os.path.exists(self.project_dir): raise FileNotFoundError("Project directory not found.")

            self.cache_dir = os.path.join(self.project_dir, ".cache"); os.makedirs(self.cache_dir, exist_ok=True)
            self.all_typeset_data = project_data.get('all_data', {})

            font_info = project_data.get('font'); self.typeset_font = QFont(); self.typeset_font.setFamily(font_info['family']); self.typeset_font.setPointSize(font_info['pointSize']); self.typeset_font.setWeight(font_info['weight']); self.typeset_font.setItalic(font_info['italic'])
            self.typeset_color = QColor(project_data.get('color', '#000000'))

            self.update_file_list()
            path_to_load = project_data.get('current_path', None)
            if path_to_load and path_to_load in self.image_files:
                self.file_list_widget.setCurrentRow(self.image_files.index(path_to_load))

            self.current_project_path = filePath
            self.setWindowTitle(f"Manga OCR & Typeset Tool v16.1.0 - {os.path.basename(self.current_project_path)}"); self.autosave_timer.start()
            QMessageBox.information(self, "Success", "Project loaded successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load project: {e}")

    def start_inline_edit(self, area):
        if not area:
            return

        dialog = AdvancedTextEditDialog(area, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            result = dialog.get_result()
            if not result:
                return

            area.set_segments(result.get('segments', []))
            area.text = result.get('plain_text', area.text)
            area.orientation = result.get('orientation', area.get_orientation())
            area.effect = result.get('effect', area.get_effect())
            area.effect_intensity = result.get('effect_intensity', area.get_effect_intensity())
            area.bezier_points = result.get('bezier_points', area.get_bezier_points())
            area.bubble_enabled = result.get('bubble_enabled', area.bubble_enabled)
            area.alignment = result.get('alignment', area.get_alignment())
            area.line_spacing = result.get('line_spacing', area.get_line_spacing())
            area.char_spacing = result.get('char_spacing', area.get_char_spacing())
            area.margins = result.get('margins', area.get_margins())

            first_segment = next((seg for seg in area.get_segments() if seg.get('text', '').strip()), None)
            if first_segment:
                area.font_info = first_segment.get('font', area.font_info)
                area.color_info = first_segment.get('color', area.color_info)

            area.ensure_defaults()
            self.redo_stack.clear()
            self.redraw_all_typeset_areas()
            self.update_undo_redo_buttons_state()
            self.statusBar().showMessage("Text updated", 2000)

    def zoom_coords(self, unzoomed_rect):
        pixmap = self.image_label.pixmap()
        if not pixmap or pixmap.isNull(): return QRect()
        label_size = self.image_label.size(); pixmap_size = pixmap.size()
        offset_x = max(0, (label_size.width() - pixmap_size.width()) // 2); offset_y = max(0, (label_size.height() - pixmap_size.height()) // 2)
        zoomed_x = int(unzoomed_rect.x() * self.zoom_factor + offset_x); zoomed_y = int(unzoomed_rect.y() * self.zoom_factor + offset_y)
        zoomed_w = int(unzoomed_rect.width() * self.zoom_factor); zoomed_h = int(unzoomed_rect.height() * self.zoom_factor)
        return QRect(zoomed_x, zoomed_y, zoomed_w, zoomed_h)

    def toggle_theme(self):
        pass # Light theme TBD

    def show_about_dialog(self):
        self.load_usage_data()
        provider, model_name = self.get_selected_model_name()
        if not model_name: return
        about_text = (f"<b>Manga OCR & Typeset Tool v16.1.0</b><br><br>This tool was created to streamline the process of translating manga.<br><br>Powered by Python, PyQt5, and various AI APIs.<br>Enhanced with new features by Gemini.<br><br>Copyright © 2024")
        QMessageBox.about(self, "About & API Usage", about_text)
    def export_to_pdf(self):
        if not self.project_dir:
            QMessageBox.warning(self, "No Folder Loaded", "Please load a folder containing images first.")
            return

        image_files_to_export = []
        for file_path in self.image_files:
            if "_typeset" in file_path.lower():
                continue

            path_part, ext = os.path.splitext(file_path)
            typeset_path = f"{path_part}_typeset.png"

            if os.path.exists(typeset_path):
                image_files_to_export.append(typeset_path)

        if not image_files_to_export:
            QMessageBox.warning(self, "No Typeset Files Found", "No '_typeset.png' files were found in the current folder to export.")
            return

        folder_name = os.path.basename(self.project_dir)
        save_suggestion = os.path.join(self.project_dir, f"{folder_name}_typeset.pdf")

        pdf_path, _ = QFileDialog.getSaveFileName(self, "Save Typeset PDF As", save_suggestion, "PDF Files (*.pdf)")
        if not pdf_path: return

        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', os.path.basename(s))]

        image_files_to_export.sort(key=natural_sort_key)

        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage("Exporting to PDF... Please wait.")

        try:
            images_pil = []
            for i, f in enumerate(image_files_to_export):
                self.overall_progress_bar.setVisible(True)
                self.update_overall_progress(int((i/len(image_files_to_export))*100), f"Converting {os.path.basename(f)}...")
                img = Image.open(f).convert("RGB")
                images_pil.append(img)

            if images_pil:
                self.update_overall_progress(100, "Saving PDF...")
                images_pil[0].save(pdf_path, "PDF", resolution=100.0, save_all=True, append_images=images_pil[1:])
                QMessageBox.information(self, "Success", f"Successfully exported {len(images_pil)} typeset images to:\n{pdf_path}")
            else:
                raise Exception("No images could be processed.")

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"An error occurred while exporting to PDF:\n{e}")
        finally:
            QApplication.restoreOverrideCursor() # DIUBAH: hapus argumen
            self.overall_progress_bar.setVisible(False)
            self.statusBar().showMessage("Ready", 3000)

    def wheelEvent(self, event):
        if self.pdf_document and not (self.detection_thread and self.detection_thread.isRunning()):
            if event.angleDelta().y() < 0: self.load_next_image()
            elif event.angleDelta().y() > 0: self.load_prev_image()
        super().wheelEvent(event)

    def on_dl_detector_state_changed(self, state):
        is_checked = (state == Qt.Checked)
        provider = self.dl_model_provider_combo.currentText()
        model_file = self.dl_model_file_combo.currentText()
        is_available = True
        tooltip = f"Uses {provider}'s {model_file} for advanced bubble detection."

        if not model_file: is_available = False; tooltip = "No model selected or available."
        elif model_file.endswith('.onnx'):
            if not self.is_onnx_available: is_available = False; tooltip = "Disabled: 'onnxruntime' not installed."
            elif not os.path.exists(self.dl_models['kitsumed_onnx']['path']): is_available = False; tooltip = f"Disabled: Model file not found."
        elif model_file.endswith('.pt'):
            if not self.is_yolo_available: is_available = False; tooltip = "Disabled: 'ultralytics' not installed."
            else:
                key = 'ogkalu_pt' if provider == 'Ogkalu' else 'kitsumed_pt'
                if not os.path.exists(self.dl_models[key]['path']): is_available = False; tooltip = f"Disabled: Model file not found."

        self.dl_bubble_detector_checkbox.setEnabled(is_available)
        self.dl_bubble_detector_checkbox.setToolTip(tooltip)
        if not is_available: self.dl_bubble_detector_checkbox.setChecked(False)

    def on_dl_provider_changed(self, provider):
        self.dl_model_file_combo.clear()
        if provider == "Kitsumed":
            self.dl_model_file_combo.addItems(['model_dynamic.onnx', 'model.pt'])
        elif provider == "Ogkalu":
            self.dl_model_file_combo.addItems(['comic-speech-bubble-detector.pt'])
        self.on_dl_detector_state_changed(self.dl_bubble_detector_checkbox.checkState())

    def on_ai_model_changed(self, text):
        self.update_usage_display(); self.check_limits_and_update_ui()

    # [DIUBAH] Mengambil nama model dan provider dari combo box
    def get_selected_model_name(self):
        selected_text = self.ai_model_combo.currentText()
        if not selected_text:
            return None, None
        
        # Ekstrak provider dan nama tampilan dari format "[Provider] Display Name"
        match = re.match(r"\[(.*?)\] (.*)", selected_text)
        if not match:
            return None, None
            
        provider_name = match.group(1)
        display_name = match.group(2)

        if provider_name in self.AI_PROVIDERS:
            for model_id, properties in self.AI_PROVIDERS[provider_name].items():
                if properties['display'] == display_name:
                    return provider_name, model_id
        return None, None

    def on_batch_mode_changed(self, state):
        is_checked = (state == Qt.Checked)
        self.process_batch_button.setVisible(is_checked)
        if not is_checked and self.batch_processing_queue:
            reply = QMessageBox.question(self, 'Clear Batch Queue?',
                                           f"You have {len(self.batch_processing_queue)} items in the batch. Do you want to process them now? \n\nChoosing 'No' will discard them.",
                                           QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.Yes: self.start_batch_processing()
            else: self.batch_processing_queue.clear(); self.update_batch_button_text()

    def add_to_batch_queue(self, job):
        self.batch_processing_queue.append(job); self.update_batch_button_text()
        self.statusBar().showMessage(f"Added to batch. Queue has {len(self.batch_processing_queue)} items.")
        if len(self.batch_processing_queue) >= self.BATCH_SIZE_LIMIT:
            self.statusBar().showMessage(f"Batch limit of {self.BATCH_SIZE_LIMIT} reached. Processing automatically...")
            self.start_batch_processing()

    def update_batch_button_text(self):
        count = len(self.batch_processing_queue)
        self.process_batch_button.setText(f"Process Batch Now ({count} items)")
        self.process_batch_button.setEnabled(count > 0)

    def start_batch_processing(self):
        if not self.batch_processing_queue: return
        if self.batch_processor_thread and self.batch_processor_thread.isRunning():
            QMessageBox.warning(self, "Busy", "A batch is already being processed."); return

        self.statusBar().showMessage(f"Starting to process batch of {len(self.batch_processing_queue)} items...")

        queue_to_process = self.batch_processing_queue[:]; self.batch_processing_queue.clear(); self.update_batch_button_text()

        settings = self.get_current_settings()
        self.batch_processor_thread = QThread()
        self.batch_processor_worker = BatchProcessorWorker(self, queue_to_process, settings)
        self.batch_processor_worker.moveToThread(self.batch_processor_thread)
        self.batch_processor_worker.signals.batch_job_complete.connect(self.on_queue_job_complete) # Re-use the single job complete handler
        self.batch_processor_worker.signals.batch_finished.connect(self.on_api_batch_finished)
        self.batch_processor_worker.signals.error.connect(self.on_worker_error)
        self.batch_processor_thread.started.connect(self.batch_processor_worker.run)
        self.batch_processor_thread.finished.connect(self.batch_processor_thread.deleteLater)
        self.batch_processor_thread.start()

    def on_api_batch_finished(self):
        self.statusBar().showMessage("Batch processing finished.", 5000)
        self.batch_processor_thread.quit()

    def split_extended_bubbles(self, detections, split_threshold=2.5):
        new_detections = []
        for item in detections:
            poly = item['polygon']
            bbox = poly.boundingRect()
            if bbox.width() <= 0 or bbox.height() <= 0: continue
            aspect_ratio = bbox.width() / bbox.height()

            if aspect_ratio > split_threshold:
                mid_x = bbox.left() + bbox.width() // 2
                poly1 = QPolygon(QRect(bbox.left(), bbox.top(), bbox.width() // 2, bbox.height()))
                poly2 = QPolygon(QRect(mid_x, bbox.top(), bbox.width() // 2, bbox.height()))
                new_detections.append({'polygon': poly1, 'text': None}) # Teks akan di-OCR ulang
                new_detections.append({'polygon': poly2, 'text': None})
            elif (1 / aspect_ratio) > split_threshold:
                mid_y = bbox.top() + bbox.height() // 2
                poly1 = QPolygon(QRect(bbox.left(), bbox.top(), bbox.width(), bbox.height() // 2))
                poly2 = QPolygon(QRect(bbox.left(), mid_y, bbox.width(), bbox.height() // 2))
                new_detections.append({'polygon': poly1, 'text': None})
                new_detections.append({'polygon': poly2, 'text': None})
            else:
                new_detections.append(item)
        return new_detections

    def start_interactive_batch_detection(self):
        if not self.image_files:
            QMessageBox.warning(self, "No Files Loaded", "Please load a folder first to use this feature.")
            return

        if self.detection_thread and self.detection_thread.isRunning():
            QMessageBox.warning(self, "Busy", "A detection process is already running.")
            return
        
        # [DIUBAH] Menggunakan mode deteksi yang dipilih user
        detection_mode = "Text" if self.text_detect_radio.isChecked() else "Bubble"

        reply = QMessageBox.question(self, f'Confirm Full {detection_mode} Detection',
                                       f"This will detect {detection_mode.lower()}s in all {len(self.image_files)} files in the current folder. This may take a while.\n\nDo you want to continue?",
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.No: return

        self.detected_items_map.clear()
        self.set_ui_for_detection(True)

        settings = self.get_current_settings()
        self.detection_thread = QThread()
        self.detection_worker = AutoDetectorWorker(self, self.image_files, settings, detection_mode)
        self.detection_worker.moveToThread(self.detection_thread)
        self.detection_worker.signals.detection_complete.connect(self.on_detection_complete)
        self.detection_worker.signals.overall_progress.connect(self.update_overall_progress)
        self.detection_worker.signals.error.connect(self.on_worker_error)
        self.detection_worker.signals.finished.connect(self.on_detection_finished)
        self.detection_thread.started.connect(self.detection_worker.run)
        self.detection_thread.start()

    def on_detection_complete(self, image_path, detections):
        self.detected_items_map[image_path] = detections
        current_key = self._resolve_detection_key(self.get_current_data_key())
        if current_key == image_path:
            self.image_label.set_detected_items(detections)

    def on_detection_finished(self):
        self.overall_progress_bar.setVisible(False)
        if self.get_current_settings()['auto_split_bubbles']:
            self.statusBar().showMessage("Splitting extended items...", 3000)
            QApplication.processEvents()
            for path, detections in self.detected_items_map.items():
                self.detected_items_map[path] = self.split_extended_bubbles(detections)

        self.statusBar().showMessage("Detection complete. Please review the highlighted areas.", 5000)
        self.set_ui_for_confirmation(True)

    def process_confirmed_detections(self):
        self.statusBar().showMessage("Processing confirmed items...")
        QApplication.processEvents()

        total_items = sum(len(items) for items in self.detected_items_map.values())
        if total_items == 0:
            QMessageBox.information(self, "No Items", "No items were confirmed for processing.")
            self.cancel_interactive_batch()
            return

        settings = self.get_current_settings()
        # Paksa AI-only untuk batch processing agar lebih cepat & murah
        settings['use_ai_only_translate'] = True
        settings['use_deepl_only_translate'] = False

        # Simpan halaman saat ini untuk kembali nanti
        current_image_path = self.current_image_path
        current_pdf_page = self.current_pdf_page

        try:
            for image_path, detections in self.detected_items_map.items():
                # Muat gambar untuk halaman ini jika berbeda dengan yang sedang aktif
                if image_path != self.get_current_data_key():
                    # Untuk file gambar biasa
                    if not image_path.lower().endswith('.pdf'):
                        if image_path != self.current_image_path:
                            # Simpan data halaman saat ini
                            current_key = self.get_current_data_key()
                            if current_key:
                                self.all_typeset_data[current_key] = {
                                    'areas': self.typeset_areas[:], 
                                    'redo': self.redo_stack[:]
                                }
                            
                            # Muat gambar baru
                            self.current_image_path = image_path
                            self.load_image(image_path)
                    # Untuk PDF (handle khusus)
                    elif '::page::' in image_path:
                        # Ekstrak path dan page number
                        path_part, page_str = image_path.split('::page::')
                        page_num = int(page_str)
                        
                        # Muat halaman PDF yang sesuai
                        if self.pdf_document and self.pdf_document.name == path_part:
                            self.load_pdf_page(page_num)
                        else:
                            # Jika PDF belum dimuat, muat dulu
                            self.load_item(path_part)
                            self.load_pdf_page(page_num)

                # Proses setiap deteksi untuk halaman ini
                for item in detections:
                    polygon = item['polygon']
                    text = item['text'] # Bisa None jika dari Bubble Detect
                    self.process_confirmed_polygon(polygon, pre_detected_text=text)
                    
        except Exception as e:
            self.on_worker_error(f"Error processing batch: {e}")
        finally:
            # Kembali ke halaman asal
            try:
                if current_image_path != self.get_current_data_key():
                    if current_image_path.lower().endswith('.pdf') and current_pdf_page != -1:
                        self.load_pdf_page(current_pdf_page)
                    else:
                        self.load_item(current_image_path)
            except:
                pass

        # Worker sudah mulai dari process_confirmed_polygon, jadi kita hanya perlu membersihkan UI
        self.cancel_interactive_batch()

    def cancel_interactive_batch(self):
        if self.detection_worker: self.detection_worker.cancel()
        if self.detection_thread: self.detection_thread.quit(); self.detection_thread.wait()

        self.detection_thread = None; self.detection_worker = None
        self.detected_items_map.clear(); self.image_label.clear_detected_items()
        self.set_ui_for_detection(False); self.set_ui_for_confirmation(False)
        self.statusBar().showMessage("Batch detection cancelled.", 3000)

    def remove_detected_item(self, index_to_remove):
        current_key = self.get_current_data_key()
        resolved_key = self._resolve_detection_key(current_key) or current_key
        if resolved_key in self.detected_items_map and 0 <= index_to_remove < len(self.detected_items_map[resolved_key]):
            del self.detected_items_map[resolved_key][index_to_remove]
            if self.detected_items_map.get(resolved_key):
                self.image_label.set_detected_items(self.detected_items_map[resolved_key])
            else:
                self.image_label.clear_detected_items()
            self.update_confirmation_button_text()

    def set_ui_for_detection(self, is_detecting):
        self.batch_process_button.setEnabled(not is_detecting)
        self.file_list_widget.setEnabled(not is_detecting)
        self.prev_button.setEnabled(not is_detecting); self.next_button.setEnabled(not is_detecting)
        self.cancel_detection_button.setVisible(is_detecting)
        self.overall_progress_bar.setVisible(is_detecting)
        if is_detecting: self.overall_progress_bar.setValue(0); self.statusBar().showMessage("Starting detection...")
        else: self.overall_progress_bar.setVisible(False)

    def set_ui_for_confirmation(self, is_confirming):
        self.is_in_confirmation_mode = is_confirming
        self.batch_process_button.setEnabled(not is_confirming)
        self.confirm_items_button.setVisible(is_confirming)
        if is_confirming: self.update_confirmation_button_text()
        self._refresh_detection_overlay()

    def _resolve_detection_key(self, key):
        if not key:
            return None
        if key in self.detected_items_map:
            return key
        if "::page::" in key:
            base_key = key.split('::page::')[0]
            if base_key in self.detected_items_map:
                return base_key
        return None

    def _refresh_detection_overlay(self):
        if not self.image_label:
            return
        if not self.is_in_confirmation_mode:
            self.image_label.clear_detected_items()
            return
        current_key = self._resolve_detection_key(self.get_current_data_key())
        if current_key and current_key in self.detected_items_map:
            self.image_label.set_detected_items(self.detected_items_map[current_key])
        else:
            self.image_label.clear_detected_items()

    def update_confirmation_button_text(self):
        total_items = sum(len(items) for items in self.detected_items_map.values())
        self.confirm_items_button.setText(f"Confirm & Process ({total_items}) Items")

    def open_batch_save_dialog(self):
        if not self.image_files:
            QMessageBox.warning(self, "No Folder Loaded", "Please load a folder to use the batch save feature.")
            return

        dialog = BatchSaveDialog(self.image_files, self)
        if dialog.exec_() == QDialog.Accepted:
            files_to_save = dialog.get_selected_files()
            if files_to_save: self.execute_batch_save(files_to_save)
            else: self.statusBar().showMessage("No files were selected to save.", 3000)

    def execute_batch_save(self, files_to_save):
        if self.batch_save_thread and self.batch_save_thread.isRunning():
            QMessageBox.warning(self, "Busy", "A batch save process is already running.")
            return

        self.overall_progress_bar.setVisible(True); self.overall_progress_bar.setValue(0)
        self.statusBar().showMessage("Starting batch save...")

        self.batch_save_thread = QThread()
        self.batch_save_worker = BatchSaveWorker(self, files_to_save)
        self.batch_save_worker.moveToThread(self.batch_save_thread)
        self.batch_save_worker.signals.progress.connect(self.update_overall_progress)
        self.batch_save_worker.signals.file_saved.connect(self.on_batch_file_saved)
        self.batch_save_worker.signals.error.connect(self.on_worker_error)
        self.batch_save_worker.signals.finished.connect(self.on_batch_save_finished)
        self.batch_save_thread.started.connect(self.batch_save_worker.run)
        self.batch_save_thread.start()

    def on_batch_file_saved(self, file_path): pass

    def on_batch_save_finished(self):
        self.statusBar().showMessage("Batch save complete.", 5000)
        self.overall_progress_bar.setVisible(False)
        self.batch_save_thread.quit(); self.batch_save_thread.wait()
        QMessageBox.information(self, "Batch Save Complete", "All selected files have been saved.")

    def check_if_saved(self, file_path):
        path_part, ext = os.path.splitext(file_path)
        save_path = f"{path_part}_typeset.png"
        return os.path.exists(save_path)
    
    # --- Metode Baru untuk Bubble Finder ---
    def find_bubble_in_rect(self, selection_rect):
        """Menjalankan deteksi bubble pada area yang dipilih pengguna."""
        if not self.current_image_pil:
            return
        
        settings = self.get_current_settings()
        if not settings.get('use_dl_detector'):
            QMessageBox.warning(self, "Detector Disabled", "Please enable 'Gunakan DL Model untuk Bubble' in the Cleanup tab to use this feature.")
            self.image_label.clear_selection()
            return
            
        self.statusBar().showMessage(f"Finding bubble with {settings['dl_provider']} model...")
        QApplication.processEvents()
        
        try:
            # Crop image
            cropped_pil = self.current_image_pil.crop((
                selection_rect.left(), selection_rect.top(),
                selection_rect.right(), selection_rect.bottom()
            ))
            cropped_cv = cv2.cvtColor(np.array(cropped_pil), cv2.COLOR_RGB2BGR)

            # Run inference on the crop
            mask = self.detect_bubble_with_dl_model(cropped_cv, settings)
            
            if mask is None or cv2.countNonZero(mask) == 0:
                self.statusBar().showMessage("No bubble found in the selected area.", 3000)
                self.image_label.clear_selection()
                return

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                self.statusBar().showMessage("No valid bubble contour found.", 3000)
                self.image_label.clear_selection()
                return

            # Ambil kontur terbesar
            best_contour = max(contours, key=cv2.contourArea)

            # Geser koordinat poligon kembali ke sistem koordinat gambar penuh
            offset = selection_rect.topLeft()
            full_image_polygon = QPolygon([QPoint(p[0][0] + offset.x(), p[0][1] + offset.y()) for p in best_contour])

            # Tampilkan untuk konfirmasi
            self.image_label.set_pending_item(full_image_polygon)
            self.statusBar().showMessage("Bubble found! Right-click to confirm, Middle-click to cancel.", 5000)

        except Exception as e:
            self.on_worker_error(f"Error during interactive bubble detection: {e}")
            self.image_label.clear_selection()

    def confirm_pending_item(self, polygon):
        """Memproses item yang telah dikonfirmasi oleh pengguna."""
        self.statusBar().showMessage("Item confirmed. Processing for OCR...", 3000)
        self.process_confirmed_polygon(polygon)


    def closeEvent(self, event):
        self.cancel_interactive_batch()
        if self.batch_save_worker: self.batch_save_worker.cancel()
        if self.batch_save_thread: self.batch_save_thread.quit(); self.batch_save_thread.wait()
        for worker_id, (thread, worker) in list(self.worker_pool.items()):
            worker.stop(); thread.quit(); thread.wait()
        if hasattr(self, 'batch_processor_thread') and self.batch_processor_thread:
            self.batch_processor_thread.quit(); self.batch_processor_thread.wait()
        if hasattr(self, 'exchange_rate_thread') and self.exchange_rate_thread and self.exchange_rate_thread.isRunning():
            self.exchange_rate_thread.quit(); self.exchange_rate_thread.wait()
        self.save_usage_data(); event.accept()
    
    # ===================================================================
    # ======================= OCR & DETECT METHODS ======================
    # ===================================================================

    def detect_text_with_ocr_engine(self, cv_image, settings):
        """
        [DIUBAH] Mendeteksi teks, lalu menggabungkannya menjadi blok yang koheren.
        """
        engine = settings['ocr_engine']
        lang_code = settings['ocr_lang']
        raw_results = []

        try:
            if engine == 'DocTR':
                # Perbaikan implementasi DocTR
                if not self.doctr_predictor: 
                    return []
                
                # Convert BGR to RGB
                rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                
                # Predict
                result = self.doctr_predictor([rgb_image])
                
                # Extract text and bounding boxes
                for page in result.pages:
                    for block in page.blocks:
                        for line in block.lines:
                            line_text = ' '.join([word.value for word in line.words])
                            
                            # Get bounding box coordinates
                            geometry = line.geometry
                            x1, y1 = geometry[0][0] * cv_image.shape[1], geometry[0][1] * cv_image.shape[0]
                            x2, y2 = geometry[1][0] * cv_image.shape[1], geometry[1][1] * cv_image.shape[0]
                            
                            points = [
                                QPoint(int(x1), int(y1)),
                                QPoint(int(x2), int(y1)),
                                QPoint(int(x2), int(y2)),
                                QPoint(int(x1), int(y2))
                            ]
                            raw_results.append((line_text, QPolygon(points)))
                
                return self._merge_text_boxes_to_blocks(raw_results, cv_image.shape)
            
            elif engine in ['RapidOCR', 'PaddleOCR', 'EasyOCR']:
                # Engine yang mengembalikan bounding box per baris/kata
                if engine == 'RapidOCR':
                    if not self.rapid_ocr_reader: return []
                    ocr_result, _ = self.rapid_ocr_reader(cv_image)
                    if ocr_result:
                        for box_info in ocr_result:
                            points = [QPoint(int(p[0]), int(p[1])) for p in box_info[0]]
                            raw_results.append((box_info[1], QPolygon(points)))
                elif engine == 'PaddleOCR':
                    if not self.paddle_ocr_reader: return []
                    ocr_result = self.paddle_ocr_reader.ocr(cv_image, cls=True)
                    if ocr_result and ocr_result[0]:
                        for line in ocr_result[0]:
                            points = [QPoint(int(p[0]), int(p[1])) for p in line[0]]
                            raw_results.append((line[1][0], QPolygon(points)))
                elif engine == 'EasyOCR':
                    if not self.easyocr_reader: return []
                    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                    ocr_result = self.easyocr_reader.readtext(gray, detail=1)
                    for (bbox, text, prob) in ocr_result:
                        points = [QPoint(int(p[0]), int(p[1])) for p in bbox]
                        raw_results.append((text, QPolygon(points)))
                
                return self._merge_text_boxes_to_blocks(raw_results, cv_image.shape)

            else: # Tesseract sebagai default
                return self._detect_text_with_tesseract(cv_image, lang_code)

        except Exception as e:
            print(f"Error during text detection with {engine}: {e}")

        return []
    
    # [BARU] Logika penggabungan teks untuk Tesseract
    def _detect_text_with_tesseract(self, cv_image, lang_code):
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        data = pytesseract.image_to_data(gray, lang=lang_code, config='--oem 1 --psm 3', output_type=pytesseract.Output.DICT)
        
        blocks = {}
        # Kumpulkan semua kata yang valid ke dalam bloknya masing-masing
        for i in range(len(data['text'])):
            conf = int(data['conf'][i])
            text = data['text'][i].strip()
            if conf > 40 and text:
                block_num = data['block_num'][i]
                if block_num not in blocks:
                    blocks[block_num] = []
                
                blocks[block_num].append({
                    'text': text,
                    'rect': QRect(data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                })
        
        final_results = []
        # Gabungkan kata-kata dalam setiap blok menjadi satu kesatuan
        for block_num, words in blocks.items():
            if not words: continue
            
            full_text = ' '.join(w['text'] for w in words)
            
            # Buat bounding box gabungan untuk seluruh blok
            combined_rect = QRect()
            for w in words:
                if combined_rect.isEmpty():
                    combined_rect = w['rect']
                else:
                    combined_rect = combined_rect.united(w['rect'])
            
            final_results.append((full_text, QPolygon(combined_rect)))
            
        return final_results


    # [BARU] Algoritma untuk menggabungkan kotak teks yang terdeteksi
    def _merge_text_boxes_to_blocks(self, boxes, image_shape):
        if not boxes:
            return []

        # Konversi ke format yang lebih mudah diolah dan hitung tinggi rata-rata
        lines = []
        total_height = 0
        for text, poly in boxes:
            rect = poly.boundingRect()
            if rect.width() > 0 and rect.height() > 0:
                lines.append({'text': text, 'rect': rect, 'poly': poly, 'merged': False})
                total_height += rect.height()
        
        if not lines: return []
        avg_height = total_height / len(lines)
        
        # Urutkan berdasarkan posisi vertikal
        lines.sort(key=lambda l: l['rect'].top())
        
        # Gabungkan baris-baris yang berdekatan menjadi blok
        blocks = []
        for i in range(len(lines)):
            if lines[i]['merged']:
                continue
            
            current_block_lines = [lines[i]]
            lines[i]['merged'] = True
            
            for j in range(i + 1, len(lines)):
                if lines[j]['merged']:
                    continue
                
                # Cek jarak vertikal antara baris saat ini (i) dan baris kandidat (j)
                last_line_rect = current_block_lines[-1]['rect']
                candidate_rect = lines[j]['rect']
                
                vertical_gap = candidate_rect.top() - last_line_rect.bottom()
                
                # Cek overlap horizontal
                is_horizontally_aligned = (last_line_rect.left() < candidate_rect.right() and
                                           last_line_rect.right() > candidate_rect.left())
                
                # Kondisi penggabungan: jarak vertikal kecil dan ada overlap horizontal
                if vertical_gap < (avg_height * 0.8) and is_horizontally_aligned:
                    current_block_lines.append(lines[j])
                    lines[j]['merged'] = True

            blocks.append(current_block_lines)

        # Finalisasi blok: gabungkan teks dan poligon
        final_results = []
        for block_lines in blocks:
            if not block_lines: continue
            
            # Urutkan baris dalam blok secara vertikal
            block_lines.sort(key=lambda l: l['rect'].top())
            
            full_text = ' '.join(l['text'] for l in block_lines)
            
            combined_poly = QPolygon()
            for line in block_lines:
                combined_poly = combined_poly.united(line['poly'])

            final_results.append((full_text, combined_poly))
            
        return final_results


    def perform_ocr(self, image_to_process, settings: dict) -> str:
        """
        [DIUBAH] Menjalankan OCR pada gambar yang diberikan berdasarkan pengaturan.
        """
        ocr_engine = settings['ocr_engine']
        orientation = settings['orientation']
        ocr_lang = settings.get('ocr_lang', 'ja')
        raw_text = ""

        # Penyesuaian rotasi sesuai orientasi
        h, w = image_to_process.shape[:2]
        if ocr_engine == "Manga-OCR":
            if orientation == "Vertical" and w > h:
                image_to_process = cv2.rotate(image_to_process, cv2.ROTATE_90_CLOCKWISE)
            elif orientation == "Horizontal" and h > w:
                image_to_process = cv2.rotate(image_to_process, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Jalankan OCR sesuai engine
        if ocr_engine == "Manga-OCR":
            if self.manga_ocr_reader:
                pil_img = Image.fromarray(cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB))
                raw_text = self.manga_ocr_reader(pil_img)
            else:
                return "[ERROR: Manga-OCR not installed or initialized]"

        elif ocr_engine == "EasyOCR":
            if not self.easyocr_reader:
                return "[ERROR: EasyOCR not initialized. Select language and apply.]"
            gray = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2GRAY)
            results = self.easyocr_reader.readtext(gray, detail=0, paragraph=True)
            raw_text = "\n".join(results)

        elif ocr_engine == "PaddleOCR":
            if not self.paddle_ocr_reader:
                return f"[ERROR: PaddleOCR for '{ocr_lang}' not ready. Please select it in the UI first.]"
            try:
                # PaddleOCR mengharapkan gambar BGR
                # Gunakan predict() bukan ocr(), dan hapus parameter cls
                result = self.paddle_ocr_reader.predict(image_to_process)
                
                # Ekstrak teks dari hasil - format hasil mungkin berbeda
                texts = []
                if result and result[0]:
                    for line in result[0]:
                        if line and len(line) >= 2:
                            text = line[1][0]  # Teks ada di index 1, subindex 0
                            texts.append(text)
                
                raw_text = "\n".join(texts)
            except Exception as e:
                print(f"Error during PaddleOCR execution: {e}")
                raw_text = "[PADDLEOCR RUNTIME ERROR]"

        elif ocr_engine == "DocTR":
            if not self.doctr_predictor: 
                return "[ERROR: DocTR not initialized]"
            
            try:
                # Convert BGR to RGB
                rgb_image = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB)
                
                # Predict
                result = self.doctr_predictor([rgb_image])
                
                # Extract text
                texts = []
                for page in result.pages:
                    for block in page.blocks:
                        for line in block.lines:
                            line_text = ' '.join([word.value for word in line.words])
                            texts.append(line_text)
                
                raw_text = "\n".join(texts)
            except Exception as e:
                print(f"Error during DocTR execution: {e}")
                raw_text = "[DOCTR RUNTIME ERROR]"

        elif ocr_engine == "RapidOCR":
            if not self.rapid_ocr_reader: return "[ERROR: RapidOCR not initialized]"
            result, _ = self.rapid_ocr_reader(image_to_process)
            if result:
                raw_text = "\n".join([res[1] for res in result])

        elif ocr_engine == "Tesseract":
            try:
                gray = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2GRAY)
                psm = 5 if orientation == "Vertical" else 6
                custom_config = f'--oem 1 --psm {psm}'
                raw_text = pytesseract.image_to_string(gray, lang=ocr_lang, config=custom_config).strip()
            except pytesseract.TesseractError as e:
                print(f"Tesseract Error in Worker: {e}")
                return f"[TESSERACT ERROR: {e}]"

        return raw_text

class BatchSaveDialog(QDialog):
    save_requested = pyqtSignal(list)

    def __init__(self, all_files, parent=None):
        super().__init__(parent)
        self.main_app = parent
        self.all_files = all_files
        self.setWindowTitle("Batch Save Images")
        self.setMinimumSize(600, 700)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        selection_layout = QGridLayout()
        presets = [5, 10, 15, 20, 25]
        for i, num in enumerate(presets):
            btn = QPushButton(f"Select Next {num} Unsaved")
            btn.clicked.connect(lambda _, n=num: self.select_next_unsaved(n))
            selection_layout.addWidget(btn, 0, i)

        select_all_btn = QPushButton("Select All Unsaved"); select_all_btn.clicked.connect(self.select_all_unsaved); selection_layout.addWidget(select_all_btn, 1, 0, 1, 2)
        deselect_all_btn = QPushButton("Deselect All"); deselect_all_btn.clicked.connect(self.deselect_all); selection_layout.addWidget(deselect_all_btn, 1, 2, 1, 3)
        layout.addLayout(selection_layout)

        self.list_widget = QListWidget(); self.populate_list(); layout.addWidget(self.list_widget)

        button_layout = QHBoxLayout()
        self.save_button = QPushButton("Save Selected"); self.save_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel"); self.cancel_button.clicked.connect(self.reject)
        button_layout.addStretch(); button_layout.addWidget(self.cancel_button); button_layout.addWidget(self.save_button)
        layout.addLayout(button_layout)

    def populate_list(self):
        self.list_widget.clear()
        for file_path in self.all_files:
            if "_typeset" in file_path.lower(): continue
            item = QListWidgetItem(os.path.basename(file_path))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)

            is_saved = self.main_app.check_if_saved(file_path)
            if is_saved:
                item.setCheckState(Qt.Unchecked); item.setForeground(QColor("gray")); item.setText(f"{os.path.basename(file_path)} [SAVED]")
            else:
                item.setCheckState(Qt.Unchecked)

            item.setData(Qt.UserRole, file_path)
            self.list_widget.addItem(item)

    def select_next_unsaved(self, count):
        self.deselect_all()
        selected_count = 0
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            file_path = item.data(Qt.UserRole)
            if not self.main_app.check_if_saved(file_path):
                item.setCheckState(Qt.Checked); selected_count += 1
                if selected_count >= count: break

    def select_all_unsaved(self):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if not self.main_app.check_if_saved(item.data(Qt.UserRole)): item.setCheckState(Qt.Checked)
            else: item.setCheckState(Qt.Unchecked)

    def deselect_all(self):
        for i in range(self.list_widget.count()): self.list_widget.item(i).setCheckState(Qt.Unchecked)

    def get_selected_files(self):
        selected = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.Checked: selected.append(item.data(Qt.UserRole))
        return selected

if __name__ == "__main__":
    app = QApplication(sys.argv)

    try: import fitz
    except ImportError: QMessageBox.critical(None, "Dependency Missing", "PyMuPDF not installed. 'pip install PyMuPDF'."); sys.exit()

    if not DEEPL_API_KEY or "your_deepl_key_here" in DEEPL_API_KEY: QMessageBox.warning(None, "DeepL API Key Missing", "Please provide your valid DeepL API key in 'config.ini'.")
    if not GEMINI_API_KEY or "your_gemini_key_here" in GEMINI_API_KEY: QMessageBox.warning(None, "Gemini API Key Missing", "Please provide your valid Gemini API key in 'config.ini'.")

    try: pytesseract.get_tesseract_version()
    except Exception: QMessageBox.warning(None, "Tesseract Not Found", f"Tesseract not found at: {TESSERACT_PATH}\nPlease install it or correct the path in config.ini.")

    if not MangaOcr: QMessageBox.warning(None, "Manga-OCR Not Found", "'pip install manga-ocr' to enable the Manga-OCR engine.")
    if not onnxruntime: QMessageBox.warning(None, "ONNX Runtime Not Found", "'pip install onnxruntime' to enable some DL detectors.")
    if not paddleocr: QMessageBox.warning(None, "PaddleOCR Not Found", "'pip install paddleocr paddlepaddle' to enable the PaddleOCR engine.")
    if not YOLO: QMessageBox.warning(None, "Ultralytics Not Found", "'pip install ultralytics' to enable some DL detectors.")
    # [BARU] Peringatan untuk dependensi inpainting
    if not lama_cleaner: QMessageBox.warning(None, "Lama-Cleaner Not Found", "'pip install lama-cleaner' to enable advanced inpainting models.")

    window = MangaOCRApp()
    window.showMaximized()
    sys.exit(app.exec_())
