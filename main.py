# Manga OCR & Typeset Tool v13.0.1
# Ditingkatkan oleh Gemini

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
from datetime import date

# ==============================
# 📦 Library pihak ketiga
# ==============================
import numpy as np
import cv2
import pytesseract
import requests
import easyocr
import fitz  # from PyMuPDF
import google.generativeai as genai
from PIL import Image
from PIL.ImageQt import ImageQt

# ==============================
# 📦 PyQt5 (dibagi per kategori)
# ==============================
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QFileDialog, QTextEdit, QScrollArea, QComboBox, QMessageBox,
    QProgressBar, QShortcut, QListWidget, QListWidgetItem, QColorDialog, QFontDialog,
    QLineEdit, QAction, QDialog, QCheckBox, QStatusBar, QAbstractItemView, QSpinBox,
    QTabWidget, QGroupBox, QGridLayout, QFrame, QSplitter
)
from PyQt5.QtGui import (
    QPixmap, QPainter, QPen, QColor, QFont, QKeySequence, QPolygon,
    QPainterPath, QPolygonF, QImage, QIcon
)
from PyQt5.QtCore import (
    Qt, QRect, QPoint, pyqtSignal, QTimer, QThread, QObject,
    QFileSystemWatcher, QRectF, QMutex
)

# ==============================
# 📦 Dependensi opsional (DL)
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
        print(f"Peringatan: Pustaka '{import_name}' tidak ditemukan. {pip_msg}", file=sys.stderr)
        return None if alias is None else alias

onnxruntime = check_dependency("onnxruntime", "onnxruntime")
paddleocr = check_dependency("paddleocr", "paddleocr paddlepaddle-gpu")
YOLO = None
try:
    from ultralytics import YOLO as YOLO
except ImportError:
    print("Peringatan: Pustaka 'ultralytics' tidak ditemukan. (pip install ultralytics)", file=sys.stderr)

# --- Fungsi untuk menangani config.ini ---
def create_default_config(config_path: str = "config.ini"):
    """
    Membuat file config.ini default jika belum ada.
    """
    config = configparser.ConfigParser()
    config['API'] = {
        'DEEPL_KEY': 'your_deepl_key_here',
        'GEMINI_KEY': 'your_gemini_key_here'
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
    finished = pyqtSignal()              # Sinyal jika proses selesai
    error = pyqtSignal(str)              # Sinyal jika terjadi error, membawa pesan error
    progress = pyqtSignal(int, str)    # Sinyal progress dengan persentase dan pesan status


# Sinyal khusus untuk deteksi bubble (misalnya balon teks komik/manga)
class BubbleDetectorSignals(WorkerSignals):
    detection_complete = pyqtSignal(str, list)  # Sinyal jika deteksi selesai, membawa path gambar & list QPolygon
    overall_progress = pyqtSignal(int, str)      # Sinyal progress keseluruhan (persentase & status)


# Sinyal khusus untuk pemrosesan antrian pekerjaan
class QueueProcessorSignals(WorkerSignals):
    job_complete = pyqtSignal(str, object, str, str)  # image_path, new_area, original_text, translated_text
    queue_status = pyqtSignal(int)               # Sinyal jumlah item dalam antrian
    worker_finished = pyqtSignal(int)            # Sinyal saat 1 worker selesai (dengan ID worker)
    status_update = pyqtSignal(str)              # Sinyal update status bar (aman dari thread)


# Sinyal khusus untuk pemrosesan batch (sekumpulan pekerjaan)
class BatchProcessorSignals(WorkerSignals):
    batch_job_complete = pyqtSignal(str, object)  # Sinyal jika 1 job batch selesai
    batch_finished = pyqtSignal()                 # Sinyal jika semua batch selesai

# Sinyal untuk worker saran glosarium
class GlossarySuggestionSignals(WorkerSignals):
    suggestions_ready = pyqtSignal(list)  # Mengirimkan daftar saran (list of dicts)


# Sinyal khusus untuk penyimpanan hasil batch
class BatchSaveSignals(WorkerSignals):
    file_saved = pyqtSignal(str)                  # Sinyal jika file berhasil disimpan


# Kelas untuk menyimpan hasil teks dengan informasi apakah ada isinya atau tidak
class EnhancedResult:
    def __init__(self, text: str):
        self.text = text
        self.parts = bool(text)                      # True jika ada teks, False jika kosong

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

                original_text, translated_text = self.process_job(cropped_cv_img, settings)

                if translated_text:
                    new_area = TypesetArea(
                        job['rect'], translated_text,
                        settings['font'], settings['color'],
                        job.get('polygon')
                    )
                    self.signals.job_complete.emit(image_path, new_area, original_text, translated_text)

            except Exception as e:
                print(f"Error in Worker {self.worker_id}: {e}")
                self.signals.error.emit(str(e))
                continue

        print(f"Worker {self.worker_id} finished.")
        self.signals.worker_finished.emit(self.worker_id)

    # Menentukan pipeline mana yang akan dipakai (standar / enhanced)
    def process_job(self, cropped_cv_img, settings: dict):
        return (
            self.run_enhanced_pipeline(cropped_cv_img, settings)
            if settings.get('enhanced_pipeline')
            else self.run_standard_pipeline(cropped_cv_img, settings)
        )

    # Melakukan OCR sesuai engine yang dipilih
    def perform_ocr(self, image_to_process, settings: dict) -> str:
        ocr_engine = settings['ocr_engine']
        orientation = settings['orientation']
        raw_text = ""

        # Penyesuaian rotasi sesuai orientasi
        if ocr_engine == "Manga-OCR":
            h, w = image_to_process.shape[:2]
            if orientation == "Vertical" and w > h:
                image_to_process = cv2.rotate(image_to_process, cv2.ROTATE_90_CLOCKWISE)
            elif orientation == "Horizontal" and h > w:
                image_to_process = cv2.rotate(image_to_process, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Jalankan OCR sesuai engine
        if ocr_engine == "Manga-OCR":
            pil_img = Image.fromarray(cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB))
            raw_text = self.main_app.manga_ocr_reader(pil_img)

        elif ocr_engine == "EasyOCR":
            gray = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2GRAY)
            results = self.main_app.easyocr_reader.readtext(
                gray, detail=0, paragraph=True, vertical=(orientation == "Vertical")
            )
            raw_text = "\n".join(results)

        elif ocr_engine == "PaddleOCR" and self.main_app.paddle_ocr_reader:
            lang_map = {"Japanese": "japan", "English": "en", "Chinese": "ch", "Korean": "korean"}
            lang_code = lang_map.get(settings['ocr_lang'], 'japan')

            # Re-initialize if language changed
            if not hasattr(self.main_app, 'paddle_lang') or self.main_app.paddle_lang != lang_code:
                self.main_app.initialize_paddle_ocr(lang_code)

            result = self.main_app.paddle_ocr_reader.ocr(image_to_process, cls=True)
            if result and result[0]:
                lines = [line[1][0] for line in result[0]]
                raw_text = "\n".join(lines)

        else:  # Default: Tesseract
            gray = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2GRAY)
            lang_map = {"Japanese": "jpn", "English": "eng", "Chinese": "chi_sim", "Korean": "kor", "Chinese Traditional": "chi_tra"}
            tess_lang = lang_map.get(settings.get('ocr_lang'), "jpn")
            psm = 5 if orientation == "Vertical" else 6
            custom_config = f'--oem 1 --psm {psm} -l {tess_lang}'
            raw_text = pytesseract.image_to_string(gray, config=custom_config).strip()

        return raw_text

    # Pipeline standar: OCR → Cleaning → Translate → Naturalize (opsional)
    def run_standard_pipeline(self, cropped_cv_img, settings: dict):
        preprocessed_image, _ = self.main_app.preprocess_for_ocr(cropped_cv_img, settings['orientation'])
        raw_text = self.perform_ocr(preprocessed_image, settings)
        processed_text = self.main_app.clean_and_join_text(raw_text)

        if not processed_text:
            return "", ""

        if settings.get('use_gemini_only_translate'):
            if not self.wait_for_api_slot(settings['ai_model']):
                return processed_text, None

            # Dapatkan hasil terjemahan dari Gemini
            translated_text = self.main_app.translate_with_gemini(processed_text, settings['target_lang'], settings['ai_model'], settings)
            return processed_text, translated_text


        text_for_translation = processed_text
        if settings.get('use_ai'):
            if not self.wait_for_api_slot(settings['ai_model']):
                return processed_text, None
            corrected_text = self.main_app.correct_text_with_gemini(processed_text, settings['ai_model'])
            if corrected_text:
                text_for_translation = corrected_text

        if not text_for_translation:
            return processed_text, ""

        translated_text = self.main_app.translate_text(text_for_translation, settings['target_lang'])

        if settings.get('naturalize_translation') and translated_text:
            if not self.wait_for_api_slot(settings['ai_model']):
                return processed_text, None
            naturalized_text = self.main_app.naturalize_text_with_gemini(translated_text, settings['ai_model'])
            if naturalized_text:
                translated_text = naturalized_text

        return processed_text, translated_text

    # Pipeline enhanced: gabungkan hasil Manga-OCR + Tesseract → Gemini
    def run_enhanced_pipeline(self, cropped_cv_img, settings: dict):
        preprocessed_image, _ = self.main_app.preprocess_for_ocr(cropped_cv_img, "Auto-Detect")

        manga_ocr_text = self.perform_ocr(preprocessed_image, {**settings, 'ocr_engine': 'Manga-OCR'})
        tesseract_text = self.perform_ocr(preprocessed_image, {**settings, 'ocr_engine': 'Tesseract', 'ocr_lang': 'Japanese'})

        original_text = manga_ocr_text if len(manga_ocr_text) > len(tesseract_text) else tesseract_text

        model_name = settings['ai_model']
        if not self.wait_for_api_slot(model_name):
            return original_text, None

        try:
            model = genai.GenerativeModel(model_name)
            prompt_enhancements = self.main_app._build_prompt_enhancements(settings)

            prompt = f"""
You are an expert manga translator. Your task is to accurately translate a text bubble from a Japanese manga into natural, colloquial Indonesian.
Analyze and merge the following two OCR results to deduce the most accurate original text, then provide the translation.
1. Manga-OCR Result: "{manga_ocr_text}"
2. Tesseract OCR Result: "{tesseract_text}"

IMPORTANT RULES:
- IMPORTANT CENSORSHIP RULES:
- If the text contains the word "vagina" or any explicit equivalent, you MUST replace it with the word "meong".
- If the text contains the word "penis" or any explicit equivalent, you MUST replace it with the word "burung".
- After applying these replacements, ensure the final translation is still natural, coherent, and makes sense in the context of the manga. Do not just blindly replace the words; adapt the sentence structure if necessary to maintain flow.
{prompt_enhancements}

Your final output must ONLY be the translated Indonesian text. No explanations, no markdown.
"""
            response = model.generate_content(prompt)

            if response.parts:
                self.main_app.add_api_cost(len(prompt), len(response.text), model_name)
                return original_text, response.text.strip()
            return original_text, "[GEMINI FAILED]"

        except Exception as e:
            print(f"Gemini API call failed in pipeline: {e}")
            return original_text, "[GEMINI ERROR]"


    # Mekanisme tunggu jika API slot penuh (rate limit)
    def wait_for_api_slot(self, model_name: str) -> bool:
        while self.is_running:
            if self.main_app.check_and_increment_usage(model_name):
                return True
            now = time.time()
            wait_sec = 61 - int(time.strftime('%S', time.localtime(now)))
            self.signals.status_update.emit(f"API limit {model_name} tercapai. Tunggu {wait_sec}s...")
            time.sleep(wait_sec)
        return False

    # Hentikan worker
    def stop(self):
        self.is_running = False

class BubbleDetectorWorker(QObject):
    def __init__(self, main_app, file_paths, settings):
        super().__init__()
        self.main_app = main_app
        self.file_paths = file_paths
        self.settings = settings
        self.signals = BubbleDetectorSignals()
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

                # Gunakan model yang dipilih dari settings, bukan hardcoded
                combined_mask = self.main_app.detect_bubble_with_dl_model(cv_image, self.settings)

                if combined_mask is not None:
                    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    polygons = [QPolygon([QPoint(p[0][0], p[0][1]) for p in cnt]) for cnt in contours]
                    self.signals.detection_complete.emit(file_path, polygons)
                else:
                    self.signals.detection_complete.emit(file_path, []) # Kirim list kosong jika gagal

            except Exception as e:
                self.signals.error.emit(f"Error detecting bubbles in {file_path}: {e}")
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
        model_name = self.settings['ai_model']

        # 1. Lakukan OCR untuk semua pekerjaan
        ocr_texts = []
        for job in jobs:
            try:
                preprocessed, _ = self.main_app.preprocess_for_ocr(
                    job['cropped_cv_img'], self.settings['orientation']
                )
                raw_text = self.main_app.perform_ocr_for_batch(preprocessed, self.settings)
                cleaned_text = self.main_app.clean_and_join_text(raw_text)
                ocr_texts.append(cleaned_text)
            except Exception as e:
                ocr_texts.append("")  # agar tetap sinkron dengan urutan job
                self.signals.error.emit(f"OCR failed on {image_path}: {e}")

        # Filter teks kosong
        prompt_lines = [f"{i+1}. {text}" for i, text in enumerate(ocr_texts) if text]
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

IMPORTANT RULES:
- CENSORSHIP: If the text contains "vagina", replace it with "meong". If it contains "penis", replace it with "burung". Ensure the final sentence is still coherent.
{prompt_enhancements}

Snippets to Translate:
{numbered_ocr_text}

Your final output must ONLY be the translated {target_lang} text, with each translation on a new line and correctly numbered.
"""

        # 3. Panggil API sekali untuk batch ini
        if not self.main_app.wait_for_api_slot(model_name):
            return

        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)

            if not response or not response.parts:
                raise Exception("API response was empty or invalid.")

            # Catat penggunaan API
            self.main_app.add_api_cost(len(prompt), len(response.text), model_name)

            # 4. Parsing respons dan petakan kembali ke pekerjaan
            translated_lines = response.text.strip().splitlines()
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
                if translated_text and "[N/A]" not in translated_text:
                    new_area = TypesetArea(
                        job['rect'], translated_text,
                        self.settings['font'], self.settings['color'],
                        job.get('polygon')
                    )
                    # Note: We can't easily get original/translated pair here for glossary suggestion
                    # in batch mode without a more complex mapping.
                    self.signals.batch_job_complete.emit(image_path, new_area)

        except Exception as e:
            self.signals.error.emit(
                f"Failed to process batch for {os.path.basename(image_path)}: {e}"
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

                # [FIX v9.9.2] Konversi PIL.Image ke QPixmap dengan cara yang lebih andal
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

# Worker untuk saran Glosarium
class GlossarySuggestionWorker(QObject):
    def __init__(self, main_app, original_text, translated_text, model_name):
        super().__init__()
        self.main_app = main_app
        self.original_text = original_text
        self.translated_text = translated_text
        self.model_name = model_name
        self.signals = GlossarySuggestionSignals()

    def run(self):
        if not self.original_text or not self.translated_text:
            self.signals.finished.emit()
            return

        try:
            model = genai.GenerativeModel(self.model_name)
            prompt = f"""
You are a linguistic analyst specializing in Japanese manga.
Given the original Japanese text and its Indonesian translation, identify potential glossary terms.
Focus on:
1.  Proper Nouns (Names of people, places, organizations).
2.  Unique Terminology (Special attacks, magic spells, fictional concepts).
3.  Significant Honorifics that should be consistently translated or kept.

Do not suggest common words or phrases. The terms must be specific and important for translation consistency.

- Original Japanese: "{self.original_text}"
- Indonesian Translation: "{self.translated_text}"

Your task is to provide a valid JSON array of potential glossary entries.
The JSON format must be: `[{"source": "Japanese Term", "target": "Indonesian Translation"}]`
If no valid terms are found, return an empty array `[]`.

Your final output must ONLY be the JSON array and nothing else.
"""
            response = model.generate_content(prompt)
            if response.parts:
                self.main_app.add_api_cost(len(prompt), len(response.text), self.model_name)
                # Clean the response text to get only the JSON part
                json_text = response.text.strip()
                match = re.search(r'\[.*\]', json_text, re.DOTALL)
                if match:
                    json_text = match.group(0)
                    suggestions = json.loads(json_text)
                    self.signals.suggestions_ready.emit(suggestions)
                else:
                    self.signals.suggestions_ready.emit([]) # Send empty list if no JSON found
            else:
                 self.signals.suggestions_ready.emit([])
        except Exception as e:
            self.signals.error.emit(f"Glossary suggestion failed: {e}")
        finally:
            self.signals.finished.emit()


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

class TypesetArea:
    def __init__(self, rect, text, font, color, polygon=None):
        self.rect = rect
        self.text = text
        self.font_info = self.font_to_dict(font)
        self.color_info = color.name()
        self.polygon = polygon

    @staticmethod
    def font_to_dict(font):
        return {'family': font.family(), 'pointSize': font.pointSize(), 'weight': font.weight(), 'italic': font.italic()}

    def get_font(self):
        font = QFont()
        font.setFamily(self.font_info['family'])
        font.setPointSize(self.font_info['pointSize'])
        font.setWeight(self.font_info['weight'])
        font.setItalic(self.font_info['italic'])
        return font

    def get_color(self):
        return QColor(self.color_info)

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

        self.detected_bubbles = []
        self.hovered_bubble_index = -1

    def get_selection_mode(self):
        return self.main_window.selection_mode_combo.currentText()

    def get_polygon_points(self):
        return self.polygon_points

    def set_detected_bubbles(self, bubbles):
        self.detected_bubbles = bubbles
        self.hovered_bubble_index = -1
        self.update()

    def clear_detected_bubbles(self):
        self.detected_bubbles = []
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

        if self.main_window.is_in_confirmation_mode:
            if event.button() == Qt.LeftButton and self.hovered_bubble_index != -1:
                self.main_window.remove_detected_bubble(self.hovered_bubble_index)
                self.hovered_bubble_index = -1
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
            if mode == "Rectangle":
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

        if self.main_window.is_in_confirmation_mode:
            unzoomed_pos = self.main_window.unzoom_coords(self.current_mouse_pos, as_point=True)
            new_hover_index = -1
            if unzoomed_pos:
                for i, bubble in reversed(list(enumerate(self.detected_bubbles))):
                    if bubble.containsPoint(unzoomed_pos, Qt.OddEvenFill):
                        new_hover_index = i
                        break
            if self.hovered_bubble_index != new_hover_index:
                self.hovered_bubble_index = new_hover_index
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
        if mode == "Rectangle":
            if self.dragging:
                self.selection_end = self.current_mouse_pos
                self.update()
        elif mode == "Pen Tool":
            if self.polygon_points:
                self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.get_selection_mode() == "Rectangle":
            if self.dragging:
                self.dragging = False
                self.selection_rect = QRect(self.selection_start, self.selection_end).normalized()
                if self.selection_rect.width() > 5 and self.selection_rect.height() > 5:
                    self.main_window.process_rect_area(self.selection_rect)
                else:
                    self.clear_selection()
                self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.main_window.original_pixmap: return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self.main_window.is_in_confirmation_mode and self.detected_bubbles:
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

            for i, bubble in enumerate(self.detected_bubbles):
                path = QPainterPath()
                path.addPolygon(QPolygonF(bubble))

                if i == self.hovered_bubble_index:
                    painter.fillPath(path, QColor(255, 80, 80, 150))
                    painter.setPen(QPen(QColor(255, 100, 100), 3 / scale))
                else:
                    painter.fillPath(path, QColor(0, 120, 215, 100))
                    painter.setPen(QPen(QColor(90, 180, 255), 2 / scale))

                painter.drawPath(path)
            painter.restore()

        mode = self.get_selection_mode()
        if mode == "Rectangle":
            if self.dragging:
                rect = QRect(self.selection_start, self.selection_end).normalized()
                painter.setPen(QPen(QColor(0, 120, 215), 2, Qt.DashLine))
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
        if self.main_window:
            self.main_window.update_pen_tool_buttons_visibility(False)
        self.update()

class MangaOCRApp(QMainWindow):
    DARK_THEME_STYLESHEET = """
        QMainWindow, QDialog { background-color: #2e2e2e; }
        QMenuBar { background-color: #3c3c3c; color: #f0f0f0; }
        QMenuBar::item:selected { background-color: #007acc; }
        QMenu { background-color: #3c3c3c; color: #f0f0f0; border: 1px solid #555; }
        QMenu::item:selected { background-color: #007acc; }
        QWidget { color: #f0f0f0; background-color: #2e2e2e; font-size: 10pt; }
        QLabel { padding: 2px; background-color: transparent; }
        QLabel#h3 { color: #55aaff; font-size: 12pt; font-weight: bold; margin-top: 10px; border-bottom: 1px solid #444; padding-bottom: 4px; }
        QPushButton { background-color: #007acc; color: white; padding: 8px; border: 1px solid #005f9e; border-radius: 4px; margin: 2px; }
        QPushButton:hover { background-color: #008ae6; }
        QPushButton:pressed { background-color: #005f9e; }
        QPushButton:disabled { background-color: #555555; border-color: #444; color: #999; }
        QTextEdit, QComboBox, QListWidget, QLineEdit, QSpinBox { background-color: #3c3c3c; color: #f0f0f0; border: 1px solid #555; padding: 5px; border-radius: 4px; }
        QListWidget::item { padding: 5px; }
        QListWidget::item:selected { background-color: #007acc; }
        QScrollArea { border: none; }
        QProgressBar { border: 1px solid #555; border-radius: 4px; text-align: center; color: #f0f0f0; }
        QProgressBar::chunk { background-color: #007acc; border-radius: 3px; }
        QStatusBar { background-color: #252525; color: #f0f0f0; }
        QGroupBox { border: 1px solid #444; border-radius: 6px; margin-top: 10px; padding: 10px; }
        QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; background-color: #2e2e2e; }
        QTabWidget::pane { border: 1px solid #444; border-top: none; }
        QTabBar::tab { background: #3c3c3c; color: #f0f0f0; padding: 10px 20px; border: 1px solid #444; border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; }
        QTabBar::tab:selected { background: #2e2e2e; border-color: #444; color: #55aaff; }
        QTabBar::tab:hover { background: #4a4a4a; }
        QSplitter::handle { background-color: #444; }
        QFrame[frameShape="5"] { color: #444; } /* VLine */
    """
    LIGHT_THEME_STYLESHEET = """
        /* TODO: Implement a light theme if needed */
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Manga OCR & Typeset Tool v13.0.1")
        self.image_files = []
        self.current_image_path = None
        self.current_image_pil = None
        self.original_pixmap = None
        self.typeset_pixmap = None
        self.zoom_factor = 1.0

        self.all_typeset_data = {}
        self.typeset_areas = []
        self.redo_stack = []

        self.easyocr_reader = None
        self.manga_ocr_reader = None
        self.paddle_ocr_reader = None
        self.paddle_lang = None

        self.current_project_path = None
        self.current_theme = 'dark'
        self.typeset_font = QFont("Arial", 12, QFont.Bold)
        self.typeset_color = QColor(Qt.black)
        self.inline_editor = None
        self.editing_area = None

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

        self.usage_file_path = os.path.join(os.path.expanduser("~"), "manga_ocr_usage_v13.dat")
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
            'ogkalu_pt':     {'path': 'models/comic-speech-bubble-detector.pt', 'instance': None, 'type': 'yolo'}
        }
        self.is_yolo_available = YOLO is not None
        self.is_onnx_available = onnxruntime is not None
        self.is_paddle_available = paddleocr is not None
        self.total_cost = 0.0
        self.usd_to_idr_rate = 16000.0

        self.exchange_rate_thread = None
        self.exchange_rate_worker = None

        self.AI_MODELS = {
            'gemini-2.5-flash-lite': {
                'display': 'Gemini 2.5 Flash Lite (Utama - Paling Cepat & Murah)',
                'pricing': {'input': 0.0001 / 1000, 'output': 0.0002 / 1000},  # per char
                'limits': {'rpm': 4000, 'rpd': 10000000}
            },
            'gemini-2.5-flash': {
                'display': 'Gemini 2.5 Flash (Fallback 1 - Akurasi Lebih Tinggi)',
                'pricing': {'input': 0.000125 / 1000, 'output': 0.00025 / 1000},  # per char
                'limits': {'rpm': 1000, 'rpd': 10000}
            },
            'gemini-2.5-pro': {
                'display': 'Gemini 2.5 Pro (Fallback 2 - Teks Rumit & Penting)',
                'pricing': {'input': 0.0025 / 1000, 'output': 0.0025 / 1000},  # per char
                'limits': {'rpm': 150, 'rpd': 10000}
            },
            'gemini-2.0-flash-lite': {
                'display': 'Gemini 2.0 Flash Lite (Darurat - Semua 2.5 Limit)',
                'pricing': {'input': 0.0001 / 1000, 'output': 0.0002 / 1000},  # per char
                'limits': {'rpm': 4000, 'rpd': 10000000}
            }
        }

        self.batch_processing_queue = []
        self.batch_processor_thread = None
        self.batch_processor_worker = None
        self.BATCH_SIZE_LIMIT = 20

        self.worker_pool = {}
        self.next_worker_id = 0
        self.MAX_WORKERS = 15
        self.WORKER_SPAWN_THRESHOLD = 3 # [v13.0.1] Changed from 5 to 3 for more aggressive scaling

        self.is_processing_selection = False

        self.ui_update_queue = []
        self.ui_update_mutex = QMutex()
        self.ui_update_timer = QTimer(self)
        self.ui_update_timer.setSingleShot(True)
        self.ui_update_timer.timeout.connect(self.process_ui_updates)
        self.is_processing_ui_updates = False

        self.is_in_confirmation_mode = False
        self.bubble_detection_thread = None
        self.bubble_detection_worker = None
        self.detected_bubbles_map = {}

        self.glossary = {} # source: target
        self.glossary_suggestion_worker = None
        self.glossary_suggestion_thread = None

        self.init_ui()
        self.setup_styles()
        self.setup_shortcuts()
        self.initialize_ocr_engines()
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
        left_panel_layout = QVBoxLayout(left_panel_widget)
        left_panel_layout.setContentsMargins(10, 10, 10, 10)
        left_panel_layout.addWidget(QLabel("<h3>Image Files</h3>"))
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
        right_panel_widget = QWidget()
        right_panel_layout = self.setup_right_panel()
        right_panel_widget.setLayout(right_panel_layout)
        splitter.addWidget(right_panel_widget)

        # Set initial sizes for the splitter
        splitter.setSizes([200, 700, 350])
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

    def setup_right_panel(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10,10,10,10)

        # Tabs for main functions
        tabs = QTabWidget()
        tabs.addTab(self._create_translate_tab(), "Translate")
        tabs.addTab(self._create_glossary_tab(), "Glossary")
        tabs.addTab(self._create_cleanup_tab(), "Cleanup")
        tabs.addTab(self._create_typeset_tab(), "Typeset")
        main_layout.addWidget(tabs)
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
        self.batch_process_button.setToolTip("Detects all bubbles in every file in the folder, lets you confirm, then processes them.")
        self.batch_process_button.clicked.connect(self.start_interactive_batch_detection)
        bottom_layout.addWidget(self.batch_process_button)

        self.confirm_bubbles_button = QPushButton("Confirm Bubbles (0)")
        self.confirm_bubbles_button.clicked.connect(self.process_confirmed_bubbles); self.confirm_bubbles_button.setVisible(False)
        bottom_layout.addWidget(self.confirm_bubbles_button)

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
        ocr_engines = ["Manga-OCR", "EasyOCR", "Tesseract"]
        if self.is_paddle_available: ocr_engines.insert(1, "PaddleOCR")

        self.ocr_engine_combo = self._create_combo_box(ocr_layout, "OCR Engine:", ocr_engines, 0, 0)
        self.lang_combo = self._create_combo_box(ocr_layout, "OCR Language:", ["Japanese", "Korean", "Chinese", "Chinese Traditional", "English"], 1, 0, default="Japanese")
        self.translate_combo = self._create_combo_box(ocr_layout, "Translate to:", ["Indonesian", "English"], 2, 0, default="Indonesian")
        self.orientation_combo = self._create_combo_box(ocr_layout, "Orientation:", ["Auto-Detect", "Horizontal", "Vertical"], 3, 0)

        self.ocr_engine_combo.currentTextChanged.connect(self.on_ocr_engine_changed)
        self.lang_combo.currentTextChanged.connect(self.on_lang_changed)
        layout.addWidget(ocr_group)
        self.on_lang_changed(self.lang_combo.currentText())

        # AI Group
        ai_group = QGroupBox("AI Enhancement")
        ai_layout = QGridLayout(ai_group)
        self.ai_model_combo = self._create_combo_box(ai_layout, "AI Model:", [m['display'] for m in self.AI_MODELS.values()], 0, 0)
        self.ai_model_combo.currentTextChanged.connect(self.on_ai_model_changed)

        styles = [
            "Santai (Default)",
            "Formal (Ke Atasan)",
            "Akrab (Ke Teman/Pacar)",
            "Vulgar/Dewasa (Adegan Seks)",
            "Sesuai Konteks Manga"
        ]
        self.style_combo = self._create_combo_box(ai_layout, "Translation Style:", styles, 1, 0)

        checkbox_layout = QVBoxLayout(); checkbox_layout.setSpacing(10)
        self.enhanced_pipeline_checkbox = QCheckBox("Enhanced Pipeline (JP Only, More API)"); checkbox_layout.addWidget(self.enhanced_pipeline_checkbox)
        self.gemini_only_translate_checkbox = QCheckBox("Gemini-Only Translate"); checkbox_layout.addWidget(self.gemini_only_translate_checkbox)
        self.batch_mode_checkbox = QCheckBox("Enable Batch Processing"); checkbox_layout.addWidget(self.batch_mode_checkbox)
        self.gemini_only_translate_checkbox.stateChanged.connect(self.on_gemini_only_mode_changed)
        self.batch_mode_checkbox.stateChanged.connect(self.on_batch_mode_changed)
        ai_layout.addLayout(checkbox_layout, 2, 0, 1, 2)
        layout.addWidget(ai_group)

        layout.addStretch()
        return tab

    def _create_glossary_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 15, 10, 10)

        # Glossary Management Group
        glossary_group = QGroupBox("Glossary Manager")
        glossary_layout = QVBoxLayout(glossary_group)

        # List of current glossary items
        self.glossary_list_widget = QListWidget()
        self.glossary_list_widget.itemDoubleClicked.connect(self.edit_glossary_item)
        glossary_layout.addWidget(self.glossary_list_widget)

        # Input fields
        entry_layout = QGridLayout()
        entry_layout.addWidget(QLabel("Source (Japanese):"), 0, 0)
        self.glossary_source_edit = QLineEdit()
        entry_layout.addWidget(self.glossary_source_edit, 0, 1)
        entry_layout.addWidget(QLabel("Target (Indonesian):"), 1, 0)
        self.glossary_target_edit = QLineEdit()
        entry_layout.addWidget(self.glossary_target_edit, 1, 1)
        glossary_layout.addLayout(entry_layout)

        # Action buttons
        action_layout = QHBoxLayout()
        self.add_glossary_button = QPushButton("Add / Update Entry")
        self.add_glossary_button.clicked.connect(self.add_or_update_glossary_entry)
        self.delete_glossary_button = QPushButton("Delete Selected")
        self.delete_glossary_button.clicked.connect(self.delete_glossary_entry)
        action_layout.addWidget(self.add_glossary_button)
        action_layout.addWidget(self.delete_glossary_button)
        glossary_layout.addLayout(action_layout)

        # File IO buttons
        file_io_layout = QHBoxLayout()
        self.save_glossary_button = QPushButton("Save Glossary to File")
        self.save_glossary_button.clicked.connect(self.save_glossary_manual)
        self.load_glossary_button = QPushButton("Load Glossary from File")
        self.load_glossary_button.clicked.connect(self.load_glossary_manual)
        file_io_layout.addWidget(self.save_glossary_button)
        file_io_layout.addWidget(self.load_glossary_button)
        glossary_layout.addLayout(file_io_layout)

        layout.addWidget(glossary_group)

        # AI Suggestions Group
        suggestions_group = QGroupBox("AI Glossary Suggestions")
        suggestions_layout = QVBoxLayout(suggestions_group)
        self.glossary_suggestions_list = QListWidget()
        self.glossary_suggestions_list.itemDoubleClicked.connect(self.add_suggestion_to_glossary)
        suggestions_layout.addWidget(self.glossary_suggestions_list)

        sug_btn_layout = QHBoxLayout()
        self.add_all_suggestions_button = QPushButton("Add All Suggestions")
        self.add_all_suggestions_button.clicked.connect(self.add_all_suggestions_to_glossary)
        self.clear_suggestions_button = QPushButton("Clear Suggestions")
        self.clear_suggestions_button.clicked.connect(self.glossary_suggestions_list.clear)
        sug_btn_layout.addWidget(self.add_all_suggestions_button)
        sug_btn_layout.addWidget(self.clear_suggestions_button)
        suggestions_layout.addLayout(sug_btn_layout)

        layout.addWidget(suggestions_group)

        return tab

    def _create_cleanup_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 15, 10, 10)

        selection_group = QGroupBox("Selection Tool")
        selection_layout = QGridLayout(selection_group)
        self.selection_mode_combo = self._create_combo_box(selection_layout, "Mode:", ["Rectangle", "Pen Tool"], 0, 0)
        self.selection_mode_combo.currentTextChanged.connect(self.selection_mode_changed)
        pen_buttons_layout = QHBoxLayout()
        self.confirm_button = QPushButton("Confirm"); self.confirm_button.clicked.connect(self.confirm_pen_selection); self.confirm_button.setVisible(False)
        self.cancel_button = QPushButton("Cancel"); self.cancel_button.clicked.connect(self.cancel_pen_selection); self.cancel_button.setVisible(False)
        pen_buttons_layout.addWidget(self.confirm_button); pen_buttons_layout.addWidget(self.cancel_button)
        selection_layout.addLayout(pen_buttons_layout, 1, 0, 1, 2)
        layout.addWidget(selection_group)

        inpaint_group = QGroupBox("Inpainting (Cleanup)")
        inpaint_layout = QGridLayout(inpaint_group)
        self.inpaint_checkbox = QCheckBox("Gunakan Inpainting"); self.inpaint_checkbox.setChecked(True)
        inpaint_layout.addWidget(self.inpaint_checkbox, 0, 0, 1, 2)
        self.inpaint_algo_combo = self._create_combo_box(inpaint_layout, "Algoritma:", ["Navier-Stokes (NS)", "Telea"], 1, 0)
        self.inpaint_padding_spinbox = self._create_spin_box(inpaint_layout, "Padding (px):", 1, 25, 5, 2, 0)
        layout.addWidget(inpaint_group)

        dl_detect_group = QGroupBox("Bubble Auto-Detection")
        dl_layout = QGridLayout(dl_detect_group)
        self.dl_bubble_detector_checkbox = QCheckBox("Gunakan DL Detector"); dl_layout.addWidget(self.dl_bubble_detector_checkbox, 0, 0, 1, 2)
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

    def _create_combo_box(self, parent_layout, label_text, items, row, col, default=None):
        label = QLabel(label_text); parent_layout.addWidget(label, row, col)
        combo = QComboBox(); combo.addItems(items)
        if default: combo.setCurrentText(default)
        parent_layout.addWidget(combo, row, col + 1)
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

    def initialize_ocr_engines(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage("Initializing OCR engines...")

        # cek GPU
        gpu_available = False
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            if not gpu_available:
                print("CUDA not available, OCR engines will run on CPU.")
        except ImportError:
            print("PyTorch not found, OCR engines will run on CPU.")

        # EasyOCR
        try:
            # bahasa awal (sesuai kebutuhan kamu)
            lang_list = ['en', 'ja', 'ko', 'ch_sim', 'ch_tra']

            # ---- aturan fix untuk EasyOCR ----
            if 'ch_tra' in lang_list:
                # kalau ada ch_tra, harus cuma ['en','ch_tra']
                lang_list = ['en', 'ch_tra']

            elif 'ch_sim' in lang_list:
                # kalau ada ch_sim, harus cuma ['en','ch_sim']
                lang_list = ['en', 'ch_sim']

            # inisialisasi reader
            self.easyocr_reader = easyocr.Reader(lang_list, gpu=gpu_available)
            self.statusBar().showMessage(
                f"EasyOCR initialized with languages: {lang_list}", 3000
            )
        except Exception as e:
            QMessageBox.critical(
                self, "EasyOCR Error", f"Could not initialize EasyOCR.\nError: {e}"
            )
            self.easyocr_reader = None

        # Manga-OCR
        if MangaOcr:
            try:
                self.manga_ocr_reader = MangaOcr()
                self.statusBar().showMessage("Manga-OCR initialized.", 3000)
            except Exception as e:
                QMessageBox.critical(
                    self, "Manga-OCR Error", f"Could not initialize Manga-OCR.\nError: {e}"
                )
                self.manga_ocr_reader = None

        # PaddleOCR → tetap on-demand
        QApplication.restoreOverrideCursor()
        self.statusBar().showMessage("Ready", 3000)

    def initialize_paddle_ocr(self, lang_code):
        if not self.is_paddle_available: return
        try:
            self.statusBar().showMessage(f"Initializing PaddleOCR for language: {lang_code}...")
            QApplication.processEvents()
            self.paddle_ocr_reader = paddleocr.PaddleOCR(use_angle_cls=True, lang=lang_code)
            self.paddle_lang = lang_code
            self.statusBar().showMessage(f"PaddleOCR ({lang_code}) initialized.", 3000)
        except Exception as e:
            QMessageBox.critical(self, "PaddleOCR Error", f"Could not initialize PaddleOCR.\nError: {e}")
            self.paddle_ocr_reader = None

    def add_api_cost(self, input_chars, output_chars, model_name):
        model_info = self.AI_MODELS.get(model_name)
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
        if is_enhanced and self.gemini_only_translate_checkbox.isChecked():
            self.gemini_only_translate_checkbox.setChecked(False)

        self.ocr_engine_combo.setEnabled(not is_enhanced)
        if is_enhanced:
            self.ocr_engine_combo.setToolTip("Disabled in Enhanced Pipeline mode.")
        else:
            self.on_gemini_only_mode_changed(self.gemini_only_translate_checkbox.checkState())

    def on_gemini_only_mode_changed(self, state):
        is_gemini_only = (state == Qt.Checked)
        if is_gemini_only and self.enhanced_pipeline_checkbox.isChecked():
            self.enhanced_pipeline_checkbox.setChecked(False)
        self.translate_combo.setEnabled(not is_gemini_only)

    def on_lang_changed(self, lang):
        """Sarankan OCR engine terbaik berdasarkan bahasa."""
        if lang == "Japanese":
            if self.manga_ocr_reader: self.ocr_engine_combo.setCurrentText("Manga-OCR")
            else: self.ocr_engine_combo.setCurrentText("Tesseract")
        elif lang in ["Korean", "Chinese", "Chinese Traditional"]:
            if self.is_paddle_available: self.ocr_engine_combo.setCurrentText("PaddleOCR")
            else: self.ocr_engine_combo.setCurrentText("EasyOCR")
        else: # English
             self.ocr_engine_combo.setCurrentText("Tesseract")

    def on_ocr_engine_changed(self, engine):
        is_manga_ocr = (engine == "Manga-OCR")
        self.lang_combo.setEnabled(not is_manga_ocr)
        if is_manga_ocr:
            self.lang_combo.setCurrentText("Japanese")


    def clean_and_join_text(self, raw_text):
        return ' '.join(raw_text.split())

    def correct_text_with_gemini(self, text_to_correct, model_name):
        # This function might be deprecated in favor of the all-in-one translate prompt
        # but is kept for the standard pipeline.
        if not text_to_correct.strip(): return ""
        if not GEMINI_API_KEY or "your_gemini_key_here" in GEMINI_API_KEY: return text_to_correct
        try:
            model = genai.GenerativeModel(model_name)
            prompt = (f"Correct OCR errors in this Japanese manga text: \"{text_to_correct}\". Return only the corrected Japanese text.")
            response = model.generate_content(prompt)
            if response.parts:
                self.add_api_cost(len(prompt), len(response.text), model_name)
                return response.text.strip()
            return text_to_correct
        except Exception as e:
            print(f"Error calling Gemini API for correction: {e}"); return text_to_correct

    def naturalize_text_with_gemini(self, text_to_naturalize, model_name):
        # Kept for standard pipeline
        if not text_to_naturalize.strip(): return ""
        try:
            model = genai.GenerativeModel(model_name)
            prompt = f"Refine this Indonesian translation to sound more natural and colloquial for a manga: \"{text_to_naturalize}\". Return only the refined Indonesian text."
            response = model.generate_content(prompt)
            if response.parts:
                self.add_api_cost(len(prompt), len(response.text), model_name)
                return response.text.strip()
            return text_to_naturalize
        except Exception as e:
            print(f"Error calling Gemini API for naturalization: {e}"); return text_to_naturalize

    def _build_prompt_enhancements(self, settings):
        """Membangun string tambahan untuk prompt Gemini berdasarkan pengaturan."""
        enhancements = ""

        style_map = {
            "Santai (Default)": "Your tone MUST be casual and colloquial, like everyday conversation.",
            "Formal (Ke Atasan)": "Your tone MUST be formal and respectful, suitable for addressing a superior or elder.",
            "Akrab (Ke Teman/Pacar)": "Your tone MUST be intimate and very casual, suitable for talking to a close friend or romantic partner.",
            "Vulgar/Dewasa (Adegan Seks)": "Your tone MUST be vulgar and explicit, suitable for an adult/sexual scene. Use direct language but still adhere to censorship rules.",
            "Sesuai Konteks Manga": "Analyze the text and provide a translation that best fits the likely context of a manga scene (e.g., action, comedy, drama)."
        }
        style = settings.get('translation_style', 'Santai (Default)')
        style_instruction = style_map.get(style, style_map["Santai (Default)"])
        enhancements += f"\n- Translation Style: {style_instruction}"

        if self.glossary:
            enhancements += "\n- Custom Glossary (You MUST strictly follow these rules, it is a top priority, no exceptions):\n"
            for source, target in self.glossary.items():
                 enhancements += f'  - Always translate the exact term "{source}" as "{target}".\n'
        return enhancements

    def translate_with_gemini(self, text_to_translate, target_lang, model_name, settings):
        if not text_to_translate.strip(): return ""
        if not GEMINI_API_KEY or "your_gemini_key_here" in GEMINI_API_KEY: return "[GEMINI API KEY NOT CONFIGURED]"
        try:
            model = genai.GenerativeModel(model_name)
            prompt_enhancements = self._build_prompt_enhancements(settings)

            prompt = f"""As an expert manga translator, your task is to process a raw OCR text from a Japanese manga.
1.  **Correct** any OCR errors in the original Japanese text.
2.  **Translate** the corrected text into natural, colloquial {target_lang}.
3.  **Ensure** the final translation sounds authentic and fits a casual manga context.

**IMPORTANT RULES:**
- CENSORSHIP: If the text contains "vagina" or equivalent, replace it with "meong". If it contains "penis", replace it with "burung". Ensure the final sentence is still coherent.
{prompt_enhancements}

**Raw OCR Text:** "{text_to_translate}"

**Your final output must ONLY be the translated {target_lang} text. Do not include explanations, the original text, or any markdown formatting.**
"""
            response = model.generate_content(prompt)
            if response.parts:
                self.add_api_cost(len(prompt), len(response.text), model_name)
                # This now happens inside the worker after this function returns
                return response.text.strip()
            return "[GEMINI FAILED]"
        except Exception as e:
            print(f"Error calling Gemini API for full translation: {e}")
            return "[GEMINI ERROR]"

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

        # Start glossary suggestion worker
        self.start_glossary_suggestion(original_text, translated_text)

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

            updates_by_image = {}
            for image_path, new_area in self.ui_update_queue:
                if image_path not in updates_by_image:
                    updates_by_image[image_path] = []
                updates_by_image[image_path].append(new_area)

            self.ui_update_queue.clear()
            self.ui_update_mutex.unlock()

            for image_path, new_areas in updates_by_image.items():
                if image_path not in self.all_typeset_data:
                    self.all_typeset_data[image_path] = {'areas': [], 'redo': []}

                self.all_typeset_data[image_path]['areas'].extend(new_areas)
                self.all_typeset_data[image_path]['redo'].clear()

            current_key = self.get_current_data_key()
            if current_key in updates_by_image:
                self.typeset_areas = self.all_typeset_data[current_key]['areas']
                self.redo_stack = self.all_typeset_data[current_key]['redo']
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
        return {
            'ocr_engine': self.ocr_engine_combo.currentText(),
            'ocr_lang': self.lang_combo.currentText(),
            'orientation': self.orientation_combo.currentText(),
            'target_lang': self.translate_combo.currentText(),
            'use_ai': True, # Implied by new workflows
            'font': self.typeset_font,
            'color': self.typeset_color,
            'enhanced_pipeline': self.enhanced_pipeline_checkbox.isChecked(),
            'use_gemini_only_translate': self.gemini_only_translate_checkbox.isChecked(),
            'inpaint_algo': self.inpaint_algo_combo.currentText(),
            'use_dl_detector': self.dl_bubble_detector_checkbox.isChecked(),
            'dl_provider': self.dl_model_provider_combo.currentText(),
            'dl_model_file': self.dl_model_file_combo.currentText(),
            'inpaint_padding': self.inpaint_padding_spinbox.value(),
            'ai_model': self.get_selected_model_name(),
            'translation_style': self.style_combo.currentText(),
            'auto_split_bubbles': self.split_bubbles_checkbox.isChecked()
        }

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

            if 'model_usage' not in self.usage_data:
                self.usage_data['model_usage'] = {}

            for model_name in self.AI_MODELS:
                if model_name not in self.usage_data['model_usage']:
                    self.usage_data['model_usage'][model_name] = {'daily_count': 0, 'minute_count': 0, 'current_minute': ''}

            if 'date' not in self.usage_data or self.usage_data.get('date') != str(date.today()):
                self.usage_data['date'] = str(date.today())
                for model_name in self.AI_MODELS:
                    self.usage_data['model_usage'][model_name]['daily_count'] = 0
                    self.usage_data['model_usage'][model_name]['minute_count'] = 0

            self.total_cost = self.usage_data.get('total_cost', 0.0)
            self.update_cost_display()
            self.save_usage_data()
        except Exception as e:
            print(f"Could not load or create usage data file: {e}")
            self.usage_data = {'date': str(date.today()), 'total_cost': 0.0, 'model_usage': {}}
            for model_name in self.AI_MODELS:
                self.usage_data['model_usage'][model_name] = {'daily_count': 0, 'minute_count': 0, 'current_minute': ''}

    def save_usage_data(self):
        try:
            self.usage_data['total_cost'] = self.total_cost
            with open(self.usage_file_path, 'wb') as f: pickle.dump(self.usage_data, f)
        except Exception as e: print(f"Could not save usage data: {e}")

    def check_and_increment_usage(self, model_name):
        now = time.time()
        current_minute_str = time.strftime('%Y-%m-%d %H:%M', time.localtime(now))

        model_usage = self.usage_data['model_usage'][model_name]
        model_limits = self.AI_MODELS[model_name]['limits']

        if self.usage_data.get('date') != str(date.today()):
            self.usage_data['date'] = str(date.today())
            for m in self.usage_data['model_usage']:
                self.usage_data['model_usage'][m]['daily_count'] = 0
                self.usage_data['model_usage'][m]['minute_count'] = 0

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
        model_name = self.get_selected_model_name()
        if not model_name: return

        model_usage = self.usage_data['model_usage'][model_name]
        model_limits = self.AI_MODELS[model_name]['limits']

        rpm = model_usage.get('minute_count', 0)
        rpd = model_usage.get('daily_count', 0)

        self.rpm_label.setText(f"RPM: {rpm} / {model_limits['rpm']}")
        self.rpd_label.setText(f"RPD: {rpd} / {model_limits['rpd']}")

    def check_limits_and_update_ui(self):
        self.load_usage_data()

        model_name = self.get_selected_model_name()
        if not model_name: return

        model_usage = self.usage_data['model_usage'][model_name]
        model_limits = self.AI_MODELS[model_name]['limits']

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
                                      (self.bubble_detection_thread and self.bubble_detection_thread.isRunning()) or \
                                      (self.batch_processor_thread and self.batch_processor_thread.isRunning())

        enabled_state = not ai_disabled and not is_worker_running

        self.enhanced_pipeline_checkbox.setEnabled(enabled_state)
        self.gemini_only_translate_checkbox.setEnabled(enabled_state)

        if ai_disabled:
            self.enhanced_pipeline_checkbox.setChecked(False)
            self.gemini_only_translate_checkbox.setChecked(False)
            self.enhanced_pipeline_checkbox.setToolTip(tooltip_message)
            self.gemini_only_translate_checkbox.setToolTip(tooltip_message)
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
            self.load_glossary()

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
        if self.is_in_confirmation_mode:
            self.cancel_interactive_batch()

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
        self.image_label.clear_detected_bubbles()

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

            key = self.get_current_data_key()
            img_data = self.all_typeset_data.get(key, {'areas': [], 'redo': []})
            self.typeset_areas = img_data['areas']
            self.redo_stack = img_data['redo']

            self.redraw_all_typeset_areas()
            self.update_undo_redo_buttons_state()
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
            current_key = self.get_current_data_key()
            if current_key in self.detected_bubbles_map:
                self.detected_bubbles_map[current_key].append(poly)
                self.image_label.set_detected_bubbles(self.detected_bubbles_map[current_key])
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
            current_key = self.get_current_data_key()
            if current_key in self.detected_bubbles_map:
                self.detected_bubbles_map[current_key].append(unzoomed_polygon)
                self.image_label.set_detected_bubbles(self.detected_bubbles_map[current_key])
                self.update_confirmation_button_text()
            self.image_label.clear_selection()
            return

        if self.is_processing_selection: return

        self.is_processing_selection = True
        try:
            if not self.current_image_pil: return
            result = self.unzoom_coords(scaled_points)
            if not result: return
            unzoomed_polygon, unzoomed_bbox = result
            if not unzoomed_bbox or unzoomed_bbox.width() <= 0 or unzoomed_bbox.height() <= 0: return
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

            job = {
                'image_path': self.get_current_data_key(),
                'rect': unzoomed_bbox,
                'polygon': unzoomed_polygon,
                'cropped_cv_img': img_for_ocr,
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

        # [FIX v12.0.3] Cegah crash jika area crop menghasilkan gambar kosong
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
            try: model_info['instance'] = onnxruntime.InferenceSession(model_info['path'])
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
            results = model(full_cv_image, verbose=False)
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

    def draw_single_area(self, painter, area, source_pil_image, for_saving=False):
        settings = self.get_current_settings()
        cv_original = cv2.cvtColor(np.array(source_pil_image), cv2.COLOR_RGB2BGR)

        text_mask = np.zeros(cv_original.shape[:2], dtype=np.uint8)
        if area.polygon:
            cv_poly_points = np.array([[p.x(), p.y()] for p in area.polygon], dtype=np.int32)
            cv2.fillPoly(text_mask, [cv_poly_points], 255)
        else:
            cv2.rectangle(text_mask, (area.rect.x(), area.rect.y()), (area.rect.right(), area.rect.bottom()), 255, -1)

        padding = settings['inpaint_padding']
        if padding > 0:
            kernel = np.ones((padding, padding), np.uint8)
            dilated_text_mask = cv2.dilate(text_mask, kernel, iterations=1)
        else:
            dilated_text_mask = text_mask

        bubble_mask = self.find_speech_bubble_mask(cv_original, area.rect, settings, for_saving=for_saving)

        if bubble_mask is not None: final_inpaint_mask = cv2.bitwise_and(dilated_text_mask, bubble_mask)
        else: final_inpaint_mask = dilated_text_mask

        if self.inpaint_checkbox.isChecked():
            algo_map = {"Navier-Stokes (NS)": cv2.INPAINT_NS, "Telea": cv2.INPAINT_TELEA}
            inpaint_algorithm = algo_map.get(settings['inpaint_algo'], cv2.INPAINT_NS)
            inpainted_cv = cv2.inpaint(cv_original, final_inpaint_mask, 3, inpaint_algorithm)

            height, width, channel = inpainted_cv.shape; bytes_per_line = 3 * width
            q_image = QImage(inpainted_cv.tobytes(), width, height, bytes_per_line, QImage.Format_BGR888).rgbSwapped()

            painter.save()
            clip_path = QPainterPath()
            contours, _ = cv2.findContours(final_inpaint_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                polygon = QPolygonF([QPoint(p[0][0], p[0][1]) for p in cnt])
                clip_path.addPolygon(polygon)

            painter.setClipPath(clip_path); painter.drawImage(0, 0, q_image); painter.restore()
        else:
            background_color = self.get_background_color(cv_original, area.rect)
            painter.save()
            path = QPainterPath()
            contours, _ = cv2.findContours(final_inpaint_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                polygon = QPolygonF([QPoint(p[0][0], p[0][1]) for p in cnt])
                path.addPolygon(polygon)
            painter.setClipPath(path); painter.fillRect(painter.window(), background_color); painter.restore()

        painter.save()
        is_vertical = self.vertical_typeset_checkbox.isChecked()
        self.draw_text_with_options(painter, area.rect, area.text, area.get_font(), area.get_color(), is_vertical)
        painter.restore()

    def draw_text_with_options(self, painter, rect, text, font, color, is_vertical=False):
        if is_vertical: self.draw_rotated_vertical_text(painter, rect, text, font, color); return
        margin_x = int(rect.width() * 0.05); margin_y = int(rect.height() * 0.05)
        inner_rect = rect.adjusted(margin_x, margin_y, -margin_x, -margin_y)
        if inner_rect.width() <= 0 or inner_rect.height() <= 0: return
        current_font = QFont(font)
        low, high, best_size = 6, 200, 6
        while low <= high:
            mid = (low + high) // 2; current_font.setPointSizeF(mid); painter.setFont(current_font)
            bounding_rect = painter.boundingRect(inner_rect, Qt.TextWordWrap, text)
            if bounding_rect.height() <= inner_rect.height() and bounding_rect.width() <= inner_rect.width():
                best_size = mid; low = mid + 1
            else: high = mid - 1
        current_font.setPointSizeF(best_size); painter.setFont(current_font)
        outline_color = Qt.white if color.value() < 128 else Qt.black
        painter.setPen(outline_color)
        for dx, dy in [(-1,-1), (1,-1), (-1,1), (1,1), (-2,0), (2,0), (0,-2), (0,2)]:
            painter.drawText(inner_rect.translated(dx, dy), Qt.AlignCenter | Qt.TextWordWrap, text)
        painter.setPen(color)
        painter.drawText(inner_rect, Qt.AlignCenter | Qt.TextWordWrap, text)

    def draw_rotated_vertical_text(self, painter, rect, text, font, color):
        painter.save(); center = rect.center()
        painter.translate(center.x(), center.y()); painter.rotate(90); painter.translate(-center.x(), -center.y())
        rotated_rect = QRect(0, 0, rect.height(), rect.width()); rotated_rect.moveCenter(rect.center())
        self.draw_text_with_options(painter, rotated_rect, text, font, color, is_vertical=False)
        painter.restore()

    def selection_mode_changed(self, mode):
        self.image_label.clear_selection()
        self.image_label.setCursor(Qt.CrossCursor if mode == "Rectangle" else Qt.PointingHandCursor)

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

    def save_project(self):
        if not self.image_files: return False

        if self.current_image_path:
            self.all_typeset_data[self.get_current_data_key()] = {'areas': self.typeset_areas, 'redo': self.redo_stack}

        self.save_glossary() # Save glossary with project

        if not self.current_project_path:
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
            self.setWindowTitle(f"Manga OCR & Typeset Tool v13.0.1 - {os.path.basename(self.current_project_path)}"); self.autosave_timer.start(); return True
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save project: {e}"); return False

    def auto_save_project(self):
        if self.current_project_path and os.path.exists(os.path.dirname(self.current_project_path)):
            if self.save_project(): self.statusBar().showMessage(f"Project auto-saved at {time.strftime('%H:%M:%S')}", 3000)

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
            self.load_glossary() # Load associated glossary

            self.update_file_list()
            path_to_load = project_data.get('current_path', None)
            if path_to_load and path_to_load in self.image_files:
                self.file_list_widget.setCurrentRow(self.image_files.index(path_to_load))

            self.current_project_path = filePath
            self.setWindowTitle(f"Manga OCR & Typeset Tool v13.0.1 - {os.path.basename(self.current_project_path)}"); self.autosave_timer.start()
            QMessageBox.information(self, "Success", "Project loaded successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load project: {e}")

    def start_inline_edit(self, area):
        if self.inline_editor: self.finish_inline_edit()
        self.editing_area = area; self.inline_editor = QLineEdit(self.image_label)
        zoomed_rect = self.zoom_coords(area.rect); self.inline_editor.setGeometry(zoomed_rect); self.inline_editor.setText(area.text); self.inline_editor.setFont(area.get_font())
        if self.current_theme == 'dark': self.inline_editor.setStyleSheet("background-color: rgba(255, 255, 255, 0.9); color: black; border: 2px solid #007acc;")
        else: self.inline_editor.setStyleSheet("background-color: rgba(50, 50, 50, 0.9); color: white; border: 2px solid #0078d7;")
        self.inline_editor.editingFinished.connect(self.finish_inline_edit); self.inline_editor.show(); self.inline_editor.setFocus()

    def finish_inline_edit(self):
        if not self.inline_editor: return
        self.editing_area.text = self.inline_editor.text(); self.redo_stack.clear()
        self.redraw_all_typeset_areas(); self.update_undo_redo_buttons_state()
        self.inline_editor.deleteLater(); self.inline_editor = None; self.editing_area = None

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
        model_name = self.get_selected_model_name()
        if not model_name: return
        about_text = (f"<b>Manga OCR & Typeset Tool v13.0.1</b><br><br>This tool was created to streamline the process of translating manga.<br><br>Powered by Python, PyQt5, and Gemini API.<br>Enhanced with new features by Gemini.<br><br>Copyright © 2024")
        QMessageBox.about(self, "About & API Usage", about_text)
    def export_to_pdf(self):
        if not self.project_dir:
            QMessageBox.warning(self, "No Folder Loaded", "Please load a folder containing images first.")
            return

        image_files_to_export = []
        for file_path in self.image_files:
            # Skip checking files that are already typeset versions themselves to avoid duplicates
            if "_typeset" in file_path.lower():
                continue

            path_part, ext = os.path.splitext(file_path)
            typeset_path = f"{path_part}_typeset.png"

            # Check if the typeset version of the file exists
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
            QApplication.restoreOverrideCursor()
            self.overall_progress_bar.setVisible(False)
            self.statusBar().showMessage("Ready", 3000)

    def wheelEvent(self, event):
        if self.pdf_document and not (self.bubble_detection_thread and self.bubble_detection_thread.isRunning()):
            if event.angleDelta().y() < 0: self.load_next_image()
            elif event.angleDelta().y() > 0: self.load_prev_image()
        super().wheelEvent(event)

    def on_dl_detector_state_changed(self, state):
        is_checked = (state == Qt.Checked)
        # self.dl_options_widget.setVisible(is_checked) # Widget ini sudah tidak ada, diganti layout

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

    def get_selected_model_name(self):
        selected_display_name = self.ai_model_combo.currentText()
        for name, properties in self.AI_MODELS.items():
            if properties['display'] == selected_display_name: return name
        return None

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

    def perform_ocr_for_batch(self, image_to_process, settings):
        # This function is now mostly redundant with the main one but kept for the batch worker
        return self.perform_ocr(image_to_process, settings)

    def split_extended_bubbles(self, polygons, split_threshold=2.5):
        new_polygons = []
        for poly in polygons:
            bbox = poly.boundingRect()
            if bbox.width() <= 0 or bbox.height() <= 0: continue
            aspect_ratio = bbox.width() / bbox.height()

            if aspect_ratio > split_threshold:
                mid_x = bbox.left() + bbox.width() // 2
                new_polygons.append(QPolygon(QRect(bbox.left(), bbox.top(), bbox.width() // 2, bbox.height())))
                new_polygons.append(QPolygon(QRect(mid_x, bbox.top(), bbox.width() // 2, bbox.height())))
            elif (1 / aspect_ratio) > split_threshold:
                mid_y = bbox.top() + bbox.height() // 2
                new_polygons.append(QPolygon(QRect(bbox.left(), bbox.top(), bbox.width(), bbox.height() // 2)))
                new_polygons.append(QPolygon(QRect(bbox.left(), mid_y, bbox.width(), bbox.height() // 2)))
            else:
                new_polygons.append(poly)
        return new_polygons

    def start_interactive_batch_detection(self):
        if not self.image_files:
            QMessageBox.warning(self, "No Files Loaded", "Please load a folder first to use this feature.")
            return

        if self.bubble_detection_thread and self.bubble_detection_thread.isRunning():
            QMessageBox.warning(self, "Busy", "A bubble detection process is already running.")
            return

        reply = QMessageBox.question(self, 'Confirm Full Batch Detection',
                                     f"This will detect speech bubbles in all {len(self.image_files)} files in the current folder. This may take a while.\n\nDo you want to continue?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.No: return

        self.detected_bubbles_map.clear()
        self.set_ui_for_detection(True)

        settings = self.get_current_settings()
        self.bubble_detection_thread = QThread()
        self.bubble_detection_worker = BubbleDetectorWorker(self, self.image_files, settings)
        self.bubble_detection_worker.moveToThread(self.bubble_detection_thread)
        self.bubble_detection_worker.signals.detection_complete.connect(self.on_bubble_detection_complete)
        self.bubble_detection_worker.signals.overall_progress.connect(self.update_overall_progress)
        self.bubble_detection_worker.signals.error.connect(self.on_worker_error)
        self.bubble_detection_worker.signals.finished.connect(self.on_bubble_detection_finished)
        self.bubble_detection_thread.started.connect(self.bubble_detection_worker.run)
        self.bubble_detection_thread.start()

    def on_bubble_detection_complete(self, image_path, polygons):
        self.detected_bubbles_map[image_path] = polygons
        if image_path == self.get_current_data_key():
            self.image_label.set_detected_bubbles(polygons)

    def on_bubble_detection_finished(self):
        self.overall_progress_bar.setVisible(False)
        if self.get_current_settings()['auto_split_bubbles']:
            self.statusBar().showMessage("Splitting extended bubbles...", 3000)
            QApplication.processEvents()
            for path, polygons in self.detected_bubbles_map.items():
                self.detected_bubbles_map[path] = self.split_extended_bubbles(polygons)

            current_key = self.get_current_data_key()
            if current_key in self.detected_bubbles_map:
                self.image_label.set_detected_bubbles(self.detected_bubbles_map[current_key])

        self.statusBar().showMessage("Detection complete. Please review the bubbles.", 5000)
        self.set_ui_for_confirmation(True)

    def process_confirmed_bubbles(self):
        self.statusBar().showMessage("Processing confirmed bubbles...")
        QApplication.processEvents()

        total_bubbles = sum(len(bubbles) for bubbles in self.detected_bubbles_map.values())
        if total_bubbles == 0:
            QMessageBox.information(self, "No Bubbles", "No bubbles were confirmed for processing.")
            self.cancel_interactive_batch()
            return

        settings = self.get_current_settings()
        settings['use_gemini_only_translate'] = True

        for image_path, polygons in self.detected_bubbles_map.items():
            try:
                pil_image = Image.open(image_path).convert('RGB')
                for polygon in polygons:
                    bbox = polygon.boundingRect()
                    if bbox.width() <= 0 or bbox.height() <= 0: continue
                    cropped_pil_img = pil_image.crop((bbox.x(), bbox.y(), bbox.right(), bbox.bottom()))
                    cropped_cv_img = cv2.cvtColor(np.array(cropped_pil_img), cv2.COLOR_RGB2BGR)
                    mask = np.zeros(cropped_cv_img.shape[:2], dtype=np.uint8)
                    relative_poly_points = [QPoint(p.x() - bbox.x(), p.y() - bbox.y()) for p in polygon]
                    cv_poly_points = np.array([[p.x(), p.y()] for p in relative_poly_points], dtype=np.int32)
                    cv2.fillPoly(mask, [cv_poly_points], 255)

                    white_bg = np.full(cropped_cv_img.shape, 255, dtype=np.uint8)
                    fg = cv2.bitwise_and(cropped_cv_img, cropped_cv_img, mask=mask)
                    bg = cv2.bitwise_and(white_bg, white_bg, mask=cv2.bitwise_not(mask))
                    img_for_ocr = cv2.add(fg, bg)

                    job = { 'image_path': image_path, 'rect': bbox, 'polygon': polygon, 'cropped_cv_img': img_for_ocr, 'settings': settings }
                    self.processing_queue.append(job)
            except Exception as e:
                self.on_worker_error(f"Error preparing job for {os.path.basename(image_path)}: {e}")

        self.update_queue_status(len(self.processing_queue))
        self.manage_worker_pool()
        self.cancel_interactive_batch()

    def cancel_interactive_batch(self):
        if self.bubble_detection_worker: self.bubble_detection_worker.cancel()
        if self.bubble_detection_thread: self.bubble_detection_thread.quit(); self.bubble_detection_thread.wait()

        self.bubble_detection_thread = None; self.bubble_detection_worker = None
        self.detected_bubbles_map.clear(); self.image_label.clear_detected_bubbles()
        self.set_ui_for_detection(False); self.set_ui_for_confirmation(False)
        self.statusBar().showMessage("Batch detection cancelled.", 3000)

    def remove_detected_bubble(self, index_to_remove):
        current_key = self.get_current_data_key()
        if current_key in self.detected_bubbles_map and 0 <= index_to_remove < len(self.detected_bubbles_map[current_key]):
            del self.detected_bubbles_map[current_key][index_to_remove]
            self.image_label.set_detected_bubbles(self.detected_bubbles_map[current_key])
            self.update_confirmation_button_text()

    def set_ui_for_detection(self, is_detecting):
        self.batch_process_button.setEnabled(not is_detecting)
        self.file_list_widget.setEnabled(not is_detecting)
        self.prev_button.setEnabled(not is_detecting); self.next_button.setEnabled(not is_detecting)
        self.cancel_detection_button.setVisible(is_detecting)
        self.overall_progress_bar.setVisible(is_detecting)
        if is_detecting: self.overall_progress_bar.setValue(0); self.statusBar().showMessage("Starting bubble detection...")
        else: self.overall_progress_bar.setVisible(False)

    def set_ui_for_confirmation(self, is_confirming):
        self.is_in_confirmation_mode = is_confirming
        self.batch_process_button.setEnabled(not is_confirming)
        self.confirm_bubbles_button.setVisible(is_confirming)
        if is_confirming: self.update_confirmation_button_text()

    def update_confirmation_button_text(self):
        total_bubbles = sum(len(bubbles) for bubbles in self.detected_bubbles_map.values())
        self.confirm_bubbles_button.setText(f"Confirm & Process ({total_bubbles}) Bubbles")

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

    # --- GLOSSARY METHODS ---
    def add_or_update_glossary_entry(self, source=None, target=None):
        src = source if source is not None else self.glossary_source_edit.text().strip()
        tgt = target if target is not None else self.glossary_target_edit.text().strip()

        if not src or not tgt:
            QMessageBox.warning(self, "Input Error", "Both source and target fields must be filled.")
            return

        self.glossary[src] = tgt
        self.glossary_source_edit.clear()
        self.glossary_target_edit.clear()
        self.update_glossary_list()
        self.statusBar().showMessage(f"Glossary updated: '{src}' -> '{tgt}'", 3000)

    def delete_glossary_entry(self):
        selected_items = self.glossary_list_widget.selectedItems()
        if not selected_items:
            return

        for item in selected_items:
            source_text = item.text().split(' -> ')[0]
            if source_text in self.glossary:
                del self.glossary[source_text]

        self.update_glossary_list()
        self.statusBar().showMessage(f"Deleted {len(selected_items)} glossary entries.", 3000)

    def edit_glossary_item(self, item):
        source_text, target_text = item.text().split(' -> ')
        self.glossary_source_edit.setText(source_text)
        self.glossary_target_edit.setText(target_text)

    def update_glossary_list(self):
        self.glossary_list_widget.clear()
        sorted_glossary = sorted(self.glossary.items())
        for src, tgt in sorted_glossary:
            self.glossary_list_widget.addItem(f"{src} -> {tgt}")

    def save_glossary(self):
        if not self.project_dir: return False
        glossary_path = os.path.join(self.project_dir, "glossary.json")
        try:
            with open(glossary_path, 'w', encoding='utf-8') as f:
                json.dump(self.glossary, f, ensure_ascii=False, indent=4)
            return True
        except Exception as e:
            print(f"Error saving glossary: {e}")
            return False

    def load_glossary(self):
        if not self.project_dir: return False
        glossary_path = os.path.join(self.project_dir, "glossary.json")
        try:
            if os.path.exists(glossary_path):
                with open(glossary_path, 'r', encoding='utf-8') as f:
                    self.glossary = json.load(f)
                self.update_glossary_list()
                self.statusBar().showMessage("Glossary loaded from project.", 3000)
                return True
        except Exception as e:
            self.glossary = {}
            self.update_glossary_list()
            print(f"Error loading glossary: {e}")
        return False

    def save_glossary_manual(self):
        if not self.project_dir:
            QMessageBox.warning(self, "No Project", "Please load a project folder first.")
            return
        if self.save_glossary():
            self.statusBar().showMessage("Glossary saved successfully.", 3000)
        else:
            QMessageBox.critical(self, "Error", "Failed to save glossary.")

    def load_glossary_manual(self):
        if not self.project_dir:
            QMessageBox.warning(self, "No Project", "Please load a project folder first.")
            return
        if self.load_glossary():
             self.statusBar().showMessage("Glossary loaded successfully.", 3000)
        else:
             QMessageBox.information(self, "Info", "No glossary.json file found in the project directory.")

    def start_glossary_suggestion(self, original_text, translated_text):
        if self.glossary_suggestion_thread and self.glossary_suggestion_thread.isRunning():
            return # Don't start a new one if it's already running

        model_name = self.get_selected_model_name()
        if not model_name: return

        self.glossary_suggestion_thread = QThread()
        self.glossary_suggestion_worker = GlossarySuggestionWorker(self, original_text, translated_text, model_name)
        self.glossary_suggestion_worker.moveToThread(self.glossary_suggestion_thread)

        self.glossary_suggestion_worker.signals.suggestions_ready.connect(self.on_suggestions_ready)
        self.glossary_suggestion_worker.signals.error.connect(self.on_worker_error)
        self.glossary_suggestion_worker.signals.finished.connect(self.glossary_suggestion_thread.quit)

        # [FIX v13.0.1] Proper cleanup connection
        self.glossary_suggestion_thread.finished.connect(self.glossary_suggestion_worker.deleteLater)
        self.glossary_suggestion_thread.finished.connect(self.glossary_suggestion_thread.deleteLater)
        self.glossary_suggestion_thread.finished.connect(self.on_glossary_thread_finished)

        self.glossary_suggestion_thread.start()

    def on_glossary_thread_finished(self):
        """[FIX v13.0.1] Cleans up references to the glossary thread and worker after it has finished."""
        self.glossary_suggestion_thread = None
        self.glossary_suggestion_worker = None

    def on_suggestions_ready(self, suggestions):
        for suggestion in suggestions:
            source = suggestion.get('source')
            target = suggestion.get('target')
            if source and target and source not in self.glossary:
                # Add to suggestion list only if not already in main glossary
                # and not already in the suggestion list
                items = [self.glossary_suggestions_list.item(i).text() for i in range(self.glossary_suggestions_list.count())]
                if f"{source} -> {target}" not in items:
                    self.glossary_suggestions_list.addItem(f"{source} -> {target}")

    def add_suggestion_to_glossary(self, item):
        source, target = item.text().split(' -> ')
        self.add_or_update_glossary_entry(source, target)
        # Remove from suggestion list
        self.glossary_suggestions_list.takeItem(self.glossary_suggestions_list.row(item))

    def add_all_suggestions_to_glossary(self):
        for i in range(self.glossary_suggestions_list.count()):
            item = self.glossary_suggestions_list.item(i)
            source, target = item.text().split(' -> ')
            self.glossary[source] = target
        self.glossary_suggestions_list.clear()
        self.update_glossary_list()
        self.statusBar().showMessage("All suggestions added to glossary.", 3000)


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
        if hasattr(self, 'glossary_suggestion_thread') and self.glossary_suggestion_thread and self.glossary_suggestion_thread.isRunning():
            self.glossary_suggestion_thread.quit(); self.glossary_suggestion_thread.wait()
        if hasattr(self, 'inline_editor') and self.inline_editor:
            self.inline_editor.hide(); self.inline_editor.deleteLater(); self.inline_editor = None
        self.save_usage_data(); event.accept()

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
    if not paddleocr: QMessageBox.warning(None, "PaddleOCR Not Found", "'pip install paddleocr paddlepaddle-gpu' to enable the PaddleOCR engine.")
    if not YOLO: QMessageBox.warning(None, "Ultralytics Not Found", "'pip install ultralytics' to enable some DL detectors.")

    window = MangaOCRApp()
    window.showMaximized()
    sys.exit(app.exec_())