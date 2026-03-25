# Manga OCR & Typeset Tool v14.3.4
# ==============================
# ?? Import modul bawaan Python
# ==============================
import os
import sys
import time
import json
import re
import hashlib
import pickle
import configparser
import base64
from datetime import date
from openai import OpenAI

# ==============================
# ?? Library pihak ketiga
# ==============================
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*resume_download.*")
# Suppress a noisy transformers/torch FutureWarning about torch.load weights_only default change
warnings.filterwarnings("ignore", message=r".*You are using `torch.load` with `weights_only=False`.*")
import subprocess
import importlib
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
import weakref
import traceback
import copy
import shutil
from functools import partial

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==============================
# ?? PyQt5 (dibagi per kategori)
# ==============================
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QFileDialog, QTextEdit, QScrollArea, QComboBox, QMessageBox,
    QProgressBar, QShortcut, QListWidget, QListWidgetItem, QColorDialog, QFontDialog,
    QLineEdit, QAction, QDialog, QDialogButtonBox, QCheckBox, QStatusBar, QAbstractItemView, QSpinBox,
    QInputDialog,
    QTabWidget, QGroupBox, QGridLayout, QFrame, QSplitter, QRadioButton, QToolButton, QButtonGroup,
    QFormLayout,
    QFontComboBox, QDoubleSpinBox, QMenu, QTableWidget, QTableWidgetItem, QHeaderView, QSlider,
    QKeySequenceEdit
)
from PyQt5.QtGui import (
    QPixmap, QPainter, QPen, QColor, QFont, QKeySequence, QPolygon,
    QPainterPath, QPolygonF, QImage, QIcon, QWheelEvent, QTextDocument,
    QTextCharFormat, QTextCursor, QBrush, QFontMetrics, QTransform, QTextBlockFormat,
    QFontDatabase
)
from PyQt5.QtGui import QCursor
from PyQt5.QtCore import (
    Qt, QRect, QPoint, pyqtSignal, QTimer, QThread, QObject,
    QFileSystemWatcher, QRectF, QMutex, QPointF, QSignalBlocker, QSize, QEvent
)

from src.core.config import *
from src.utils.helpers import *
from src.utils.geometry import *
from src.core.models import EnhancedResult

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
    batch_job_complete = pyqtSignal(str, object, str, str)  # image_path, new_area, original_text, translated_text
    batch_finished = pyqtSignal()           # Sinyal jika semua batch selesai


# Sinyal khusus untuk penyimpanan hasil batch
class BatchSaveSignals(WorkerSignals):
    file_saved = pyqtSignal(str)            # Sinyal jika file berhasil disimpan


# Worker untuk menyimpan project di background agar UI tidak menjadi not responding
class ProjectSaveWorker(QObject):
    finished = pyqtSignal(bool, str)  # success, message
    progress = pyqtSignal(int, str)
    error = pyqtSignal(str)

    def __init__(self, target_path, snapshot):
        super().__init__()
        self.target_path = target_path
        self.snapshot = snapshot
        self._is_cancelled = False

    def run(self):
        tmp_path = self.target_path + '.tmp'
        try:
            # Build payload from already-serialized snapshot (snapshot['typeset_data'] contains primitive dicts)
            payload = {
                'schema_version': 2,
                'project_dir': os.path.abspath(self.snapshot.get('project_dir')) if self.snapshot.get('project_dir') else None,
                'current_image_path': self.snapshot.get('current_image_path'),
                'current_pdf_page': int(self.snapshot.get('current_pdf_page', -1)) if isinstance(self.snapshot.get('current_pdf_page'), int) else int(self.snapshot.get('current_pdf_page', -1)),
                'typeset_data': copy.deepcopy(self.snapshot.get('typeset_data', {})),
                'history_entries': copy.deepcopy(self.snapshot.get('history_entries', [])),
                'proofreader_entries': copy.deepcopy(self.snapshot.get('proofreader_entries', [])),
                'quality_entries': copy.deepcopy(self.snapshot.get('quality_entries', [])),
                'history_counter': int(self.snapshot.get('history_counter', 0)),
                'typeset_font': self.snapshot.get('typeset_font'),
                'typeset_color': self.snapshot.get('typeset_color'),
                'settings': copy.deepcopy(self.snapshot.get('settings', {})),
                'saved_at': time.time(),
                'app_version': self.snapshot.get('app_version', '16.1.0'),
            }

            # Write to temporary file then replace atomically
            with open(tmp_path, 'w', encoding='utf-8') as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.target_path)

            self.finished.emit(True, "Project saved.")
        except Exception as exc:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            self.error.emit(str(exc))
            self.finished.emit(False, f"Failed to save project: {exc}")


# Worker untuk menyimpan gambar (save image) di background
class ImageSaveWorker(QObject):
    finished = pyqtSignal(bool, str)  # success, message
    error = pyqtSignal(str)

    def __init__(self, qimage: QImage, target_path: str, fmt: str = None, quality: int = -1):
        super().__init__()
        self.qimage = qimage
        self.target_path = target_path
        self.fmt = fmt
        self.quality = quality

    def run(self):
        # Determine format from path if not provided
        fmt = self.fmt
        if not fmt:
            _, ext = os.path.splitext(self.target_path)
            if ext:
                fmt = ext.lstrip('.').upper()
                if fmt == 'JPG': fmt = 'JPEG'
            else:
                fmt = 'PNG'
        
        tmp_path = self.target_path + '.tmp'
        try:
            # QImage.save is reentrant and can be used from worker thread
            if not self.qimage.save(tmp_path, fmt, self.quality):
                raise Exception(f'Failed to save temporary image as {fmt}')
            
            if os.path.exists(self.target_path):
                os.remove(self.target_path)
            os.replace(tmp_path, self.target_path)
            
            self.finished.emit(True, f"Image saved to:\n{self.target_path}")
        except Exception as exc:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            self.error.emit(str(exc))
            self.finished.emit(False, f"Failed to save image: {exc}")

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
                    new_area = self.main_app._create_typeset_area(
                        job['rect'],
                        translated_text,
                        settings,
                        polygon=job.get('polygon'),
                        original_text=original_text
                    )
                    if settings.get('ai_model_label'):
                        if not isinstance(new_area.review_notes, dict):
                            new_area.review_notes = {}
                        new_area.review_notes['ai_model'] = settings.get('ai_model_label')
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


    # Pipeline standar: OCR ? Cleaning ? Translate ? Naturalize (opsional)
    def run_standard_pipeline(self, cropped_cv_img, settings: dict, pre_detected_text: str = None):
        # Jika teks sudah dideteksi sebelumnya (mode Text Detect), lewati OCR
        if pre_detected_text:
            raw_text = pre_detected_text
        else:
            # Decide whether to run preprocessing first: only for English and non-Manga engines
            ocr_engine = settings.get('ocr_engine', '')
            ocr_lang = settings.get('ocr_lang', '')
            # If engine is Manga-OCR, always pass the RAW crop (possibly rotated by orientation) because
            # Manga-OCR expects unmodified PIL images and preprocessing often harms its results.
            # Special-case: AI-based OCR engines should receive the pure raw crop without any preprocessing or contrast/threshold changes.
            if ocr_engine in ('AI_OCR', 'MOFRL-GPT'):
                raw_text = self.perform_ocr(cropped_cv_img, settings)
                # skip the rest of preprocessing logic
                processed_text = self.main_app.clean_and_join_text(raw_text)
                if not processed_text or "[ERROR:" in raw_text or "[TESSERACT ERROR:" in raw_text:
                    return raw_text, ""
                # If user requested AI-only translation (or an AI model is configured), use the selected AI model
                ai_model_cfg = settings.get('ai_model') if isinstance(settings, dict) else None
                use_ai_translate = bool(settings.get('use_ai_only_translate')) or bool(ai_model_cfg)
                if use_ai_translate and ai_model_cfg:
                    provider, model_name = ai_model_cfg
                    if not self.wait_for_api_slot(provider, model_name):
                        return processed_text, None
                    try:
                        translated_text = self.main_app.translate_with_ai(processed_text, settings['target_lang'], provider, model_name, settings)
                    except Exception as exc:
                        # fallback to DeepL if AI translate fails
                        try:
                            translated_text = self.main_app.translate_text(processed_text, settings['target_lang'])
                        except Exception:
                            translated_text = f"[TRANSLATE ERROR: {exc}]"
                    return processed_text, translated_text
                else:
                    translated_text = self.main_app.translate_text(processed_text, settings['target_lang'])
                    return processed_text, translated_text
            
            if ocr_engine.lower() in ('manga-ocr', 'mangaocr'):
                # Apply only orientation-based rotation (preserve raw pixel data otherwise)
                orientation = get_effective_orientation(settings, ocr_lang)
                raw_crop = cropped_cv_img
                h, w = raw_crop.shape[:2]
                if orientation == "Vertical" and w > h:
                    raw_crop = cv2.rotate(raw_crop, cv2.ROTATE_90_CLOCKWISE)
                elif orientation == "Horizontal" and h > w:
                    raw_crop = cv2.rotate(raw_crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
                raw_text = self.perform_ocr(raw_crop, settings)
            else:
                needs_preprocessing = (ocr_lang and 'en' in ocr_lang.lower()) and (ocr_engine.lower() not in ('manga-ocr', 'mangaocr'))
                if needs_preprocessing:
                    preprocessed_image, _ = self.main_app.preprocess_for_ocr(cropped_cv_img, get_effective_orientation(settings, ocr_lang))
                    raw_text = self.perform_ocr(preprocessed_image, settings)
                    # fallback to raw crop if preprocessing produced empty/whitespace-only result
                    def is_empty_result(r):
                        if r is None:
                            return True
                        if isinstance(r, (list, tuple)):
                            return all((not (t or '').strip() for t in r))
                        return not (str(r) or '').strip()
                    if is_empty_result(raw_text):
                        # pass raw crop with orientation-only rotation
                        orientation = get_effective_orientation(settings, ocr_lang)
                        raw_crop = cropped_cv_img
                        h, w = raw_crop.shape[:2]
                        if orientation == "Vertical" and w > h:
                            raw_crop = cv2.rotate(raw_crop, cv2.ROTATE_90_CLOCKWISE)
                        elif orientation == "Horizontal" and h > w:
                            raw_crop = cv2.rotate(raw_crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        raw_text = self.perform_ocr(raw_crop, settings)
                else:
                    preprocessed_image, _ = self.main_app.preprocess_for_ocr(cropped_cv_img, get_effective_orientation(settings, ocr_lang))
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

    # Pipeline enhanced: gabungkan hasil Manga-OCR + Tesseract ? AI Pilihan
    def run_enhanced_pipeline(self, cropped_cv_img, settings: dict):
        # For the enhanced pipeline, prefer Manga-OCR on the raw crop (orientation applied only),
        # while Tesseract uses the preprocessed image.
        preprocessed_image, _ = self.main_app.preprocess_for_ocr(cropped_cv_img, "Auto-Detect")

        # Prepare raw crop for Manga-OCR with orientation-only rotation
        orientation = "Auto-Detect"
        raw_crop = cropped_cv_img
        h, w = raw_crop.shape[:2]
        if orientation == "Vertical" and w > h:
            raw_crop = cv2.rotate(raw_crop, cv2.ROTATE_90_CLOCKWISE)
        elif orientation == "Horizontal" and h > w:
            raw_crop = cv2.rotate(raw_crop, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Paksa penggunaan engine yang sesuai untuk pipeline ini
        manga_ocr_settings = {**settings, 'ocr_engine': 'Manga-OCR', 'ocr_lang': 'ja'}
        tesseract_settings = {**settings, 'ocr_engine': 'Tesseract', 'ocr_lang': 'jpn'}

        manga_ocr_text = self.perform_ocr(raw_crop, manga_ocr_settings)
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

    def run(self):
        try:
            jobs_by_image = {}
            for job in self.batch_queue:
                jobs_by_image.setdefault(job['image_path'], []).append(job)

            for image_path, jobs in jobs_by_image.items():
                self.process_image_batch(image_path, jobs)
        except Exception as e:
            self.signals.error.emit(f"Error in batch processor: {e}")
        finally:
            self.signals.batch_finished.emit()

    def process_image_batch(self, image_path, jobs):
        provider, model_name = self.settings['ai_model']

        # 1. OCR per job
        ocr_texts = []
        for job in jobs:
            try:
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
                ocr_texts.append("")
                self.signals.error.emit(f"OCR failed on {image_path}: {e}")

        prompt_lines = [f"{i+1}. {text}" for i, text in enumerate(ocr_texts) if text and "[ERROR:" not in text]
        if not prompt_lines:
            return

        target_lang = self.settings['target_lang']
        prompt_enhancements = self.main_app._build_prompt_enhancements(self.settings)

        # 2. Kalau provider = OPENAI ? gunakan endpoint batch resmi
        if provider.lower() == "openai":
            try:
                client = getattr(self.main_app, "openai_client", None)
                if client is None:
                    client = openai_client

                requests = []
                for i, text in enumerate(ocr_texts):
                    if not text:
                        continue
                    requests.append({
                        "custom_id": f"job-{i+1}",
                        "body": {
                            "model": model_name,
                            "messages": [
                                {
                                    "role": "system",
                                    "content": (
                                        f"You are an expert manga translator. Translate into {target_lang}. "
                                        f"Only return raw translation text."
                                    )
                                },
                                {"role": "user", "content": text}
                            ]
                        }
                    })

                # Submit batch job ke OpenAI
                batch_job = client.batches.create(
                    input_file=requests,
                    endpoint="/v1/chat/completions",
                    completion_window="24h"
                )
                self.signals.info.emit(f"Submitted OpenAI batch {batch_job.id} for {os.path.basename(image_path)}")
                return  # hasil batch akan di-polling async, bukan langsung

            except Exception as e:
                self.signals.error.emit(f"OpenAI batch error on {image_path}: {e}")
                return

        # 3. Kalau provider = GEMINI ? tetap pakai prompt batch (dengan limit aman)
        numbered_ocr_text = "\n".join(prompt_lines)
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

        if not self.main_app.wait_for_api_slot(provider, model_name):
            return

        response_text = self.main_app.call_ai_for_batch(
            prompt,
            provider,
            model_name,
            max_output_tokens=self.settings.get("max_output_tokens", 500012)  # default aman
        )

        if not response_text or "[ERROR]" in response_text or "[FAILED]" in response_text:
            self.signals.error.emit(
                f"Failed to process batch for {os.path.basename(image_path)}: API call failed."
            )
            return

        try:
            translated_lines = response_text.strip().splitlines()
            translation_map = {}
            for line in translated_lines:
                match = re.match(r"^\s*(\d+)\.\s*(.*)", line)
                if match:
                    translation_map[int(match.group(1))] = match.group(2).strip()

            for i, job in enumerate(jobs):
                if not ocr_texts[i]:
                    continue
                translated_text = translation_map.get(i + 1)
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
                        margins=self.settings.get('margins', {'top': 0, 'right': 0, 'bottom': 0, 'left': 0}),
                        original_text=ocr_texts[i],
                        translation_style=self.settings.get('translation_style', '')
                    )
                    if self.settings.get('ai_model_label'):
                        if not isinstance(new_area.review_notes, dict):
                            new_area.review_notes = {}
                        new_area.review_notes['ai_model'] = self.settings.get('ai_model_label')
                    self.signals.batch_job_complete.emit(image_path, new_area, ocr_texts[i], translated_text)

        except Exception as e:
            self.signals.error.emit(
                f"Failed to parse batch response for {os.path.basename(image_path)}: {e}"
            )

# --- Baru: Worker untuk Batch Save ---
class BatchSaveWorker(QObject):
    def __init__(self, main_app, files_to_save, fmt='PNG', quality=-1):
        super().__init__()
        self.main_app = main_app
        self.files_to_save = files_to_save
        self.fmt = fmt.upper()
        self.quality = quality
        self.signals = BatchSaveSignals()
        self.is_cancelled = False

    def run(self):
        total_files = len(self.files_to_save)
        # Map format to extension
        ext_map = {'PNG': '.png', 'JPG': '.jpg', 'JPEG': '.jpg', 'WEBP': '.webp'}
        out_ext = ext_map.get(self.fmt, '.png')

        for i, file_path in enumerate(self.files_to_save):
            if self.is_cancelled:
                break

            self.signals.progress.emit(int(((i + 1) / total_files) * 100), f"Saving {os.path.basename(file_path)}...")

            try:
                # Tentukan nama file output
                path_part, ext = os.path.splitext(file_path)
                save_path = f"{path_part}_typeset{out_ext}"

                # Muat gambar asli
                pil_image = Image.open(file_path).convert('RGB')

                # Konversi PIL.Image ke QImage (lebih aman untuk digunakan dari thread)
                data = pil_image.tobytes('raw', 'RGB')
                qimage = QImage(data, pil_image.width, pil_image.height, pil_image.width * 3, QImage.Format_RGB888).copy()

                # Dapatkan data typeset untuk gambar ini
                data_key = self.main_app.get_current_data_key(path=file_path)
                typeset_data = self.main_app.all_typeset_data.get(data_key, {'areas': []})
                areas = typeset_data['areas']

                if not areas:
                    continue # Lewati jika tidak ada yang perlu di-typeset

                # Gambar ulang semua area ke QImage (thread-safe)
                painter = QPainter()
                try:
                    painter.begin(qimage)
                    for area in areas:
                        # Panggil draw_single_area dengan flag for_saving=True untuk mencegah pembaruan UI
                        self.main_app.draw_single_area(painter, area, pil_image, for_saving=True)
                finally:
                    try:
                        painter.end()
                    except Exception:
                        pass

                # Simpan QImage
                if not qimage.save(save_path, self.fmt, self.quality):
                    raise Exception(f"Failed to save image to {save_path}")

                self.signals.file_saved.emit(file_path)

            except Exception as e:
                self.signals.error.emit(f"Error saving {os.path.basename(file_path)}: {e}")
                continue

        self.signals.finished.emit()

    def cancel(self):
        self.is_cancelled = True
