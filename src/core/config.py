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

from src.utils.helpers import check_dependency, ensure_dependencies

def get_effective_orientation(settings: dict, ocr_lang: str = ''):
    # SETTINGS may contain a 'lang_orientation' mapping like {'en': 'Horizontal', 'ja': 'Auto-Detect'}
    lang_map = SETTINGS.get('lang_orientation', {})
    # normalize short code
    code = (ocr_lang or '').lower()
    if code.startswith('en') and 'en' in lang_map:
        return lang_map.get('en')
    if code.startswith('ja') and 'ja' in lang_map:
        return lang_map.get('ja')
    # Fallback to per-job or global orientation
    return settings.get('orientation', SETTINGS.get('orientation', 'Auto-Detect'))

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



# --- JSON-based settings (settings.json) ---
# Menentukan ROOT direktori aplikasi agar settings.json diakses secara seragam
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SETTINGS_PATH = os.path.join(ROOT_DIR, 'settings.json')


def default_settings() -> dict:
    if sys.platform.startswith('win'):
        default_tess = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    elif sys.platform.startswith('darwin'):
        default_tess = "/usr/local/bin/tesseract"
    else:
        default_tess = "/usr/bin/tesseract"
    return {
        "general": {
            "save_format": "PNG",  # PNG, WEBP, JPG
            "save_quality": 95
        },
        "apis": {
            "gemini": {"keys": []},
            "openai": {"keys": []},
            "deepl": {"keys": []},
            "google": {"keys": []}
        },
        "tesseract": {
            "path": default_tess,
            "auto_detected": False
        }
        ,
        "cleanup": {
            "use_background_box": True,
            "use_inpaint": True,
            "apply_mode": "selected",
            "text_color_threshold": 128,
            "auto_text_color": True,   # <— BARU: bisa dimatikan dari Settings
            # When true, debug/temp files created by AI OCR and MOFRL (under ./temp/) will be removed after a run
            "remove_ai_temp_files": False,
        },
        "typeset": {
            "outline_enabled": True,
            "outline_thickness": 2,  # legacy key, kept for backward compatibility
            "outline_width": 2.0,
            "outline_color": "#000000",
            "outline_style": "stroke",
        },
        "ocr": {
            "openrouter": {
                "url": "https://openrouter.ai/api/v1/chat/completions",
                "api_key": "",
                "models": []
            },
            "other": {
                "url": "",
                "api_key": "",
                "models": []
            }
        },
        "translate": {
            "openrouter": {
                "url": "https://openrouter.ai/api/v1/chat/completions",
                "api_key": "",
                "models": []
            },
            "other": {
                "url": "",
                "api_key": "",
                "models": []
            }
        },
        "autosave": {
            "enabled": True,
            "interval_ms": 300000
        }
    }


def save_settings(settings: dict, path: str = SETTINGS_PATH):
    try:
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(settings, fh, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed to save settings.json: {e}", file=sys.stderr)


def load_or_create_settings(path: str = SETTINGS_PATH) -> dict:
    try:
        if not os.path.exists(path):
            s = default_settings()
            save_settings(s, path)
            return s
        with open(path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        base = default_settings()
        # Shallow merge for top-level keys
        merged = base
        if isinstance(data, dict):
            merged.update(data)
        # Ensure providers exist
        merged.setdefault('apis', base['apis'])
        for p in base['apis'].keys():
            merged['apis'].setdefault(p, {'keys': []})
        merged.setdefault('tesseract', base['tesseract'])
        cleanup_defaults = base.get('cleanup', {})
        cleanup_settings = merged.setdefault('cleanup', {})
        for key, value in cleanup_defaults.items():
            cleanup_settings.setdefault(key, value)
        ocr_defaults = base.get('ocr', {})
        ocr_settings = merged.setdefault('ocr', {})
        for provider, defaults in ocr_defaults.items():
            provider_cfg = ocr_settings.setdefault(provider, {})
            provider_cfg.setdefault('url', defaults.get('url', ''))
            provider_cfg.setdefault('api_key', defaults.get('api_key', ''))
            models = provider_cfg.get('models')
            if not isinstance(models, list):
                provider_cfg['models'] = []
                models = provider_cfg['models']
            for model in models:
                if not isinstance(model, dict):
                    continue
                model.setdefault('name', '')
                model.setdefault('id', '')
                model['active'] = bool(model.get('active', False))
        translate_defaults = base.get('translate', {})
        translate_settings = merged.setdefault('translate', {})
        for provider, defaults in translate_defaults.items():
            provider_cfg = translate_settings.setdefault(provider, {})
            provider_cfg.setdefault('url', defaults.get('url', ''))
            provider_cfg.setdefault('api_key', defaults.get('api_key', ''))
            models = provider_cfg.get('models')
            if not isinstance(models, list):
                provider_cfg['models'] = []
                models = provider_cfg['models']
            for model in models:
                if not isinstance(model, dict):
                    continue
                model.setdefault('name', '')
                model.setdefault('id', '')
                model.setdefault('description', '')
                model['active'] = bool(model.get('active', True))
        autosave_defaults = base.get('autosave', {})
        autosave_settings = merged.setdefault('autosave', {})
        autosave_settings['enabled'] = bool(autosave_settings.get('enabled', autosave_defaults.get('enabled', True)))
        try:
            interval = int(autosave_settings.get('interval_ms', autosave_defaults.get('interval_ms', 300000)))
        except Exception:
            interval = autosave_defaults.get('interval_ms', 300000)
        autosave_settings['interval_ms'] = max(5000, interval)
        return merged
    except Exception as e:
        print(f"Failed to load settings.json: {e}", file=sys.stderr)
        return default_settings()


# Load settings and expose simple getters for compatibility
SETTINGS = load_or_create_settings()


def get_active_key(provider_name: str) -> str:
    try:
        prov = SETTINGS.get('apis', {}).get(provider_name.lower(), {})
        for k in prov.get('keys', []) or []:
            if k.get('active'):
                return k.get('value') or ''
    except Exception:
        pass
    return ''


DEEPL_API_KEY = get_active_key('deepl')
GEMINI_API_KEY = get_active_key('gemini')
OPENAI_API_KEY = get_active_key('openai')
TESSERACT_PATH = SETTINGS.get('tesseract', {}).get('path', '')


def get_translate_provider_settings(provider_name: str) -> dict:
    try:
        translate_cfg = SETTINGS.get('translate', {})
        return translate_cfg.get(provider_name.lower(), {}) or {}
    except Exception:
        return {}

def refresh_api_clients():
    """Refresh global API key variables and reconfigure provider clients.

    Call this after modifying `SETTINGS` so the running application picks up
    newly added or activated API keys without needing to restart.
    """
    global DEEPL_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY, openai_client
    DEEPL_API_KEY = get_active_key('deepl')
    GEMINI_API_KEY = get_active_key('gemini')
    OPENAI_API_KEY = get_active_key('openai')

    # Reconfigure Gemini (google.generativeai)
    try:
        if GEMINI_API_KEY:
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                print("Gemini configured with active key.", file=sys.stderr)
            except Exception as e:
                print(f"Gagal mengkonfigurasi Gemini API: {e}", file=sys.stderr)
        else:
            print("Gemini: no active key", file=sys.stderr)
    except Exception:
        pass

    # Recreate OpenAI client if available
    openai_client = None
    try:
        if openai and OPENAI_API_KEY:
            try:
                openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
                print("OpenAI client created.", file=sys.stderr)
            except Exception as e:
                print(f"Gagal mengkonfigurasi OpenAI API: {e}", file=sys.stderr)
        else:
            print("OpenAI: no active key or library missing", file=sys.stderr)
    except Exception:
        pass

def detect_tesseract_and_update_settings():
    found = None
    if sys.platform.startswith('win'):
        cand = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(cand):
            found = cand
    if not found and sys.platform.startswith('darwin'):
        cand = '/usr/local/bin/tesseract'
        if os.path.exists(cand):
            found = cand
    if not found:
        cand = '/usr/bin/tesseract'
        if os.path.exists(cand):
            found = cand
    if not found:
        try:
            which_out = shutil.which('tesseract')
            if which_out and os.path.exists(which_out):
                found = which_out
        except Exception:
            pass
    if found:
        SETTINGS.setdefault('tesseract', {})['path'] = found
        SETTINGS.setdefault('tesseract', {})['auto_detected'] = True
        save_settings(SETTINGS)
        return found
    SETTINGS.setdefault('tesseract', {})['auto_detected'] = False
    save_settings(SETTINGS)
    return ''


# If no valid tess path, attempt autodetect
if not TESSERACT_PATH or not os.path.exists(TESSERACT_PATH):
    detected = detect_tesseract_and_update_settings()
    if detected:
        TESSERACT_PATH = detected


# Configure APIs with active keys (best-effort) using centralized helper
openai_client = None

try:
    refresh_api_clients()
except Exception:
    pass


SELECTION_MODE_LABELS = [
    "Bubble Finder (Rect)",
    "Bubble Finder (Oval)",
    "Direct OCR (Rect)",
    "Direct OCR (Oval)",
    "Manual Text (Rect)",
    "Manual Text (Pen)",
    "Pen Tool",
    "Transform (Hand)"
]

SELECTION_MODE_SHORTCUT_KEYS = ["7", "8", "3", "4", "5", "6", "2", "1"]

DEFAULT_SHORTCUTS = {
    'save_project': 'Ctrl+S',
    'load_project': 'Ctrl+O',
    'save_image': 'Ctrl+Shift+S',
    'undo': 'Ctrl+Z',
    'redo': 'Ctrl+Y',
}

for idx, default_key in enumerate(SELECTION_MODE_SHORTCUT_KEYS):
    DEFAULT_SHORTCUTS[f'selection_mode_{idx}'] = default_key

SHORTCUT_DEFINITIONS = [
    ('save_project', "Save Project", "File"),
    ('save_image', "Save Typeset Image", "File"),
    ('load_project', "Load Project", "File"),
    ('undo', "Undo Last Action", "Editing"),
    ('redo', "Redo Last Action", "Editing"),
    ('confirm_pen', "Confirm Pen Selection", "Selection Actions"),
    ('next', "Next Image/Page", "Navigation"),
    ('prev', "Previous Image/Page", "Navigation"),
]

for idx, label in enumerate(SELECTION_MODE_LABELS):
    SHORTCUT_DEFINITIONS.append(
        (f'selection_mode_{idx}', f"Switch to {label}", "Selection Modes")
    )

MOUSE_BUTTON_NAME_MAP = {
    Qt.LeftButton: 'Left',
    Qt.RightButton: 'Right',
    Qt.MiddleButton: 'Middle',
    Qt.BackButton: 'Back',
    Qt.ForwardButton: 'Forward',
    Qt.TaskButton: 'Task',
}

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
        print("Peringatan: Tesseract path dari settings.json tidak valid atau kosong.", file=sys.stderr)
except Exception as e:
    print(f"Gagal mengatur Tesseract path: {e}", file=sys.stderr)
