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
    QFileSystemWatcher, QRectF, QMutex, QPointF, QSignalBlocker, QSize,
    QBuffer, QIODevice
)


# Helper: determine effective orientation considering language-specific overrides
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

# ==============================
# ?? Dependensi opsional (DL & OCR)
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


def ensure_dependencies(parent=None, required=None):
    """Check a list of required packages and optionally prompt to install missing ones.

    required: list of tuples (import_name, pip_name)
    Returns a dict import_name->module or None
    """
    missing = []
    results = {}
    for imp_name, pip_name in required:
        try:
            results[imp_name] = importlib.import_module(imp_name)
        except Exception:
            missing.append((imp_name, pip_name))

    if missing and parent is not None:
        names = ', '.join([m[0] for m in missing])
        resp = QMessageBox.question(parent, "Missing dependencies",
                                    f"The application is missing the following packages: {names}.\nInstall them now?",
                                    QMessageBox.Yes | QMessageBox.No)
        if resp == QMessageBox.Yes:
            for imp_name, pip_name in missing:
                pkg = pip_name or imp_name
                try:
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])
                    results[imp_name] = importlib.import_module(imp_name)
                except Exception as e:
                    QMessageBox.warning(parent, "Install failed", f"Failed to install {pkg}: {e}")
    return results

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


# --- JSON-based settings (settings.json) ---
SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.json')


def default_settings() -> dict:
    if sys.platform.startswith('win'):
        default_tess = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    elif sys.platform.startswith('darwin'):
        default_tess = "/usr/local/bin/tesseract"
    else:
        default_tess = "/usr/bin/tesseract"
    return {
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
            # If true, results from LaMA inpainting will be saved into the project's workshop folder.
            # If false (default), the inpaint result is applied directly to the current image in-memory
            # and not written to disk automatically.
            "lama_save_to_workshop": False,
        },
        "typeset": {
            "outline_enabled": True,
            "outline_thickness": 2,  # legacy key, kept for backward compatibility
            "outline_width": 2.0,
            "outline_color": "#000000"
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


def robust_post(url: str, headers: dict | None = None, json_payload: dict | None = None,
                timeout: int = 60, max_retries: int = 3, backoff_factor: float = 1.5):
    """Perform a POST request with retries and exponential backoff.

    Returns: requests.Response on success.
    Raises: requests.RequestException on persistent failures.
    """
    headers = headers or {}
    attempt = 0
    last_exc = None
    while attempt <= max_retries:
        try:
            attempt += 1
            resp = requests.post(url, headers=headers, json=json_payload, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            # If we've exhausted retries, re-raise
            if attempt > max_retries:
                raise
            # For some client errors (4xx) it's not worth retrying except 429
            if isinstance(exc, requests.HTTPError):
                status = getattr(exc.response, 'status_code', None)
                if status and 400 <= status < 500 and status != 429:
                    # Non-retriable client error
                    raise
            # Sleep with exponential backoff (jitter)
            sleep_time = backoff_factor * (2 ** (attempt - 1))
            # add small jitter
            sleep_time = sleep_time * (0.8 + 0.4 * (os.urandom(1)[0] / 255.0))
            time.sleep(sleep_time)
    # If here, raise last exception
    if last_exc:
        raise last_exc
    raise requests.RequestException("Unknown error in robust_post")
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
    "Brush Inpaint (LaMA)",
    "Transform (Hand)"
]

SELECTION_MODE_SHORTCUT_KEYS = ["7", "8", "3", "4", "5", "6", "2", "0", "1"]

DEFAULT_BIG_LAMA_ENDPOINT = SETTINGS.get('cleanup', {}).get('lama_endpoint', "http://127.0.0.1:5000/inpaint")
DEFAULT_BIG_LAMA_MODEL = SETTINGS.get('cleanup', {}).get('lama_model', "cleaner")

DEFAULT_SHORTCUTS = {
    'save_project': 'Ctrl+S',
    'load_project': 'Ctrl+O',
    'undo': 'Ctrl+Z',
    'redo': 'Ctrl+Y',
}

for idx, default_key in enumerate(SELECTION_MODE_SHORTCUT_KEYS):
    DEFAULT_SHORTCUTS[f'selection_mode_{idx}'] = default_key

SHORTCUT_DEFINITIONS = [
    ('save_project', "Save Project", "File"),
    ('load_project', "Load Project", "File"),
    ('undo', "Undo Last Action", "Editing"),
    ('redo', "Redo Last Action", "Editing"),
]

for idx, label in enumerate(SELECTION_MODE_LABELS):
    SHORTCUT_DEFINITIONS.append(
        (f'selection_mode_{idx}', f"Switch to {label}", "Selection Modes")
    )


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


class APIManagerPanel(QWidget):
    """Panel widget to manage translation and AI OCR API settings."""

    TRANSLATION_PROVIDERS = ['gemini', 'openai', 'deepl', 'google']
    OCR_PROVIDERS = {
        'openrouter': "OpenRouter",
        'other': "Other"
    }

    def __init__(self, initial_settings=None, parent=None):
        super().__init__(parent)
        base_settings = initial_settings or SETTINGS
        self.temp_settings = copy.deepcopy(base_settings)

        self.translation_provider_widgets = {}
        self.ocr_provider_widgets = {}

        main_layout = QVBoxLayout(self)
        self.tab_widget = QTabWidget(self)
        main_layout.addWidget(self.tab_widget)

        self.translation_tab = QWidget()
        self._build_translation_tab()
        self.tab_widget.addTab(self.translation_tab, "Translation APIs")

        self.ocr_tab = QWidget()
        self._build_ocr_tab()
        self.tab_widget.addTab(self.ocr_tab, "AI OCR")

        # Tesseract section below tabs
        tess_group = QGroupBox('Tesseract OCR Path')
        tess_layout = QHBoxLayout(tess_group)
        self.tess_path_edit = QLineEdit(self.temp_settings.get('tesseract', {}).get('path', ''))
        self.tess_path_edit.setPlaceholderText('Path to tesseract executable')
        browse_btn = QPushButton('Browse')
        browse_btn.clicked.connect(self._browse_tesseract)
        tess_layout.addWidget(self.tess_path_edit)
        tess_layout.addWidget(browse_btn)
        tess_hint = QLabel("Set the executable path Tesseract OCR should use.")
        tess_hint.setStyleSheet("color: #9cb4d0; font-size: 11px;")
        tess_layout.addWidget(tess_hint)
        main_layout.addWidget(tess_group)

        self._load_from_settings()

    # ------------------------------------------------------------------
    # UI Builders
    # ------------------------------------------------------------------
    def _build_translation_tab(self):
        layout = QVBoxLayout(self.translation_tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        description = QLabel("Manage translation API keys and select the active key per provider.")
        description.setWordWrap(True)
        description.setStyleSheet("color: #9cb4d0;")
        layout.addWidget(description)

        for provider in self.TRANSLATION_PROVIDERS:
            group = QGroupBox(provider.capitalize())
            gl = QHBoxLayout(group)
            list_widget = QListWidget()
            list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
            list_widget.setMinimumHeight(120)
            gl.addWidget(list_widget, 1)

            button_col = QVBoxLayout()
            add_btn = QPushButton('Add Key')
            add_btn.setToolTip('Add a new API key for this provider')
            rem_btn = QPushButton('Remove Key')
            rem_btn.setToolTip('Remove the selected API key')
            set_btn = QPushButton('Set Active')
            set_btn.setToolTip('Mark the selected key as active')
            button_col.addWidget(add_btn)
            button_col.addWidget(rem_btn)
            button_col.addWidget(set_btn)
            button_col.addStretch()
            gl.addLayout(button_col)

            self.translation_provider_widgets[provider] = {
                'list': list_widget,
                'add': add_btn,
                'remove': rem_btn,
                'set': set_btn
            }

            add_btn.clicked.connect(partial(self._on_add_translation_key, provider))
            rem_btn.clicked.connect(partial(self._on_remove_translation_key, provider))
            set_btn.clicked.connect(partial(self._on_set_translation_active, provider))

            layout.addWidget(group)

        layout.addStretch()

    def _build_ocr_tab(self):
        layout = QVBoxLayout(self.ocr_tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        description = QLabel("Configure AI-powered OCR providers. Set API URL, API key, and manage available models.")
        description.setWordWrap(True)
        description.setStyleSheet("color: #9cb4d0;")
        layout.addWidget(description)

        for provider_key, display_name in self.OCR_PROVIDERS.items():
            group = QGroupBox(display_name)
            group_layout = QVBoxLayout(group)
            group_layout.setSpacing(8)

            info_layout = QGridLayout()
            info_layout.setColumnStretch(1, 1)

            url_edit = QLineEdit()
            url_edit.setPlaceholderText('API endpoint URL')
            url_edit.setToolTip('Enter the full endpoint URL for the OCR provider')
            api_key_edit = QLineEdit()
            api_key_edit.setPlaceholderText('API key')
            api_key_edit.setEchoMode(QLineEdit.Password)
            api_key_edit.setToolTip('Enter the API key for this provider')
            # Small toggle to reveal/hide API key for convenience
            show_key_btn = QPushButton('Show')
            show_key_btn.setCheckable(True)
            show_key_btn.setToolTip('Show/Hide API key')
            def _toggle_key_visibility(ckb, edit=api_key_edit, btn=show_key_btn):
                try:
                    if ckb.isChecked():
                        edit.setEchoMode(QLineEdit.Normal)
                        btn.setText('Hide')
                    else:
                        edit.setEchoMode(QLineEdit.Password)
                        btn.setText('Show')
                except Exception:
                    pass
            show_key_btn.toggled.connect(_toggle_key_visibility)

            info_layout.addWidget(QLabel("API URL"), 0, 0)
            info_layout.addWidget(url_edit, 0, 1)
            info_layout.addWidget(QLabel("API Key"), 1, 0)
            # place API key edit and show/hide button in a small horizontal layout
            key_widget = QWidget()
            key_layout = QHBoxLayout(key_widget)
            key_layout.setContentsMargins(0, 0, 0, 0)
            key_layout.addWidget(api_key_edit)
            key_layout.addWidget(show_key_btn)
            info_layout.addWidget(key_widget, 1, 1)

            warning_label = QLabel("")
            warning_label.setStyleSheet("color: #ff6b6b;")
            info_layout.addWidget(warning_label, 2, 0, 1, 2)

            group_layout.addLayout(info_layout)

            active_label = QLabel("Active Model: None")
            active_label.setStyleSheet("font-weight: 600;")
            group_layout.addWidget(active_label)

            models_table = QTableWidget(0, 3)
            models_table.setHorizontalHeaderLabels(["Model Name", "Model ID", "Active"])
            header = models_table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.Stretch)
            header.setSectionResizeMode(1, QHeaderView.Stretch)
            header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
            models_table.verticalHeader().setVisible(False)
            models_table.setSelectionBehavior(QAbstractItemView.SelectRows)
            models_table.setSelectionMode(QAbstractItemView.SingleSelection)
            models_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            models_table.setAlternatingRowColors(True)
            group_layout.addWidget(models_table)

            button_row = QHBoxLayout()
            add_model_btn = QPushButton("Add Model")
            add_model_btn.setToolTip("Add a new OCR model definition")
            remove_model_btn = QPushButton("Remove Model")
            remove_model_btn.setToolTip("Remove the selected model")
            set_active_btn = QPushButton("Set Active")
            set_active_btn.setToolTip("Mark the selected model as active")
            button_row.addWidget(add_model_btn)
            button_row.addWidget(remove_model_btn)
            button_row.addWidget(set_active_btn)
            button_row.addStretch()
            group_layout.addLayout(button_row)

            add_model_btn.clicked.connect(partial(self._add_ocr_model, provider_key))
            remove_model_btn.clicked.connect(partial(self._remove_ocr_model, provider_key))
            set_active_btn.clicked.connect(partial(self._set_active_ocr_model, provider_key))

            self.ocr_provider_widgets[provider_key] = {
                'url_edit': url_edit,
                'api_key_edit': api_key_edit,
                'models_table': models_table,
                'warning_label': warning_label,
                'active_label': active_label,
                'model_radio_group': None
            }

            layout.addWidget(group)

        layout.addStretch()

    # ------------------------------------------------------------------
    # Loading Helpers
    # ------------------------------------------------------------------
    def _load_from_settings(self):
        self._load_translation_settings()
        self._load_ocr_settings()

    def _load_translation_settings(self):
        apis = self.temp_settings.get('apis', {})
        for provider, widgets in self.translation_provider_widgets.items():
            list_widget: QListWidget = widgets['list']
            list_widget.clear()
            keys = apis.get(provider, {}).get('keys', [])
            for key_info in keys:
                name = key_info.get('name') or (key_info.get('value') or '')[:8]
                active = bool(key_info.get('active'))
                display = f"[ON] {name}" if active else f"[OFF] {name}"
                item = QListWidgetItem(display)
                item.setData(Qt.UserRole, key_info)
                if active:
                    item.setForeground(QColor('#5de6c1'))
                list_widget.addItem(item)

    def _load_ocr_settings(self):
        ocr_settings = self.temp_settings.get('ocr', {})
        for provider, widgets in self.ocr_provider_widgets.items():
            cfg = ocr_settings.get(provider, {})
            widgets['url_edit'].setText(cfg.get('url', ''))
            widgets['api_key_edit'].setText(cfg.get('api_key', ''))
            self._populate_ocr_models_table(provider)

    # ------------------------------------------------------------------
    # Translation API handlers
    # ------------------------------------------------------------------
    def _on_add_translation_key(self, provider):
        name, ok = QInputDialog.getText(self, 'Add API Key', 'Key name (label):')
        if not ok or not name.strip():
            return
        value, ok2 = QInputDialog.getText(self, 'Add API Key', 'Key value:')
        if not ok2 or not value.strip():
            return
        keys = self.temp_settings.setdefault('apis', {}).setdefault(provider, {}).setdefault('keys', [])
        keys.append({'name': name.strip(), 'value': value.strip(), 'active': False})
        self._load_translation_settings()

    def _on_remove_translation_key(self, provider):
        widgets = self.translation_provider_widgets[provider]
        list_widget: QListWidget = widgets['list']
        item = list_widget.currentItem()
        if not item:
            return
        key_data = item.data(Qt.UserRole)
        keys = self.temp_settings.setdefault('apis', {}).setdefault(provider, {}).setdefault('keys', [])
        self.temp_settings['apis'][provider]['keys'] = [k for k in keys if k.get('value') != key_data.get('value')]
        self._load_translation_settings()

    def _on_set_translation_active(self, provider):
        widgets = self.translation_provider_widgets[provider]
        list_widget: QListWidget = widgets['list']
        item = list_widget.currentItem()
        if not item:
            return
        key_data = item.data(Qt.UserRole)
        keys = self.temp_settings.setdefault('apis', {}).setdefault(provider, {}).setdefault('keys', [])
        for entry in keys:
            entry['active'] = (entry.get('value') == key_data.get('value'))
        self._load_translation_settings()

    # ------------------------------------------------------------------
    # AI OCR handlers
    # ------------------------------------------------------------------
    def _populate_ocr_models_table(self, provider):
        widgets = self.ocr_provider_widgets[provider]
        table: QTableWidget = widgets['models_table']
        # Disconnect previous signal if any
        old_group = widgets.get('model_radio_group')
        if old_group is not None:
            try:
                old_group.buttonToggled.disconnect()
            except Exception:
                pass

        table.setRowCount(0)
        models = self.temp_settings.setdefault('ocr', {}).setdefault(provider, {}).setdefault('models', [])
        radio_group = QButtonGroup(table)
        radio_group.setExclusive(True)

        for row, model in enumerate(models):
            table.insertRow(row)
            name_item = QTableWidgetItem(model.get('name', ''))
            id_item = QTableWidgetItem(model.get('id', ''))
            name_item.setFlags(name_item.flags() ^ Qt.ItemIsEditable)
            id_item.setFlags(id_item.flags() ^ Qt.ItemIsEditable)
            table.setItem(row, 0, name_item)
            table.setItem(row, 1, id_item)

            radio = QRadioButton()
            radio.setProperty('row', row)
            radio.setChecked(bool(model.get('active')))
            radio_widget = QWidget()
            radio_layout = QHBoxLayout(radio_widget)
            radio_layout.setContentsMargins(0, 0, 0, 0)
            radio_layout.setAlignment(Qt.AlignCenter)
            radio_layout.addWidget(radio)
            table.setCellWidget(row, 2, radio_widget)
            radio_group.addButton(radio, row)

        def on_radio_toggled(button, checked, prov=provider):
            self._on_ocr_radio_toggled(prov, button, checked)

        radio_group.buttonToggled.connect(on_radio_toggled)
        widgets['model_radio_group'] = radio_group
        self._update_ocr_active_display(provider)

    def _on_ocr_radio_toggled(self, provider, button, checked):
        if not checked:
            return
        row = button.property('row')
        models = self.temp_settings.setdefault('ocr', {}).setdefault(provider, {}).setdefault('models', [])
        for idx, model in enumerate(models):
            model['active'] = (idx == row)
        self._update_ocr_active_display(provider)

    def _update_ocr_active_display(self, provider):
        widgets = self.ocr_provider_widgets[provider]
        table: QTableWidget = widgets['models_table']
        models = self.temp_settings.setdefault('ocr', {}).setdefault(provider, {}).setdefault('models', [])
        active_index = next((idx for idx, m in enumerate(models) if m.get('active')), -1)

        if active_index >= 0:
            active_name = models[active_index].get('name', '(Unnamed)')
            widgets['active_label'].setText(f"Active Model: {active_name}")
        else:
            widgets['active_label'].setText("Active Model: None")

        for row in range(table.rowCount()):
            is_active = (row == active_index)
            for col in range(2):
                item = table.item(row, col)
                if not item:
                    continue
                if is_active:
                    item.setBackground(QColor('#234162'))
                    item.setForeground(QColor('#f3f6fb'))
                else:
                    item.setBackground(QColor('#1a2634'))
                    item.setForeground(QColor('#e0e8f5'))
            widget = table.cellWidget(row, 2)
            if widget:
                pal = widget.palette()
                pal.setColor(widget.backgroundRole(), QColor('#234162') if is_active else QColor('#1a2634'))
                widget.setPalette(pal)
                widget.setAutoFillBackground(True)

    def _add_ocr_model(self, provider):
        name, ok = QInputDialog.getText(self, "Add OCR Model", "Model Name:")
        if not ok or not name.strip():
            return
        model_id, ok2 = QInputDialog.getText(self, "Add OCR Model", "Model ID:")
        if not ok2 or not model_id.strip():
            return

        models = self.temp_settings.setdefault('ocr', {}).setdefault(provider, {}).setdefault('models', [])
        is_first = len(models) == 0
        models.append({'name': name.strip(), 'id': model_id.strip(), 'active': is_first})
        if is_first:
            # ensure only one active
            for model in models[1:]:
                model['active'] = False
        self._populate_ocr_models_table(provider)

    def _remove_ocr_model(self, provider):
        widgets = self.ocr_provider_widgets[provider]
        table: QTableWidget = widgets['models_table']
        row = table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Remove Model", "Select a model to remove.")
            return
        confirm = QMessageBox.question(self, "Remove Model", "Remove selected model?",
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if confirm != QMessageBox.Yes:
            return
        models = self.temp_settings.setdefault('ocr', {}).setdefault(provider, {}).setdefault('models', [])
        if row >= len(models):
            return
        was_active = bool(models[row].get('active'))
        models.pop(row)
        if was_active and models:
            models[0]['active'] = True
            for model in models[1:]:
                model['active'] = False
        self._populate_ocr_models_table(provider)

    def _set_active_ocr_model(self, provider):
        widgets = self.ocr_provider_widgets[provider]
        table: QTableWidget = widgets['models_table']
        row = table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Set Active", "Select a model to activate.")
            return
        models = self.temp_settings.setdefault('ocr', {}).setdefault(provider, {}).setdefault('models', [])
        if row >= len(models):
            return
        for idx, model in enumerate(models):
            model['active'] = (idx == row)
        self._populate_ocr_models_table(provider)

    # ------------------------------------------------------------------
    # Save / Close helpers
    # ------------------------------------------------------------------
    def _browse_tesseract(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Select Tesseract executable')
        if path:
            self.tess_path_edit.setText(path)

    def _apply_latest_form_values(self):
        self._validation_messages = []
        ocr_settings = self.temp_settings.setdefault('ocr', {})
        for provider, widgets in self.ocr_provider_widgets.items():
            provider_cfg = ocr_settings.setdefault(provider, {})
            url = widgets['url_edit'].text().strip()
            api_key = widgets['api_key_edit'].text().strip()
            provider_cfg['url'] = url
            provider_cfg['api_key'] = api_key
            models = provider_cfg.setdefault('models', [])

            warnings = []
            if models:
                if not url:
                    warnings.append("API URL is required when models are configured.")
                if not api_key:
                    warnings.append("API Key is required when models are configured.")
            widgets['warning_label'].setText("\n".join(warnings))
            if warnings:
                display_name = self.OCR_PROVIDERS.get(provider, provider.title())
                for msg in warnings:
                    self._validation_messages.append(f"{display_name}: {msg}")

        tess_cfg = self.temp_settings.setdefault('tesseract', {})
        tess_cfg['path'] = self.tess_path_edit.text().strip()
        tess_cfg['auto_detected'] = False

        return not self._validation_messages

    def export_settings(self):
        if not self._apply_latest_form_values():
            return None
        return {
            'apis': copy.deepcopy(self.temp_settings.get('apis', {})),
            'ocr': copy.deepcopy(self.temp_settings.get('ocr', {})),
            'tesseract': copy.deepcopy(self.temp_settings.get('tesseract', {})),
        }

    def validation_messages(self):
        return list(getattr(self, '_validation_messages', []))


class APIManagerDialog(QDialog):
    """Modal wrapper that reuses APIManagerPanel and persists changes."""

    TRANSLATION_PROVIDERS = APIManagerPanel.TRANSLATION_PROVIDERS
    OCR_PROVIDERS = APIManagerPanel.OCR_PROVIDERS

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('API Manager')
        self.setModal(True)
        self.resize(860, 520)

        layout = QVBoxLayout(self)
        self.panel = APIManagerPanel(SETTINGS, self)
        layout.addWidget(self.panel)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel, self)
        save_btn = buttons.button(QDialogButtonBox.Save)
        save_btn.setText("Save")
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_save(self):
        exported = self.panel.export_settings()
        if exported is None:
            warnings = "\n".join(self.panel.validation_messages())
            details = f"\n\n{warnings}" if warnings else ""
            QMessageBox.warning(self, "Validation", f"Please fix the highlighted issues before saving.{details}")
            return

        SETTINGS.setdefault('apis', {})
        SETTINGS.setdefault('ocr', {})
        SETTINGS.setdefault('tesseract', {})
        SETTINGS['apis'] = copy.deepcopy(exported.get('apis', {}))
        SETTINGS['ocr'] = copy.deepcopy(exported.get('ocr', {}))
        SETTINGS['tesseract'] = copy.deepcopy(exported.get('tesseract', {}))
        save_settings(SETTINGS)

        try:
            refresh_api_clients()
        except Exception:
            pass

        parent = self.parent()
        if parent and hasattr(parent, 'populate_ocr_languages'):
            try:
                parent.populate_ocr_languages()
            except Exception:
                pass

        QMessageBox.information(self, "Success", "API settings updated successfully.")
        self.accept()
# --------------------------------------------------------------------
# Font management helpers


class FontManager:
    """Utility class for loading and tracking custom font files."""

    SUPPORTED_EXTENSIONS = {".ttf", ".otf", ".ttc", ".otc"}

    def __init__(self, font_dir: str):
        self.font_dir = os.path.abspath(font_dir)
        self._fonts = {}
        self._family_lookup = {}
        self._default_display = "System Default"
        self.ensure_font_dir()
        self.reload_fonts()

    def ensure_font_dir(self):
        try:
            os.makedirs(self.font_dir, exist_ok=True)
        except OSError:
            pass

    def reload_fonts(self):
        self._fonts.clear()
        self._family_lookup.clear()
        self.ensure_font_dir()

        default_font = QFont()
        default_family = default_font.family()
        self._fonts[self._default_display] = {
            'display': self._default_display,
            'path': None,
            'families': [default_family] if default_family else [],
            'font_id': None,
            'is_system': True,
        }
        if default_family:
            self._family_lookup[default_family] = self._default_display

        for entry in sorted(os.listdir(self.font_dir)):
            path = os.path.join(self.font_dir, entry)
            if not os.path.isfile(path):
                continue
            ext = os.path.splitext(entry)[1].lower()
            if ext not in self.SUPPORTED_EXTENSIONS:
                continue
            display_name = os.path.splitext(entry)[0]
            font_id = QFontDatabase.addApplicationFont(path)
            if font_id == -1:
                continue
            families = QFontDatabase.applicationFontFamilies(font_id)
            if not families:
                QFontDatabase.removeApplicationFont(font_id)
                continue
            self._fonts[display_name] = {
                'display': display_name,
                'path': path,
                'families': families,
                'font_id': font_id,
                'is_system': False,
            }
            for fam in families:
                self._family_lookup[fam] = display_name

    def list_fonts(self):
        names = sorted(name for name, meta in self._fonts.items() if not meta.get('is_system'))
        return [self._default_display] + names

    def has_font(self, display_name: str) -> bool:
        return display_name in self._fonts

    @property
    def default_display(self) -> str:
        return self._default_display

    def create_qfont(self, display_name: str, base_font: QFont | None = None) -> QFont:
        font = QFont(base_font) if isinstance(base_font, QFont) else QFont()
        meta = self._fonts.get(display_name)
        if not meta:
            return font
        families = meta.get('families') or []
        if families:
            font.setFamily(families[0])
        return font

    def display_name_for_font(self, font: QFont | None, fallback: str | None = None) -> str:
        if not isinstance(font, QFont):
            return fallback or self._default_display
        family = font.family()
        if family and family in self._family_lookup:
            return self._family_lookup[family]
        return fallback or self._default_display

    def import_font(self, source_path: str) -> str:
        if not source_path:
            raise ValueError("Missing font path")

        ext = os.path.splitext(source_path)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported font format: {ext}")

        display_name = os.path.splitext(os.path.basename(source_path))[0]
        dest_path = os.path.join(self.font_dir, os.path.basename(source_path))

        if os.path.exists(dest_path) or self.has_font(display_name):
            raise FileExistsError(f"Font '{display_name}' already exists")

        shutil.copy2(source_path, dest_path)
        registered = self._register_font_file(dest_path)
        if not registered:
            try:
                os.remove(dest_path)
            except OSError:
                pass
            raise RuntimeError(f"Failed to load font '{display_name}'")
        return registered

    def _register_font_file(self, path: str) -> str | None:
        font_id = QFontDatabase.addApplicationFont(path)
        if font_id == -1:
            return None
        families = QFontDatabase.applicationFontFamilies(font_id)
        if not families:
            QFontDatabase.removeApplicationFont(font_id)
            return None
        display_name = os.path.splitext(os.path.basename(path))[0]
        self._fonts[display_name] = {
            'display': display_name,
            'path': path,
            'families': families,
            'font_id': font_id,
            'is_system': False,
        }
        for fam in families:
            self._family_lookup[fam] = display_name
        return display_name


GLOBAL_FONT_MANAGER: FontManager | None = None


def set_global_font_manager(manager: FontManager):
    global GLOBAL_FONT_MANAGER
    GLOBAL_FONT_MANAGER = manager


def get_font_manager() -> FontManager | None:
    return GLOBAL_FONT_MANAGER


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

    def __init__(self, qimage: QImage, target_path: str):
        super().__init__()
        self.qimage = qimage
        self.target_path = target_path

    def run(self):
        tmp_path = self.target_path + '.tmp'
        try:
            # QImage.save is reentrant and can be used from worker thread
            if not self.qimage.save(tmp_path, 'PNG'):
                raise Exception('Failed to save temporary image')
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


# Simple settings dialog to configure autosave
class SettingsDialog(QDialog):
    def __init__(self, parent=None, autosave_enabled: bool = True, autosave_interval_ms: int = 300000):
        super().__init__(parent)
        self.setWindowTitle('Preferences')
        self.setModal(True)
        self.resize(360, 140)

        layout = QVBoxLayout(self)

        # Autosave enable
        self.autosave_checkbox = QCheckBox('Enable autosave', self)
        self.autosave_checkbox.setChecked(bool(autosave_enabled))
        layout.addWidget(self.autosave_checkbox)

        # Interval (seconds)
        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel('Autosave interval (seconds):'))
        self.interval_spin = QSpinBox(self)
        self.interval_spin.setRange(5, 3600)  # 5s..1h
        # store/display in seconds for user convenience
        self.interval_spin.setValue(max(5, int(autosave_interval_ms / 1000)))
        interval_layout.addWidget(self.interval_spin)
        layout.addLayout(interval_layout)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_values(self):
        return {
            'enabled': bool(self.autosave_checkbox.isChecked()),
            'interval_ms': int(self.interval_spin.value() * 1000)
        }


class ModelEditDialog(QDialog):
    def __init__(self, parent=None, name: str = "", model_id: str = "", description: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Model Settings")
        self.setModal(True)
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name_edit = QLineEdit(name)
        self.name_edit.setPlaceholderText("Display name")
        self.id_edit = QLineEdit(model_id)
        self.id_edit.setPlaceholderText("provider/model-id")
        self.description_edit = QLineEdit(description)
        self.description_edit.setPlaceholderText("Optional description")
        form.addRow("Model Name", self.name_edit)
        form.addRow("Model ID", self.id_edit)
        form.addRow("Description", self.description_edit)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_values(self):
        return {
            "name": self.name_edit.text().strip(),
            "id": self.id_edit.text().strip(),
            "description": self.description_edit.text().strip()
        }


class OpenRouterSettingsPanel(QWidget):
    def __init__(self, initial_settings=None, parent=None):
        super().__init__(parent)

        translate_cfg = (initial_settings or SETTINGS).get('translate', {})
        self.data = copy.deepcopy(translate_cfg.get('openrouter', {}))
        self.data.setdefault('url', "https://openrouter.ai/api/v1/chat/completions")
        self.data.setdefault('api_key', "")
        self.data.setdefault('models', [])
        self.models = copy.deepcopy(self.data.get('models', []))

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget(self)
        layout.addWidget(self.tabs, 1)

        self._build_api_tab()
        self._build_models_tab()

    def _build_api_tab(self):
        api_widget = QWidget()
        form = QFormLayout(api_widget)
        self.url_edit = QLineEdit(self.data.get('url', ''))
        self.api_key_edit = QLineEdit(self.data.get('api_key', ''))
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        key_layout = QHBoxLayout()
        key_layout.setContentsMargins(0, 0, 0, 0)
        key_layout.addWidget(self.api_key_edit, 1)
        self.api_key_toggle = QToolButton()
        self.api_key_toggle.setText("Show")
        self.api_key_toggle.setCheckable(True)
        self.api_key_toggle.toggled.connect(self._toggle_api_key_visibility)
        key_layout.addWidget(self.api_key_toggle)
        key_widget = QWidget()
        key_widget.setLayout(key_layout)
        form.addRow("API URL", self.url_edit)
        form.addRow("API Key", key_widget)
        # Provider tuning
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(5, 600)
        self.timeout_spin.setValue(int(self.data.get('timeout', 60) or 60))
        form.addRow("Timeout (s)", self.timeout_spin)
        self.retries_spin = QSpinBox()
        self.retries_spin.setRange(0, 10)
        self.retries_spin.setValue(int(self.data.get('retries', 3) or 3))
        form.addRow("Retries", self.retries_spin)
        self.backoff_spin = QDoubleSpinBox()
        self.backoff_spin.setRange(0.1, 10.0)
        self.backoff_spin.setSingleStep(0.1)
        self.backoff_spin.setValue(float(self.data.get('backoff', 1.5) or 1.5))
        form.addRow("Backoff factor", self.backoff_spin)
        help_label = QLabel("Tip: Find your OpenRouter API key at https://openrouter.ai/account")
        help_label.setStyleSheet("color: #9cb4d0;")
        help_label.setWordWrap(True)
        form.addRow(help_label)
        self.tabs.addTab(api_widget, "API Configuration")

    def _build_models_tab(self):
        models_widget = QWidget()
        vbox = QVBoxLayout(models_widget)
        info = QLabel("Add translation models to call via OpenRouter. Multiple models can be active at the same time.")
        info.setWordWrap(True)
        info.setStyleSheet("color: #9cb4d0;")
        vbox.addWidget(info)

        self.models_table = QTableWidget(0, 4, self)
        self.models_table.setHorizontalHeaderLabels(["Model Name", "Model ID", "Description", "Active"])
        header = self.models_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.models_table.verticalHeader().setVisible(False)
        self.models_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.models_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.models_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.models_table.setAlternatingRowColors(True)
        vbox.addWidget(self.models_table, 1)

        button_row = QHBoxLayout()
        self.add_button = QPushButton("Add Model")
        self.add_button.clicked.connect(self._add_model)
        self.edit_button = QPushButton("Edit Model")
        self.edit_button.clicked.connect(self._edit_model)
        self.remove_button = QPushButton("Remove Model")
        self.remove_button.clicked.connect(self._remove_model)
        button_row.addWidget(self.add_button)
        button_row.addWidget(self.edit_button)
        button_row.addWidget(self.remove_button)
        button_row.addStretch()
        vbox.addLayout(button_row)

        self.tabs.addTab(models_widget, "Models")
        self._refresh_models_table()

    def _toggle_api_key_visibility(self, checked: bool):
        self.api_key_edit.setEchoMode(QLineEdit.Normal if checked else QLineEdit.Password)
        self.api_key_toggle.setText("Hide" if checked else "Show")

    def _refresh_models_table(self):
        self.models_table.setRowCount(0)
        for row, model in enumerate(self.models):
            name = model.get('name', '')
            model_id = model.get('id', '')
            desc = model.get('description', '')
            active = bool(model.get('active', True))
            self.models_table.insertRow(row)
            name_item = QTableWidgetItem(name)
            model_item = QTableWidgetItem(model_id)
            desc_item = QTableWidgetItem(desc)
            for item in (name_item, model_item, desc_item):
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
            self.models_table.setItem(row, 0, name_item)
            self.models_table.setItem(row, 1, model_item)
            self.models_table.setItem(row, 2, desc_item)
            checkbox = QCheckBox()
            checkbox.setChecked(active)
            checkbox.stateChanged.connect(lambda state, r=row: self._set_model_active(r, state))
            container = QWidget()
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setAlignment(Qt.AlignCenter)
            layout.addWidget(checkbox)
            self.models_table.setCellWidget(row, 3, container)

    def _set_model_active(self, row: int, state: int):
        if 0 <= row < len(self.models):
            self.models[row]['active'] = (state == Qt.Checked)

    def _add_model(self):
        dialog = ModelEditDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            values = dialog.get_values()
            if not values['name'] or not values['id']:
                QMessageBox.warning(self, "Invalid Model", "Model name and ID are required.")
                return
            self.models.append({
                'name': values['name'],
                'id': values['id'],
                'description': values['description'],
                'active': True
            })
            self._refresh_models_table()

    def _edit_model(self):
        row = self.models_table.currentRow()
        if not (0 <= row < len(self.models)):
            QMessageBox.information(self, "Edit Model", "Select a model to edit.")
            return
        model = self.models[row]
        dialog = ModelEditDialog(self, name=model.get('name', ''), model_id=model.get('id', ''), description=model.get('description', ''))
        if dialog.exec_() == QDialog.Accepted:
            values = dialog.get_values()
            if not values['name'] or not values['id']:
                QMessageBox.warning(self, "Invalid Model", "Model name and ID are required.")
                return
            model.update({
                'name': values['name'],
                'id': values['id'],
                'description': values['description']
            })
            self._refresh_models_table()

    def _remove_model(self):
        row = self.models_table.currentRow()
        if not (0 <= row < len(self.models)):
            QMessageBox.information(self, "Remove Model", "Select a model to remove.")
            return
        confirm = QMessageBox.question(self, "Remove Model", f"Remove model '{self.models[row].get('name', '')}'?",
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if confirm == QMessageBox.Yes:
            self.models.pop(row)
            self._refresh_models_table()

    def _on_lang_orientation_changed(self, lang_code, value):
        # Ensure mapping exists
        lang_map = SETTINGS.setdefault('lang_orientation', {})
        lang_map[lang_code] = value
        save_settings(SETTINGS)

    def export_settings(self):
        self.data['url'] = self.url_edit.text().strip() or "https://openrouter.ai/api/v1/chat/completions"
        self.data['api_key'] = self.api_key_edit.text().strip()
        try:
            self.data['timeout'] = int(self.timeout_spin.value())
            self.data['retries'] = int(self.retries_spin.value())
            self.data['backoff'] = float(self.backoff_spin.value())
        except Exception:
            pass
        self.data['models'] = copy.deepcopy(self.models)
        return copy.deepcopy(self.data)

    def get_settings(self):
        return self.export_settings()


class OpenRouterSettingsDialog(QDialog):
    """Backward-compatible dialog wrapper around OpenRouterSettingsPanel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("OpenRouter Translate Settings")
        self.setModal(True)
        self.resize(640, 480)

        layout = QVBoxLayout(self)
        self.panel = OpenRouterSettingsPanel(SETTINGS, self)
        layout.addWidget(self.panel)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self):
        self.panel.export_settings()
        self.accept()

    def get_settings(self):
        return self.panel.get_settings()


class SettingsCenterDialog(QDialog):
    """Unified settings dialog that groups the most-used options by category."""

    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.setWindowTitle("Settings")
        self.setModal(True)
        # Start with a reasonable size and cap height so the dialog doesn't grow off-screen.
        self.resize(860, 620)
        try:
            # Prevent the dialog from becoming excessively tall on multi-monitor setups
            self.setMaximumHeight(900)
        except Exception:
            pass

        self._initial_autosave_enabled, self._initial_autosave_interval = self._current_autosave_state()
        self._initial_cleanup = copy.deepcopy(SETTINGS.get('cleanup', {}))
        self._initial_api = {
            'apis': copy.deepcopy(SETTINGS.get('apis', {})),
            'ocr': copy.deepcopy(SETTINGS.get('ocr', {})),
            'tesseract': copy.deepcopy(SETTINGS.get('tesseract', {})),
        }
        self._initial_translate = copy.deepcopy(SETTINGS.get('translate', {}).get('openrouter', {}))
        self._initial_shortcuts = copy.deepcopy(SETTINGS.get('shortcuts', {}))
        self.shortcut_editors = {}

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.tabs = QTabWidget(self)
        self.tabs.setDocumentMode(True)
        self.tabs.setTabPosition(QTabWidget.West)
        self.tabs.setMovable(False)

        # Wrap tabs in a scroll area so the dialog height stays bounded and long content scrolls
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.addWidget(self.tabs)
        scroll.setWidget(container)
        layout.addWidget(scroll, 1)

        self.general_tab = self._create_general_tab()
        self.cleanup_tab = self._create_cleanup_tab()
        self.translation_tab = self._create_translation_tab()
        self.shortcuts_tab = self._create_shortcuts_tab()
        self.api_tab = self._create_api_tab()

        self.tabs.addTab(self.general_tab, "General")
        self.tabs.addTab(self.cleanup_tab, "Cleanup")
        self.tabs.addTab(self.translation_tab, "Translation")
        self.tabs.addTab(self.shortcuts_tab, "Shortcuts")
        self.tabs.addTab(self.api_tab, "API Keys")

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _create_general_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        autosave_group = QGroupBox("Autosave")
        autosave_layout = QVBoxLayout(autosave_group)
        autosave_layout.setSpacing(8)

        self.autosave_checkbox = QCheckBox("Enable autosave", autosave_group)
        autosave_layout.addWidget(self.autosave_checkbox)

        interval_layout = QHBoxLayout()
        interval_layout.setContentsMargins(0, 0, 0, 0)
        interval_layout.setSpacing(8)
        interval_layout.addWidget(QLabel("Autosave interval (seconds):"))
        self.autosave_interval_spin = QSpinBox()
        self.autosave_interval_spin.setRange(5, 3600)
        self.autosave_interval_spin.setSingleStep(5)
        interval_layout.addWidget(self.autosave_interval_spin, 1)
        autosave_layout.addLayout(interval_layout)

        note = QLabel("Autosave stores a backup of your current project at the selected interval.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #9cb4d0;")
        autosave_layout.addWidget(note)

        layout.addWidget(autosave_group)
        layout.addStretch(1)

        self.autosave_checkbox.setChecked(self._initial_autosave_enabled)
        self.autosave_interval_spin.setValue(max(5, int(self._initial_autosave_interval / 1000)))

        return widget

    def _create_cleanup_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        cleanup_group = QGroupBox("Text cleanup defaults")
        form = QFormLayout(cleanup_group)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(12)

        self.auto_text_color_checkbox = QCheckBox("Automatically pick contrasting text color")
        form.addRow("Auto text color", self.auto_text_color_checkbox)

        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(0, 255)
        self.threshold_spin.setToolTip("Threshold for deciding when to invert text color. Higher values prefer brighter text.")
        form.addRow("Color threshold", self.threshold_spin)

        self.use_background_box_checkbox = QCheckBox("Draw a background box behind new text by default")
        form.addRow("Background box", self.use_background_box_checkbox)

        hint = QLabel("These preferences act as defaults for new cleanup and translation areas.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #9cb4d0;")

        layout.addWidget(cleanup_group)
        layout.addWidget(hint)
        layout.addStretch(1)

        cleanup_cfg = SETTINGS.get('cleanup', {})
        self.auto_text_color_checkbox.setChecked(bool(cleanup_cfg.get('auto_text_color', True)))
        self.threshold_spin.setValue(int(cleanup_cfg.get('text_color_threshold', 128)))
        self.use_background_box_checkbox.setChecked(bool(cleanup_cfg.get('use_background_box', True)))
        # New: option to remove AI debug/temp files after OCR/translation runs
        self.remove_ai_temp_checkbox = QCheckBox("Remove AI debug/temp files after run (AI OCR & MOFRL)")
        self.remove_ai_temp_checkbox.setToolTip("If checked, temporary debug files saved under the project's temp/ folder will be deleted after successful runs.")
        form.addRow("Remove AI temp files", self.remove_ai_temp_checkbox)
        self.remove_ai_temp_checkbox.setChecked(bool(cleanup_cfg.get('remove_ai_temp_files', False)))

        return widget

    def _create_translation_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.openrouter_panel = OpenRouterSettingsPanel(SETTINGS, widget)
        layout.addWidget(self.openrouter_panel)

        return widget

    def _create_shortcuts_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(20, 20, 20, 20)
        container_layout.setSpacing(16)

        hint = QLabel("Assign custom keyboard shortcuts for frequently used actions. "
                      "Leave blank to disable a shortcut, or press Default to restore the original binding.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #9cb4d0;")
        container_layout.addWidget(hint)

        category_order = []
        grouped = {}
        for key, label, category in SHORTCUT_DEFINITIONS:
            grouped.setdefault(category, []).append((key, label))
            if category not in category_order:
                category_order.append(category)

        user_shortcuts = SETTINGS.get('shortcuts', {}) or {}
        for category in category_order:
            entries = grouped.get(category, [])
            if not entries:
                continue
            group_box = QGroupBox(category)
            form = QFormLayout(group_box)
            form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            form.setHorizontalSpacing(18)
            form.setVerticalSpacing(12)

            for key, label in entries:
                editor = QKeySequenceEdit()
                seq = user_shortcuts.get(key)
                if seq is None:
                    seq = DEFAULT_SHORTCUTS.get(key, '')
                if seq:
                    editor.setKeySequence(QKeySequence(seq))
                else:
                    editor.clear()

                row_widget = QWidget()
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(6)
                row_layout.addWidget(editor, 1)

                clear_btn = QToolButton()
                clear_btn.setText("Clear")
                clear_btn.clicked.connect(editor.clear)
                row_layout.addWidget(clear_btn)

                default_btn = QToolButton()
                default_btn.setText("Default")

                def _reset_editor(target_editor=editor, target_key=key):
                    default_seq = DEFAULT_SHORTCUTS.get(target_key, '')
                    if default_seq:
                        target_editor.setKeySequence(QKeySequence(default_seq))
                    else:
                        target_editor.clear()

                default_btn.clicked.connect(_reset_editor)
                row_layout.addWidget(default_btn)

                form.addRow(label, row_widget)
                self.shortcut_editors[key] = editor

            container_layout.addWidget(group_box)

        container_layout.addStretch(1)
        scroll.setWidget(container)
        layout.addWidget(scroll)

        return widget

    def _create_api_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.api_panel = APIManagerPanel(SETTINGS, widget)
        layout.addWidget(self.api_panel)

        return widget

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def set_active_tab(self, key: str):
        if not key:
            return
        tab_map = {
            'general': self.general_tab,
            'cleanup': self.cleanup_tab,
            'translation': self.translation_tab,
            'shortcuts': self.shortcuts_tab,
            'shortcut': self.shortcuts_tab,
            'api': self.api_tab,
            'apis': self.api_tab,
            'keys': self.api_tab,
        }
        widget = tab_map.get(str(key).lower())
        if widget is not None:
            index = self.tabs.indexOf(widget)
            if index >= 0:
                self.tabs.setCurrentIndex(index)

    def _current_autosave_state(self):
        enabled = False
        interval_ms = 300000
        try:
            timer = getattr(self.main_window, 'autosave_timer', None)
            if timer is not None:
                interval_ms = timer.interval()
                enabled = timer.isActive()
        except Exception:
            pass
        if hasattr(self.main_window, 'autosave_enabled'):
            enabled = bool(getattr(self.main_window, 'autosave_enabled'))
        return enabled, max(5000, int(interval_ms))

    def _apply_autosave_settings(self, enabled, interval_ms):
        try:
            timer = getattr(self.main_window, 'autosave_timer', None)
            if timer is None:
                timer = QTimer(self.main_window)
                timer.timeout.connect(self.main_window.auto_save_project)
                self.main_window.autosave_timer = timer
            timer.setInterval(int(interval_ms))
            self.main_window.autosave_enabled = bool(enabled)
            if enabled:
                timer.start()
            else:
                timer.stop()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Save handler
    # ------------------------------------------------------------------
    def _on_save(self):
        api_export = self.api_panel.export_settings()
        if api_export is None:
            warnings = "\n".join(self.api_panel.validation_messages())
            details = f"\n\n{warnings}" if warnings else ""
            QMessageBox.warning(self, "API Settings", f"Please fix the highlighted API settings issues before saving.{details}")
            self.tabs.setCurrentWidget(self.api_tab)
            return

        openrouter_settings = self.openrouter_panel.export_settings()

        autosave_enabled = bool(self.autosave_checkbox.isChecked())
        autosave_interval_ms = int(self.autosave_interval_spin.value() * 1000)
        self._apply_autosave_settings(autosave_enabled, autosave_interval_ms)

        cleanup_cfg = SETTINGS.setdefault('cleanup', {})
        prev_auto_color = bool(cleanup_cfg.get('auto_text_color', True))
        prev_threshold = int(cleanup_cfg.get('text_color_threshold', 128))
        prev_use_box = bool(cleanup_cfg.get('use_background_box', True))

        cleanup_cfg['auto_text_color'] = bool(self.auto_text_color_checkbox.isChecked())
        cleanup_cfg['text_color_threshold'] = int(self.threshold_spin.value())
        use_box_value = bool(self.use_background_box_checkbox.isChecked())
        if hasattr(self.main_window, '_set_global_cleanup_default'):
            try:
                self.main_window._set_global_cleanup_default('use_background_box', use_box_value, persist=False)
            except Exception:
                cleanup_cfg['use_background_box'] = use_box_value
        cleanup_cfg['use_background_box'] = use_box_value
        # Persist new AI temp cleanup option
        try:
            cleanup_cfg['remove_ai_temp_files'] = bool(self.remove_ai_temp_checkbox.isChecked())
        except Exception:
            cleanup_cfg['remove_ai_temp_files'] = bool(cleanup_cfg.get('remove_ai_temp_files', False))

        shortcut_settings = {}
        for key, editor in self.shortcut_editors.items():
            seq_obj = editor.keySequence()
            sequence = seq_obj.toString(QKeySequence.PortableText).strip()
            default_seq = DEFAULT_SHORTCUTS.get(key, '')
            if not sequence:
                shortcut_settings[key] = ''
            elif sequence != default_seq:
                shortcut_settings[key] = sequence

        shortcuts_changed = shortcut_settings != self._initial_shortcuts

        SETTINGS.setdefault('translate', {})
        SETTINGS['translate']['openrouter'] = copy.deepcopy(openrouter_settings)
        SETTINGS['apis'] = copy.deepcopy(api_export.get('apis', {}))
        SETTINGS['ocr'] = copy.deepcopy(api_export.get('ocr', {}))
        SETTINGS['tesseract'] = copy.deepcopy(api_export.get('tesseract', {}))
        SETTINGS['shortcuts'] = shortcut_settings

        save_settings(SETTINGS)

        try:
            refresh_api_clients()
        except Exception:
            pass

        if hasattr(self.main_window, 'reload_shortcuts'):
            try:
                self.main_window.reload_shortcuts()
            except Exception:
                pass

        if hasattr(self.main_window, 'populate_ocr_languages'):
            try:
                self.main_window.populate_ocr_languages()
            except Exception:
                pass

        if hasattr(self.main_window, 'populate_ai_models'):
            try:
                self.main_window.populate_ai_models()
            except Exception:
                pass

        if (cleanup_cfg.get('auto_text_color') != prev_auto_color) or (cleanup_cfg.get('text_color_threshold') != prev_threshold):
            try:
                self.main_window.redraw_all_typeset_areas()
            except Exception:
                pass

        status_parts = []
        if (cleanup_cfg.get('auto_text_color') != prev_auto_color) or (cleanup_cfg.get('text_color_threshold') != prev_threshold) or (cleanup_cfg.get('use_background_box') != prev_use_box):
            status_parts.append("Cleanup defaults updated")
        if openrouter_settings != self._initial_translate:
            status_parts.append("Translation settings updated")
        if any(api_export.get(key, {}) != self._initial_api.get(key, {}) for key in ('apis', 'ocr', 'tesseract')):
            status_parts.append("API settings updated")
        if (autosave_enabled != self._initial_autosave_enabled) or (abs(autosave_interval_ms - self._initial_autosave_interval) > 1):
            status_parts.append("Autosave preferences updated")
        if shortcuts_changed:
            status_parts.append("Shortcuts updated")

        if status_parts and hasattr(self.main_window, 'statusBar'):
            try:
                self.main_window.statusBar().showMessage(" · ".join(status_parts), 4000)
            except Exception:
                pass

        self.accept()
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
                if not qimage.save(save_path, "PNG"):
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
        ("Heart", "?"),
        ("Sparkle", "?"),
        ("Star", "?"),
        ("Music", "?"),
        ("Shock", "!?"),
        ("Sweat", "??"),
        ("Smile", "??"),
        ("Angry", "??"),
    ]

    def __init__(self, area, font_manager, parent=None):
        super().__init__(parent)
        self.area = area
        self.font_manager = font_manager
        self.result = None
        self.setWindowTitle("Advanced Text Editor")
        self.setModal(True)
        self.resize(820, 600)

        main_layout = QVBoxLayout(self)
        header = QLabel("Fine-tune text formatting. Select a range to style only that portion or press Ctrl+A to target the whole bubble.")
        header.setWordWrap(True)
        main_layout.addWidget(header)

        toolbar_layout = QHBoxLayout()
        # Font group selector for the advanced editor (if the main application provided groups)
        self.font_group_combo = QComboBox()
        self.font_group_combo.setMinimumWidth(160)
        self.font_group_combo.addItem("All")
        toolbar_layout.addWidget(self.font_group_combo)
        self.font_combo = QComboBox()
        self.font_combo.setMinimumWidth(220)
        toolbar_layout.addWidget(self.font_combo)

        self.font_preview = QLabel("AaBb123")
        self.font_preview.setFixedWidth(140)
        self.font_preview.setAlignment(Qt.AlignCenter)
        self.font_preview.setStyleSheet("border: 1px solid #1f2b3b; border-radius: 6px; padding: 4px;")
        toolbar_layout.addWidget(self.font_preview)

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

        # After creating widgets, populate group combo if possible and hook handlers
        try:
            main_win = self.parent()
            if getattr(main_win, 'font_groups', None):
                # Clear default 'All' then add groups
                self.font_group_combo.clear(); self.font_group_combo.addItem('All')
                for g in main_win.font_groups.keys():
                    self.font_group_combo.addItem(g)
            # connect group changes to a local handler
            self.font_group_combo.currentTextChanged.connect(self._on_dialog_font_group_changed)
        except Exception:
            pass

        # Populate the font combo initially (respecting any selected group)
        try:
            sel = self.font_group_combo.currentText()
            if sel == 'All':
                self._populate_dialog_fonts(None)
            else:
                self._populate_dialog_fonts(sel)
        except Exception:
            pass

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
        self._last_text_cursor = QTextCursor(self.text_edit.document())
        self.text_edit.cursorPositionChanged.connect(self._remember_cursor_state)
        self.text_edit.cursorPositionChanged.connect(self._sync_toolbar_from_cursor)

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

        # Revert to global defaults button
        self.revert_button = QPushButton("Revert to Global Defaults")
        self.revert_button.setToolTip("Clear per-area overrides and use global defaults for this area.")
        def _on_revert_clicked():
            notes = self.area.review_notes if isinstance(getattr(self.area, 'review_notes', {}), dict) else {}
            for legacy in ('manual_inpaint', 'manual'):
                if legacy in notes:
                    notes.pop(legacy, None)
            self.area.review_notes = notes
            if hasattr(self.area, 'clear_override'):
                self.area.clear_override('use_inpaint')
                self.area.clear_override('use_background_box')
            parent = self.parent()
            if parent and hasattr(parent, 'redraw_all_typeset_areas'):
                try:
                    parent.redraw_all_typeset_areas()
                except Exception:
                    pass
            if parent and getattr(parent, 'selected_typeset_area', None) is self.area:
                try:
                    parent._sync_cleanup_controls_from_selection()
                except Exception:
                    pass
            QMessageBox.information(self, "Reverted", "This area's overrides have been cleared and global defaults will be used.")
        self.revert_button.clicked.connect(_on_revert_clicked)
        main_layout.addWidget(self.revert_button)

        self.font_combo.currentTextChanged.connect(self._change_font_family)
        self.font_size_spin.valueChanged.connect(self._change_font_size)
        self.bold_button.toggled.connect(self._toggle_bold)
        self.italic_button.toggled.connect(self._toggle_italic)
        self.underline_button.toggled.connect(self._toggle_underline)
        self.color_button.clicked.connect(self._choose_color)
        self.text_edit.cursorPositionChanged.connect(self._sync_toolbar_from_cursor)
        self.alignment_combo.currentIndexChanged.connect(self._apply_alignment)
        self.line_spacing_spin.valueChanged.connect(self._apply_line_spacing)
        self.char_spacing_spin.valueChanged.connect(self._apply_char_spacing)

        self._populate_font_combo()
        self._load_area_into_editor()
        self._sync_toolbar_from_cursor()

    def _create_bezier_spin(self):
        spin = QDoubleSpinBox()
        spin.setRange(-1.0, 2.0)
        spin.setDecimals(3)
        spin.setSingleStep(0.05)
        return spin

    def _remember_cursor_state(self):
        try:
            cursor = self.text_edit.textCursor()
            if self._is_cursor_valid(cursor):
                self._last_text_cursor = QTextCursor(cursor)
            else:
                self._last_text_cursor = QTextCursor(self.text_edit.document())
        except Exception:
            traceback.print_exc()

    def _is_cursor_valid(self, cursor: QTextCursor | None) -> bool:
        try:
            if cursor is None or cursor.isNull():
                return False
            doc = self.text_edit.document()
            if cursor.document() is not doc:
                return False
            pos = cursor.position()
            anchor = cursor.anchor()
            length = doc.characterCount()
            if pos < 0 or anchor < 0 or pos > length or anchor > length:
                return False
            return True
        except Exception:
            return False

    def _populate_font_combo(self):
        if not self.font_manager:
            return
        fonts = self.font_manager.list_fonts()
        with QSignalBlocker(self.font_combo):
            self.font_combo.clear()
            for name in fonts:
                self.font_combo.addItem(name)
                preview_font = self.font_manager.create_qfont(name)
                preview_font.setPointSize(16)
                idx = self.font_combo.count() - 1
                self.font_combo.setItemData(idx, preview_font, Qt.FontRole)

    def _update_font_preview(self, display_name):
        if not self.font_manager or not display_name:
            return
        base_font = self.text_edit.currentCharFormat().font()
        if not base_font.family():
            base_font = self.area.get_font()
        preview_font = self.font_manager.create_qfont(display_name, base_font=base_font)
        preview_font.setPointSizeF(self.font_size_spin.value())
        preview_font.setWeight(base_font.weight())
        preview_font.setItalic(base_font.italic())
        self.font_preview.setFont(preview_font)
        self.font_preview.setToolTip(display_name)

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
        self.margin_top_spin.setValue(int(margins.get('top', 0)))
        self.margin_right_spin.setValue(int(margins.get('right', 0)))
        self.margin_bottom_spin.setValue(int(margins.get('bottom', 0)))
        self.margin_left_spin.setValue(int(margins.get('left', 0)))

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
        display_name = None
        if self.font_manager:
            display_name = self.font_manager.display_name_for_font(base_font)
        if not display_name:
            display_name = base_font.family()
        with QSignalBlocker(self.font_combo):
            if display_name:
                self.font_combo.setCurrentText(display_name)
        with QSignalBlocker(self.font_size_spin):
            self.font_size_spin.setValue(base_font.pointSizeF() or base_font.pointSize())
        self._update_font_preview(display_name)
        self._last_text_cursor = QTextCursor(self.text_edit.document())

    def _insert_emoji(self, text):
        try:
            if not text:
                return
            cursor = self.text_edit.textCursor()
            if not self._is_cursor_valid(cursor):
                cursor = QTextCursor(self.text_edit.document())
                cursor.movePosition(QTextCursor.End)
            cursor.insertText(text)
            self.text_edit.setTextCursor(cursor)
            self._last_text_cursor = QTextCursor(cursor)
        except Exception:
            traceback.print_exc()

    def _populate_dialog_fonts(self, group: str | None = None):
        """Populate the dialog font combo, optionally filtering by a group name provided by the main window."""
        try:
            fonts = self.font_manager.list_fonts() if self.font_manager else []
            if group and getattr(self.parent(), 'font_groups', None):
                allowed = set(self.parent().font_groups.get(group, []))
                fonts = [f for f in fonts if f in allowed]
            with QSignalBlocker(self.font_combo):
                self.font_combo.clear()
                for name in fonts:
                    self.font_combo.addItem(name)
                    preview_font = self.font_manager.create_qfont(name) if self.font_manager else QFont(name)
                    preview_font.setPointSize(14)
                    idx = self.font_combo.count() - 1
                    self.font_combo.setItemData(idx, preview_font, Qt.FontRole)
        except Exception:
            traceback.print_exc()

    def _on_dialog_font_group_changed(self, group_name: str):
        if group_name == 'All':
            self._populate_dialog_fonts(None)
        else:
            self._populate_dialog_fonts(group=group_name)

    def _merge_char_format(self, fmt):
        if fmt is None:
            return
        try:
            cursor = self.text_edit.textCursor()
            if cursor is None or cursor.isNull():
                return
            if not cursor.hasSelection():
                stored = getattr(self, '_last_text_cursor', None)
                if self._is_cursor_valid(stored) and stored.hasSelection():
                    cursor = QTextCursor(stored)
                    self.text_edit.setTextCursor(cursor)
                else:
                    cursor.select(QTextCursor.WordUnderCursor)
            self.text_edit.setTextCursor(cursor)
            cursor.mergeCharFormat(fmt)
            self.text_edit.mergeCurrentCharFormat(fmt)
            self.text_edit.ensureCursorVisible()
            self._last_text_cursor = QTextCursor(self.text_edit.textCursor())
        except Exception:
            traceback.print_exc()

    def _change_font_family(self, display_name):
        try:
            if not display_name:
                return
            cursor_font = self.text_edit.currentCharFormat().font()
            if not cursor_font.family():
                cursor_font = self.area.get_font()
            if self.font_manager:
                new_font = self.font_manager.create_qfont(display_name, base_font=cursor_font)
            else:
                new_font = QFont(cursor_font)
                new_font.setFamily(display_name)
            new_font.setPointSizeF(self.font_size_spin.value())
            new_font.setLetterSpacing(cursor_font.letterSpacingType(), cursor_font.letterSpacing())
            new_font.setWeight(cursor_font.weight())
            new_font.setItalic(cursor_font.italic())
            new_font.setUnderline(cursor_font.underline())
            fmt = QTextCharFormat()
            fmt.setFont(new_font)
            self._merge_char_format(fmt)
            self._update_font_preview(display_name)
            self._last_text_cursor = QTextCursor(self.text_edit.textCursor())
        except Exception:
            traceback.print_exc()

    def _change_font_size(self, value):
        try:
            fmt = QTextCharFormat()
            fmt.setFontPointSize(float(value))
            self._merge_char_format(fmt)
            current = self.font_combo.currentText()
            if current:
                self._update_font_preview(current)
            self._last_text_cursor = QTextCursor(self.text_edit.textCursor())
        except Exception:
            traceback.print_exc()

    def _toggle_bold(self, checked):
        try:
            fmt = QTextCharFormat()
            fmt.setFontWeight(QFont.Bold if checked else QFont.Normal)
            self._merge_char_format(fmt)
            self._last_text_cursor = QTextCursor(self.text_edit.textCursor())
        except Exception:
            traceback.print_exc()

    def _toggle_italic(self, checked):
        try:
            fmt = QTextCharFormat()
            fmt.setFontItalic(checked)
            self._merge_char_format(fmt)
            self._last_text_cursor = QTextCursor(self.text_edit.textCursor())
        except Exception:
            traceback.print_exc()

    def _toggle_underline(self, checked):
        try:
            fmt = QTextCharFormat()
            fmt.setFontUnderline(checked)
            self._merge_char_format(fmt)
            self._last_text_cursor = QTextCursor(self.text_edit.textCursor())
        except Exception:
            traceback.print_exc()

    def _choose_color(self):
        try:
            color = QColorDialog.getColor(self._current_color_from_cursor(), self, "Select Text Color")
            if color.isValid():
                fmt = QTextCharFormat()
                fmt.setForeground(QBrush(color))
                self._merge_char_format(fmt)
                self._update_color_button(color)
                self._last_text_cursor = QTextCursor(self.text_edit.textCursor())
        except Exception:
            traceback.print_exc()

    def _current_color_from_cursor(self):
        fmt = self.text_edit.currentCharFormat()
        brush = fmt.foreground()
        return brush.color() if brush.style() != Qt.NoBrush else self.area.get_color()

    def _update_color_button(self, color):
        try:
            if color.isValid():
                self.color_button.setStyleSheet(f"background-color: {color.name()}; border: 1px solid #222; color: white;")
            else:
                self.color_button.setStyleSheet("")
        except Exception:
            traceback.print_exc()

    def _apply_alignment(self):
        try:
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
        except Exception:
            traceback.print_exc()

    def _apply_line_spacing(self):
        try:
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
        except Exception:
            traceback.print_exc()

    def _apply_char_spacing(self):
        try:
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
        except Exception:
            traceback.print_exc()

    def _sync_toolbar_from_cursor(self):
        fmt = self.text_edit.currentCharFormat()
        current_font = fmt.font()
        if not current_font.family():
            current_font = self.area.get_font()
        display = None
        if self.font_manager:
            display = self.font_manager.display_name_for_font(current_font)
        if not display:
            display = current_font.family()
        if display:
            with QSignalBlocker(self.font_combo):
                self.font_combo.setCurrentText(display)
        point = fmt.fontPointSize() or current_font.pointSizeF() or current_font.pointSize()
        with QSignalBlocker(self.font_size_spin): self.font_size_spin.setValue(point)
        with QSignalBlocker(self.bold_button): self.bold_button.setChecked(fmt.fontWeight() >= QFont.Bold)
        with QSignalBlocker(self.italic_button): self.italic_button.setChecked(fmt.fontItalic())
        with QSignalBlocker(self.underline_button): self.underline_button.setChecked(fmt.fontUnderline())
        self._update_color_button(self._current_color_from_cursor())
        if display:
            self._update_font_preview(display)

    def _extract_segments(self):
        segments = []
        try:
            doc = self.text_edit.document()
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
        except Exception:
            traceback.print_exc()
        return segments

    def _handle_accept(self):
        try:
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
            self._last_text_cursor = QTextCursor(self.text_edit.document())
        except Exception:
            traceback.print_exc()
            QMessageBox.warning(self, "Apply Failed", "Failed to apply text changes due to an unexpected error. Please try again.")

    def get_result(self):
        return self.result


class HistoryEditDialog(QDialog):
    def __init__(self, entry, styles, allow_original=True, allow_style=True, parent=None):
        super().__init__(parent)
        self.entry_style = entry.get('translation_style', '')
        self.result = None

        self.setWindowTitle("Edit Entry")
        self.setModal(True)
        self.resize(520, 420)

        layout = QVBoxLayout(self)
        info_label = QLabel("Adjust the text below. Use Confirm to apply the changes when you are ready.")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        original_label = QLabel("Original OCR")
        layout.addWidget(original_label)
        self.original_edit = QTextEdit()
        self.original_edit.setPlainText(entry.get('original_text', ''))
        self.original_edit.setMinimumHeight(100)
        if not allow_original:
            self.original_edit.setReadOnly(True)
            self.original_edit.setStyleSheet("background-color: #111824; color: #7f8ba7;")
        layout.addWidget(self.original_edit)

        translated_label = QLabel("Translated Text")
        layout.addWidget(translated_label)
        self.translated_edit = QTextEdit()
        self.translated_edit.setPlainText(entry.get('translated_text', ''))
        self.translated_edit.setMinimumHeight(140)
        layout.addWidget(self.translated_edit)

        # Translate button: sends current OCR text through active pipeline and fills translated_edit
        translate_btn_layout = QHBoxLayout()
        translate_btn_layout.addStretch()
        self.translate_button = QPushButton("Translate")
        self.translate_button.setToolTip("Translate the OCR text using the active translation provider and show result here")
        translate_btn_layout.addWidget(self.translate_button)
        layout.addLayout(translate_btn_layout)

        self.translate_button.clicked.connect(self._on_translate_clicked)

        if allow_style:
            style_layout = QHBoxLayout()
            style_label = QLabel("Translation Style")
            style_layout.addWidget(style_label)
            self.style_combo = QComboBox()
            self.style_combo.addItems(styles or [])
            current_style = entry.get('translation_style', '')
            if current_style and current_style in (styles or []):
                self.style_combo.setCurrentText(current_style)
            elif styles:
                self.style_combo.setCurrentIndex(0)
            style_layout.addWidget(self.style_combo)
            style_layout.addStretch()
            layout.addLayout(style_layout)
        else:
            self.style_combo = None
            style_label = QLabel(f"Style: {self.entry_style or '-'}")
            style_label.setWordWrap(True)
            layout.addWidget(style_label)

        button_box = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        button_box.button(QDialogButtonBox.Ok).setText("Confirm")
        button_box.accepted.connect(self.handle_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def handle_accept(self):
        self.result = {
            'original_text': self.original_edit.toPlainText(),
            'translated_text': self.translated_edit.toPlainText(),
            'translation_style': self.style_combo.currentText() if self.style_combo else self.entry_style,
        }
        self.accept()

    def _on_translate_clicked(self):
        """Translate current OCR text using parent app's active pipeline and place result into translated_edit.

        The priority is:
        - If parent has translate_with_ai and an AI provider is selected -> use translate_with_ai
        - Else, use translate_text (DeepL or fallback libs)
        """
        parent = self.parent()
        ocr_text = self.original_edit.toPlainText() or ''
        if not ocr_text.strip():
            QMessageBox.information(self, "Nothing to translate", "Original OCR text is empty.")
            return

        # Try to find translation functions on parent (MainWindow)
        translated = None
        try:
            # Prefer AI pipeline if available and configured
            if parent is not None and hasattr(parent, 'get_selected_model_name') and hasattr(parent, 'translate_with_ai'):
                provider, model_name = parent.get_selected_model_name() or (None, None)
                # If provider is set and provider is an AI provider, attempt AI translation
                if provider and provider.lower() in (k.lower() for k in getattr(parent, 'AI_PROVIDERS', {}).keys()):
                    # Build minimal settings dict to pass style/temperature if available
                    settings = {}
                    try:
                        settings = getattr(parent, 'settings', {}) or getattr(parent, 'settings', None) or {}
                    except Exception:
                        settings = {}
                    # Respect translation style if available
                    if hasattr(self, 'style_combo') and self.style_combo:
                        settings = dict(settings)
                        settings['translation_style'] = self.style_combo.currentText()

                    # Call AI translator
                    try:
                        translated = parent.translate_with_ai(ocr_text, settings.get('target_lang', 'Indonesian'), provider, model_name, settings)
                    except Exception as e:
                        translated = f"[AI translation error: {e}]"

            # Fallback to non-AI translation function
            if translated is None or (isinstance(translated, str) and translated.startswith('[AI translation error')):
                if parent is not None and hasattr(parent, 'translate_text'):
                    target_lang = 'Indonesian'
                    try:
                        if hasattr(parent, 'settings') and isinstance(getattr(parent, 'settings', None), dict):
                            target_lang = parent.settings.get('target_lang', target_lang)
                    except Exception:
                        pass
                    try:
                        translated = parent.translate_text(ocr_text, target_lang)
                    except Exception as e:
                        translated = f"[Translation error: {e}]"

        except Exception as outer_e:
            translated = f"[Translation error: {outer_e}]"

        # Put translated text into translated_edit for further manual edits
        if translated is None:
            translated = ''
        self.translated_edit.setPlainText(str(translated))

    def get_result(self):
        return self.result



class ManualTextDialog(QDialog):
    def __init__(self, default_inpaint=True, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manual Text Input")
        self.setModal(True)
        self.resize(420, 340)

        main_layout = QVBoxLayout(self)

        instructions = QLabel("Type the text you want to place inside the selected area.")
        instructions.setWordWrap(True)
        main_layout.addWidget(instructions)

        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Enter manual text here...")
        self.text_edit.setMinimumHeight(160)
        main_layout.addWidget(self.text_edit)

        self.inpaint_checkbox = QCheckBox("Apply inpainting before adding text")
        self.inpaint_checkbox.setChecked(default_inpaint)
        main_layout.addWidget(self.inpaint_checkbox)

        button_box = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        button_box.button(QDialogButtonBox.Ok).setText("Apply")
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

    def get_text(self):
        return self.text_edit.toPlainText()

    def use_inpainting(self):
        return self.inpaint_checkbox.isChecked()

# --- Project serialization helpers ---
def coerce_int(value, default=0, minimum=None, maximum=None):
    try:
        ivalue = int(round(float(value)))
    except (TypeError, ValueError):
        return default
    if minimum is not None:
        ivalue = max(minimum, ivalue)
    if maximum is not None:
        ivalue = min(maximum, ivalue)
    return ivalue


def coerce_float(value, default=0.0, minimum=None, maximum=None):
    try:
        fvalue = float(value)
    except (TypeError, ValueError):
        return default
    if minimum is not None:
        fvalue = max(minimum, fvalue)
    if maximum is not None:
        fvalue = min(maximum, fvalue)
    return fvalue


def rect_to_dict(rect):
    if rect is None:
        return None
    if isinstance(rect, dict):
        return {
            'x': coerce_int(rect.get('x', 0)),
            'y': coerce_int(rect.get('y', 0)),
            'width': coerce_int(rect.get('width', 0), minimum=0),
            'height': coerce_int(rect.get('height', 0), minimum=0),
        }
    try:
        return {
            'x': coerce_int(rect.x()),
            'y': coerce_int(rect.y()),
            'width': coerce_int(rect.width(), minimum=0),
            'height': coerce_int(rect.height(), minimum=0),
        }
    except AttributeError:
        return {'x': 0, 'y': 0, 'width': 0, 'height': 0}


def dict_to_rect(data):
    if not data:
        return QRect()
    try:
        x = coerce_int(data.get('x', 0))
        y = coerce_int(data.get('y', 0))
        w = coerce_int(data.get('width', 0), minimum=0)
        h = coerce_int(data.get('height', 0), minimum=0)
        return QRect(x, y, w, h)
    except Exception:
        return QRect()


def polygon_to_list(polygon):
    if polygon is None:
        return None
    points = []
    try:
        for pt in polygon:
            if hasattr(pt, 'x') and hasattr(pt, 'y'):
                points.append({'x': coerce_int(pt.x()), 'y': coerce_int(pt.y())})
            elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                points.append({'x': coerce_int(pt[0]), 'y': coerce_int(pt[1])})
    except TypeError:
        return None
    return points if points else None


def list_to_polygon(data):
    if not data:
        return None
    points = []
    for pt in data:
        if isinstance(pt, dict):
            x = coerce_int(pt.get('x', 0))
            y = coerce_int(pt.get('y', 0))
        elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
            x = coerce_int(pt[0])
            y = coerce_int(pt[1])
        else:
            continue
        points.append(QPoint(x, y))
    return QPolygon(points) if points else None

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
        text_outline=False,
        text_outline_width=2.0,
        text_outline_color='#000000',
        alignment='center',
        line_spacing=1.1,
        char_spacing=100.0,
        margins=None,
        history_id=None,
        original_text="",
        translation_style="",
        review_notes=None,
        overrides=None,
        rotation=0.0,
    ):
        self.rect = rect
        self.rotation = float(rotation) if rotation is not None else 0.0
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
        self.text_outline = bool(text_outline)
        try:
            self.text_outline_width = max(0.0, float(text_outline_width))
        except Exception:
            self.text_outline_width = 2.0
        if isinstance(text_outline_color, QColor):
            self.text_outline_color = text_outline_color.name()
        else:
            self.text_outline_color = str(text_outline_color or '#000000')
        self.alignment = alignment
        self.line_spacing = line_spacing
        self.char_spacing = char_spacing
        self.margins = margins or {'top': 0, 'right': 0, 'bottom': 0, 'left': 0}
        self.history_id = history_id
        self.original_text = original_text or ""
        self.translation_style = translation_style or ""
        self.review_notes = review_notes or {}
        self.overrides = overrides if isinstance(overrides, dict) else {}
        self.text_segments = segments if segments is not None else self._build_segments_from_plain(self.text, font, color)
        self.ensure_defaults()

    def ensure_defaults(self):
        if not hasattr(self, 'overrides') or not isinstance(self.overrides, dict):
            self.overrides = {}
        if not hasattr(self, 'history_id'): self.history_id = None
        if not hasattr(self, 'original_text') or self.original_text is None: self.original_text = ''
        if not hasattr(self, 'translation_style') or self.translation_style is None: self.translation_style = ''
        if not hasattr(self, 'review_notes') or self.review_notes is None: self.review_notes = {}
        if not hasattr(self, 'orientation'): self.orientation = 'horizontal'
        if not hasattr(self, 'effect'): self.effect = 'none'
        if not hasattr(self, 'effect_intensity'): self.effect_intensity = 20.0
        if not hasattr(self, 'rotation') or self.rotation is None:
            try:
                self.rotation = float(self.rotation)
            except Exception:
                self.rotation = 0.0

        if not getattr(self, 'bezier_points', None):
            self.bezier_points = [{'x': 0.25, 'y': 0.2}, {'x': 0.75, 'y': 0.2}]
        if not hasattr(self, 'bubble_enabled'): self.bubble_enabled = False
        if not getattr(self, 'bubble_fill', None): self.bubble_fill = '#ffffff'
        if not getattr(self, 'bubble_outline', None): self.bubble_outline = '#000000'
        if not hasattr(self, 'bubble_outline_width'): self.bubble_outline_width = 3.0
        if not hasattr(self, 'text_outline'): self.text_outline = False
        if not hasattr(self, 'text_outline_width'):
            self.text_outline_width = 2.0
        else:
            try:
                self.text_outline_width = max(0.0, float(self.text_outline_width))
            except Exception:
                self.text_outline_width = 2.0
        if not getattr(self, 'text_outline_color', None):
            self.text_outline_color = '#000000'
        if not hasattr(self, 'alignment'): self.alignment = 'center'
        if not hasattr(self, 'line_spacing') or self.line_spacing is None: self.line_spacing = 1.1
        if not hasattr(self, 'char_spacing') or self.char_spacing is None: self.char_spacing = 100.0
        if 'letterSpacing' not in self.font_info:
            self.font_info['letterSpacing'] = self.char_spacing
            self.font_info['letterSpacingType'] = QFont.PercentageSpacing
        if not getattr(self, 'margins', None): self.margins = {'top': 0, 'right': 0, 'bottom': 0, 'left': 0}
        if not getattr(self, 'text_segments', None):
            self.text_segments = self._build_segments_from_plain(self.text, self.get_font(), self.get_color())
        if not getattr(self, 'text', None):
            self.text = self._segments_to_plain_text(self.text_segments)

    def get_overrides(self):
        return self.overrides

    def has_override(self, key):
        return isinstance(self.overrides, dict) and key in self.overrides

    def get_override(self, key, default=None):
        if self.has_override(key):
            return self.overrides[key]
        return default

    def set_override(self, key, value):
        if not isinstance(self.overrides, dict):
            self.overrides = {}
        self.overrides[key] = value

    def clear_override(self, key):
        if isinstance(self.overrides, dict):
            self.overrides.pop(key, None)

    def clear_overrides(self):
        if isinstance(self.overrides, dict):
            self.overrides.clear()

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.ensure_defaults()

    @staticmethod
    def font_to_dict(font):
        manager = get_font_manager()
        data = {
            'family': font.family(),
            'pointSize': font.pointSizeF() if hasattr(font, 'pointSizeF') else font.pointSize(),
            'weight': font.weight(),
            'italic': font.italic(),
            'underline': font.underline(),
            'letterSpacing': font.letterSpacing(),
            'letterSpacingType': font.letterSpacingType(),
        }
        if manager:
            display_name = manager.display_name_for_font(font, fallback=None)
            if display_name:
                data['displayName'] = display_name
        return data

    @staticmethod
    def font_from_dict(info):
        if isinstance(info, QFont):
            return QFont(info)

        font_manager = get_font_manager()
        base_font = None
        if isinstance(info, dict) and font_manager:
            display_name = info.get('displayName')
            if display_name:
                base_font = font_manager.create_qfont(display_name)

        font = QFont(base_font) if isinstance(base_font, QFont) else QFont()
        if not isinstance(info, dict):
            if not font.family():
                font.setFamily('Arial')
            if font.pointSize() <= 0:
                font.setPointSize(14)
            if font.weight() < 0:
                font.setWeight(QFont.Normal)
            return font

        family = info.get('family')
        if family:
            font.setFamily(str(family))

        point_size = coerce_float(info.get('pointSize', 14.0), default=14.0, minimum=1.0)
        if hasattr(font, 'setPointSizeF'):
            font.setPointSizeF(point_size)
        else:
            font.setPointSize(coerce_int(point_size, default=14, minimum=1))

        weight_value = info.get('weight', QFont.Normal)
        try:
            font.setWeight(coerce_int(weight_value, default=QFont.Normal))
        except Exception:
            font.setWeight(QFont.Normal)

        font.setItalic(bool(info.get('italic', False)))
        font.setUnderline(bool(info.get('underline', False)))

        spacing_type = coerce_int(info.get('letterSpacingType', QFont.PercentageSpacing), default=QFont.PercentageSpacing)
        spacing_value = coerce_float(info.get('letterSpacing', 100.0), default=100.0)
        font.setLetterSpacing(spacing_type, spacing_value)
        return font
    
    @classmethod
    def _sanitize_segments(cls, segments, fallback_font_dict, fallback_color):
        sanitized = []
        if not isinstance(segments, list):
            return sanitized
        fallback_font_dict = fallback_font_dict or cls.font_to_dict(cls.font_from_dict({}))
        fallback_color = fallback_color or '#000000'
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            seg_copy = copy.deepcopy(seg)
            font_info = seg_copy.get('font')
            if isinstance(font_info, QFont):
                seg_copy['font'] = cls.font_to_dict(font_info)
            elif isinstance(font_info, dict):
                seg_copy['font'] = cls.font_to_dict(cls.font_from_dict(font_info))
            else:
                seg_copy['font'] = fallback_font_dict
            seg_copy['color'] = seg_copy.get('color') or fallback_color
            seg_copy['text'] = seg_copy.get('text', '')
            seg_copy['underline'] = bool(seg_copy.get('underline', False))
            sanitized.append(seg_copy)
        return sanitized
    
    def to_payload(self):
        base_font = copy.deepcopy(self.font_info)
        segments = self._sanitize_segments(self.text_segments or [], base_font, self.color_info)
        margins = self.get_margins() if hasattr(self, 'get_margins') else getattr(self, 'margins', {})
        sanitized_margins = {key: coerce_int(margins.get(key, 0)) for key in ('top', 'right', 'bottom', 'left')}
        bezier_points = []
        for pt in self.bezier_points or []:
            if isinstance(pt, dict):
                bx = coerce_float(pt.get('x', 0.0))
                by = coerce_float(pt.get('y', 0.0))
                bezier_points.append({'x': bx, 'y': by})
            elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                bx = coerce_float(pt[0])
                by = coerce_float(pt[1])
                bezier_points.append({'x': bx, 'y': by})
        bezier_points = bezier_points or None
        return {
            'rect': rect_to_dict(self.rect),
            'text': self.text or '',
            'font': base_font,
            'color': self.color_info,
            'polygon': polygon_to_list(self.polygon),
            'orientation': self.get_orientation(),
            'effect': self.get_effect(),
            'effect_intensity': float(self.get_effect_intensity()),
            'bezier_points': bezier_points,
            'bubble_enabled': bool(self.bubble_enabled),
            'segments': segments,
            'bubble_fill': getattr(self, 'bubble_fill', '#ffffff') or '#ffffff',
            'bubble_outline': getattr(self, 'bubble_outline', '#000000') or '#000000',
            'bubble_outline_width': float(getattr(self, 'bubble_outline_width', 3.0) or 3.0),
            'text_outline': bool(self.has_text_outline()),
            'text_outline_width': float(self.get_text_outline_width()),
            'text_outline_color': self.get_text_outline_color().name(),
            'alignment': self.get_alignment(),
            'line_spacing': float(self.get_line_spacing()),
            'char_spacing': float(self.get_char_spacing()),
            'rotation': float(self.get_rotation()),
            'margins': sanitized_margins,
            'history_id': self.history_id,
            'original_text': self.original_text or '',
            'translation_style': self.translation_style or '',
            'review_notes': copy.deepcopy(self.review_notes if isinstance(self.review_notes, dict) else {}),
            'overrides': copy.deepcopy(self.get_overrides() if isinstance(self.get_overrides(), dict) else {}),
        }
    
    @classmethod
    def from_payload(cls, data, fallback_font=None, fallback_color=None):
        if fallback_font is None:
            fallback_font = QFont('Arial', 12)
        if fallback_color is None:
            fallback_color = QColor('#000000')
        if not isinstance(data, dict):
            return cls(QRect(), '', fallback_font, fallback_color)
        rect = dict_to_rect(data.get('rect'))
        font = cls.font_from_dict(data.get('font')) or fallback_font
        color_value = data.get('color', fallback_color.name()) or fallback_color.name()
        color = QColor(color_value)
        polygon = list_to_polygon(data.get('polygon'))
        orientation = data.get('orientation', 'horizontal') or 'horizontal'
        effect = data.get('effect', 'none') or 'none'
        effect_intensity = coerce_float(data.get('effect_intensity'), default=20.0)
        bezier_raw = data.get('bezier_points')
        bezier_points = None
        if isinstance(bezier_raw, list):
            normalized = []
            for pt in bezier_raw:
                if isinstance(pt, dict):
                    normalized.append({'x': coerce_float(pt.get('x', 0.0)), 'y': coerce_float(pt.get('y', 0.0))})
                elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    normalized.append({'x': coerce_float(pt[0]), 'y': coerce_float(pt[1])})
            bezier_points = normalized if normalized else None
        bubble_enabled = bool(data.get('bubble_enabled', False))
        bubble_fill = data.get('bubble_fill', '#ffffff') or '#ffffff'
        bubble_outline = data.get('bubble_outline', '#000000') or '#000000'
        bubble_outline_width = coerce_float(data.get('bubble_outline_width'), default=3.0, minimum=0.0)
        text_outline = bool(data.get('text_outline', False))
        text_outline_width = coerce_float(data.get('text_outline_width'), default=2.0, minimum=0.0)
        text_outline_color = data.get('text_outline_color', '#000000') or '#000000'
        alignment = data.get('alignment', 'center') or 'center'
        line_spacing = coerce_float(data.get('line_spacing'), default=1.1)
        line_spacing = max(0.6, min(line_spacing, 5.0))
        char_spacing = coerce_float(data.get('char_spacing'), default=100.0)
        char_spacing = max(10.0, min(char_spacing, 500.0))
        rotation = coerce_float(data.get('rotation', 0.0))
        margins_data = data.get('margins')
        margins = {'top': 0, 'right': 0, 'bottom': 0, 'left': 0}
        if isinstance(margins_data, dict):
            for key in margins:
                margins[key] = coerce_int(margins_data.get(key, 0))
        segments = cls._sanitize_segments(data.get('segments'), cls.font_to_dict(font), color.name())
        history_id = data.get('history_id') or data.get('id')
        if history_id is not None:
            history_id = str(history_id)
        else:
            history_id = None
        original_text = data.get('original_text', '') or ''
        translation_style = data.get('translation_style', '') or ''
        review_notes = data.get('review_notes')
        if not isinstance(review_notes, dict):
            review_notes = {}
        overrides = data.get('overrides')
        if not isinstance(overrides, dict):
            overrides = {}
        legacy_override_keys = ('use_inpaint', 'use_background_box')
        for key in legacy_override_keys:
            if key in review_notes and key not in overrides:
                overrides[key] = review_notes.get(key)
                review_notes.pop(key, None)
        area = cls(
            rect,
            data.get('text', '') or '',
            font,
            color,
            polygon=polygon,
            orientation=orientation,
            effect=effect,
            effect_intensity=effect_intensity,
            bezier_points=bezier_points,
            bubble_enabled=bubble_enabled,
            segments=segments or None,
            bubble_fill=bubble_fill,
            bubble_outline=bubble_outline,
            bubble_outline_width=bubble_outline_width,
            text_outline=text_outline,
            text_outline_width=text_outline_width,
            text_outline_color=text_outline_color,
            alignment=alignment,
            line_spacing=line_spacing,
            char_spacing=char_spacing,
            margins=margins,
            history_id=history_id,
            original_text=original_text,
            translation_style=translation_style,
            review_notes=review_notes,
            overrides=overrides,
            rotation=rotation,
        )
        return area
    
    def get_font(self):
        return self.font_from_dict(self.font_info)

    def get_color(self):
        return QColor(self.color_info)

    def get_rotation(self):
        try:
            return float(self.rotation)
        except Exception:
            return 0.0

    def set_rotation(self, value):
        try:
            self.rotation = float(value)
        except Exception:
            self.rotation = 0.0

    def segment_to_qfont(self, segment):
        info = segment.get('font', self.font_info) or self.font_info
        font = self.font_from_dict(info)
        underline = segment.get('underline')
        if underline is None and isinstance(info, dict):
            underline = info.get('underline', False)
        font.setUnderline(bool(underline))
        return font

    def segment_to_color(self, segment):
        return QColor(segment.get('color', self.color_info))

    def get_segments(self):
        self.ensure_defaults()
        return self.text_segments

    def has_text_outline(self):
        return bool(getattr(self, 'text_outline', False)) and self.get_text_outline_width() > 0.0

    def get_text_outline_width(self):
        try:
            return max(0.0, float(getattr(self, 'text_outline_width', 2.0)))
        except Exception:
            return 2.0

    def get_text_outline_color(self):
        value = getattr(self, 'text_outline_color', '#000000')
        if isinstance(value, QColor):
            color = QColor(value)
        else:
            color = QColor(str(value))
        if not color.isValid():
            color = QColor('#000000')
        return color

    def set_segments(self, segments):
        self.text_segments = segments or []
        self.text = self._segments_to_plain_text(self.text_segments)

    def update_plain_text(self, text):
        font = self.get_font()
        color = self.get_color()
        self.text = text or ''
        self.text_segments = self._build_segments_from_plain(self.text, font, color)

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
        margins = getattr(self, 'margins', {'top': 0, 'right': 0, 'bottom': 0, 'left': 0})
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
        self.hovering_edit_icon = False

        self.transform_mode = False
        self.transform_handles = {}
        self.transform_hover_handle = None
        self._set_active_transform(None)
        self.transform_handle_size = 14

        # --- Variabel baru untuk deteksi interaktif ---
        self.pending_bubble_polygon = None
        self.pending_bubble_rect = QRect()
        self.pending_trash_icon_rect = QRect()
        self.hovering_pending_trash = False

        self.detected_items = [] # List of dicts {'polygon': QPolygon, 'text': str}
        self.hovered_item_index = -1
        # Hovered handle index for pen tool vertex highlighting
        self.hovered_handle_index = -1
        # Debug flag: when True, draw large red markers for polygon points and print coords
        self._debug_draw_pen_points = False

        self._transform_update_timer = QTimer(self)
        self._transform_update_timer.setSingleShot(True)
        self._transform_update_timer.timeout.connect(self._emit_transform_redraw)
        # Inpainting brush state
        self.brush_mask = None
        self.brush_last_image_pos = None
        self.is_brushing = False
        self.brush_size = 45
        self.brush_dirty = False
        self._brush_mask_array = None
        self.brush_color = QColor(255, 235, 59, 180)
        self._brush_tint_cache = None
        self._brush_cursor = None
        self._cached_cursor_size = None

    def _set_active_transform(self, transform_data):
        self.active_transform = transform_data
        try:
            self.main_window.set_transform_preview_active(bool(transform_data))
        except Exception:
            pass

    def get_selection_mode(self):
        return self.main_window.selection_mode_combo.currentText()

    def get_polygon_points(self):
        return self.polygon_points
    
    def set_transform_mode(self, enabled):
        if self.transform_mode == enabled:
            return
        self.transform_mode = bool(enabled)
        self._set_active_transform(None)
        self.transform_handles.clear()
        self.transform_hover_handle = None
        if self.transform_mode:
            self._refresh_transform_handles()
        self.update()
    
    def _rotate_point(self, point, angle_deg):
        rad = math.radians(angle_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        return QPointF(point.x() * cos_a - point.y() * sin_a,
                       point.x() * sin_a + point.y() * cos_a)

    def _image_point_to_widget(self, point):
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull():
            return QPointF(point)
        scale = getattr(self.main_window, "zoom_factor", 1.0) or 1.0
        label_size = self.size()
        pix_size = pixmap.size()
        offset_x = max(0.0, (label_size.width() - pix_size.width()) / 2.0)
        offset_y = max(0.0, (label_size.height() - pix_size.height()) / 2.0)
        return QPointF(point.x() * scale + offset_x, point.y() * scale + offset_y)

    def _widget_point_to_image(self, pos):
        result = self.main_window.unzoom_coords(pos, as_point=True)
        if isinstance(result, QPoint):
            return QPointF(result)
        if isinstance(result, QPointF):
            return result
        return None

    def _create_brush_mask_storage(self, width, height):
        if width <= 0 or height <= 0:
            return False
        self.brush_mask = QImage(width, height, QImage.Format_Alpha8)
        self.brush_mask.fill(0)
        try:
            ptr = self.brush_mask.bits()
            ptr.setsize(self.brush_mask.bytesPerLine() * height)
            buffer = np.frombuffer(ptr, dtype=np.uint8)
            buffer.setflags(write=True)
            buffer = buffer.reshape((height, self.brush_mask.bytesPerLine()))
            self._brush_mask_array = buffer[:, :width]
        except Exception:
            self._brush_mask_array = None
            return False
        self.brush_dirty = False
        self._brush_tint_cache = None
        return True

    def _ensure_brush_mask(self):
        if self.brush_mask is not None:
            return True
        base_pixmap = getattr(self.main_window, 'original_pixmap', None)
        if base_pixmap and not base_pixmap.isNull():
            size = base_pixmap.size()
            return self._create_brush_mask_storage(size.width(), size.height())
        pil_img = getattr(self.main_window, 'current_image_pil', None)
        if pil_img is not None:
            try:
                width = int(getattr(pil_img, 'width', pil_img.size[0]))
                height = int(getattr(pil_img, 'height', pil_img.size[1]))
            except Exception:
                width, height = pil_img.size if hasattr(pil_img, 'size') else (0, 0)
            return self._create_brush_mask_storage(width, height)
        return False

    def reset_brush_mask(self):
        self.brush_mask = None
        self._brush_mask_array = None
        self.brush_last_image_pos = None
        self.is_brushing = False
        self.brush_dirty = False
        self._brush_tint_cache = None
        self.update()
        if self.main_window:
            self.main_window.on_brush_mask_updated()

    def clear_brush_mask(self):
        if not self._ensure_brush_mask():
            return
        if self._brush_mask_array is not None:
            self._brush_mask_array.fill(0)
        self.brush_last_image_pos = None
        self.is_brushing = False
        self.brush_dirty = False
        self._brush_tint_cache = None
        self.update()
        if self.main_window:
            self.main_window.on_brush_mask_updated()

    def stop_brush_session(self):
        self.is_brushing = False
        self.brush_last_image_pos = None

    def has_brush_mask(self):
        return bool(self.brush_mask is not None and self.brush_dirty)

    def export_brush_mask(self):
        if not self.has_brush_mask():
            return None
        return self.brush_mask.copy()

    def _clamp_image_point(self, point):
        if self.brush_mask is None:
            return None
        width = max(1, self.brush_mask.width())
        height = max(1, self.brush_mask.height())
        x = max(0.0, min(point.x(), width - 1))
        y = max(0.0, min(point.y(), height - 1))
        return QPointF(x, y)

    def _paint_brush_segment(self, start_point, end_point=None):
        if not self._ensure_brush_mask():
            return
        if self._brush_mask_array is None:
            return
        start = self._clamp_image_point(QPointF(start_point))
        end = self._clamp_image_point(QPointF(end_point)) if end_point is not None else start
        if start is None or end is None:
            return
        x1 = int(round(start.x()))
        y1 = int(round(start.y()))
        x2 = int(round(end.x()))
        y2 = int(round(end.y()))
        height, width = self._brush_mask_array.shape
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))
        max_dimension = max(width, height)
        thickness = max(1, min(int(round(self.brush_size)), max_dimension))
        if x1 == x2 and y1 == y2:
            cv2.circle(self._brush_mask_array, (x1, y1), max(1, thickness // 2), 255, thickness=-1, lineType=cv2.LINE_AA)
        else:
            cv2.line(self._brush_mask_array, (x1, y1), (x2, y2), 255, thickness=thickness, lineType=cv2.LINE_AA)
            cv2.circle(self._brush_mask_array, (x2, y2), max(1, thickness // 2), 255, thickness=-1, lineType=cv2.LINE_AA)
        self.brush_dirty = True
        self._brush_tint_cache = None
        if self.main_window:
            self.main_window.on_brush_mask_updated()

    def start_brush_stroke(self, image_point):
        if image_point is None:
            return
        self.is_brushing = True
        clamped = self._clamp_image_point(QPointF(image_point))
        if clamped is None:
            return
        self.brush_last_image_pos = QPointF(clamped)
        self._paint_brush_segment(clamped)
        self.update()

    def continue_brush_stroke(self, image_point):
        if not self.is_brushing or self.brush_last_image_pos is None or image_point is None:
            return
        clamped = self._clamp_image_point(QPointF(image_point))
        if clamped is None:
            return
        self._paint_brush_segment(self.brush_last_image_pos, clamped)
        self.brush_last_image_pos = QPointF(clamped)
        self.update()

    def set_brush_size(self, size):
        try:
            size = int(size)
        except Exception:
            size = self.brush_size
        size = max(2, min(size, 400))
        if size == self.brush_size:
            return
        self.brush_size = size
        self.stop_brush_session()
        self._brush_tint_cache = None
        self.update_brush_cursor()
        self.update()

    def update_brush_cursor(self):
        if self.get_selection_mode() != "Brush Inpaint (LaMA)":
            self._brush_cursor = None
            self._cached_cursor_size = None
            self.setCursor(Qt.CrossCursor)
            return
        zoom = getattr(self.main_window, 'zoom_factor', 1.0) or 1.0
        diameter = max(6, int(round(self.brush_size * zoom)))
        diameter = min(diameter, 256)
        if self._brush_cursor is not None and self._cached_cursor_size == diameter:
            self.setCursor(self._brush_cursor)
            return
        pm_size = diameter + 6
        pixmap = QPixmap(pm_size, pm_size)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(255, 235, 59))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawEllipse(QRectF(3, 3, diameter, diameter))
        painter.end()
        hotspot = pm_size // 2
        self._brush_cursor = QCursor(pixmap, hotspot, hotspot)
        self._cached_cursor_size = diameter
        self.setCursor(self._brush_cursor)

    def _get_tinted_brush_image(self):
        if self.brush_mask is None or not self.brush_dirty:
            return None
        if self._brush_tint_cache is None:
            tinted = QImage(self.brush_mask.size(), QImage.Format_ARGB32_Premultiplied)
            tinted.fill(Qt.transparent)
            painter = QPainter(tinted)
            painter.setCompositionMode(QPainter.CompositionMode_Source)
            painter.fillRect(tinted.rect(), self.brush_color)
            painter.setCompositionMode(QPainter.CompositionMode_DestinationIn)
            painter.drawImage(0, 0, self.brush_mask)
            painter.end()
            self._brush_tint_cache = tinted
        return self._brush_tint_cache

    def _area_polygon(self, area):
        rect = QRectF(area.rect)
        points = [QPointF(rect.topLeft()), QPointF(rect.topRight()), QPointF(rect.bottomRight()), QPointF(rect.bottomLeft())]
        rotation = area.get_rotation() if hasattr(area, 'get_rotation') else float(getattr(area, 'rotation', 0.0))
        if abs(rotation) > 0.01:
            center = rect.center()
            transform = QTransform()
            transform.translate(center.x(), center.y())
            transform.rotate(rotation)
            transform.translate(-center.x(), -center.y())
            points = [transform.map(pt) for pt in points]
        return QPolygonF(points)

    def _area_polygon_widget(self, area):
        return [self._image_point_to_widget(pt) for pt in self._area_polygon(area)]

    def _point_in_area(self, area, image_point):
        if image_point is None:
            return False
        polygon = self._area_polygon(area)
        return polygon.containsPoint(QPointF(image_point), Qt.OddEvenFill)

    def _area_at_widget_pos(self, pos):
        image_point = self._widget_point_to_image(pos)
        if image_point is None:
            return None
        for area in reversed(getattr(self.main_window, 'typeset_areas', [])):
            if self._point_in_area(area, image_point):
                return area
        return None

    def _image_dimensions(self):
        pixmap = getattr(self.main_window, 'original_pixmap', None)
        if pixmap is not None and not pixmap.isNull():
            return pixmap.width(), pixmap.height()
        pil_image = getattr(self.main_window, 'current_image_pil', None)
        if pil_image is not None:
            try:
                return pil_image.width, pil_image.height
            except AttributeError:
                try:
                    w, h = pil_image.size
                    return w, h
                except Exception:
                    pass
        return None, None

    def _clamp_rectf_to_image(self, rectf):
        w, h = self._image_dimensions()
        if rectf is None or w is None or h is None or w <= 0 or h <= 0:
            return QRectF(rectf)
        x = max(0.0, min(rectf.x(), float(w - 1)))
        y = max(0.0, min(rectf.y(), float(h - 1)))
        width = rectf.width()
        height = rectf.height()
        width = max(1.0, min(width, float(w) - x))
        height = max(1.0, min(height, float(h) - y))
        return QRectF(x, y, width, height)

    def _update_area_polygon_from_delta(self, area, orig_polygon_points, dx, dy):
        if not orig_polygon_points:
            return
        w, h = self._image_dimensions()
        updated_points = []
        for pt in orig_polygon_points:
            new_x = pt.x() + dx
            new_y = pt.y() + dy
            if w is not None and w > 0:
                new_x = max(0.0, min(new_x, w - 1))
            if h is not None and h > 0:
                new_y = max(0.0, min(new_y, h - 1))
            updated_points.append(QPoint(int(round(new_x)), int(round(new_y))))
        if updated_points:
            area.polygon = QPolygon(updated_points)

    def _update_area_polygon_for_scale(self, area, orig_polygon_points, anchor_point, orig_rect, new_rect):
        if not orig_polygon_points or anchor_point is None:
            return
        w, h = self._image_dimensions()
        orig_width = max(1.0, float(orig_rect.width()))
        orig_height = max(1.0, float(orig_rect.height()))
        new_width = max(1.0, float(new_rect.width()))
        new_height = max(1.0, float(new_rect.height()))
        scale_x = new_width / orig_width
        scale_y = new_height / orig_height
        anchor_x = anchor_point.x()
        anchor_y = anchor_point.y()
        updated_points = []
        for pt in orig_polygon_points:
            vec_x = pt.x() - anchor_x
            vec_y = pt.y() - anchor_y
            new_x = anchor_x + vec_x * scale_x
            new_y = anchor_y + vec_y * scale_y
            if w is not None and w > 0:
                new_x = max(0.0, min(new_x, w - 1))
            if h is not None and h > 0:
                new_y = max(0.0, min(new_y, h - 1))
            updated_points.append(QPoint(int(round(new_x)), int(round(new_y))))
        if updated_points:
            area.polygon = QPolygon(updated_points)

    def _build_local_transforms(self, area):
        rotation = area.get_rotation() if hasattr(area, 'get_rotation') else float(getattr(area, 'rotation', 0.0))
        center = QPointF(area.rect.center())
        to_local = QTransform()
        to_local.translate(-center.x(), -center.y())
        to_local.rotate(-rotation)
        from_local, invertible = to_local.inverted()
        if not invertible:
            from_local = QTransform()
            from_local.translate(center.x(), center.y())
        return to_local, from_local, center, rotation

    def _refresh_transform_handles(self):
        self.transform_handles.clear()
        if not self.transform_mode:
            return
        area = getattr(self.main_window, 'selected_typeset_area', None)
        if not area or area not in getattr(self.main_window, 'typeset_areas', []):
            return
        polygon = self._area_polygon_widget(area)
        if len(polygon) < 4:
            return
        handle_size = float(self.transform_handle_size)
        half = handle_size / 2.0
        for key, point in zip(('nw', 'ne', 'se', 'sw'), polygon):
            self.transform_handles[key] = QRectF(point.x() - half, point.y() - half, handle_size, handle_size)
        center_widget = self._image_point_to_widget(QPointF(area.rect.center()))
        top_center = QPointF((polygon[0].x() + polygon[1].x()) / 2.0, (polygon[0].y() + polygon[1].y()) / 2.0)
        direction = QPointF(top_center.x() - center_widget.x(), top_center.y() - center_widget.y())
        length = math.hypot(direction.x(), direction.y())
        if length < 1e-3:
            direction = QPointF(0.0, -1.0)
        else:
            direction = QPointF(direction.x() / length, direction.y() / length)
        rotation_offset = max(30.0, handle_size * 2.0)
        rotation_center = QPointF(top_center.x() + direction.x() * rotation_offset,
                                  top_center.y() + direction.y() * rotation_offset)
        self.transform_handles['rotate'] = QRectF(rotation_center.x() - half, rotation_center.y() - half, handle_size, handle_size)
        self.transform_handles['_points_widget'] = polygon
        self.transform_handles['_rotation_line'] = (top_center, rotation_center)
        self.transform_handles['_center_widget'] = center_widget

    def _emit_transform_redraw(self):
        if not self.main_window:
            return
        try:
            self.main_window.schedule_typeset_redraw(45)
        except Exception:
            traceback.print_exc()

    def _set_transform_hover_handle(self, handle):
        if self.transform_hover_handle == handle:
            return
        self.transform_hover_handle = handle
        if not self.transform_mode or self.active_transform:
            return
        if handle == 'rotate':
            self.setCursor(Qt.SizeAllCursor)
        elif handle in ('nw', 'se'):
            self.setCursor(Qt.SizeFDiagCursor)
        elif handle in ('ne', 'sw'):
            self.setCursor(Qt.SizeBDiagCursor)
        else:
            self.setCursor(Qt.OpenHandCursor)

    def _handle_transform_mouse_press(self, event):
        if not self.transform_mode or self.main_window.is_in_confirmation_mode:
            return False
        if event.button() not in (Qt.LeftButton, Qt.RightButton):
            return False
        self._refresh_transform_handles()
        selected_area = getattr(self.main_window, 'selected_typeset_area', None)
        area_at = self._area_at_widget_pos(event.pos())
        if area_at and area_at is not selected_area:
            self.main_window.set_selected_area(area_at)
            selected_area = area_at
            self._refresh_transform_handles()
        if event.button() == Qt.RightButton:
            return selected_area is not None
        if not selected_area:
            self._set_active_transform(None)
            self.update()
            return False
        image_point = self._widget_point_to_image(event.pos())
        if image_point is None:
            return False
        handle = None
        for key, rect in self.transform_handles.items():
            if key.startswith('_'):
                continue
            try:
                if rect and hasattr(rect, 'contains') and rect.contains(QPointF(event.pos())):
                    handle = key
                    break
            except Exception:
                # Defensive: ignore malformed transform handle entries
                continue
        if handle == 'rotate':
            center = QPointF(selected_area.rect.center())
            vector = QPointF(image_point.x() - center.x(), image_point.y() - center.y())
            if math.hypot(vector.x(), vector.y()) < 1e-3:
                return False
            base_rotation = selected_area.get_rotation() if hasattr(selected_area, 'get_rotation') else float(getattr(selected_area, 'rotation', 0.0))
            start_angle = math.degrees(math.atan2(vector.y(), vector.x()))
            transform = {
                'type': 'rotate',
                'area': selected_area,
                'center': center,
                'base_rotation': base_rotation,
                'start_angle': start_angle,
                'orig_polygon': [QPointF(pt) for pt in selected_area.polygon] if getattr(selected_area, 'polygon', None) else None,
            }
            self._set_active_transform(transform)
            self.transform_hover_handle = None
            self.setCursor(Qt.ClosedHandCursor)
            return True
        elif handle in ('nw', 'ne', 'se', 'sw'):
            anchor_map = {
                'nw': ('bottomRight', 'topLeft'),
                'ne': ('bottomLeft', 'topRight'),
                'se': ('topLeft', 'bottomRight'),
                'sw': ('topRight', 'bottomLeft'),
            }
            anchor_attr, _ = anchor_map[handle]
            orig_rect = QRectF(selected_area.rect)
            rotation = selected_area.get_rotation() if hasattr(selected_area, 'get_rotation') else float(getattr(selected_area, 'rotation', 0.0))
            center = QPointF(orig_rect.center())
            to_local = QTransform()
            to_local.translate(-center.x(), -center.y())
            to_local.rotate(-rotation)
            from_local, invertible = to_local.inverted()
            if not invertible:
                from_local = QTransform()
                from_local.rotate(rotation)
                from_local.translate(center.x(), center.y())
            anchor_point = QPointF(getattr(orig_rect, anchor_attr)())
            x_dir = -1 if handle in ('nw', 'sw') else 1
            y_dir = -1 if handle in ('nw', 'ne') else 1
            transform = {
                'type': 'scale',
                'area': selected_area,
                'handle': handle,
                'orig_rect': orig_rect,
                'anchor_point': anchor_point,
                'to_local': to_local,
                'from_local': from_local,
                'rotation': rotation,
                'x_dir': x_dir,
                'y_dir': y_dir,
                'min_size': 12.0,
                'orig_polygon': [QPointF(pt) for pt in selected_area.polygon] if getattr(selected_area, 'polygon', None) else None,
            }
            self._set_active_transform(transform)
            self.transform_hover_handle = None
            self.setCursor(Qt.SizeFDiagCursor if handle in ('nw', 'se') else Qt.SizeBDiagCursor)
            return True
        elif self._point_in_area(selected_area, image_point):
            transform = {
                'type': 'move',
                'area': selected_area,
                'start_mouse': QPointF(image_point),
                'orig_rect': QRectF(selected_area.rect),
                'orig_polygon': [QPointF(pt) for pt in selected_area.polygon] if getattr(selected_area, 'polygon', None) else None,
            }
            self._set_active_transform(transform)
            self.transform_hover_handle = None
            self.setCursor(Qt.ClosedHandCursor)
            return True
        else:
            self.main_window.clear_selected_area()
            self._set_active_transform(None)
            self.update()
            return False

    def _handle_transform_mouse_move(self, event):
        if not self.transform_mode:
            return False
        self.current_mouse_pos = event.pos()
        if self.active_transform:
            try:
                image_point = self._widget_point_to_image(event.pos())
                transform_type = self.active_transform.get('type') if isinstance(self.active_transform, dict) else None
                if transform_type == 'move':
                    self._update_transform_move(image_point)
                elif transform_type == 'rotate':
                    self._update_transform_rotate(image_point)
                elif transform_type == 'scale':
                    self._update_transform_scale(image_point)
            except Exception as exc:
                # Defensive: if something unexpected happens during transform, cancel active transform to avoid crash
                try:
                    print(f"Warning: transform move error: {exc}")
                except Exception:
                    pass
                self._set_active_transform(None)
                self._refresh_transform_handles()
                self.update()
                return False
            self._refresh_transform_handles()
            self.update()
            return True
        self._refresh_transform_handles()
        handle = None
        for key, rect in self.transform_handles.items():
            if key.startswith('_'):
                continue
            if rect.contains(QPointF(event.pos())):
                handle = key
                break
        self._set_transform_hover_handle(handle)
        self.hovered_area = self._area_at_widget_pos(event.pos())
        self.update()
        return True

    def _handle_transform_mouse_release(self, event):
        if not self.transform_mode or not self.active_transform:
            return False
        if event.button() != Qt.LeftButton:
            return False
        self._set_active_transform(None)
        self._refresh_transform_handles()
        self._set_transform_hover_handle(None)
        self.setCursor(Qt.OpenHandCursor)
        if hasattr(self.main_window, 'redraw_all_typeset_areas'):
            self.main_window.redraw_all_typeset_areas()
        self.update()
        return True

    def _update_transform_move(self, image_point):
        try:
            if image_point is None:
                return
            info = self.active_transform
            # Validate transform info
            if not info or not isinstance(info, dict):
                return
            if any(k not in info for k in ('area', 'orig_rect', 'start_mouse')):
                return

            area = info.get('area')
            start_rect = QRectF(info.get('orig_rect'))
            start_mouse = info.get('start_mouse')
            if start_mouse is None:
                return

            # Compute deltas and translate
            delta_x = image_point.x() - start_mouse.x()
            delta_y = image_point.y() - start_mouse.y()
            start_rect.translate(delta_x, delta_y)

            # Ensure area object is valid and has rect attribute
            if area is None:
                return
            clamped_rectf = self._clamp_rectf_to_image(start_rect)
            new_rect = QRect(
                int(round(clamped_rectf.x())),
                int(round(clamped_rectf.y())),
                max(1, int(round(clamped_rectf.width()))),
                max(1, int(round(clamped_rectf.height())))
            )

            orig_rect = QRectF(info.get('orig_rect'))
            dx = new_rect.x() - int(round(orig_rect.x()))
            dy = new_rect.y() - int(round(orig_rect.y()))

            try:
                area.rect = new_rect
            except Exception:
                return

            if info.get('orig_polygon'):
                try:
                    self._update_area_polygon_from_delta(area, info['orig_polygon'], dx, dy)
                except Exception:
                    area.polygon = QPolygon([new_rect.topLeft(), new_rect.topRight(), new_rect.bottomRight(), new_rect.bottomLeft()])
        except Exception as exc:
            try:
                print(f"Warning: _update_transform_move failed: {exc}")
            except Exception:
                pass
            return
        if hasattr(self.main_window, 'redo_stack'):
            try:
                self.main_window.redo_stack.clear()
            except Exception:
                pass
        if not self._transform_update_timer.isActive():
            self._transform_update_timer.start(16)

    def _update_transform_rotate(self, image_point):
        try:
            if image_point is None:
                return
            info = self.active_transform
            if not info or 'area' not in info or 'center' not in info or 'base_rotation' not in info or 'start_angle' not in info:
                return
            area = info.get('area')
            center = info.get('center')
            base_rotation = info.get('base_rotation')
            start_angle = info.get('start_angle')
            vector = QPointF(image_point.x() - center.x(), image_point.y() - center.y())
            if math.hypot(vector.x(), vector.y()) < 1e-3:
                return
            current_angle = math.degrees(math.atan2(vector.y(), vector.x()))
            new_rotation = (base_rotation + (current_angle - start_angle)) % 360.0
            if hasattr(area, 'set_rotation'):
                area.set_rotation(new_rotation)
            else:
                try:
                    area.rotation = float(new_rotation)
                except Exception:
                    pass
            if hasattr(self.main_window, 'redo_stack'):
                try:
                    self.main_window.redo_stack.clear()
                except Exception:
                    pass
            if not self._transform_update_timer.isActive():
                self._transform_update_timer.start(16)
        except Exception as exc:
            try:
                print(f"Warning: _update_transform_rotate failed: {exc}")
            except Exception:
                pass
            return

    def _update_transform_scale(self, image_point):
        try:
            if image_point is None:
                return
            info = self.active_transform
            if not info:
                return
            # Required fields
            required = ('area', 'to_local', 'from_local', 'anchor_point', 'x_dir', 'y_dir')
            if any(k not in info for k in required):
                return
            area = info['area']
            to_local = info['to_local']
            from_local = info['from_local']
            anchor_local = to_local.map(info['anchor_point'])
            current_local = to_local.map(QPointF(image_point))
            min_size = float(info.get('min_size', 12.0))
            x_dir = info['x_dir']
            y_dir = info['y_dir']
            raw_width = (anchor_local.x() - current_local.x()) if x_dir == -1 else (current_local.x() - anchor_local.x())
            raw_height = (anchor_local.y() - current_local.y()) if y_dir == -1 else (current_local.y() - anchor_local.y())
            width = max(min_size, raw_width)
            height = max(min_size, raw_height)
            new_center_local = QPointF(anchor_local.x() - x_dir * width / 2.0,
                                       anchor_local.y() - y_dir * height / 2.0)
            new_center_global = from_local.map(new_center_local)
            new_rectf = QRectF(new_center_global.x() - width / 2.0,
                               new_center_global.y() - height / 2.0,
                               width,
                               height)
            clamped_rectf = self._clamp_rectf_to_image(new_rectf)
            new_rect = QRect(
                int(round(clamped_rectf.x())),
                int(round(clamped_rectf.y())),
                max(1, int(round(clamped_rectf.width()))),
                max(1, int(round(clamped_rectf.height())))
            )
            try:
                area.rect = new_rect
            except Exception:
                return

            orig_rect = QRectF(info.get('orig_rect', QRectF(area.rect)))
            orig_polygon = info.get('orig_polygon')
            anchor_point = info.get('anchor_point')
            if orig_polygon and anchor_point is not None:
                try:
                    self._update_area_polygon_for_scale(
                        area,
                        orig_polygon,
                        anchor_point,
                        orig_rect,
                        clamped_rectf
                    )
                except Exception:
                    area.polygon = QPolygon([new_rect.topLeft(), new_rect.topRight(), new_rect.bottomRight(), new_rect.bottomLeft()])

            if hasattr(self.main_window, 'redo_stack'):
                try:
                    self.main_window.redo_stack.clear()
                except Exception:
                    pass
            if not self._transform_update_timer.isActive():
                self._transform_update_timer.start(16)
        except Exception as exc:
            try:
                print(f"Warning: _update_transform_scale failed: {exc}")
            except Exception:
                pass
            return

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
            posf = QPointF(unzoomed_pos) if isinstance(unzoomed_pos, QPoint) else unzoomed_pos
            for area in reversed(self.main_window.typeset_areas):
                if self._point_in_area(area, posf):
                    self.areaDoubleClicked.emit(area)
                    return

    def mousePressEvent(self, event):
        if not self.main_window.original_pixmap: return
        # Allow app-level mouse shortcuts to intercept the event
        try:
            if getattr(self.main_window, 'dispatch_mouse_shortcut', None):
                if self.main_window.dispatch_mouse_shortcut('press', event.button()):
                    return
        except Exception:
            pass
        
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

        # Determine selection mode early so pen/manual modes aren't blocked by hovered area logic
        mode = self.get_selection_mode()
        manual_rect = "Manual Text (Rect)" in mode
        manual_polygon = "Manual Text (Pen)" in mode
        pen_mode = (mode == "Pen Tool")
        brush_mode = (mode == "Brush Inpaint (LaMA)")
        transform_mode = (mode == "Transform (Hand)")

        if transform_mode and not self.main_window.is_in_confirmation_mode:
            if self._handle_transform_mouse_press(event):
                return

        if brush_mode:
            if self.main_window.is_in_confirmation_mode:
                return
            if event.button() == Qt.LeftButton:
                image_point = self._widget_point_to_image(event.pos())
                self.start_brush_stroke(image_point)
                if self.main_window:
                    self.main_window.update_selection_action_buttons(True)
            return

        # If hovering over an existing area, allow area-level actions unless we're in pen/manual-pen mode
        if self.hovered_area and not (pen_mode or manual_polygon or brush_mode or transform_mode):
            if self.trash_icon_rect.contains(event.pos()):
                if self.main_window.is_in_confirmation_mode: return
                self.main_window.delete_typeset_area(self.hovered_area)
                self.hovered_area = None
                self.update()
                return
            if self.edit_icon_rect.contains(event.pos()):
                if self.main_window.is_in_confirmation_mode: return
                self.hovering_edit_icon = False
                self.main_window.start_inline_edit(self.hovered_area)
                self.hovered_area = None
                self.update()
                return
            # Right-click on an area shows context menu for area-level actions
            if event.button() == Qt.RightButton:
                try:
                    from PyQt5.QtWidgets import QMenu
                    menu = QMenu(self)
                    revert_action = menu.addAction("Revert to Global Defaults")
                    action = menu.exec_(self.mapToGlobal(event.pos()))
                    if action == revert_action:
                        notes = self.hovered_area.review_notes if isinstance(getattr(self.hovered_area, 'review_notes', {}), dict) else {}
                        for legacy in ('manual_inpaint', 'manual'):
                            if legacy in notes:
                                notes.pop(legacy, None)
                        self.hovered_area.review_notes = notes
                        if hasattr(self.hovered_area, 'clear_override'):
                            self.hovered_area.clear_override('use_inpaint')
                            self.hovered_area.clear_override('use_background_box')
                        try:
                            self.main_window.redraw_all_typeset_areas()
                        except Exception:
                            pass
                        if self.main_window.selected_typeset_area is self.hovered_area:
                            self.main_window._sync_cleanup_controls_from_selection()
                        self.main_window.statusBar().showMessage("Area reverted to global defaults.", 2500)
                        self.update()
                        return
                except Exception:
                    pass

        if event.button() == Qt.LeftButton:
            # If in pen/manual-pen mode, do not let hovered_area clicks block pen selection
            if (
                self.hovered_area
                and not self.trash_icon_rect.contains(event.pos())
                and not self.edit_icon_rect.contains(event.pos())
                and not (pen_mode or manual_polygon or brush_mode)
            ):
                if not self.main_window.is_in_confirmation_mode:
                    self.main_window.set_selected_area(self.hovered_area)
                self.update()
                return
            if not self.main_window.is_in_confirmation_mode and not self.hovered_area:
                self.main_window.clear_selected_area()

        if event.button() == Qt.LeftButton:
            mode = self.get_selection_mode()
            manual_rect = "Manual Text (Rect)" in mode
            manual_polygon = "Manual Text (Pen)" in mode
            if ("Bubble Finder" in mode or "Direct OCR" in mode or manual_rect) and not manual_polygon:
                self.clear_selection()
                self.selection_start = event.pos()
                self.selection_end = event.pos()
                self.dragging = True
            elif mode == "Pen Tool" or manual_polygon:
                if not self.polygon_points:
                    self.clear_selection()
                self.polygon_points.append(event.pos())
                # If debug flag enabled, log the added point (widget coords)
                if getattr(self, '_debug_draw_pen_points', False):
                    try:
                        print(f"[DEBUG] Added polygon point: {event.pos().x()},{event.pos().y()}")
                    except Exception:
                        pass
                self.main_window.update_selection_action_buttons(True)
            self.update()
    def mouseMoveEvent(self, event):
        self.current_mouse_pos = event.pos()
        mode = self.get_selection_mode()
        transform_mode = False
        brush_mode = (mode == "Brush Inpaint (LaMA)")

        # Cek hover di atas ikon tong sampah untuk item yang menunggu
        if self.pending_bubble_polygon:
            new_hover_state = self.pending_trash_icon_rect.contains(self.current_mouse_pos)
            if self.hovering_pending_trash != new_hover_state:
                self.hovering_pending_trash = new_hover_state
                self.update() # Perbarui untuk mengubah warna ikon

        if brush_mode and self.is_brushing and not self.main_window.is_in_confirmation_mode:
            image_point = self._widget_point_to_image(event.pos())
            self.continue_brush_stroke(image_point)
            return

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
            transform_mode = (mode == "Transform (Hand)")
            if transform_mode and self._handle_transform_mouse_move(event):
                return
            manual_polygon = "Manual Text (Pen)" in mode
            pen_mode = (mode == "Pen Tool")

            if pen_mode or manual_polygon or brush_mode:
                # In pen drawing modes we avoid changing hovered_area to prevent interference
                new_hover_area = self.hovered_area
            else:
                unzoomed_pos = self.main_window.unzoom_coords(self.current_mouse_pos, as_point=True)
                new_hover_area = None
                if unzoomed_pos:
                    posf = QPointF(unzoomed_pos) if isinstance(unzoomed_pos, QPoint) else unzoomed_pos
                    for area in reversed(self.main_window.typeset_areas):
                        if self._point_in_area(area, posf):
                            new_hover_area = area
                            break

            if self.hovered_area != new_hover_area:
                self.hovered_area = new_hover_area
                self.update()

        edit_hover_state = (
            bool(self.hovered_area)
            and not self.main_window.is_in_confirmation_mode
            and not transform_mode
            and not self.edit_icon_rect.isNull()
            and self.edit_icon_rect.contains(event.pos())
        )
        if edit_hover_state != self.hovering_edit_icon:
            self.hovering_edit_icon = edit_hover_state
            self.update()

        manual_rect = "Manual Text (Rect)" in mode
        manual_polygon = "Manual Text (Pen)" in mode
        if ("Bubble Finder" in mode or "Direct OCR" in mode or manual_rect) and not manual_polygon:
            if self.dragging:
                self.selection_end = self.current_mouse_pos
                self.update()
        elif mode == "Pen Tool" or manual_polygon:
            if self.polygon_points:
                # detect nearby handles for hover feedback
                self.hovered_handle_index = -1
                if self.current_mouse_pos:
                    for i, p in enumerate(self.polygon_points):
                        if QPointF(p).toPoint().manhattanLength() is not None:
                            dist = (p - self.current_mouse_pos).manhattanLength()
                            if dist <= 10:
                                self.hovered_handle_index = i
                                break
                self.update()
    def mouseReleaseEvent(self, event):
        mode = self.get_selection_mode()
        brush_mode = (mode == "Brush Inpaint (LaMA)")
        # Allow app-level mouse shortcuts to intercept release events
        try:
            if getattr(self.main_window, 'dispatch_mouse_shortcut', None):
                if self.main_window.dispatch_mouse_shortcut('release', event.button()):
                    return
        except Exception:
            pass
        manual_polygon = "Manual Text (Pen)" in mode
        manual_rect = "Manual Text (Rect)" in mode

        if mode == "Transform (Hand)" and self._handle_transform_mouse_release(event):
            return

        if event.button() == Qt.RightButton and (mode == "Pen Tool" or manual_polygon):
            if len(self.polygon_points) >= 3:
                self.main_window.confirm_pen_selection()
            else:
                self.main_window.cancel_pen_selection()
            return

        if brush_mode and event.button() == Qt.LeftButton:
            if self.is_brushing:
                image_point = self._widget_point_to_image(event.pos())
                self.continue_brush_stroke(image_point)
            self.stop_brush_session()
            if self.main_window:
                self.main_window.on_brush_mask_updated()
            return

        if event.button() == Qt.LeftButton:
            if ("Bubble Finder" in mode or "Direct OCR" in mode or manual_rect) and not manual_polygon:
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
                                path = QPainterPath()
                                path.addEllipse(QRectF(self.selection_rect))
                                polygon = path.toFillPolygon().toPolygon()
                                self.main_window.process_polygon_area(list(polygon))
                            elif manual_rect:
                                self.main_window.process_rect_area(self.selection_rect)
                        else:
                            self.clear_selection()
                    
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
                # Glow effect: draw a soft semi-transparent larger rect behind
                glow_pen = QPen(QColor(30, 150, 240, 40), 10)
                glow_pen.setStyle(Qt.SolidLine)
                painter.setPen(glow_pen)
                if "Oval" in mode:
                    painter.drawEllipse(rect.adjusted(-4, -4, 4, 4))
                else:
                    painter.drawRoundedRect(rect.adjusted(-4, -4, 4, 4), 6, 6)

                # Main outline dashed
                outline_pen = QPen(QColor(0, 140, 255), 2, Qt.DashLine)
                painter.setPen(outline_pen)
                painter.setBrush(QColor(0, 140, 255, 55))
                if "Oval" in mode:
                    painter.drawEllipse(rect)
                else:
                    painter.drawRoundedRect(rect, 6, 6)
        elif (mode == "Pen Tool" or "Manual Text (Pen)" in mode) and self.polygon_points:
            base_pen = QPen(QColor(90, 220, 130), 2, Qt.SolidLine)
            base_pen.setCapStyle(Qt.RoundCap)
            painter.setPen(base_pen)
            # draw filled translucent polygon if more than 2 points
            if len(self.polygon_points) > 2:
                poly = QPolygon(self.polygon_points)
                painter.setBrush(QColor(80, 200, 120, 50))
                painter.drawPolygon(poly)
            # draw polyline
            painter.setBrush(Qt.NoBrush)
            painter.drawPolyline(QPolygon(self.polygon_points))

            # draw handles (stronger with hover) and small index labels
            handle_pen = QPen(QColor(20, 20, 20), 1)
            painter.setPen(handle_pen)
            for idx, pt in enumerate(self.polygon_points):
                hover = (idx == getattr(self, 'hovered_handle_index', -1))
                size = 14 if hover else 12
                handle_rect = QRect(pt.x() - size//2, pt.y() - size//2, size, size)
                # draw outline for high contrast
                painter.setPen(QPen(QColor(10, 10, 10), 2))
                painter.setBrush(QColor(255, 255, 255) if not hover else QColor(255, 200, 120))
                painter.drawEllipse(handle_rect)
                painter.setPen(QPen(QColor(30, 30, 30), 1))
                painter.drawText(handle_rect, Qt.AlignCenter, str(idx + 1))
                painter.setPen(handle_pen)

            # rubber-band to current mouse
            if self.current_mouse_pos:
                rubber_pen = QPen(QColor(160, 255, 180), 1, Qt.DashLine)
                painter.setPen(rubber_pen)
                painter.drawLine(self.polygon_points[-1], self.current_mouse_pos)

            # Debug overlay: draw big red markers on each polygon point when enabled
            if getattr(self, '_debug_draw_pen_points', False):
                dbg_pen = QPen(QColor(200, 20, 20), 2)
                dbg_brush = QBrush(QColor(200, 20, 20, 180))
                painter.setPen(dbg_pen)
                painter.setBrush(dbg_brush)
                for pt in self.polygon_points:
                    r = QRect(pt.x() - 8, pt.y() - 8, 16, 16)
                    painter.drawEllipse(r)
        
        if self.brush_mask is not None and self.brush_dirty and self.get_selection_mode() == "Brush Inpaint (LaMA)":
            tinted_image = self._get_tinted_brush_image()
            if tinted_image is not None:
                painter.save()
                pixmap = self.pixmap()
                offset_x = offset_y = 0
                if pixmap and not pixmap.isNull():
                    label_size = self.size()
                    pixmap_size = pixmap.size()
                    offset_x = max(0, (label_size.width() - pixmap_size.width()) // 2)
                    offset_y = max(0, (label_size.height() - pixmap_size.height()) // 2)
                    painter.translate(offset_x, offset_y)
                    zoom_factor = getattr(self.main_window, "zoom_factor", 1.0) or 1.0
                    painter.scale(zoom_factor, zoom_factor)
                painter.drawImage(0, 0, tinted_image)
                painter.restore()

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
            painter.fillPath(path, QColor(255, 215, 120, 120))
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

            # Draw small hint text near pending polygon
            hint = "Right-click to confirm • Middle-click to cancel"
            hint_rect = QRect(self.pending_trash_icon_rect.left() - 8 - 200, self.pending_trash_icon_rect.top() - 28, 200, 20)
            painter.setPen(QPen(QColor(240, 240, 240, 220)))
            painter.setBrush(QColor(0, 0, 0, 120))
            painter.drawRoundedRect(hint_rect, 6, 6)
            painter.drawText(hint_rect, Qt.AlignCenter, hint)


        # Draw override badges for areas that have per-area overrides
        try:
            for area in (self.main_window.typeset_areas or []):
                overrides = []
                default_inpaint = self.main_window._default_cleanup_value('use_inpaint')
                default_box = self.main_window._default_cleanup_value('use_background_box')
                override_inpaint = area.get_override('use_inpaint', None)
                override_box = area.get_override('use_background_box', None)
                if override_inpaint is not None and bool(override_inpaint) != bool(default_inpaint):
                    overrides.append('inpaint')
                if override_box is not None and bool(override_box) != bool(default_box):
                    overrides.append('background box')
                if overrides:
                    # draw a small badge at area's top-left (zoomed)
                    try:
                        zoomed = self.main_window.zoom_coords(area.rect)
                        badge_rect = QRect(zoomed.left() + 4, zoomed.top() + 4, 12, 12)
                        painter.setBrush(QColor(255, 200, 60, 230))
                        painter.setPen(Qt.NoPen)
                        painter.drawEllipse(badge_rect)
                    except Exception:
                        pass
        except Exception:
            pass

        selected_area = getattr(self.main_window, 'selected_typeset_area', None)
        if selected_area and selected_area in getattr(self.main_window, 'typeset_areas', []):
            try:
                polygon_points = self._area_polygon_widget(selected_area)
                if polygon_points:
                    path = QPainterPath()
                    path.moveTo(polygon_points[0])
                    for pt in polygon_points[1:]:
                        path.lineTo(pt)
                    path.closeSubpath()
                    painter.setPen(QPen(QColor(90, 180, 255, 200), 2, Qt.DashLine))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawPath(path)
                    if self.transform_mode:
                        self._refresh_transform_handles()
                        rotation_line = self.transform_handles.get('_rotation_line')
                        if rotation_line:
                            painter.setPen(QPen(QColor(160, 160, 160, 200), 1, Qt.SolidLine))
                            painter.drawLine(rotation_line[0], rotation_line[1])
                        for key, rect in self.transform_handles.items():
                            if key.startswith('_'):
                                continue
                            is_hover = (self.transform_hover_handle == key)
                            is_active = False
                            if self.active_transform:
                                t_type = self.active_transform.get('type')
                                if key == 'rotate' and t_type == 'rotate':
                                    is_active = True
                                elif key in ('nw', 'ne', 'se', 'sw') and t_type == 'scale' and self.active_transform.get('handle') == key:
                                    is_active = True
                            if key == 'rotate':
                                base_color = QColor(255, 200, 80)
                                if is_hover or is_active:
                                    base_color = QColor(255, 230, 120)
                                painter.setPen(QPen(QColor(120, 120, 120), 1))
                                painter.setBrush(base_color)
                            else:
                                base_color = QColor(245, 245, 245)
                                if is_hover or is_active:
                                    base_color = QColor(120, 200, 255)
                                painter.setPen(QPen(QColor(30, 30, 30), 1))
                                painter.setBrush(base_color)
                            painter.drawEllipse(rect)
            except Exception:
                pass

        if self.hovered_area and not self.main_window.is_in_confirmation_mode and mode != "Transform (Hand)":
            zoomed_rect = self.main_window.zoom_coords(self.hovered_area.rect)
            icon_size = 32
            margin = 6

            # Trash Icon
            self.trash_icon_rect = QRect(zoomed_rect.topRight().x() - icon_size - margin, zoomed_rect.topRight().y() + margin, icon_size, icon_size)
            painter.setBrush(QColor(255, 80, 80, 200)); painter.setPen(Qt.NoPen)
            painter.drawEllipse(self.trash_icon_rect)
            pen = QPen(Qt.white, 2); painter.setPen(pen)
            painter.drawLine(self.trash_icon_rect.topLeft() + QPoint(6,6), self.trash_icon_rect.bottomRight() - QPoint(6,6))
            painter.drawLine(self.trash_icon_rect.topRight() - QPoint(6,-6), self.trash_icon_rect.bottomLeft() + QPoint(6,-6))

            # Edit Icon
            self.edit_icon_rect = QRect(self.trash_icon_rect.left() - icon_size - margin, self.trash_icon_rect.top(), icon_size, icon_size)
            edit_color = QColor(80, 150, 255, 200)
            if self.hovering_edit_icon:
                edit_color = QColor(110, 180, 255, 230)
            painter.setBrush(edit_color); painter.setPen(Qt.NoPen)
            painter.drawEllipse(self.edit_icon_rect)
            painter.setPen(pen)
            # Draw a simple pencil
            poly = QPolygon([
                QPoint(10, 26), QPoint(10, 20), QPoint(20, 10),
                QPoint(24, 14), QPoint(14, 26), QPoint(10, 26)
            ])
            painter.drawPolyline(poly.translated(self.edit_icon_rect.topLeft()))
            painter.drawLine(QPoint(19,11)+self.edit_icon_rect.topLeft(), QPoint(22,14)+self.edit_icon_rect.topLeft())

            # If hovered area has overrides, show tooltip text near icons
            try:
                override_list = []
                default_inpaint = self.main_window._default_cleanup_value('use_inpaint')
                default_box = self.main_window._default_cleanup_value('use_background_box')
                override_inpaint = self.hovered_area.get_override('use_inpaint', None)
                override_box = self.hovered_area.get_override('use_background_box', None)
                if override_inpaint is not None and bool(override_inpaint) != bool(default_inpaint):
                    override_list.append(f"Inpaint: {override_inpaint}")
                if override_box is not None and bool(override_box) != bool(default_box):
                    override_list.append(f"Background box: {override_box}")
                if override_list:
                    hint = "Overrides: " + ", ".join(override_list)
                    hint_rect = QRect(zoomed_rect.left(), zoomed_rect.top() - 22, min(300, zoomed_rect.width()), 18)
                    painter.setPen(QPen(QColor(240, 240, 240, 220)))
                    painter.setBrush(QColor(0, 0, 0, 160))
                    painter.drawRoundedRect(hint_rect, 6, 6)
                    painter.drawText(hint_rect, Qt.AlignCenter, hint)
            except Exception:
                pass

        if self.get_selection_mode() == "Brush Inpaint (LaMA)" and self.current_mouse_pos:
            image_point = self._widget_point_to_image(self.current_mouse_pos)
            if image_point is not None:
                painter.save()
                center = QPointF(self.current_mouse_pos)
                zoom = getattr(self.main_window, 'zoom_factor', 1.0) or 1.0
                radius = max(3.0, (self.brush_size * zoom) / 2.0)
                pen = QPen(QColor(255, 235, 59, 230))
                pen.setWidth(1)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(center, radius, radius)
                painter.restore()


    def clear_selection(self):
        self.selection_start = None
        self.selection_end = None
        self.selection_rect = QRect()
        self.dragging = False
        self.polygon_points = []
        self.current_mouse_pos = None
        # Jangan hapus item yang menunggu konfirmasi saat seleksi dibersihkan
        if self.main_window:
            self.main_window.update_selection_action_buttons(False)
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
        self._action_shortcut_map = {}
        self._shortcut_callbacks = {}
        self._active_shortcuts = {}
        self.shortcut_sequences = {}
        # Mouse-based shortcut registry: keys are tuples (event_type, Qt.MouseButton) -> callback
        # event_type: 'press' | 'release' | 'double'
        self._mouse_shortcuts = {}
        # Ensure required packages are present; offer to install missing ones.
        required_pkgs = [
            ("torch", "torch"),
            ("transformers", "transformers"),
            ("onnxruntime", "onnxruntime"),
            ("pytesseract", "pytesseract"),
            ("PIL", "Pillow"),
            ("google", "google-generative-ai"),
            ("PyQt5", "PyQt5"),
        ]
        try:
            ensure_dependencies(self, required_pkgs)
        except Exception:
            pass
        self.setWindowTitle("Manga OCR & Typeset Tool v14.3.4")
        self.image_files = []
        self.current_image_path = None
        self.current_image_pil = None
        self.original_pixmap = None
        self.typeset_pixmap = None
        self.zoom_factor = 1.0

        self.all_typeset_data = {}
        self.typeset_areas = []
        self.selected_typeset_area = None
        self.redo_stack = []
        self.history_entries = []
        self.proofreader_entries = []
        self.quality_entries = []
        self.history_lookup = {}
        self.history_counter = 0
        self.history_preview_limit = 5
        self.review_batch_limit = 20
        self.translation_styles = [
            "Santai (Default)",
            "Formal (Ke Atasan)",
            "Akrab (Ke Teman/Pacar)",
            "Vulgar/Dewasa (Adegan Seks)",
            "Sesuai Konteks Manga"
        ]

        # Initial font grouping mapping (group name -> list of font family names)
        # Populated with the example fonts the user requested. These are used
        # to filter the font dropdown in Typeset and Advanced Text Edit.
        self.font_groups = {
            "Dialog Normal": [
                "CC Wild Words",
                "Anime Ace 3",
            ],
            "Marah / Berteriak": [
                "Badaboom BB",
                "Komika Axis Bold",
            ],
            "Berbisik / Pelan": [
                "Patrick Hand",
                "Shadows Into Light",
            ],
            "Santai / Ke-enakan": [
                "Amatic SC",
                "Caveat",
            ],
            "Sexy / Intim": [
                "Sacramento",
                "Great Vibes",
            ],
            "Kaget / Shock": [
                "SF Comic Script Bold",
                "Komika Slick",
            ],
            "Tegang / Horor": [
                "Creepster",
                "Feast of Flesh BB",
            ],
            "Aksi / SFX": [
                "Komika Display",
            ],
        }

        # Path for persisting user-defined translation styles
        try:
            self._styles_storage_path = os.path.join(self.project_dir or os.path.expanduser('~'), '.manga_translation_styles.json')
        except Exception:
            self._styles_storage_path = os.path.join(os.path.expanduser('~'), '.manga_translation_styles.json')
        self.history_table = None
        self.proofreader_table = None
        self.quality_table = None
        self.history_view_all_button = None
        # Buttons for running batch reviews (may be created later in UI setup)
        self.run_proofreader_button = None
        self.run_quality_button = None
        self.proofreader_view_all_button = None
        self.quality_view_all_button = None
        self.proofreader_empty_label = None
        self.quality_empty_label = None
        self.proofreader_tab_widget = None
        self.quality_tab_widget = None
        self.batch_pf_btn = None
        self.batch_qc_btn = None
        self.proofreader_confirm_all_button = None
        self.quality_confirm_all_button = None
        self.result_table_registry = {
            'history': weakref.WeakSet(),
            'proofreader': weakref.WeakSet(),
            'quality': weakref.WeakSet(),
        }

        base_dir = os.path.dirname(os.path.abspath(__file__))
        font_root = os.path.join(base_dir, 'src', 'fonts')
        self.font_manager = FontManager(font_root)
        set_global_font_manager(self.font_manager)

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

        default_display = self.font_manager.list_fonts()[0]
        default_font = self.font_manager.create_qfont(default_display)
        if default_font.pointSizeF() <= 0:
            default_font.setPointSize(9)
        default_font.setWeight(QFont.Bold)
        self.typeset_font = default_font
        self.typeset_color = QColor(Qt.black)
        self.typeset_font.setLetterSpacing(QFont.PercentageSpacing, 100.0)

        typeset_cfg = SETTINGS.get('typeset', {}) if isinstance(SETTINGS.get('typeset'), dict) else {}
        outline_width_default = typeset_cfg.get('outline_width', typeset_cfg.get('outline_thickness', 2.0))
        try:
            outline_width_default = float(outline_width_default)
        except Exception:
            outline_width_default = 2.0
        outline_width_default = max(0.0, min(outline_width_default, 12.0))
        outline_color_value = typeset_cfg.get('outline_color', '#000000') or '#000000'
        outline_color = QColor(outline_color_value)
        if not outline_color.isValid():
            outline_color = QColor('#000000')

        self.typeset_line_spacing_value = 1.1
        self.typeset_char_spacing_value = 100.0
        self.typeset_alignment = 'center'
        self.typeset_orientation = 'horizontal'
        self.typeset_outline_enabled = bool(typeset_cfg.get('outline_enabled', False))
        self.typeset_outline_width = outline_width_default
        self.typeset_outline_color = outline_color
        self.preview_sample_text = "Aa Bb Cc"
        self.typeset_defaults = self._create_initial_typeset_defaults()

        self.processing_queue = []
        self.queue_mutex = QMutex()

        # Mutex to protect painting operations that use QPixmap/QImage paint devices
        self.paint_mutex = QMutex()

        self.batch_save_worker = None
        self.batch_save_thread = None

        self.project_dir = None
        self.cache_dir = None
        self.file_watcher = QFileSystemWatcher(self)
        self.file_watcher.directoryChanged.connect(self.on_directory_changed)

        # Cached custom cursor for pen/manual pen modes
        self.pen_cursor = None

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
        # Whether autosave is allowed to run (user preference). Default True.
        self.autosave_enabled = True

        self.dl_models = {
            'kitsumed_onnx': {'path': 'src/models/model_dynamic.onnx', 'instance': None, 'type': 'onnx'},
            'kitsumed_pt':   {'path': 'src/models/model.pt', 'instance': None, 'type': 'yolo'},
            'ogkalu_pt':     {'path': 'src/models/comic-speech-bubble-detector.pt', 'instance': None, 'type': 'yolo'},
            # [BARU] Inpainting Models
            'big_lama':      {'path': 'src/models/big-lama/models/best.ckpt', 'instance': None, 'type': 'inpaint'},
            'anime_inpaint': {'path': 'src/models/lama_large_512px.ckpt', 'instance': None, 'type': 'inpaint'},
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

        # Counter for how many snippets have been translated during this session
        # Initialized here to avoid AttributeError when translation routines increment it.
        self.translated_count = 0

        self.exchange_rate_thread = None
        self.exchange_rate_worker = None
        
        # [DIUBAH] Struktur data model AI untuk mendukung beberapa provider
        # Harga OpenAI per karakter diperkirakan dari harga per token (asumsi 1 token ~ 4 karakter)
        self.AI_PROVIDERS = {
            'Gemini': {
                'gemini-2.5-flash-lite': {
                    'display': 'Gemini 2.5 Flash Lite (Utama - Cepat & Murah)',
                    'pricing': {
                        'input': 0.0000001,   # USD per token
                        'output': 0.0000002
                    },
                    'limits': {'rpm': 4000, 'rpd': 10000000}
                },
                'gemini-2.5-flash': {
                    'display': 'Gemini 2.5 Flash (Akurasi Lebih Tinggi)',
                    'pricing': {
                        'input': 0.000000125,
                        'output': 0.00000025
                    },
                    'limits': {'rpm': 1000, 'rpd': 10000}
                },
                'gemini-2.5-pro': {
                    'display': 'Gemini 2.5 Pro (Teks Rumit & Penting)',
                    'pricing': {
                        'input': 0.0000025,
                        'output': 0.0000025
                    },
                    'limits': {'rpm': 150, 'rpd': 10000}
                }
            },
            'OpenAI': {
                'gpt-4o-mini': {
                    'display': 'GPT-4o Mini (Alternatif Cepat)',
                    'pricing': {
                        'input': 0.00000015,
                        'output': 0.00000060
                    },
                    'limits': {'rpm': 10000, 'rpd': 1000000}
                },
                'gpt-5-nano': {
                    'display': 'GPT-5 Nano (Super Hemat)',
                    'pricing': {
                        'input': 0.00000005,
                        'output': 0.00000040
                    },
                    'limits': {'rpm': 10000, 'rpd': 1000000}
                },
                'gpt-5-mini': {
                    'display': 'GPT-5 Mini (Seimbang)',
                    'pricing': {
                        'input': 0.00000015,
                        'output': 0.00000060
                    },
                    'limits': {'rpm': 10000, 'rpd': 1000000}
                }
            },
            'OpenRouter': {}
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
        self.is_transform_preview = False
        self._transform_preview_pixmap = None

        self.ui_update_queue = []
        self.ui_update_mutex = QMutex()
        self.ui_update_timer = QTimer(self)
        self.ui_update_timer.setSingleShot(True)
        self.ui_update_timer.timeout.connect(self.process_ui_updates)
        self.is_processing_ui_updates = False

        self.deferred_typeset_timer = QTimer(self)
        self.deferred_typeset_timer.setSingleShot(True)
        self.deferred_typeset_timer.timeout.connect(self.redraw_all_typeset_areas)
        self._last_redraw_request = 0.0

        self.is_in_confirmation_mode = False
        self.detection_thread = None
        self.detection_worker = None
        self.detected_items_map = {} # path -> list of dicts
        self.last_detection_mode = None
        self.preview_mode_active = False

        self.init_ui()
        self.setup_styles()
        self.setup_shortcuts()
        self.initialize_core_engines()
        self.load_usage_data()
        # load any saved custom translation styles
        try:
            self.load_translation_styles_from_disk()
        except Exception:
            pass
        self.check_limits_and_update_ui()
        self.fetch_exchange_rate()

    def setup_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('&File')
        self.save_project_action = QAction('Save Project', self)
        self.save_project_action.triggered.connect(self.save_project)
        file_menu.addAction(self.save_project_action)
        self.load_project_action = QAction('Load Project', self)
        self.load_project_action.triggered.connect(self.load_project)
        file_menu.addAction(self.load_project_action)
        self._action_shortcut_map.update({
            'save_project': self.save_project_action,
            'load_project': self.load_project_action,
        })

        file_menu.addSeparator()
        batch_save_action = QAction('Batch Save...', self)
        batch_save_action.triggered.connect(self.open_batch_save_dialog)
        file_menu.addAction(batch_save_action)
        export_pdf_action = QAction('Export Typeset to PDF...', self)
        export_pdf_action.triggered.connect(self.export_to_pdf)
        file_menu.addAction(export_pdf_action)

        view_menu = menu_bar.addMenu('&View')
        toggle_theme_action = QAction('Toggle Light/Dark Mode', self); toggle_theme_action.triggered.connect(self.toggle_theme); view_menu.addAction(toggle_theme_action)
        settings_menu = menu_bar.addMenu('&Settings')
        self.use_box_action = None
        settings_center_action = QAction('Settings...', self)
        settings_center_action.triggered.connect(self.open_settings_dialog)
        settings_menu.addAction(settings_center_action)
        help_menu = menu_bar.addMenu('&Help / Usage')
        about_action = QAction('About & API Usage', self); about_action.triggered.connect(self.show_about_dialog); help_menu.addAction(about_action)

    def open_settings_dialog(self, focus_tab: str = 'general'):
        dialog = SettingsCenterDialog(self)
        dialog.set_active_tab(focus_tab)
        dialog.exec_()

    def open_openrouter_settings_dialog(self):
        dialog = SettingsCenterDialog(self)
        dialog.set_active_tab('translation')
        dialog.exec_()

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
        # Use wrapper so we can trigger an auto-save before navigating
        self.next_button.clicked.connect(self.on_next_clicked)
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
        # Modernized right-panel layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Top header
        header = QLabel("Tools & Workflows")
        header.setStyleSheet("font-size:16px; font-weight:700; padding:4px 6px;")
        main_layout.addWidget(header)

        # Tabs area
        tabs_frame = QFrame()
        tabs_frame.setFrameShape(QFrame.NoFrame)
        tabs_layout = QVBoxLayout(tabs_frame)
        tabs_layout.setContentsMargins(0, 0, 0, 0)
        tabs_layout.setSpacing(6)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("main-tabs")
        self.tabs.setDocumentMode(True)
        self.tabs.setMovable(True)
        # Add tabs in preferred order
        tab_order = [
            (self._create_translate_tab(), "Translate"),
            (self._create_ai_hardware_tab(), "AI Hardware"),
            (self._create_typeset_tab(), "Typeset"),
            (self._create_history_tab(), "History"),
            (self._create_cleanup_tab(), "Cleanup"),
            (self._create_proofreader_tab(), "Proofreader"),
            (self._create_quality_tab(), "Quality Checker"),
        ]
        for widget, label in tab_order:
            self.tabs.addTab(widget, label)

        # Tidy tab bar appearance
        tab_bar = self.tabs.tabBar()
        try:
            tab_bar.setExpanding(False)
            tab_bar.setUsesScrollButtons(True)
            tab_bar.setElideMode(Qt.ElideRight)
            tab_bar.setToolTip('Drag tabs to reorder or use the scroll buttons when many tabs are present')
        except Exception:
            pass
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: none; }
            QTabBar::tab { background: transparent; color: #cfd9e6; padding: 6px 12px; margin: 2px; min-width: 84px; max-width: 220px; border-radius: 10px; }
            QTabBar::tab:selected { background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #6ac0ff, stop:1 #2b8ee6); color: #082036; font-weight: 700; }
            QTabBar::tab:hover { background: #32414d; color: #fff; }
            QTabBar::tab:!selected { background: rgba(30,30,30,0.55); }
        """)

        tabs_layout.addWidget(self.tabs)

        # Horizontal tab index scrollbar (handy on small windows)
        try:
            from PyQt5.QtWidgets import QScrollBar
            self.tab_index_scroll = QScrollBar(Qt.Horizontal, self)
            self.tab_index_scroll.setMinimum(0)
            self.tab_index_scroll.setMaximum(max(0, self.tabs.count() - 1))
            self.tab_index_scroll.setPageStep(1)
            self.tab_index_scroll.setSingleStep(1)
            self.tab_index_scroll.valueChanged.connect(lambda v: self.tabs.setCurrentIndex(int(v)))
            self.tabs.currentChanged.connect(lambda idx: self.tab_index_scroll.setValue(int(idx)))
            self.tab_index_scroll.setVisible(self.tabs.count() > 1)
            tabs_layout.addWidget(self.tab_index_scroll)
        except Exception:
            pass

        main_layout.addWidget(tabs_frame)

        # Expandable bottom area inside scroll area so controls remain accessible on small screens
        bottom_scroll = QScrollArea()
        bottom_scroll.setWidgetResizable(True)
        bottom_container = QWidget()
        bottom_layout = QVBoxLayout(bottom_container)
        bottom_layout.setContentsMargins(6, 6, 6, 6)
        bottom_layout.setSpacing(8)

        # Actions section
        actions_frame = QFrame()
        actions_frame.setFrameShape(QFrame.NoFrame)
        actions_layout = QVBoxLayout(actions_frame)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(6)

        actions_label = QLabel("Actions")
        actions_label.setStyleSheet("font-size:13px; font-weight:600;")
        actions_layout.addWidget(actions_label)

        # Buttons (kept same names for compatibility)
        self.process_batch_button = QPushButton("Process Batch Now (0 items)")
        self.process_batch_button.clicked.connect(self.start_batch_processing)
        actions_layout.addWidget(self.process_batch_button)
        self.on_batch_mode_changed(False)

        self.batch_process_button = QPushButton("Detect All Files")
        self.batch_process_button.setToolTip("Detects all bubbles/text in every file in the folder, lets you confirm, then processes them.")
        self.batch_process_button.clicked.connect(self.start_interactive_batch_detection)
        actions_layout.addWidget(self.batch_process_button)

        btns_row = QHBoxLayout()
        self.confirm_items_button = QPushButton("Confirm Items (0)")
        self.confirm_items_button.clicked.connect(self.process_confirmed_detections)
        self.confirm_items_button.setVisible(False)
        btns_row.addWidget(self.confirm_items_button)
        self.cancel_detection_button = QPushButton("Cancel Detection")
        self.cancel_detection_button.clicked.connect(self.cancel_interactive_batch)
        self.cancel_detection_button.setVisible(False)
        btns_row.addWidget(self.cancel_detection_button)
        actions_layout.addLayout(btns_row)

        # Undo/Redo and Save/Reset compact row
        ctrl_row = QHBoxLayout()
        self.undo_button = QPushButton("Undo"); self.undo_button.clicked.connect(self.undo_last_action); self.undo_button.setEnabled(False)
        self.redo_button = QPushButton("Redo"); self.redo_button.clicked.connect(self.redo_last_action); self.redo_button.setEnabled(False)
        ctrl_row.addWidget(self.undo_button); ctrl_row.addWidget(self.redo_button)
        actions_layout.addLayout(ctrl_row)

        save_row = QHBoxLayout()
        self.reset_button = QPushButton("Reset Image"); self.reset_button.clicked.connect(self.reset_view_to_original)
        self.save_button = QPushButton("Save Image"); self.save_button.clicked.connect(self.save_image)
        save_row.addWidget(self.reset_button); save_row.addWidget(self.save_button)
        actions_layout.addLayout(save_row)

        bottom_layout.addWidget(actions_frame)

        # API Status section (compact grid)
        api_frame = QFrame(); api_frame.setFrameShape(QFrame.NoFrame)
        api_layout = QGridLayout(api_frame)
        api_layout.setContentsMargins(0,0,0,0)
        api_layout.setSpacing(6)

        api_layout.addWidget(QLabel("Active Workers:"), 0, 0); self.active_workers_label = QLabel("0"); api_layout.addWidget(self.active_workers_label, 0, 1)
        api_layout.addWidget(QLabel("RPM:"), 1, 0); self.rpm_label = QLabel("0 / 0"); api_layout.addWidget(self.rpm_label, 1, 1)
        api_layout.addWidget(QLabel("RPD:"), 2, 0); self.rpd_label = QLabel("0 / 0"); api_layout.addWidget(self.rpd_label, 2, 1)
        api_layout.addWidget(QLabel("Cost (USD):"), 3, 0); self.cost_label = QLabel("$0.0000"); api_layout.addWidget(self.cost_label, 3, 1)
        api_layout.addWidget(QLabel("Cost (IDR):"), 4, 0); self.cost_idr_label = QLabel("Rp 0"); api_layout.addWidget(self.cost_idr_label, 4, 1)
        api_layout.addWidget(QLabel("Provider:"), 5, 0); self.provider_label = QLabel("-"); api_layout.addWidget(self.provider_label, 5, 1)
        api_layout.addWidget(QLabel("Model:"), 6, 0); self.model_label = QLabel("-"); api_layout.addWidget(self.model_label, 6, 1)

        bottom_layout.addWidget(api_frame)

        # Small status labels
        self.input_tokens_label = QLabel("Input Tokens: 0")
        self.output_tokens_label = QLabel("Output Tokens: 0")
        tokens_row = QHBoxLayout(); tokens_row.addWidget(self.input_tokens_label); tokens_row.addWidget(self.output_tokens_label)
        bottom_layout.addLayout(tokens_row)

        self.rate_label_input = QLabel("Rate Input: $0.0000000")
        self.rate_label_output = QLabel("Rate Output: $0.0000000")
        rates_row = QHBoxLayout(); rates_row.addWidget(self.rate_label_input); rates_row.addWidget(self.rate_label_output)
        bottom_layout.addLayout(rates_row)

        self.translated_label = QLabel("Translated Snippets: 0")
        bottom_layout.addWidget(self.translated_label)

        self.countdown_label = QLabel("Cooldown: 60s")
        self.countdown_label.setStyleSheet("color: #ffc107;"); self.countdown_label.setVisible(False)
        bottom_layout.addWidget(self.countdown_label)

        bottom_layout.addStretch()

        bottom_container.setLayout(bottom_layout)
        bottom_scroll.setWidget(bottom_container)
        main_layout.addWidget(bottom_scroll, 1)

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
        # Global orientation selector (kept for compatibility)
        self.orientation_combo = self._create_combo_box(ocr_layout, "Orientation:", ["Auto-Detect", "Horizontal", "Vertical"], 4, 0, 1, 2)

        # Per-language orientation overrides
        ocr_layout.addWidget(QLabel("EN Orientation:"), 5, 0)
        self.en_orientation_combo = QComboBox()
        self.en_orientation_combo.addItems(["Auto-Detect", "Horizontal", "Vertical"])
        self.en_orientation_combo.setCurrentText(SETTINGS.get('lang_orientation', {}).get('en', 'Auto-Detect'))
        self.en_orientation_combo.currentTextChanged.connect(lambda val: self._on_lang_orientation_changed('en', val))
        ocr_layout.addWidget(self.en_orientation_combo, 5, 1)

        ocr_layout.addWidget(QLabel("JP Orientation:"), 6, 0)
        self.jp_orientation_combo = QComboBox()
        self.jp_orientation_combo.addItems(["Auto-Detect", "Horizontal", "Vertical"])
        self.jp_orientation_combo.setCurrentText(SETTINGS.get('lang_orientation', {}).get('ja', 'Auto-Detect'))
        self.jp_orientation_combo.currentTextChanged.connect(lambda val: self._on_lang_orientation_changed('ja', val))
        ocr_layout.addWidget(self.jp_orientation_combo, 6, 1)

        layout.addWidget(ocr_group)

        detection_group = QGroupBox("OCR Detection Source")
        detection_layout = QGridLayout(detection_group)
        self.manga_use_easy_detection_checkbox = QCheckBox("Manga-OCR: use EasyOCR regions (recognize with Manga-OCR)")
        self.manga_use_easy_detection_checkbox.setChecked(True)
        self.manga_use_easy_detection_checkbox.setToolTip("When enabled, EasyOCR proposes text regions and Manga-OCR performs recognition. Disable to use Manga-OCR's own lightweight detection heuristic.")
        detection_layout.addWidget(self.manga_use_easy_detection_checkbox, 0, 0, 1, 2)

        self.tesseract_use_easy_detection_checkbox = QCheckBox("Tesseract: use EasyOCR regions (recognize with Tesseract)")
        self.tesseract_use_easy_detection_checkbox.setChecked(True)
        self.tesseract_use_easy_detection_checkbox.setToolTip("When enabled, EasyOCR proposes text regions before Tesseract recognition. Disable to rely on Tesseract's native detection from image_to_data.")
        detection_layout.addWidget(self.tesseract_use_easy_detection_checkbox, 1, 0, 1, 2)

        layout.addWidget(detection_group)

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
        
        selection_layout.addWidget(QLabel("Mode:"), 0, 0)
        self.selection_mode_combo = ScrollableComboBox(self)
        self.selection_mode_combo.addItems(SELECTION_MODE_LABELS)
        selection_layout.addWidget(self.selection_mode_combo, 0, 1, 1, 1)

        self.selection_mode_combo.currentTextChanged.connect(self.selection_mode_changed)
        pen_buttons_layout = QHBoxLayout()
        self.selection_confirm_button = QPushButton("Confirm")
        self.selection_confirm_button.clicked.connect(self._confirm_selection_action)
        self.selection_confirm_button.setVisible(False)
        self.selection_cancel_button = QPushButton("Cancel")
        self.selection_cancel_button.clicked.connect(self._cancel_selection_action)
        self.selection_cancel_button.setVisible(False)
        pen_buttons_layout.addWidget(self.selection_confirm_button)
        pen_buttons_layout.addWidget(self.selection_cancel_button)
        selection_layout.addLayout(pen_buttons_layout, 1, 0, 1, 2)

        self.create_bubble_checkbox = QCheckBox("Create white bubble with black outline")
        self.create_bubble_checkbox.setToolTip("When enabled, confirmed selections will render a bubble background behind the text.")
        selection_layout.addWidget(self.create_bubble_checkbox, 2, 0, 1, 2)
        # New: option to use a background box for rendered text (global cleanup option)
        self.use_background_box_checkbox = QCheckBox("Use Background Box for Text")
        # Initialize from saved SETTINGS if present
        self.use_background_box_checkbox.setChecked(bool(SETTINGS.get('cleanup', {}).get('use_background_box', True)))
        self.use_background_box_checkbox.setToolTip("When enabled, OCR/translated text will be placed inside a background box. If disabled, text is drawn directly over the image (transparent background).")
        selection_layout.addWidget(self.use_background_box_checkbox, 3, 0, 1, 2)
        # Small control: apply mode (selected area vs global)
        apply_mode_layout = QHBoxLayout()
        self.apply_mode_selected_radio = QRadioButton("Apply to Selected Area")
        self.apply_mode_global_radio = QRadioButton("Apply Globally")
        # Restore saved apply mode if present in SETTINGS
        saved_mode = self._default_cleanup_value('apply_mode') or 'selected'
        if saved_mode == 'global':
            self.apply_mode_global_radio.setChecked(True)
        else:
            self.apply_mode_selected_radio.setChecked(True)
        apply_mode_layout.addWidget(self.apply_mode_selected_radio)
        apply_mode_layout.addWidget(self.apply_mode_global_radio)
        # small status label to show which mode is active
        self.apply_mode_status_label = QLabel()
        def _update_apply_mode_label():
            if self.apply_mode_global_radio.isChecked():
                self.apply_mode_status_label.setText("Mode: Global")
            else:
                self.apply_mode_status_label.setText("Mode: Selected Area")
        _update_apply_mode_label()
        apply_mode_layout.addWidget(self.apply_mode_status_label)
        selection_layout.addLayout(apply_mode_layout, 4, 0, 1, 2)

        # Apply to All Areas button
        self.apply_all_button = QPushButton("Apply to All Areas")
        self.apply_all_button.setToolTip("Apply the selected action either to update defaults only or force update every existing area.")
        def _on_apply_all_clicked():
            # Dialog with two choices
            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QMessageBox

            dlg = QDialog(self)
            dlg.setWindowTitle("Apply to All Areas")
            v = QVBoxLayout(dlg)
            v.addWidget(QLabel("Choose how to apply the change to all existing areas:"))
            btn_layout = QHBoxLayout()
            update_defaults_btn = QPushButton("Update Defaults Only")
            force_update_btn = QPushButton("Force Update All")
            cancel_btn = QPushButton("Cancel")
            btn_layout.addWidget(update_defaults_btn)
            btn_layout.addWidget(force_update_btn)
            btn_layout.addWidget(cancel_btn)
            v.addLayout(btn_layout)

            def _update_defaults_only():
                use_box_val = bool(self.use_background_box_checkbox.isChecked())
                use_inpaint_val = bool(self.inpaint_checkbox.isChecked())
                self._set_global_cleanup_default('use_background_box', use_box_val)
                self._set_global_cleanup_default('use_inpaint', use_inpaint_val)
                QMessageBox.information(self, "Apply to All", "Global defaults updated. Existing areas keep their individual overrides.")
                self._sync_cleanup_controls_from_selection()
                dlg.accept()

            def _force_update_all():
                use_box_val = bool(self.use_background_box_checkbox.isChecked())
                use_inpaint_val = bool(self.inpaint_checkbox.isChecked())
                self._set_global_cleanup_default('use_background_box', use_box_val)
                self._set_global_cleanup_default('use_inpaint', use_inpaint_val)
                default_box = self._default_cleanup_value('use_background_box')
                default_inpaint = self._default_cleanup_value('use_inpaint')
                for record in (self.all_typeset_data or {}).values():
                    areas = record.get('areas', []) if isinstance(record, dict) else []
                    for area in areas:
                        if use_box_val == default_box:
                            area.clear_override('use_background_box')
                        else:
                            area.set_override('use_background_box', use_box_val)
                        if use_inpaint_val == default_inpaint:
                            area.clear_override('use_inpaint')
                        else:
                            area.set_override('use_inpaint', use_inpaint_val)
                try:
                    self.redraw_all_typeset_areas()
                except Exception:
                    pass
                label = getattr(self, 'image_label', None)
                if label is not None:
                    try:
                        label.update()
                    except Exception:
                        pass
                self._sync_cleanup_controls_from_selection()
                QMessageBox.information(self, "Apply to All", "Global defaults updated and applied to every typeset area.")
                dlg.accept()

            update_defaults_btn.clicked.connect(_update_defaults_only)
            force_update_btn.clicked.connect(_force_update_all)
            cancel_btn.clicked.connect(dlg.reject)
            dlg.exec_()

        self.apply_all_button.clicked.connect(_on_apply_all_clicked)
        selection_layout.addWidget(self.apply_all_button, 5, 0, 1, 2)

        # Brush size controls for inpainting mode
        selection_layout.addWidget(QLabel("Brush Size:"), 6, 0)
        brush_controls = QHBoxLayout()
        brush_controls.setContentsMargins(0, 0, 0, 0)
        self.brush_size_slider = QSlider(Qt.Horizontal)
        self.brush_size_slider.setRange(5, 200)
        initial_brush_size = getattr(self.image_label, 'brush_size', 45)
        self.brush_size_slider.setValue(initial_brush_size)
        self.brush_size_slider.valueChanged.connect(self._on_brush_size_slider_changed)
        brush_controls.addWidget(self.brush_size_slider, 2)
        self.brush_size_spin = QSpinBox()
        self.brush_size_spin.setRange(5, 200)
        self.brush_size_spin.setValue(initial_brush_size)
        self.brush_size_spin.valueChanged.connect(self._on_brush_size_spin_changed)
        brush_controls.addWidget(self.brush_size_spin)
        selection_layout.addLayout(brush_controls, 6, 1, 1, 1)
        self.image_label.set_brush_size(initial_brush_size)

        # Whenever apply-mode is toggled, save to SETTINGS
        def _on_apply_mode_changed():
            mode = 'global' if self.apply_mode_global_radio.isChecked() else 'selected'
            self._set_global_cleanup_default('apply_mode', mode)
            _update_apply_mode_label()
            self._sync_cleanup_controls_from_selection()
        self.apply_mode_selected_radio.toggled.connect(_on_apply_mode_changed)
        self.apply_mode_global_radio.toggled.connect(_on_apply_mode_changed)

        # When user toggles the checkbox in the Cleanup tab, update either the hovered area or global default depending on apply mode
        def on_tab_checkbox_toggled(state):
            self._apply_cleanup_change('use_background_box', bool(state))
        self.use_background_box_checkbox.toggled.connect(on_tab_checkbox_toggled)
        layout.addWidget(selection_group)
        
        # [DIUBAH] Inpainting Group dengan model baru
        inpaint_group = QGroupBox("Inpainting (Text Removal)")
        inpaint_layout = QGridLayout(inpaint_group)
        self.inpaint_checkbox = QCheckBox("Gunakan Inpainting")
        # Initialize from SETTINGS default if present
        self.inpaint_checkbox.setChecked(bool(SETTINGS.get('cleanup', {}).get('use_inpaint', True)))
        inpaint_layout.addWidget(self.inpaint_checkbox, 0, 0, 1, 2)

        inpaint_models = ["OpenCV-NS", "OpenCV-Telea"]
        if self.is_lama_available:
            if os.path.exists(self.dl_models['big_lama']['path']):
                inpaint_models.append("Big-LaMa")
            if os.path.exists(self.dl_models['anime_inpaint']['path']):
                inpaint_models.append("Anime-Inpainting")
        
        self.inpaint_model_combo = self._create_combo_box(inpaint_layout, "Model:", inpaint_models, 1, 0)
        self.inpaint_padding_spinbox = self._create_spin_box(inpaint_layout, "Padding (px):", 1, 25, 5, 2, 0)
        # When toggling inpaint checkbox, respect apply mode selection
        def on_inpaint_toggled(state):
            self._apply_cleanup_change('use_inpaint', bool(state))
        self.inpaint_checkbox.toggled.connect(on_inpaint_toggled)
        layout.addWidget(inpaint_group)

        # Ensure UI reflects either selected area overrides or global defaults
        self._sync_cleanup_controls_from_selection()

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

    def _create_typeset_tab_legacy(self):
        """Legacy placeholder retained for backward compatibility."""
        return self._create_typeset_tab()

    def _create_typeset_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(18)

        defaults_group = QGroupBox("Defaults")
        defaults_layout = QHBoxLayout(defaults_group)
        defaults_layout.setSpacing(12)
        defaults_description = QLabel("Save your current typography to reuse it on future text areas or restore the previously stored default.")
        defaults_description.setWordWrap(True)
        defaults_layout.addWidget(defaults_description, 1)
        actions_layout = QHBoxLayout()
        actions_layout.setSpacing(8)
        self.save_typeset_defaults_button = QPushButton("Save Current")
        self.save_typeset_defaults_button.setToolTip("Store the current typography as the default for new areas.")
        self.save_typeset_defaults_button.clicked.connect(self._handle_save_typeset_defaults)
        actions_layout.addWidget(self.save_typeset_defaults_button)
        self.reset_typeset_defaults_button = QPushButton("Reset Defaults")
        self.reset_typeset_defaults_button.setToolTip("Restore the saved default typography settings.")
        self.reset_typeset_defaults_button.clicked.connect(self._handle_reset_typeset_defaults)
        actions_layout.addWidget(self.reset_typeset_defaults_button)
        defaults_layout.addLayout(actions_layout)
        layout.addWidget(defaults_group)

        font_group = QGroupBox("Typography")
        font_layout = QGridLayout(font_group)
        font_layout.setHorizontalSpacing(12)
        font_layout.setVerticalSpacing(10)

        font_layout.addWidget(QLabel("Font Group"), 0, 0)
        group_row = QHBoxLayout()
        group_row.setSpacing(6)
        self.font_group_combo = QComboBox()
        self.font_group_combo.setMinimumWidth(180)
        self.font_group_combo.addItem("All")
        for group_name in getattr(self, 'font_groups', {}).keys():
            self.font_group_combo.addItem(group_name)
        self.font_group_combo.currentTextChanged.connect(lambda txt: self._on_font_group_changed(txt))
        group_row.addWidget(self.font_group_combo, 1)
        self.add_group_btn = QPushButton("New Group")
        self.add_group_btn.setToolTip("Create a new font group.")
        self.add_group_btn.clicked.connect(self._on_add_font_group_clicked)
        group_row.addWidget(self.add_group_btn)
        self.remove_group_btn = QPushButton("Delete")
        self.remove_group_btn.setToolTip("Remove the selected font group.")
        self.remove_group_btn.clicked.connect(self._on_remove_font_group_clicked)
        group_row.addWidget(self.remove_group_btn)
        self.add_font_to_group_btn = QPushButton("Add Font")
        self.add_font_to_group_btn.setToolTip("Add a font to the selected group.")
        self.add_font_to_group_btn.clicked.connect(self._on_add_font_to_group_clicked)
        group_row.addWidget(self.add_font_to_group_btn)
        group_row.addStretch(1)
        font_layout.addLayout(group_row, 0, 1, 1, 2)

        font_layout.addWidget(QLabel("Family"), 1, 0)
        self.font_dropdown = QComboBox()
        self.font_dropdown.setMinimumWidth(240)
        self.font_dropdown.currentTextChanged.connect(self.on_typeset_font_change)
        font_layout.addWidget(self.font_dropdown, 1, 1, 1, 2)

        font_layout.addWidget(QLabel("Preview"), 2, 0)
        self.font_preview_label = QLabel("AaBb123")
        self.font_preview_label.setAlignment(Qt.AlignCenter)
        self.font_preview_label.setMinimumHeight(64)
        self.font_preview_label.setStyleSheet("border: 1px solid #1f2b3b; border-radius: 8px; padding: 12px; background-color: #161f2b;")
        font_layout.addWidget(self.font_preview_label, 2, 1, 1, 2)

        self.import_font_button = QPushButton("Import Font...")
        self.import_font_button.setToolTip("Add new font files from your computer (TTF, OTF, TTC, OTC).")
        self.import_font_button.clicked.connect(self.import_font)
        font_layout.addWidget(self.import_font_button, 3, 1, 1, 2)
        layout.addWidget(font_group)

        appearance_group = QGroupBox("Appearance")
        appearance_layout = QGridLayout(appearance_group)
        appearance_layout.setHorizontalSpacing(12)
        appearance_layout.setVerticalSpacing(10)

        appearance_layout.addWidget(QLabel("Style"), 0, 0)
        style_row = QHBoxLayout()
        style_row.setSpacing(6)
        self.bold_toggle = self._create_tool_toggle(self._make_style_icon('B'), "Bold")
        self.bold_toggle.toggled.connect(self._on_typeset_style_changed)
        style_row.addWidget(self.bold_toggle)
        self.italic_toggle = self._create_tool_toggle(self._make_style_icon('I'), "Italic")
        self.italic_toggle.toggled.connect(self._on_typeset_style_changed)
        style_row.addWidget(self.italic_toggle)
        self.underline_toggle = self._create_tool_toggle(self._make_style_icon('U'), "Underline")
        self.underline_toggle.toggled.connect(self._on_typeset_style_changed)
        style_row.addWidget(self.underline_toggle)
        style_row.addStretch(1)
        appearance_layout.addLayout(style_row, 0, 1)

        appearance_layout.addWidget(QLabel("Text Color"), 1, 0)
        color_row = QHBoxLayout()
        color_row.setSpacing(6)
        self.color_button = QPushButton("Pick Color")
        self.color_button.setMaximumWidth(160)
        self.color_button.clicked.connect(self.choose_color)
        self.color_button.setToolTip("Pick the colour used for new text areas.")
        color_row.addWidget(self.color_button)
        color_row.addStretch(1)
        appearance_layout.addLayout(color_row, 1, 1)

        appearance_layout.addWidget(QLabel("Outline"), 2, 0)
        outline_row = QHBoxLayout()
        outline_row.setSpacing(6)
        self.outline_toggle = self._create_tool_toggle(self._make_outline_icon(), "Toggle outline")
        self.outline_toggle.toggled.connect(self._on_typeset_outline_changed)
        outline_row.addWidget(self.outline_toggle)
        self.outline_color_button = QPushButton("Outline Color")
        self.outline_color_button.setMaximumWidth(160)
        self.outline_color_button.clicked.connect(self.choose_outline_color)
        outline_row.addWidget(self.outline_color_button)
        self.outline_width_spin = QDoubleSpinBox()
        self.outline_width_spin.setRange(0.0, 12.0)
        self.outline_width_spin.setDecimals(1)
        self.outline_width_spin.setSingleStep(0.1)
        self.outline_width_spin.setSuffix(" px")
        self.outline_width_spin.valueChanged.connect(self._on_outline_width_changed)
        outline_row.addWidget(self.outline_width_spin)
        outline_row.addStretch(1)
        appearance_layout.addLayout(outline_row, 2, 1)
        layout.addWidget(appearance_group)

        spacing_group = QGroupBox("Spacing & Size")
        spacing_layout = QGridLayout(spacing_group)
        spacing_layout.setHorizontalSpacing(12)
        spacing_layout.setVerticalSpacing(10)

        spacing_layout.addWidget(QLabel("Font Size"), 0, 0)
        self.font_size_spin = QDoubleSpinBox()
        self.font_size_spin.setRange(4.0, 220.0)
        self.font_size_spin.setDecimals(1)
        self.font_size_spin.setSingleStep(1.0)
        self.font_size_spin.setSuffix(" pt")
        self.font_size_spin.valueChanged.connect(self._on_typeset_font_size_changed)
        self.font_size_spin.setToolTip("Adjust the point size for new text areas.")
        spacing_layout.addWidget(self.font_size_spin, 0, 1)

        spacing_layout.addWidget(QLabel("Line Spacing"), 1, 0)
        line_row = QHBoxLayout()
        line_row.setSpacing(8)
        self.line_spacing_slider = QSlider(Qt.Horizontal)
        self.line_spacing_slider.setRange(60, 300)
        self.line_spacing_slider.setSingleStep(5)
        self.line_spacing_slider.setPageStep(10)
        self.line_spacing_slider.valueChanged.connect(self._on_typeset_line_spacing_changed)
        self.line_spacing_slider.setToolTip("Adjust the spacing between lines (0.60x - 3.00x).")
        line_row.addWidget(self.line_spacing_slider, 1)
        self.line_spacing_value_label = QLabel("1.00x")
        self.line_spacing_value_label.setMinimumWidth(60)
        line_row.addWidget(self.line_spacing_value_label)
        spacing_layout.addLayout(line_row, 1, 1)

        spacing_layout.addWidget(QLabel("Character Spacing"), 2, 0)
        char_row = QHBoxLayout()
        char_row.setSpacing(8)
        self.char_spacing_slider = QSlider(Qt.Horizontal)
        self.char_spacing_slider.setRange(50, 400)
        self.char_spacing_slider.setSingleStep(5)
        self.char_spacing_slider.setPageStep(10)
        self.char_spacing_slider.valueChanged.connect(self._on_typeset_char_spacing_changed)
        self.char_spacing_slider.setToolTip("Adjust spacing between characters (percentage).")
        char_row.addWidget(self.char_spacing_slider, 1)
        self.char_spacing_value_label = QLabel("100%")
        self.char_spacing_value_label.setMinimumWidth(60)
        char_row.addWidget(self.char_spacing_value_label)
        spacing_layout.addLayout(char_row, 2, 1)
        layout.addWidget(spacing_group)

        layout_group = QGroupBox("Layout")
        layout_grid = QGridLayout(layout_group)
        layout_grid.setHorizontalSpacing(12)
        layout_grid.setVerticalSpacing(10)

        layout_grid.addWidget(QLabel("Alignment"), 0, 0)
        alignment_row = QHBoxLayout()
        alignment_row.setSpacing(6)
        self.alignment_group = QButtonGroup(self)
        self.alignment_group.setExclusive(True)
        self.align_left_button = self._create_tool_toggle(self._make_alignment_icon('left'), "Align left")
        self.align_center_button = self._create_tool_toggle(self._make_alignment_icon('center'), "Align center")
        self.align_right_button = self._create_tool_toggle(self._make_alignment_icon('right'), "Align right")
        for mode, button in (('left', self.align_left_button), ('center', self.align_center_button), ('right', self.align_right_button)):
            button.setCheckable(True)
            button.setProperty('align-mode', mode)
            self.alignment_group.addButton(button)
            button.toggled.connect(self._on_alignment_button_toggled)
            alignment_row.addWidget(button)
        alignment_row.addStretch(1)
        layout_grid.addLayout(alignment_row, 0, 1)

        layout_grid.addWidget(QLabel("Orientation"), 1, 0)
        orientation_row = QHBoxLayout()
        orientation_row.setSpacing(6)
        self.orientation_group = QButtonGroup(self)
        self.orientation_group.setExclusive(True)
        self.orientation_horizontal_button = self._create_tool_toggle(self._make_orientation_icon('horizontal'), "Horizontal text")
        self.orientation_vertical_button = self._create_tool_toggle(self._make_orientation_icon('vertical'), "Vertical text")
        for mode, button in (('horizontal', self.orientation_horizontal_button), ('vertical', self.orientation_vertical_button)):
            button.setCheckable(True)
            button.setProperty('orientation-mode', mode)
            self.orientation_group.addButton(button)
            button.toggled.connect(self._on_orientation_button_toggled)
            orientation_row.addWidget(button)
        orientation_row.addStretch(1)
        layout_grid.addLayout(orientation_row, 1, 1)
        layout.addWidget(layout_group)

        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(12, 10, 12, 12)
        self.typeset_preview_label = QLabel()
        self.typeset_preview_label.setAlignment(Qt.AlignCenter)
        self.typeset_preview_label.setMinimumHeight(180)
        self.typeset_preview_label.setStyleSheet("background-color: #172330; border: 1px solid #1f2b3b; border-radius: 12px;")
        preview_layout.addWidget(self.typeset_preview_label)
        layout.addWidget(preview_group)

        layout.addStretch(1)

        scroll.setWidget(container)

        selected_display = None
        if getattr(self, 'typeset_defaults', None):
            selected_display = self.typeset_defaults.get('font_display')
        self._populate_typeset_font_dropdown(selected_display)
        self._apply_typeset_defaults()
        self._refresh_outline_controls_enabled()

        return scroll
    def _create_history_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 15, 10, 10)

        description = QLabel("Review the latest translation results. Only the five most recent entries are shown here.")
        description.setWordWrap(True)
        layout.addWidget(description)

        self.history_table = self._create_result_table()
        self.history_table.setProperty('result_limit', self.history_preview_limit)
        self.result_table_registry['history'].add(self.history_table)
        layout.addWidget(self.history_table)

        controls_layout = QHBoxLayout()
        controls_layout.addStretch()
        history_view_all = QPushButton("View All")
        history_view_all.clicked.connect(self.show_history_modal)
        controls_layout.addWidget(history_view_all)
        layout.addLayout(controls_layout)

        self.history_view_all_button = history_view_all
        self.refresh_history_views()
        return tab

    def _create_proofreader_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 15, 10, 10)

        description = QLabel("Send recent translations to the AI proofreader to polish grammar and flow.")
        description.setWordWrap(True)
        layout.addWidget(description)

        self.proofreader_empty_label = QLabel("No entries sent from History.")
        self.proofreader_empty_label.setAlignment(Qt.AlignCenter)
        self.proofreader_empty_label.setStyleSheet("color: #7f8ba7; font-style: italic;")
        layout.addWidget(self.proofreader_empty_label)

        self.proofreader_table = self._create_result_table()
        self.proofreader_table.setProperty('result_limit', self.history_preview_limit)
        self.proofreader_table.setVisible(False)
        self.result_table_registry['proofreader'].add(self.proofreader_table)
        layout.addWidget(self.proofreader_table)

        proof_controls = QHBoxLayout()
        proof_controls.addStretch()
        proof_confirm_all = QPushButton("Confirm All")
        proof_confirm_all.clicked.connect(partial(self.confirm_all_result_entries, 'proofreader'))
        proof_controls.addWidget(proof_confirm_all)
        # Batch PF button
        batch_pf_btn = QPushButton("Batch PF (AI Contextual Translate)")
        batch_pf_btn.clicked.connect(self.batch_pf_contextual_translate)
        proof_controls.addWidget(batch_pf_btn)
        proof_view_all = QPushButton("View All")
        proof_view_all.clicked.connect(self.show_proofreader_modal)
        proof_controls.addWidget(proof_view_all)
        layout.addLayout(proof_controls)

        self.batch_pf_btn = batch_pf_btn
        self.proofreader_confirm_all_button = proof_confirm_all

        self.proofreader_view_all_button = proof_view_all
        self.proofreader_tab_widget = tab
        self.refresh_history_views()
        return tab

    def _create_quality_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 15, 10, 10)

        description = QLabel("Request a final quality review to check consistency and naturalness.")
        description.setWordWrap(True)
        layout.addWidget(description)

        self.quality_empty_label = QLabel("No entries sent from History.")
        self.quality_empty_label.setAlignment(Qt.AlignCenter)
        self.quality_empty_label.setStyleSheet("color: #7f8ba7; font-style: italic;")
        layout.addWidget(self.quality_empty_label)

        self.quality_table = self._create_result_table()
        self.quality_table.setProperty('result_limit', self.history_preview_limit)
        self.quality_table.setVisible(False)
        self.result_table_registry['quality'].add(self.quality_table)
        layout.addWidget(self.quality_table)

        quality_controls = QHBoxLayout()
        quality_controls.addStretch()
        quality_confirm_all = QPushButton("Confirm All")
        quality_confirm_all.clicked.connect(partial(self.confirm_all_result_entries, 'quality'))
        quality_controls.addWidget(quality_confirm_all)
        # Batch QC button
        batch_qc_btn = QPushButton("Batch QC (AI Style/Tone Check)")
        batch_qc_btn.clicked.connect(self.batch_qc_style_tone_check)
        quality_controls.addWidget(batch_qc_btn)
        quality_view_all = QPushButton("View All")
        quality_view_all.clicked.connect(self.show_quality_modal)
        quality_controls.addWidget(quality_view_all)
        layout.addLayout(quality_controls)

        self.batch_qc_btn = batch_qc_btn
        self.quality_confirm_all_button = quality_confirm_all
        self.quality_view_all_button = quality_view_all
        self.quality_tab_widget = tab
        self.refresh_history_views()
        return tab

    def batch_pf_contextual_translate(self):
        """
        Batch send all PF entries (original text only) to AI for contextual translation.
        Result: Each bubble gets updated with contextual translation.
        """
        if not self.proofreader_entries:
            QMessageBox.information(self, "No PF Entries", "Tidak ada entry PF yang bisa diproses.")
            return
        provider, model_name = self.get_selected_model_name()
        if not model_name:
            QMessageBox.warning(self, "AI Model Missing", "Pilih AI model dulu sebelum batch PF.")
            return
        # Build prompt: send all original texts, ask AI to translate contextually so text flows naturally
        pf_texts = [e.get('original_text', '') for e in self.proofreader_entries if e.get('original_text')]
        if not pf_texts:
            QMessageBox.information(self, "No Texts", "Tidak ada original text di PF entries.")
            return
        # Request JSON array first to make parsing reliable
        prompt = (
            "IMPORTANT: Return ONLY a JSON array of strings. Example: [\"dialog1\", \"dialog2\"]\n"
            "Terjemahkan dialog berikut ke bahasa Indonesia secara kontekstual sehingga hasilnya saling nyambung dan alami. "
            "Berikan hasil terjemahan dalam urutan yang sama. Jika tidak bisa mengekspor JSON, kembalikan teks setiap dialog pada baris terpisah.\n\n" +
            "\n".join(pf_texts)
        )
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            temperature = 0.35
            response_text = self._invoke_ai_review(provider, model_name, prompt, temperature=temperature)
        finally:
            QApplication.restoreOverrideCursor()
        if not response_text:
            QMessageBox.warning(self, "AI Error", "Tidak ada respon dari AI.")
            return
        # Parse response: get list of results
        results = self._parse_ai_list_response(response_text, expected_count=len(pf_texts))
        if len(results) != len(pf_texts):
            resp = QMessageBox.question(self, "Mismatch",
                                        f"AI mengembalikan {len(results)} item, tapi jumlah dialog yang dikirim {len(pf_texts)}.\n"
                                        "Terima hasil yang dapat diambil terbaik (best-effort mapping) dan lanjutkan?",
                                        QMessageBox.Yes | QMessageBox.No)
            if resp != QMessageBox.Yes:
                return
            if len(results) > len(pf_texts):
                results = results[:len(pf_texts)]
            else:
                results = results + [orig for orig in pf_texts[len(results):]]

        # Stage results: set translated_text, ai_model and staged flag, but do NOT apply to bubbles yet
        for entry, new_text in zip(self.proofreader_entries, results):
            entry['translated_text'] = new_text
            entry['ai_model'] = model_name
            entry['staged'] = True

        self.refresh_history_views()
        QMessageBox.information(self, "Batch PF Selesai", "Hasil telah di-stage. Tekan 'Confirm' pada baris untuk menerapkan ke bubble.")

    def batch_qc_style_tone_check(self):
        """
        Batch send all QC entries (translated text) to AI for style/tone validation.
        Result: Each bubble gets updated with validated/adjusted translation.
        """
        if not self.quality_entries:
            QMessageBox.information(self, "No QC Entries", "Tidak ada entry QC yang bisa diproses.")
            return
        provider, model_name = self.get_selected_model_name()
        if not model_name:
            QMessageBox.warning(self, "AI Model Missing", "Pilih AI model dulu sebelum batch QC.")
            return
        qc_texts = [e.get('translated_text', '') for e in self.quality_entries if e.get('translated_text')]
        if not qc_texts:
            QMessageBox.information(self, "No Texts", "Tidak ada hasil translate di QC entries.")
            return
        prompt = (
            "IMPORTANT: Return ONLY a JSON array of strings. Example: [\"rev1\", \"rev2\"]\n"
            "Berikut adalah hasil terjemahan dialog manga. Tolong cek gaya bahasa, suasana, dan tone agar sesuai dan alami. "
            "Jika perlu, sesuaikan gaya bahasa agar konsisten dan cocok dengan konteks manga. Berikan hasil revisi dalam urutan yang sama.\n\n" +
            "\n".join(qc_texts)
        )
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            temperature = 0.3
            response_text = self._invoke_ai_review(provider, model_name, prompt, temperature=temperature)
        finally:
            QApplication.restoreOverrideCursor()
        if not response_text:
            QMessageBox.warning(self, "AI Error", "Tidak ada respon dari AI.")
            return
        results = self._parse_ai_list_response(response_text, expected_count=len(qc_texts))
        if len(results) != len(qc_texts):
            resp = QMessageBox.question(self, "Mismatch",
                                        f"AI mengembalikan {len(results)} item, tapi jumlah dialog yang dikirim {len(qc_texts)}.\n"
                                        "Terima hasil yang dapat diambil terbaik (best-effort mapping) dan lanjutkan?",
                                        QMessageBox.Yes | QMessageBox.No)
            if resp != QMessageBox.Yes:
                return
            if len(results) > len(qc_texts):
                results = results[:len(qc_texts)]
            else:
                results = results + [orig for orig in qc_texts[len(results):]]
        for entry, new_text in zip(self.quality_entries, results):
            entry['translated_text'] = new_text
            entry['ai_model'] = model_name
            entry['staged'] = True

        self.refresh_history_views()
        QMessageBox.information(self, "Batch QC Selesai", "Hasil telah di-stage. Tekan 'Confirm' pada baris untuk menerapkan ke bubble.")

    def _create_result_table(self):
        table = QTableWidget()
        # Columns: No, Original, Translated, Style, Actions
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["No", "Original OCR", "Translated Text", "Style", "Actions"])

        # Resize mode untuk kolom
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)

        # Hilangkan header vertikal (angka baris)
        table.verticalHeader().setVisible(False)

        # Nonaktifkan edit & selection
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.NoSelection)
        table.setFocusPolicy(Qt.NoFocus)

        # Wrap teks + alternating row
        table.setWordWrap(True)
        table.setAlternatingRowColors(True)

        # Tambahkan styling
        table.setStyleSheet("""
            QTableWidget {
                background-color: #2b2b2b;       /* dark gray */
                alternate-background-color: #383838;
                color: #ffffff;                  /* teks putih */
                gridline-color: #444444;
                font-size: 13px;
            }
            QHeaderView::section {
                background-color: #3c3c3c;
                color: #ffffff;
                font-weight: bold;
                border: 1px solid #555555;
                padding: 4px;
            }
        """)

        return table

    def _get_recent_entries(self, entries, limit=None):
        if not entries:
            return []
        sorted_entries = sorted(entries, key=lambda e: e.get('timestamp', 0), reverse=True)
        if limit:
            return sorted_entries[:limit]
        return sorted_entries

    def _parse_ai_list_response(self, text: str, expected_count: int | None = None) -> list:
        """Try to parse AI response as JSON array first. If that fails, fall back to line-splitting.

        Returns a list of strings (possibly empty). Does minimal cleanup of quotes and bullets.
        """
        if not text or not text.strip():
            return []
        t = text.strip()
        # Try find a JSON array anywhere in the response
        try:
            # Attempt direct JSON parse
            cand = t
            # Sometimes model wraps reply in ```json ... ``` blocks
            if cand.startswith('```') and '```' in cand[3:]:
                cand = '\n'.join(cand.split('\n')[1:-1])
            # Find first '[' and last ']' to extract potential array
            first = cand.find('[')
            last = cand.rfind(']')
            if first != -1 and last != -1 and last > first:
                maybe = cand[first:last+1]
                try:
                    parsed = json.loads(maybe)
                    if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                        return [s.strip() for s in parsed]
                except Exception:
                    pass
        except Exception:
            pass

        # Fallback: split into lines and clean bullets/numbering
        lines = []
        for raw in t.splitlines():
            s = raw.strip()
            if not s:
                continue
            # Remove common bullet prefixes
            s = re.sub(r'^[-*\u2022\d\.\)\s]+', '', s).strip()
            if s:
                lines.append(s)

        # If the model returned a single paragraph, try splitting by ' || ' or ' / ' as last resort
        if not lines:
            parts = re.split(r'\s*\|\|\s*|\s*/\s*', t)
            lines = [p.strip() for p in parts if p.strip()]

        return lines

    def populate_result_table(self, table, entries, source):
        if table is None:
            return

        registry = self.result_table_registry.get(source)
        if registry is not None:
            registry.add(table)

        table.setProperty('result_source', source)
        self._configure_result_table_behavior(table, source)

        table.blockSignals(True)
        table.setRowCount(len(entries))

        for row, entry in enumerate(entries):
            history_id = entry.get('history_id') or entry.get('id')
            is_manual = bool(entry.get('manual'))

            original_text = entry.get('original_text', '')
            if is_manual:
                if original_text and original_text != 'Manual Input':
                    display_original = f"[Manual Input] {original_text}"
                else:
                    display_original = "[Manual Input]"
            else:
                display_original = original_text
            # Column 0: numbering
            number_item = QTableWidgetItem(str(row + 1))
            number_item.setTextAlignment(Qt.AlignTop | Qt.AlignCenter)
            table.setItem(row, 0, number_item)

            original_item = QTableWidgetItem(display_original)
            original_item.setTextAlignment(Qt.AlignTop | Qt.AlignLeft)
            if is_manual:
                original_item.setForeground(QColor('#ffd479'))
            table.setItem(row, 1, original_item)

            translated_text = entry.get('translated_text', '')
            model_label = entry.get('ai_model')
            if model_label:
                display_translated = f"{translated_text}\n(Model: {model_label})" if translated_text else f"(Model: {model_label})"
            else:
                display_translated = translated_text
            # show staged state visually
            translated_item = QTableWidgetItem(display_translated)
            translated_item.setTextAlignment(Qt.AlignTop | Qt.AlignLeft)
            if entry.get('staged'):
                translated_item.setBackground(QColor('#313a3c'))
                translated_item.setToolTip('Staged (awaiting Confirm)')
            if model_label:
                translated_item.setToolTip(f"Translated with: {model_label}")
            table.setItem(row, 2, translated_item)

            style_text = entry.get('translation_style', '') or ('Manual Input' if is_manual else '')
            style_item = QTableWidgetItem(style_text)
            style_item.setTextAlignment(Qt.AlignTop | Qt.AlignLeft)
            table.setItem(row, 3, style_item)

            # --- Kolom Actions ---
            action_widget = QWidget(table)
            action_layout = QGridLayout(action_widget)
            action_layout.setContentsMargins(2, 2, 2, 2)
            action_layout.setSpacing(4)

            # Tombol kecil biar muat
            def style_button(btn: QPushButton):
                btn.setFixedHeight(24)
                btn.setStyleSheet("""
                    QPushButton {
                        font-size: 11px;
                        padding: 2px 6px;
                    }
                """)
                return btn

            # Baris 1
            edit_button = style_button(QPushButton("Edit", action_widget))
            edit_button.clicked.connect(partial(self.open_result_editor, history_id, source))
            action_layout.addWidget(edit_button, 0, 0)

            confirm_button = style_button(QPushButton("Confirm", action_widget))
            confirm_button.clicked.connect(partial(self.confirm_result_entry, history_id, source))
            action_layout.addWidget(confirm_button, 0, 1)

            # Delete and reorder (Up/Down)
            delete_button = style_button(QPushButton("Delete", action_widget))
            delete_button.clicked.connect(partial(self.remove_result_entry, source, history_id))
            action_layout.addWidget(delete_button, 1, 0)

            up_button = style_button(QPushButton("Up", action_widget))
            up_button.clicked.connect(partial(self.move_result_entry, source, history_id, -1))
            action_layout.addWidget(up_button, 1, 1)

            down_button = style_button(QPushButton("Down", action_widget))
            down_button.clicked.connect(partial(self.move_result_entry, source, history_id, 1))
            action_layout.addWidget(down_button, 1, 2)

            # Baris 2 (hanya untuk history)
            if source == 'history':
                send_pf_button = style_button(QPushButton("To PF", action_widget))
                send_pf_button.clicked.connect(partial(self.send_history_entry_to_proofreader, history_id))
                action_layout.addWidget(send_pf_button, 2, 0)

                send_qc_button = style_button(QPushButton("To QC", action_widget))
                send_qc_button.clicked.connect(partial(self.send_history_entry_to_quality, history_id))
                action_layout.addWidget(send_qc_button, 2, 1)

            table.setCellWidget(row, 4, action_widget)

        table.blockSignals(False)
        table.resizeRowsToContents()

    def _configure_result_table_behavior(self, table, source):
        reorderable = source in ('proofreader', 'quality')
        if reorderable:
            table.setSelectionMode(QAbstractItemView.SingleSelection)
            table.setSelectionBehavior(QAbstractItemView.SelectRows)
            table.setDragEnabled(True)
            table.setAcceptDrops(True)
            table.viewport().setAcceptDrops(True)
            table.setDropIndicatorShown(True)
            table.setDragDropMode(QAbstractItemView.InternalMove)
            table.setDefaultDropAction(Qt.MoveAction)
            table.setDragDropOverwriteMode(False)
            if not table.property('rows_moved_handler_connected'):
                model = table.model()
                if model is not None:
                    try:
                        model.rowsMoved.connect(partial(self._handle_table_rows_moved, source, table))
                        table.setProperty('rows_moved_handler_connected', True)
                    except Exception:
                        pass
        else:
            table.setSelectionMode(QAbstractItemView.NoSelection)
            table.setSelectionBehavior(QAbstractItemView.SelectItems)
            table.setDragEnabled(False)
            table.setAcceptDrops(False)
            table.viewport().setAcceptDrops(False)
            table.setDropIndicatorShown(False)
            table.setDragDropMode(QAbstractItemView.NoDragDrop)
            table.setDefaultDropAction(Qt.IgnoreAction)

    def _handle_table_rows_moved(self, source, table, parent, start, end, dest_parent, dest_row):
        if source not in ('proofreader', 'quality'):
            return
        dataset = self.proofreader_entries if source == 'proofreader' else self.quality_entries
        if not dataset:
            return

        start = max(0, start)
        end = min(len(dataset) - 1, end)
        if start > end:
            return

        segment = dataset[start:end + 1]
        del dataset[start:end + 1]

        if dest_row > start:
            dest_row -= len(segment)
        dest_row = max(0, min(dest_row, len(dataset)))

        for offset, item in enumerate(segment):
            dataset.insert(dest_row + offset, item)

        self.refresh_history_views()

    def open_result_editor(self, history_id, source):
        entry = self._get_entry_by_source(source, history_id)
        if not entry:
            QMessageBox.warning(self, "Entry Missing", "Unable to find this entry. It may have been removed.")
            return

        styles = self.get_translation_styles()
        allow_original = (source == 'history')
        allow_style = (source == 'history')
        dialog = HistoryEditDialog(entry, styles, allow_original=allow_original, allow_style=allow_style, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            updated = dialog.get_result()
            entry['translated_text'] = updated.get('translated_text', entry.get('translated_text', ''))
            entry['timestamp'] = time.time()
            if source == 'history':
                entry['original_text'] = updated.get('original_text', entry.get('original_text', ''))
                entry['translation_style'] = updated.get('translation_style', entry.get('translation_style', ''))
            self.refresh_history_views()

    def confirm_result_entry(self, history_id, source, *, quiet=False):
        entry = self._get_entry_by_source(source, history_id)
        if not entry:
            if not quiet:
                QMessageBox.warning(self, "Entry Missing", "Unable to find this entry. It may have been removed.")
            return False

        ai_model_label = entry.get('ai_model')

        if source == 'history':
            success = self.apply_history_update(
                history_id,
                translated_text=entry.get('translated_text'),
                original_text=entry.get('original_text'),
                translation_style=entry.get('translation_style'),
                ai_model=ai_model_label
            )
        else:
            success = self.apply_history_update(
                history_id,
                translated_text=entry.get('translated_text'),
                ai_model=ai_model_label
            )

        if success:
            if source == 'proofreader':
                self.proofreader_entries = [e for e in self.proofreader_entries if (e.get('history_id') or e.get('id')) != history_id]
            elif source == 'quality':
                self.quality_entries = [e for e in self.quality_entries if (e.get('history_id') or e.get('id')) != history_id]
            self.refresh_history_views()
            if not quiet:
                self.statusBar().showMessage("Updated text applied.", 2500)
            return True
        else:
            if not quiet:
                QMessageBox.warning(self, "Apply Failed", "The original bubble could not be located.")
            return False

    def confirm_all_result_entries(self, source):
        source_key = (source or '').lower()
        if source_key not in ('proofreader', 'quality'):
            return

        dataset = self.proofreader_entries if source_key == 'proofreader' else self.quality_entries
        if not dataset:
            QMessageBox.information(self, "No Entries", "Tidak ada entry yang siap dikonfirmasi.")
            return

        failures = []
        history_ids = [(entry.get('history_id') or entry.get('id')) for entry in list(dataset)]
        for history_id in history_ids:
            if not history_id:
                continue
            if not self.confirm_result_entry(history_id, source_key, quiet=True):
                failures.append(history_id)

        if failures:
            QMessageBox.warning(self, "Sebagian Gagal", f"{len(failures)} entry gagal diterapkan. Periksa kembali data yang bermasalah.")
        else:
            self.statusBar().showMessage("Semua entry berhasil dikonfirmasi.", 3000)

    def send_history_entry_to_proofreader(self, history_id):
        self._stage_history_entry_for_review(history_id, 'proofreader')

    def send_history_entry_to_quality(self, history_id):
        self._stage_history_entry_for_review(history_id, 'quality')

    def _stage_history_entry_for_review(self, history_id, target):
        target = (target or '').lower()
        if target not in ('proofreader', 'quality'):
            return

        entry = self.get_history_entry(history_id)
        if not entry:
            QMessageBox.warning(self, "Entry Missing", "Unable to find this history entry. It may have been removed.")
            return

        record = {
            'history_id': history_id,
            'id': history_id,
            'original_text': entry.get('original_text', ''),
            'translated_text': entry.get('translated_text', ''),
            'translation_style': entry.get('translation_style', ''),
            'timestamp': time.time(),
        }
        if entry.get('manual'):
            record['manual'] = True
        if entry.get('manual_inpaint') is not None:
            record['manual_inpaint'] = bool(entry.get('manual_inpaint'))
        if entry.get('ai_model'):
            record['ai_model'] = entry.get('ai_model')
        if entry.get('staged'):
            record['staged'] = bool(entry.get('staged'))

        if target == 'proofreader':
            dest_list = self.proofreader_entries
            existing = self.get_proofreader_entry(history_id)
            tab_label = "Proofreader"
        else:
            dest_list = self.quality_entries
            existing = self.get_quality_entry(history_id)
            tab_label = "Quality Checker"

        if existing:
            staged_flag = existing.get('staged')
            existing.update(record)
            if staged_flag is not None:
                existing['staged'] = staged_flag
            try:
                dest_list.remove(existing)
            except ValueError:
                pass
            dest_list.insert(0, existing)
        else:
            dest_list.insert(0, record)

        self.refresh_history_views()
        self.statusBar().showMessage(f"Entry {history_id} dipindahkan ke {tab_label}.", 3000)

    def _process_single_review_request(self, history_id, mode):
        mode = (mode or '').lower()
        entry = self.get_history_entry(history_id)
        if not entry:
            QMessageBox.warning(self, "Entry Missing", "Unable to find this history entry. It may have been removed.")
            return

        provider, model_name = self.get_selected_model_name()
        if not model_name:
            QMessageBox.warning(self, "AI Model Missing", "Select an AI model before sending entries for review.")
            return

        provider_lower = (provider or '').lower()
        if provider_lower == 'gemini' and (not GEMINI_API_KEY or "your_gemini_key_here" in GEMINI_API_KEY):
            QMessageBox.warning(self, "Gemini Not Configured", "Add a valid Gemini API key before using this feature.")
            return
        if provider_lower == 'openai' and not getattr(self, 'is_openai_available', False):
            QMessageBox.warning(self, "OpenAI Not Configured", "Add a valid OpenAI API key before using this feature.")
            return

        prompt = self._build_review_prompt([entry], mode)
        if not prompt.strip():
            QMessageBox.information(self, "No Data", "There is no translation data to review for this entry.")
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if not self.check_and_increment_usage(provider, model_name):
                QMessageBox.information(self, "API Limit", "Rate limit reached for the selected model. Please wait a moment and try again.")
                return

            temperature = 0.35 if mode == 'proofreader' else 0.3
            response_text = self._invoke_ai_review(provider, model_name, prompt, temperature=temperature)
        finally:
            QApplication.restoreOverrideCursor()

        if not response_text:
            QMessageBox.warning(self, "Review Failed", "No response from AI.")
            return

        normalized = response_text.strip()
        if normalized.startswith('[') and any(token in normalized.upper() for token in ("ERROR", "NOT CONFIGURED", "FAILED")):
            QMessageBox.warning(self, "Review Failed", normalized)
            return

        # Prefer structured list responses (JSON array or one-per-line) so we don't rely on visible IDs.
        list_results = self._parse_ai_list_response(normalized, expected_count=1)
        if list_results:
            improved_text = list_results[0]
        else:
            suggestions = self._parse_review_response(normalized)
            improved_text = suggestions.get(history_id) or suggestions.get(entry.get('id')) or normalized
        improved_text = improved_text.strip()
        if not improved_text:
            QMessageBox.information(self, "No Suggestions", "The review did not return any updates.")
            return

        record = {
            'history_id': history_id,
            'id': history_id,
            'original_text': entry.get('original_text', ''),
            'translated_text': improved_text,
            'translation_style': entry.get('translation_style', ''),
            'timestamp': time.time(),
        }

        if mode == 'proofreader':
            existing = self.get_proofreader_entry(history_id)
            if existing:
                existing.update(record)
            else:
                self.proofreader_entries.append(record)
            target_tab = getattr(self, 'proofreader_tab_widget', None)
            tab_label = "Proofreader"
        else:
            existing = self.get_quality_entry(history_id)
            if existing:
                existing.update(record)
            else:
                self.quality_entries.append(record)
            target_tab = getattr(self, 'quality_tab_widget', None)
            tab_label = "Quality Checker"

        self.refresh_history_views()
        if target_tab is not None and hasattr(self, 'tabs'):
            self.tabs.setCurrentWidget(target_tab)
        self.statusBar().showMessage(f"{tab_label} processed entry {history_id}.", 4000)

    def _get_entry_by_source(self, source, history_id):
        if source == 'history':
            return self.get_history_entry(history_id)
        if source == 'proofreader':
            return self.get_proofreader_entry(history_id)
        if source == 'quality':
            return self.get_quality_entry(history_id)
        return None

    def remove_result_entry(self, source, history_id):
        """Remove a staged/result entry from proofreader or quality list."""
        if source == 'proofreader':
            self.proofreader_entries = [e for e in self.proofreader_entries if (e.get('history_id') or e.get('id')) != history_id]
        elif source == 'quality':
            self.quality_entries = [e for e in self.quality_entries if (e.get('history_id') or e.get('id')) != history_id]
        else:
            return
        self.refresh_history_views()

    def move_result_entry(self, source, history_id, delta):
        """Move an entry up or down within proofreader/quality lists by delta (-1 or +1)."""
        if source == 'proofreader':
            lst = self.proofreader_entries
        elif source == 'quality':
            lst = self.quality_entries
        else:
            return
        idx = next((i for i, e in enumerate(lst) if (e.get('history_id') or e.get('id')) == history_id), None)
        if idx is None:
            return
        new_idx = idx + delta
        if new_idx < 0 or new_idx >= len(lst):
            return
        lst[idx], lst[new_idx] = lst[new_idx], lst[idx]
        self.refresh_history_views()

    def _show_result_modal(self, source, title):
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(720, 480)
        modal_layout = QVBoxLayout(dialog)
        table = self._create_result_table()
        table.setProperty('result_limit', None)
        modal_layout.addWidget(table)

        if source == 'history':
            entries = self._get_recent_entries(self.history_entries, limit=None)
        elif source == 'proofreader':
            entries = list(self.proofreader_entries)
        else:
            entries = list(self.quality_entries)

        self.populate_result_table(table, entries, source)

        close_box = QDialogButtonBox(QDialogButtonBox.Close)
        close_box.rejected.connect(dialog.reject)
        modal_layout.addWidget(close_box)
        dialog.exec_()

    def show_history_modal(self):
        self._show_result_modal('history', 'History (All Entries)')

    def show_proofreader_modal(self):
        self._show_result_modal('proofreader', 'Proofreader Results (All Entries)')

    def show_quality_modal(self):
        self._show_result_modal('quality', 'Quality Checker Results (All Entries)')



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

        self.style_combo = self._create_combo_box(ai_layout, "Translation Style:", self.translation_styles, 1, 0, 1, 2)
        # Small controls to add/remove custom styles
        styles_controls = QWidget()
        sc_layout = QHBoxLayout(styles_controls)
        sc_layout.setContentsMargins(0, 6, 0, 0)
        self.style_input = QLineEdit(self)
        self.style_input.setPlaceholderText('Type custom style and click Add')
        add_style_btn = QPushButton('Add')
        add_style_btn.clicked.connect(lambda: (self.add_custom_style(self.style_input.text()) and self.style_input.clear()))
        remove_style_btn = QPushButton('Remove Selected')
        remove_style_btn.clicked.connect(lambda: (self.remove_selected_style()))
        sc_layout.addWidget(self.style_input)
        sc_layout.addWidget(add_style_btn)
        sc_layout.addWidget(remove_style_btn)
        layout.addWidget(ai_group)
        layout.addWidget(styles_controls)

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
        self._shortcut_callbacks = {
            'undo': self.undo_last_action,
            'redo': self.redo_last_action,
            'next': self.on_next_clicked,
            'prev': self.load_prev_image,
        }
        for idx in range(len(SELECTION_MODE_LABELS)):
            self._shortcut_callbacks[f'selection_mode_{idx}'] = partial(self.set_selection_mode_by_index, idx)
        self.reload_shortcuts()

    def on_next_clicked(self):
        """Called when user clicks Next; navigate to the next image without forcing an auto-save."""
        self.load_next_image()

    def dispatch_mouse_shortcut(self, event_type: str, button: Qt.MouseButton):
        """Dispatch a mouse shortcut if configured. event_type in {'press','release','double'}.
        Returns True if a shortcut matched and was called."""
        try:
            key = (event_type, button)
            cb = self._mouse_shortcuts.get(key)
            if cb:
                try:
                    cb()
                except TypeError:
                    # Some callbacks may expect an argument; ignore and call without
                    try:
                        cb
                    except Exception:
                        pass
                return True
        except Exception:
            pass
        return False

    def _build_shortcut_map(self):
        merged = {}
        user_map = SETTINGS.get('shortcuts', {}) or {}
        for key, default in DEFAULT_SHORTCUTS.items():
            if key in user_map:
                merged[key] = user_map.get(key) or ''
            else:
                merged[key] = default
        for key, value in user_map.items():
            if key not in merged:
                merged[key] = value or ''
        return merged

    def reload_shortcuts(self):
        self.shortcut_sequences = self._build_shortcut_map()

        # Update action-based shortcuts
        for key, action in (self._action_shortcut_map or {}).items():
            self._apply_action_shortcut(action, self.shortcut_sequences.get(key, ''))

        # Dispose old QShortcut instances
        for shortcut in self._active_shortcuts.values():
            try:
                shortcut.activated.disconnect()
            except Exception:
                pass
            shortcut.setParent(None)
            shortcut.deleteLater()
        self._active_shortcuts.clear()

        # Recreate shortcuts from definitions
        for key, callback in (self._shortcut_callbacks or {}).items():
            sequence = self.shortcut_sequences.get(key, '')
            if not sequence:
                continue
            try:
                qshortcut = QShortcut(QKeySequence(sequence), self)
                qshortcut.activated.connect(callback)
                self._active_shortcuts[key] = qshortcut
            except Exception:
                print(f"Failed to bind shortcut '{sequence}' for {key}", file=sys.stderr)
        # Also parse mouse-based shortcuts from sequences using a prefix like 'MOUSE:press:Left'
        # Supported formats: MOUSE:press:Left, MOUSE:release:Right, MOUSE:double:Left
        self._mouse_shortcuts.clear()
        for key, callback in (self._shortcut_callbacks or {}).items():
            seq = (self.shortcut_sequences.get(key, '') or '').strip()
            if not seq or not seq.upper().startswith('MOUSE:'):
                continue
            parts = seq.split(':')
            if len(parts) >= 3:
                evt = parts[1].lower()
                btn_name = parts[2].lower()
                btn = None
                if btn_name in ('left', 'l'):
                    btn = Qt.LeftButton
                elif btn_name in ('right', 'r'):
                    btn = Qt.RightButton
                elif btn_name in ('middle', 'm'):
                    btn = Qt.MiddleButton
                if btn is not None:
                    self._mouse_shortcuts[(evt, btn)] = callback

    def _apply_action_shortcut(self, action: QAction, sequence: str):
        if action is None:
            return
        try:
            action.setShortcut(QKeySequence(sequence) if sequence else QKeySequence())
        except Exception:
            action.setShortcut(QKeySequence())

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
                # If Manga-OCR is available, prefer it for Japanese (it handles manga text far better)
                if name == 'Japanese' and MangaOcr:
                    # skip adding PaddleOCR's Japanese entry to avoid selecting a poorer OCR for manga
                    continue
                if key not in self.OCR_LANGS: # Hindari duplikat jika sudah ada dari engine lain
                    self.OCR_LANGS[key] = {'code': code, 'engine': 'PaddleOCR'}
        
        # EasyOCR
        easyocr_langs = {'Afrikaans': 'af', 'Arabic': 'ar', 'Azerbaijani': 'az', 'Belarusian': 'be', 'Bulgarian': 'bg', 'Bengali': 'bn', 'Bosnian': 'bs', 'Czech': 'cs', 'Chinese (Simplified)': 'ch_sim', 'Chinese (Traditional)': 'ch_tra', 'German': 'de', 'English': 'en', 'Spanish': 'es', 'Estonian': 'et', 'French': 'fr', 'Hindi': 'hi', 'Croatian': 'hr', 'Hungarian': 'hu', 'Indonesian': 'id', 'Italian': 'it', 'Japanese': 'ja', 'Korean': 'ko', 'Lithuanian': 'lt', 'Latvian': 'lv', 'Malay': 'ms', 'Dutch': 'nl', 'Polish': 'pl', 'Portuguese': 'pt', 'Romanian': 'ro', 'Russian': 'ru', 'Slovak': 'sk', 'Slovenian': 'sl', 'Albanian': 'sq', 'Swedish': 'sv', 'Thai': 'th', 'Turkish': 'tr', 'Ukrainian': 'uk', 'Urdu': 'ur', 'Uzbek': 'uz', 'Vietnamese': 'vi'}
        for name, code in easyocr_langs.items():
            key = f'{name} (EasyOCR)'
            # Prefer Manga-OCR for Japanese manga text if available
            if name == 'Japanese' and MangaOcr:
                continue
            if key not in self.OCR_LANGS:
                self.OCR_LANGS[key] = {'code': code, 'engine': 'EasyOCR'}

        # Tesseract
        try:
            langs = [lang for lang in pytesseract.get_languages(config='--oem 1') if len(lang) == 3 and lang != 'osd']
            tess_langs = {lang.capitalize(): lang for lang in sorted(langs)}
            for name, code in tess_langs.items():
                key = f'{name} (Tesseract)'
                # If Manga-OCR exists, avoid offering Tesseract as the Japanese default
                if name.lower().startswith('jap') and MangaOcr:
                    continue
                if key not in self.OCR_LANGS:
                    self.OCR_LANGS[key] = {'code': code, 'engine': 'Tesseract'}
        except Exception as e:
            print(f"Could not get Tesseract languages: {e}")
            tess_fallback = {'English (Tesseract)': {'code': 'eng', 'engine': 'Tesseract'}, 'Japanese (Tesseract)': {'code': 'jpn', 'engine': 'Tesseract'}}
            for k,v in tess_fallback.items():
                if k not in self.OCR_LANGS: self.OCR_LANGS[k] = v
        
        # AI OCR (GPT-based via AI Translate)
        self.OCR_LANGS['AI OCR (GPT-based via AI Translate)'] = {
            'code': 'auto',
            'engine': 'MOFRL-GPT'
        }

        # Populate ComboBox
        self.ocr_lang_combo.blockSignals(True)
        self.ocr_lang_combo.clear()
        # Append AI OCR entries (active models only)
        for ai_entry in self._get_ai_ocr_entries():
            self.OCR_LANGS[ai_entry['display']] = ai_entry['data']

        for display_name, data in sorted(self.OCR_LANGS.items()):
            self.ocr_lang_combo.addItem(display_name, data)
        self.ocr_lang_combo.blockSignals(False)

        # Set default to Japanese
        jp_index = self.ocr_lang_combo.findText("Japanese (Manga-OCR)")
        if jp_index != -1:
            self.ocr_lang_combo.setCurrentIndex(jp_index)

        self.on_ocr_lang_changed(self.ocr_lang_combo.currentIndex())

    def _get_ai_ocr_entries(self):
        entries = []
        ocr_config = SETTINGS.get('ocr', {}) or {}
        provider_labels = getattr(APIManagerDialog, 'OCR_PROVIDERS', {}) if 'APIManagerDialog' in globals() else {}
        for provider_key, cfg in ocr_config.items():
            if not isinstance(cfg, dict):
                continue
            models = cfg.get('models')
            if not isinstance(models, list):
                continue
            provider_label = provider_labels.get(provider_key, provider_key.title())
            for model in models:
                if not isinstance(model, dict):
                    continue
                if not model.get('active'):
                    continue
                model_id = (model.get('id') or '').strip()
                if not model_id:
                    continue
                model_name = model.get('name', '').strip() or model_id
                display = f"AI OCR ({provider_label}: {model_name})"
                entries.append({
                    'display': display,
                    'data': {
                        'engine': 'AI_OCR',
                        'code': 'ai',
                        'provider': provider_key,
                        'provider_label': provider_label,
                        'model_id': model_id,
                        'model_name': model_name,
                    }
                })
        return entries

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
        self._load_openrouter_models()
        self.ai_model_combo.blockSignals(True)
        self.ai_model_combo.clear()
        for provider, models in self.AI_PROVIDERS.items():
            for model_key, model_info in models.items():
                if provider == 'OpenRouter' and not model_info.get('active', True):
                    continue
                display_text = f"[{provider}] {model_info.get('display', model_key)}"
                index = self.ai_model_combo.count()
                self.ai_model_combo.addItem(display_text)
                self.ai_model_combo.setItemData(index, (provider, model_key), Qt.UserRole)
                self.ai_model_combo.setItemData(index, model_info, Qt.UserRole + 1)
                description = model_info.get('description')
                if description:
                    self.ai_model_combo.setItemData(index, description, Qt.ToolTipRole)
        self.ai_model_combo.blockSignals(False)
        if self.ai_model_combo.count() > 0 and self.ai_model_combo.currentIndex() < 0:
            self.ai_model_combo.setCurrentIndex(0)

    def _load_openrouter_models(self):
        translate_cfg = SETTINGS.get('translate', {})
        openrouter_cfg = translate_cfg.get('openrouter', {}) or {}
        models = openrouter_cfg.get('models') or []
        provider_dict = self.AI_PROVIDERS.setdefault('OpenRouter', {})
        provider_dict.clear()
        for model in models:
            if not isinstance(model, dict):
                continue
            model_id = (model.get('id') or '').strip()
            if not model_id:
                continue
            name = (model.get('name') or model_id).strip()
            description = (model.get('description') or '').strip()
            provider_dict[model_id] = {
                'display': f"{name}",
                'pricing': {
                    'input': 0.0,
                    'output': 0.0
                },
                'limits': {
                    'rpm': 300,
                    'rpd': 20000
                },
                'active': bool(model.get('active', True)),
                'description': description,
                'id': model_id,
                'name': name
            }

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
                # EasyOCR expects a list of languages; include English as fallback
                lang_list = sorted(list({l for l in ('en', lang_code) if l}))
                self.easyocr_reader = easyocr.Reader(lang_list, gpu=use_gpu)
                self.easyocr_lang = lang_code
            
            # Inisialisasi PaddleOCR: try multiple constructor signatures to support different versions
            elif engine == 'PaddleOCR' and (self.paddle_ocr_reader is None or self.paddle_lang != lang_code):
                try:
                    from paddleocr import PaddleOCR

                    use_gpu_flag = bool(use_gpu and self.is_gpu_available)
                    # Prefer use_textline_orientation (newer API). Try multiple signatures to be robust.
                    try:
                        # newest variants
                        self.paddle_ocr_reader = PaddleOCR(lang=lang_code, use_textline_orientation=True, use_gpu=use_gpu_flag)
                        self.paddle_lang = lang_code
                        print(f"PaddleOCR initialized for {lang_code} (use_textline_orientation) on {'GPU' if use_gpu_flag else 'CPU'}")
                    except TypeError as te1:
                        # try deprecated/alternate arg names
                        try:
                            self.paddle_ocr_reader = PaddleOCR(lang=lang_code, use_angle_cls=True)
                            self.paddle_lang = lang_code
                            print(f"PaddleOCR initialized for {lang_code} (use_angle_cls fallback)")
                        except TypeError as te2:
                            # try minimal constructor
                            self.paddle_ocr_reader = PaddleOCR(lang=lang_code)
                            self.paddle_lang = lang_code
                            print(f"PaddleOCR initialized for {lang_code} (minimal constructor)")

                except Exception as e:
                    print(f"Error initializing PaddleOCR: {e}")
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

    def add_api_cost(self, input_tokens, output_tokens, provider, model_name):
        """
        Hitung biaya API berdasarkan jumlah token input/output.
        Update juga info token real-time & akumulasi.
        """
        provider_models = self.AI_PROVIDERS.get(provider, {})
        model_info = provider_models.get(model_name, {})
        pricing = model_info.get('pricing', {'input': 0.0, 'output': 0.0})
        # Hitung biaya total (USD)
        cost = (input_tokens * pricing['input']) + (output_tokens * pricing['output'])
        self.total_cost += cost

        # ?? Update akumulasi token
        if not hasattr(self, "total_input_tokens"):
            self.total_input_tokens = 0
        if not hasattr(self, "total_output_tokens"):
            self.total_output_tokens = 0
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        # ?? Update status detail
        self.provider_label.setText(f"Provider: {provider}")
        model_display = model_info.get('display') or model_info.get('name') or model_name
        self.model_label.setText(f"Model: {model_display}")
        self.input_tokens_label.setText(
            f"Input Tokens: {input_tokens:,} (Total: {self.total_input_tokens:,})"
        )
        self.output_tokens_label.setText(
            f"Output Tokens: {output_tokens:,} (Total: {self.total_output_tokens:,})"
        )
        self.rate_label_input.setText(f"Rate Input: ${pricing['input']:.9f} / token")
        self.rate_label_output.setText(f"Rate Output: ${pricing['output']:.9f} / token")

        # Update tampilan cost
        self.update_cost_display()
        # Simpan ke file/log
        self.save_usage_data()


    def update_cost_display(self):
        """
        Update tampilan biaya (USD & IDR).
        """
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
            # Enable/disable per-language orientation controls depending on selected OCR language/engine
            en_combo = getattr(self, 'en_orientation_combo', None)
            jp_combo = getattr(self, 'jp_orientation_combo', None)
            if en_combo is not None and jp_combo is not None:
                engine = (lang_data.get('engine') or '').lower()
                code = (lang_data.get('code') or '').lower()
                # By default allow both
                enable_en = True
                enable_jp = True
                # If engine strongly indicates Japanese (Manga-OCR) or code is 'ja', disable EN
                if 'manga' in engine or code.startswith('ja'):
                    enable_en = False
                    enable_jp = True
                # If engine is EasyOCR/Tesseract and language is English, disable JP
                elif 'easyocr' in engine or 'tesseract' in engine or code.startswith('en'):
                    enable_en = True
                    enable_jp = False
                en_combo.setEnabled(enable_en)
                jp_combo.setEnabled(enable_jp)

    def _on_lang_orientation_changed(self, lang_code, value):
        """Instance handler to persist per-language orientation overrides into SETTINGS."""
        try:
            lang_map = SETTINGS.setdefault('lang_orientation', {})
            lang_map[lang_code] = value
            save_settings(SETTINGS)
        except Exception as e:
            print(f"Failed to save lang orientation: {e}")

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
        elif provider == 'OpenRouter':
            model_info = self.AI_PROVIDERS.get('OpenRouter', {}).get(model_name, {})
            return self.translate_with_openrouter(text_to_translate, target_lang, model_name, settings, model_info, is_enhanced, ocr_results)
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

            # ? Tambah config untuk batasi output tokens + longgarkan safety_settings
            response = model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": 500012,  # batas aman
                    "temperature": settings.get("temperature", 0.5) if isinstance(settings, dict) else 0.5
                },
                safety_settings=[
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                ]
            )

            if response.parts:
                self.add_api_cost(len(prompt), len(response.text), 'Gemini', model_name)

                # ? Update counter
                self.translated_count += 1
                if hasattr(self, "translated_label"):
                    self.translated_label.setText(f"Translated Snippets: {self.translated_count}")

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
        - Gunakan caching untuk system prompt (biar hemat).
        - OCR text user tidak dicache karena selalu berbeda.
        """

        # ---------- Helper: sanitizer output ----------
        def _sanitize_output(s: str) -> str:
            if not s:
                return s
            s = s.strip()
            import re
            fence_match = re.fullmatch(r"```[a-zA-Z0-9_-]*\n([\s\S]*?)\n```", s)
            if fence_match:
                s = fence_match.group(1).strip()
            return s

        # ---------- Guard ----------
        if not text_to_translate or not text_to_translate.strip():
            return ""
        if not getattr(self, "is_openai_available", False):
            return "[OPENAI NOT CONFIGURED]"

        try:
            # --- Build prompts ---
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
                "- If OCR contains sound effects (e.g., '????', '???'), translate to natural equivalents or expressive onomatopoeia.\n"
                "- Do NOT add translator notes.\n"
            )

            if is_enhanced and ocr_results:
                system_prompt = (
                    f"You are an expert manga translator.\n"
                    f"1. Automatically detect the language of the text.\n"
                    f"2. If Japanese ? merge and correct the following OCR outputs, then translate into natural {target_lang}.\n"
                    f"3. If already in {target_lang} ? return as-is with no changes.\n"
                    f"4. If in another language ? translate into {target_lang}.\n"
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
                    f"2. If Japanese ? silently correct OCR mistakes, then translate into natural {target_lang}.\n"
                    f"3. If already in {target_lang} ? return as-is with no changes.\n"
                    f"4. If in another language ? translate into {target_lang}.\n"
                    f"{style_rules} {prompt_enhancements} {base_rule}"
                )
                user_prompt = f"Raw OCR Text:\n{text_to_translate}"

            # --- Build request ---
            model_lower = (model_name or "").lower()
            supports_temperature = not (
                model_lower.startswith("gpt-5-mini") or model_lower.startswith("gpt-5-nano")
            )
            desired_temp = settings.get("temperature", 0.5) if isinstance(settings, dict) else 0.5

            req_kwargs = {
                "model": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt,
                        # ?? simpan system prompt di cache biar hemat
                        "cache_control": {"type": "ephemeral"}
                    },
                    {"role": "user", "content": user_prompt},  # OCR text ? tidak dicache
                ],
            }
            if supports_temperature and desired_temp is not None:
                req_kwargs["temperature"] = float(desired_temp)

            # --- Call API ---
            client = getattr(self, "openai_client", None)
            if client is None:
                client = openai_client

            response = client.chat.completions.create(**req_kwargs)
            output_text = (response.choices[0].message.content or "").strip()
            output_text = _sanitize_output(output_text)

            # --- Hitung biaya dengan token usage dari API ---
            if hasattr(response, "usage"):
                in_tokens = response.usage.prompt_tokens
                out_tokens = response.usage.completion_tokens
                if hasattr(self, "add_api_cost"):
                    self.add_api_cost(in_tokens, out_tokens, "OpenAI", model_name)

            # ? Update counter
            self.translated_count += 1
            if hasattr(self, "translated_label"):
                self.translated_label.setText(f"Translated Snippets: {self.translated_count}")

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

    def translate_with_openrouter(
        self,
        text_to_translate: str,
        target_lang: str,
        model_id: str,
        settings: dict | None = None,
        model_info: dict | None = None,
        is_enhanced: bool = False,
        ocr_results: dict | None = None
    ):
        if not text_to_translate.strip():
            return ""

        provider_cfg = get_translate_provider_settings('openrouter')
        api_key = provider_cfg.get('api_key', '').strip()
        if not api_key:
            return "[OPENROUTER API KEY NOT CONFIGURED]"

        url = provider_cfg.get('url', '').strip() or "https://openrouter.ai/api/v1/chat/completions"

        # --- Dynamic prompt ---
        mode = settings.get('mode') if settings else None
        if mode == 'info':
            system_prompt = (
                f"You are an expert manga translator. Translate the user's text into clear, natural {target_lang} for narration or informational text. "
                f"Keep it smooth, neutral, and suitable for manga narration boxes.\n"
                f"IMPORTANT RULES:\n"
                f"- Avoid slang or overly casual tone.\n"
                f"- Only output the final translation in {target_lang}.\n"
                f"- No markdown, notes, or extra explanation.\n"
                f"- Preserve line breaks.\n"
            )
        else:
            system_prompt = (
                f"You are an expert manga translator. Translate the user's text into natural, fluent {target_lang} suitable for published manga dialogue. "
                f"Keep the meaning, tone, and nuances from the original text.\n"
                f"IMPORTANT RULES:\n"
                f"- Use natural and neutral tone — not overly formal, but avoid slang or street language like 'lo', 'gue', or 'nih'.\n"
                f"- Output ONLY the final translation in {target_lang}.\n"
                f"- No quotes, markdown, or explanations.\n"
                f"- Preserve line breaks.\n"
            )

        if settings and settings.get('translation_style'):
            system_prompt += f" Use the style: {settings['translation_style']}."

        messages = [{"role": "system", "content": system_prompt}]
        user_content = ""
        if is_enhanced and isinstance(ocr_results, dict):
            user_content = "\n\n".join(filter(None, [
                ocr_results.get('manga_ocr', ''),
                ocr_results.get('tesseract', '')
            ])).strip() or text_to_translate
        else:
            user_content = text_to_translate
        messages.append({"role": "user", "content": user_content})

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": float(model_info.get('temperature', 0.7)) if model_info else 0.7,
            "max_tokens": int(model_info.get('max_tokens', 1024)) if model_info else 1024,
        }

        pr_timeout = int(provider_cfg.get('timeout', 60) or 60)
        pr_retries = int(provider_cfg.get('retries', 3) or 3)
        pr_backoff = float(provider_cfg.get('backoff', 1.5) or 1.5)

        try:
            response = robust_post(url, headers=headers, json_payload=payload,
                                timeout=pr_timeout, max_retries=pr_retries, backoff_factor=pr_backoff)
            data = response.json()
        except Exception as exc:
            return f"[OPENROUTER REQUEST ERROR: {exc}]"

        choices = data.get('choices')
        output_text = ""
        if isinstance(choices, list) and choices:
            msg = choices[0].get('message', {})
            content = msg.get('content')
            if isinstance(content, list):
                output_text = "".join(part.get('text', '') for part in content if isinstance(part, dict))
            elif isinstance(content, str):
                output_text = content

        if not output_text:
            if 'error' in data:
                return f"[OPENROUTER ERROR: {data['error'].get('message', 'Unknown error')}]"
            logger.warning(f"OpenRouter returned empty response: {response.text}")
            return "[OPENROUTER ERROR: Empty response]"

        usage = data.get('usage') or {}
        self.add_api_cost(usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0), 'OpenRouter', model_id)
        return output_text.strip()

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
        # Basic orientation detection remains; preprocessing pipeline optional
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if h == 0 or w == 0:
            return cv_image, 0
        angle = 0
        if orientation_hint == "Auto-Detect":
            try:
                coords = cv2.findNonZero(cv2.bitwise_not(gray))
                if coords is not None:
                    rect = cv2.minAreaRect(coords)
                    angle = rect[-1]
                    if w < h and angle < -45:
                        angle = -(90 + angle)
                    elif w > h and angle > 45:
                        angle = 90 - angle
                    else:
                        angle = -angle
            except cv2.error:
                angle = 0
        elif orientation_hint == "Vertical":
            if w > h:
                angle = 90

        # Rotate grayscale for subsequent preprocessing
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # Preprocessing: histogram equalization, Gaussian blur, Otsu threshold
        try:
            equalized = cv2.equalizeHist(rotated_gray)
            blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
            _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        except cv2.error:
            # Fall back to original rotated gray if any op fails
            otsu = rotated_gray

        # Return BGR image expected by OCR engines
        processed_bgr = cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)
        return processed_bgr, angle
    
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
        self.ui_update_queue.append((image_path, new_area, original_text, translated_text))
        self.ui_update_mutex.unlock()

        current_key = self.get_current_data_key()
        if image_path == current_key and not self.ui_update_timer.isActive():
            self.ui_update_timer.start(100)  # Coalesce updates within 100ms

    def process_ui_updates(self):
        if self.is_processing_ui_updates:
            return

        self.is_processing_ui_updates = True
        try:
            self.ui_update_mutex.lock()
            if not self.ui_update_queue:
                self.ui_update_mutex.unlock()
                return

            current_key = self.get_current_data_key()
            relevant_updates = [
                (path, area, original, translated)
                for path, area, original, translated in self.ui_update_queue
                if path == current_key
            ]

            updates_by_image = {}
            for image_path, new_area, original_text, translated_text in self.ui_update_queue:
                updates_by_image.setdefault(image_path, []).append((new_area, original_text, translated_text))

            self.ui_update_queue.clear()
            self.ui_update_mutex.unlock()

            for image_path, entries in updates_by_image.items():
                image_record = self.all_typeset_data.setdefault(image_path, {'areas': [], 'redo': []})
                new_areas = [area for area, _original, _translated in entries]
                image_record['areas'].extend(new_areas)
                image_record['redo'].clear()

                for area, original_text, translated_text in entries:
                    self.register_history_entry(image_path, area, original_text, translated_text)

            if relevant_updates:
                self.typeset_areas = self.all_typeset_data.get(current_key, {'areas': []})['areas']
                self.redo_stack = self.all_typeset_data.get(current_key, {'redo': []})['redo']
                self.redraw_all_typeset_areas()
                newest_area = relevant_updates[-1][1]
                self.set_selected_area(newest_area, notify=True)
                self.update_undo_redo_buttons_state()

            if updates_by_image:
                self.refresh_history_views()

        finally:
            self.is_processing_ui_updates = False
            self.ui_update_mutex.lock()
            needs_another_run = bool(self.ui_update_queue)
            self.ui_update_mutex.unlock()
            if needs_another_run:
                self.ui_update_timer.start(100)

    def generate_history_id(self):
        self.history_counter += 1
        return f"H{self.history_counter:05d}"

    def get_history_entry(self, history_id):
        for entry in self.history_entries:
            if entry['id'] == history_id:
                return entry
        return None

    def get_proofreader_entry(self, history_id):
        for entry in self.proofreader_entries:
            if entry.get('history_id') == history_id:
                return entry
        return None

    def get_quality_entry(self, history_id):
        for entry in self.quality_entries:
            if entry.get('history_id') == history_id:
                return entry
        return None

    def get_translation_styles(self):
        return list(self.translation_styles)

    def load_translation_styles_from_disk(self):
        try:
            if os.path.exists(self._styles_storage_path):
                with open(self._styles_storage_path, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                    if isinstance(data, list):
                        # merge unique while preserving built-ins first
                        new_styles = []
                        for s in data:
                            if s and s not in self.translation_styles:
                                self.translation_styles.append(s)
                                new_styles.append(s)

                        # If the style combo exists (UI already created), add loaded styles to it
                        try:
                            if getattr(self, 'style_combo', None) and new_styles:
                                for s in new_styles:
                                    try:
                                        self.style_combo.addItem(s)
                                    except Exception:
                                        pass
                        except Exception:
                            pass
        except Exception:
            # ignore load failures
            pass

    def save_translation_styles_to_disk(self):
        try:
            # ensure dir
            d = os.path.dirname(self._styles_storage_path)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            with open(self._styles_storage_path, 'w', encoding='utf-8') as fh:
                json.dump([s for s in self.translation_styles if s], fh, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def add_custom_style(self, style_text: str):
        style_text = (style_text or '').strip()
        if not style_text:
            return False
        if style_text in self.translation_styles:
            return False
        self.translation_styles.append(style_text)
        # update combo if exists
        try:
            if getattr(self, 'style_combo', None):
                self.style_combo.addItem(style_text)
        except Exception:
            pass
        self.save_translation_styles_to_disk()
        return True

    def remove_selected_style(self):
        try:
            if not getattr(self, 'style_combo', None):
                return False
            sel = self.style_combo.currentText()
            if not sel:
                return False
            # protect the default core styles (first 5)
            if sel in ["Santai (Default)", "Formal (Ke Atasan)", "Akrab (Ke Teman/Pacar)", "Vulgar/Dewasa (Adegan Seks)", "Sesuai Konteks Manga"]:
                return False
            # remove
            if sel in self.translation_styles:
                self.translation_styles.remove(sel)
            index = self.style_combo.currentIndex()
            self.style_combo.removeItem(index)
            self.save_translation_styles_to_disk()
            return True
        except Exception:
            return False

    def _create_typeset_area(self, rect, text, settings, polygon=None, original_text="", translation_style=None, manual_inpaint=None, is_manual=False):
        area = TypesetArea(
            rect,
            text,
            settings['font'],
            settings['color'],
            polygon=polygon,
            orientation=settings.get('orientation_mode', 'horizontal'),
            effect=settings.get('text_effect', 'none'),
            effect_intensity=settings.get('effect_intensity', 20.0),
            bezier_points=settings.get('bezier_points'),
            bubble_enabled=settings.get('create_bubble', False),
            text_outline=settings.get('text_outline', False),
            text_outline_width=settings.get('outline_width', self.typeset_outline_width),
            text_outline_color=settings.get('outline_color', self.typeset_outline_color.name() if isinstance(self.typeset_outline_color, QColor) else '#000000'),
            alignment=settings.get('alignment', 'center'),
            line_spacing=settings.get('line_spacing', 1.1),
            char_spacing=settings.get('char_spacing', 100.0),
            margins=settings.get('margins', {'top': 0, 'right': 0, 'bottom': 0, 'left': 0}),
            original_text=original_text,
            translation_style=translation_style if translation_style is not None else settings.get('translation_style', '')
        )
        cleanup_defaults = SETTINGS.get('cleanup', {})
        use_inpaint_value = bool(manual_inpaint) if manual_inpaint is not None else bool(settings.get('use_inpaint', cleanup_defaults.get('use_inpaint', True)))
        use_background_box_value = bool(settings.get('use_background_box', cleanup_defaults.get('use_background_box', True)))
        area.set_override('use_inpaint', use_inpaint_value)
        area.set_override('use_background_box', use_background_box_value)
        notes = area.review_notes if isinstance(area.review_notes, dict) else {}
        area.review_notes = notes
        if is_manual:
            area.review_notes['manual'] = True
        if manual_inpaint is not None:
            area.review_notes['manual_inpaint'] = bool(manual_inpaint)
        area.ensure_defaults()
        return area
    def rebuild_history_for_image(self, image_key, areas):
        if not image_key or not areas:
            return
        for area in areas:
            self.register_history_entry(image_key, area, getattr(area, 'original_text', ''), getattr(area, 'text', ''))

    def register_history_entry(self, image_key, area, original_text, translated_text):
        if not getattr(area, 'history_id', None):
            area.history_id = self.generate_history_id()
        history_id = area.history_id

        if original_text is not None:
            area.original_text = original_text
        if translated_text is not None:
            preserve_segments = False
            try:
                segments = area.get_segments()
                if segments:
                    existing_plain = area._segments_to_plain_text(segments)
                    preserve_segments = (existing_plain == translated_text)
            except Exception:
                preserve_segments = False
            if preserve_segments:
                area.text = translated_text or ''
            else:
                area.update_plain_text(translated_text)

        entry = self.get_history_entry(history_id)
        notes = area.review_notes if isinstance(getattr(area, 'review_notes', {}), dict) else {}
        if not isinstance(notes, dict):
            notes = {}
            area.review_notes = notes
        manual_flag = bool(notes.get('manual'))
        manual_inpaint = notes.get('manual_inpaint')
        model_label = notes.get('ai_model')
        record = {
            'id': history_id,
            'history_id': history_id,
            'image_key': image_key,
            'original_text': area.original_text or '',
            'translated_text': translated_text if translated_text is not None else area.text or '',
            'translation_style': getattr(area, 'translation_style', ''),
            'timestamp': time.time(),
        }
        if manual_flag:
            record['manual'] = True
            if not record['original_text']:
                record['original_text'] = 'Manual Input'
        if manual_inpaint is not None:
            record['manual_inpaint'] = bool(manual_inpaint)
        if model_label:
            record['ai_model'] = model_label

        if entry:
            entry.update(record)
        else:
            self.history_entries.append(record)

        self.history_lookup[history_id] = {'image_key': image_key, 'area': area}
        return record

    def apply_history_update(self, history_id, *, translated_text=None, original_text=None, translation_style=None, ai_model=None):
        entry = self.get_history_entry(history_id)
        if not entry:
            return False

        if original_text is not None:
            entry['original_text'] = original_text
        if translated_text is not None:
            entry['translated_text'] = translated_text
        if translation_style is not None:
            entry['translation_style'] = translation_style
        if ai_model is not None:
            entry['ai_model'] = ai_model
        entry['timestamp'] = time.time()

        lookup = self.history_lookup.get(history_id)
        if not lookup:
            return False

        area = lookup.get('area')
        if not area:
            return False

        if original_text is not None:
            area.original_text = original_text
        if translation_style is not None:
            area.translation_style = translation_style
        if translated_text is not None:
            area.update_plain_text(translated_text)
        if ai_model is not None:
            notes = area.review_notes if isinstance(getattr(area, 'review_notes', {}), dict) else {}
            if not isinstance(notes, dict):
                notes = {}
            notes['ai_model'] = ai_model
            area.review_notes = notes

        image_key = lookup.get('image_key')
        image_record = self.all_typeset_data.get(image_key)
        if image_record:
            image_record.setdefault('redo', []).clear()

        if image_key == self.get_current_data_key():
            self.redo_stack.clear()
            self.redraw_all_typeset_areas()
            self.update_undo_redo_buttons_state()

        self.refresh_history_views()
        return True

    def reset_history_state(self):
        self.history_entries.clear()
        self.proofreader_entries.clear()
        self.quality_entries.clear()
        self.history_lookup.clear()
        self.history_counter = 0
        self.refresh_history_views()

    def refresh_history_views(self):
        sources = [
            ('history', self.history_entries),
            ('proofreader', self.proofreader_entries),
            ('quality', self.quality_entries),
        ]

        for source, dataset in sources:
            tables = list(self.result_table_registry.get(source, []))
            if not tables:
                continue
            dataset = dataset or []

            for table in tables:
                limit_property = table.property('result_limit')
                limit_value = None
                if limit_property not in (None, '', False):
                    try:
                        limit_value = int(limit_property)
                    except (TypeError, ValueError):
                        limit_value = None

                if source == 'history':
                    entries = self._get_recent_entries(dataset, limit_value if limit_value and limit_value > 0 else None)
                else:
                    if limit_value and limit_value > 0:
                        entries = list(dataset[:limit_value])
                    else:
                        entries = list(dataset)
                self.populate_result_table(table, entries, source)

        self.update_result_buttons_state()

    def update_result_buttons_state(self):
        has_history = bool(self.history_entries)
        if self.run_proofreader_button is not None:
            self.run_proofreader_button.setEnabled(has_history)
        if self.run_quality_button is not None:
            self.run_quality_button.setEnabled(has_history)
        if self.history_view_all_button is not None:
            self.history_view_all_button.setEnabled(has_history)

        has_proof = bool(self.proofreader_entries)
        if self.proofreader_view_all_button is not None:
            self.proofreader_view_all_button.setEnabled(has_proof)
        if getattr(self, 'batch_pf_btn', None) is not None:
            self.batch_pf_btn.setEnabled(has_proof)
        if getattr(self, 'proofreader_confirm_all_button', None) is not None:
            self.proofreader_confirm_all_button.setEnabled(has_proof)
        if self.proofreader_table is not None:
            self.proofreader_table.setVisible(has_proof)
        if self.proofreader_empty_label is not None:
            self.proofreader_empty_label.setVisible(not has_proof)

        has_quality = bool(self.quality_entries)
        if self.quality_view_all_button is not None:
            self.quality_view_all_button.setEnabled(has_quality)
        if getattr(self, 'batch_qc_btn', None) is not None:
            self.batch_qc_btn.setEnabled(has_quality)
        if getattr(self, 'quality_confirm_all_button', None) is not None:
            self.quality_confirm_all_button.setEnabled(has_quality)
        if self.quality_table is not None:
            self.quality_table.setVisible(has_quality)
        if self.quality_empty_label is not None:
            self.quality_empty_label.setVisible(not has_quality)

    def _build_review_prompt(self, entries, mode):
        if not entries:
            return ""

        mode = (mode or '').lower()
        if mode == 'proofreader':
            instruction = (
                "You are an expert bilingual proofreader. Improve grammar, flow, and clarity while keeping the meaning, tone, "
                "and requested style. Preserve honorifics and important nuances. If the current translation is already "
                "excellent, return it unchanged."
            )
        else:
            instruction = (
                "You are an expert quality reviewer. Ensure the translation reads naturally, stays faithful to the original, "
                "and keeps terminology consistent. Adjust wording to sound like native dialogue and respect the requested style. "
                "If no change is needed, return the original translation."
            )

        lines = [
            instruction,
            "IMPORTANT: Return ONLY a JSON array of strings in the same order as the entries. Example: [\"improved1\", \"improved2\"]",
            "Do not include IDs, explanations, numbering, or extra commentary. If JSON is not possible, return one improved translation per line in the same order.",
            "Entries:"
        ]

        for entry in entries:
            # keep history_id internal only; do NOT inject into prompt where the model will echo it back
            history_id = entry.get('history_id') or entry.get('id') or 'UNKNOWN'
            style = entry.get('translation_style') or 'Santai (Default)'
            original = (entry.get('original_text') or '').replace(chr(13), '').replace('\n', '').strip()
            translated = (entry.get('translated_text') or '').replace(chr(13), '').replace('\n', '').strip()
            lines.append(f"Style: {style}")
            lines.append("OCR:")
            lines.append(original)
            lines.append("Current Translation:")
            lines.append(translated)
            lines.append("---")

        return "\n".join(lines)
    def _strip_code_fences(self, text):
        if not text:
            return text
        stripped = text.strip()
        if stripped.startswith("`"):
            stripped = stripped.split("\n", 1)[-1]
        if stripped.endswith("`"):
            stripped = stripped.rsplit("\n", 1)[0]
        return stripped.strip()

    def _parse_review_response(self, response_text):
        cleaned = self._strip_code_fences(response_text)
        suggestions = {}
        for raw_line in cleaned.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            # Accept ID-prefixed lines like 'H00123|text' but also accept any prefix 'KEY|text'
            match = re.match(r"^(.+?)\s*\|\s*(.+)$", line)
            if match:
                key = match.group(1).strip()
                suggestions[key] = match.group(2).strip()
        return suggestions

    def _invoke_ai_review(self, provider, model_name, prompt, temperature=0.35):
        provider = (provider or '').lower()
        prompt = prompt.strip()
        if not prompt:
            return ''

        if provider == 'gemini':
            if not GEMINI_API_KEY or "your_gemini_key_here" in GEMINI_API_KEY:
                return "[GEMINI NOT CONFIGURED]"
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    prompt,
                    generation_config={
                        "max_output_tokens": 4096,
                        "temperature": temperature
                    },
                    safety_settings=[
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    ]
                )
                text = getattr(response, 'text', '') or ''
                text = text.strip()
                if text:
                    self.add_api_cost(len(prompt), len(text), 'Gemini', model_name)
                return text or ""
            except Exception as exc:
                return f"[GEMINI ERROR: {exc}]"

        if provider == 'openai':
            if not getattr(self, 'is_openai_available', False):
                return "[OPENAI NOT CONFIGURED]"
            try:
                client = getattr(self, 'openai_client', None) or openai_client
                # Some OpenAI models (e.g. gpt-5-mini / gpt-5-nano) do not accept a custom temperature
                model_lower = (model_name or "").lower()
                supports_temperature = not (
                    model_lower.startswith("gpt-5-mini") or model_lower.startswith("gpt-5-nano")
                )

                req_kwargs = {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are an expert editor for manga translations. Improve the provided translations "
                                "without changing their intended meaning."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                }
                if supports_temperature:
                    # Only include temperature when the model supports it
                    req_kwargs["temperature"] = float(temperature)

                response = client.chat.completions.create(**req_kwargs)
                output_text = (response.choices[0].message.content or '').strip()
                if hasattr(response, 'usage'):
                    in_tokens = getattr(response.usage, 'prompt_tokens', 0)
                    out_tokens = getattr(response.usage, 'completion_tokens', 0)
                    self.add_api_cost(in_tokens, out_tokens, 'OpenAI', model_name)
                return output_text
            except Exception as exc:
                return f"[OPENAI ERROR: {exc}]"

        return "[REVIEW PROVIDER NOT SUPPORTED]"

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
        ocr_engine = lang_data.get('engine') if isinstance(lang_data, dict) else None
        ocr_lang_code = lang_data.get('code') if isinstance(lang_data, dict) else None
        ai_provider = None
        ai_model_id = None
        ai_model_name = None
        ai_provider_label = None
        if isinstance(lang_data, dict) and ocr_engine == 'AI_OCR':
            ai_provider = lang_data.get('provider')
            ai_model_id = lang_data.get('model_id')
            ai_model_name = lang_data.get('model_name')
            ai_provider_label = lang_data.get('provider_label')
        selected_model_info = self.get_selected_model_info() or {}
        selected_model_label = selected_model_info.get('display') or selected_model_info.get('name')

        # Pastikan inpaint_model_key selalu memiliki nilai
        inpaint_model_text = self.inpaint_model_combo.currentText()
        inpaint_model_key = None
        
        if "Big-LaMa" in inpaint_model_text:
            inpaint_model_key = 'big_lama'
        elif "Anime" in inpaint_model_text:
            inpaint_model_key = 'anime_inpaint'
        elif "OpenCV" in inpaint_model_text:
            inpaint_model_key = 'opencv'  # Untuk OpenCV fallback
            
        font_for_settings = QFont(self._build_current_font())
        self.typeset_font = font_for_settings
        color_for_settings = QColor(self.typeset_color)
        char_spacing_value = float(self.typeset_char_spacing_value)
        line_spacing_value = float(self.typeset_line_spacing_value)
        apply_mode_global = getattr(self, 'apply_mode_global_radio', None) and self.apply_mode_global_radio.isChecked()
        if apply_mode_global:
            use_inpaint_value = self._default_cleanup_value('use_inpaint')
            use_background_box_value = self._default_cleanup_value('use_background_box')
        else:
            use_inpaint_value = bool(self.inpaint_checkbox.isChecked()) if getattr(self, 'inpaint_checkbox', None) else self._default_cleanup_value('use_inpaint')
            use_background_box_value = bool(self.use_background_box_checkbox.isChecked()) if getattr(self, 'use_background_box_checkbox', None) else self._default_cleanup_value('use_background_box')

        return {
            'ocr_engine': ocr_engine,
            'ocr_lang': ocr_lang_code,
            'ocr_ai_provider': ai_provider,
            'ocr_ai_provider_label': ai_provider_label,
            'ocr_ai_model_id': ai_model_id,
            'ocr_ai_model_name': ai_model_name,
            'orientation': self.orientation_combo.currentText(),
            'target_lang': self.translate_combo.currentText(),
            'use_ai': True,
            'font': font_for_settings,
            'color': color_for_settings,
            'enhanced_pipeline': self.enhanced_pipeline_checkbox.isChecked(),
            'use_ai_only_translate': self.ai_only_translate_checkbox.isChecked(),
            'use_deepl_only_translate': self.deepl_only_checkbox.isChecked(),
            'use_dl_detector': self.dl_bubble_detector_checkbox.isChecked(),
            'dl_provider': self.dl_model_provider_combo.currentText(),
            'dl_model_file': self.dl_model_file_combo.currentText(),
            'ai_model': self.get_selected_model_name(),
            'ai_model_label': selected_model_label,
            'ai_model_info': selected_model_info,
            'translation_style': self.style_combo.currentText(),
            'auto_split_bubbles': self.split_bubbles_checkbox.isChecked(),
            'safe_mode': self.safe_mode_checkbox.isChecked(),
            'use_gpu': self.use_gpu_checkbox.isChecked(),
            # Pastikan ini sesuai dengan hardware Anda
            'use_inpaint': use_inpaint_value,
            'inpaint_model_name': inpaint_model_text,
            'inpaint_model_key': inpaint_model_key,  # Pastikan ini tidak None
            'inpaint_padding': self.inpaint_padding_spinbox.value(),
            # Optimasi CPU
            'cpu_threads': 4,  # Sesuaikan dengan jumlah core CPU Anda
            'enable_mkldnn': True,  # Optimasi untuk CPU Intel
            'orientation_mode': self.typeset_orientation,
            'create_bubble': getattr(self, 'create_bubble_checkbox', None) and self.create_bubble_checkbox.isChecked(),
            'use_background_box': use_background_box_value,
            'text_effect': 'none',
            'effect_intensity': 20.0,
            'bezier_points': None,
            'alignment': self.typeset_alignment,
            'line_spacing': line_spacing_value,
            'char_spacing': char_spacing_value,
            'text_outline': bool(self.typeset_outline_enabled),
            'outline_width': float(self.typeset_outline_width),
            'outline_color': self.typeset_outline_color.name() if isinstance(self.typeset_outline_color, QColor) else '#000000',
            'margins': {'top': 0, 'right': 0, 'bottom': 0, 'left': 0},
            'manga_use_easy_detection': bool(getattr(self, 'manga_use_easy_detection_checkbox', None) and self.manga_use_easy_detection_checkbox.isChecked()),
            'tesseract_use_easy_detection': bool(getattr(self, 'tesseract_use_easy_detection_checkbox', None) and self.tesseract_use_easy_detection_checkbox.isChecked()),
        }

    def _default_cleanup_value(self, key: str):
        cleanup = SETTINGS.setdefault('cleanup', {})
        if key == 'use_background_box':
            return bool(cleanup.get('use_background_box', True))
        if key == 'use_inpaint':
            return bool(cleanup.get('use_inpaint', True))
        if key == 'apply_mode':
            return cleanup.get('apply_mode', 'selected')
        return cleanup.get(key)

    def _set_global_cleanup_default(self, key: str, value, *, persist=True):
        cleanup = SETTINGS.setdefault('cleanup', {})
        cleanup[key] = value if key == 'apply_mode' else bool(value)
        if persist:
            save_settings(SETTINGS)
        if key == 'use_background_box' and getattr(self, 'use_box_action', None):
            try:
                self.use_box_action.blockSignals(True)
                self.use_box_action.setChecked(bool(value))
            finally:
                try:
                    self.use_box_action.blockSignals(False)
                except Exception:
                    pass
        if key == 'apply_mode' and getattr(self, 'apply_mode_status_label', None):
            mode_text = 'Mode: Global' if cleanup.get('apply_mode') == 'global' else 'Mode: Selected Area'
            self.apply_mode_status_label.setText(mode_text)
        if key in ('use_background_box', 'use_inpaint', 'apply_mode'):
            self._sync_cleanup_controls_from_selection()

    def set_selected_area(self, area, *, notify=True):
        if area is not None and area not in self.typeset_areas:
            area = None
        if self.selected_typeset_area is area:
            if notify:
                self._sync_cleanup_controls_from_selection()
            return
        self.selected_typeset_area = area
        if notify:
            self._sync_cleanup_controls_from_selection()
        label = getattr(self, 'image_label', None)
        if label is not None:
            try:
                label.update()
                if getattr(label, 'transform_mode', False):
                    label._refresh_transform_handles()
            except Exception:
                pass

    def clear_selected_area(self):
        self.set_selected_area(None)

    def _active_cleanup_area(self):
        mode_radio = getattr(self, 'apply_mode_global_radio', None)
        if mode_radio is not None and mode_radio.isChecked():
            return None
        if self.selected_typeset_area and self.selected_typeset_area in self.typeset_areas:
            return self.selected_typeset_area
        return None

    def _apply_cleanup_change(self, key: str, value: bool):
        value = bool(value)
        mode_radio = getattr(self, 'apply_mode_global_radio', None)
        if mode_radio is not None and mode_radio.isChecked():
            self._set_global_cleanup_default(key, value)
            return 'global'

        area = self._active_cleanup_area()
        if area is None:
            self.statusBar().showMessage("Select a typeset area to update local settings.", 2500)
            self._sync_cleanup_controls_from_selection()
            return 'no-area'

        default_value = self._default_cleanup_value(key)
        if value == default_value:
            area.clear_override(key)
        else:
            area.set_override(key, value)

        try:
            self.redraw_all_typeset_areas()
        except Exception:
            pass
        label = getattr(self, 'image_label', None)
        if label is not None:
            try:
                label.update()
            except Exception:
                pass
        self._sync_cleanup_controls_from_selection()
        return 'area'

    def _sync_cleanup_controls_from_selection(self):
        checkbox = getattr(self, 'use_background_box_checkbox', None)
        inpaint_box = getattr(self, 'inpaint_checkbox', None)
        area = self._active_cleanup_area()

        if area is not None:
            use_box_value = area.get_override('use_background_box', self._default_cleanup_value('use_background_box'))
            use_inpaint_value = area.get_override('use_inpaint', self._default_cleanup_value('use_inpaint'))
        else:
            use_box_value = self._default_cleanup_value('use_background_box')
            use_inpaint_value = self._default_cleanup_value('use_inpaint')

        if checkbox is not None:
            with QSignalBlocker(checkbox):
                checkbox.setChecked(bool(use_box_value))
        if inpaint_box is not None:
            with QSignalBlocker(inpaint_box):
                inpaint_box.setChecked(bool(use_inpaint_value))


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
        if not text or not text.strip():
            return ""

        # If DeepL has an active key, prefer it for non-AI translations
        deepl_key = get_active_key('deepl')
        if deepl_key:
            try:
                lang_map = {"Indonesian": "ID", "English": "EN-US", "Japanese": "JA", "Chinese": "ZH", "Korean": "KO"}
                url = "https://api-free.deepl.com/v2/translate"
                params = {"auth_key": deepl_key, "text": text, "target_lang": lang_map.get(target_lang, "ID")}
                response = requests.post(url, data=params, timeout=20); response.raise_for_status()
                return response.json()["translations"][0]["text"]
            except Exception as e:
                return f"[Translation Error (DeepL): {e}]"

        # If any API provider has active key, let higher-level logic use AI providers.
        any_key = False
        for prov in SETTINGS.get('apis', {}).values():
            if any(k.get('active') for k in (prov.get('keys') or [])):
                any_key = True
                break

        if not any_key:
            # No API keys at all: fallback to free translator library
            # Try googletrans first, then deep-translator
            try:
                from googletrans import Translator as GoogleTranslator
                tr = GoogleTranslator()
                res = tr.translate(text, dest=("id" if target_lang.lower().startswith("ind") else "en"))
                return getattr(res, 'text', str(res))
            except Exception:
                try:
                    from deep_translator import GoogleTranslator as DTGoogle
                    dest = 'id' if target_lang.lower().startswith('ind') else 'en'
                    return DTGoogle(source='auto', target=dest).translate(text)
                except Exception as e:
                    return f"[No API keys and no fallback translator available: {e}]"

        return "[No translation performed: use AI providers]"

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

    def _find_project_file(self, directory):
        try:
            entries = os.listdir(directory)
        except OSError:
            return None
        base_name = os.path.basename(directory.rstrip(os.sep)) or 'project'
        preferred = os.path.join(directory, f"{base_name}.manga_proj")
        if os.path.isfile(preferred):
            return preferred
        candidates = []
        for name in entries:
            if name.lower().endswith('.manga_proj'):
                candidate_path = os.path.join(directory, name)
                try:
                    mtime = os.path.getmtime(candidate_path)
                except OSError:
                    continue
                candidates.append((mtime, candidate_path))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]
    
    def _make_project_file_path(self, preferred_name=None, base_dir=None):
        base_dir = os.path.abspath(base_dir or (self.project_dir or os.getcwd()))
        preferred = preferred_name or os.path.basename(base_dir.rstrip(os.sep)) or 'project'
        sanitized = re.sub(r'[\/:*?"<>|]', '_', preferred).strip()
        if not sanitized:
            sanitized = 'project'
        candidate = os.path.join(base_dir, f"{sanitized}.manga_proj")
        note = None
        if os.name == 'nt':
            abs_candidate = os.path.abspath(candidate)
            if len(abs_candidate) >= 245:
                digest = hashlib.sha1(abs_candidate.encode('utf-8')).hexdigest()[:10]
                candidate = os.path.join(base_dir, f"project_{digest}.manga_proj")
                note = f"Project filename shortened to avoid Windows path limit (using {os.path.basename(candidate)})."
        return candidate, note

    def _initialize_new_project(self, directory, status_message=None):
        self.project_dir = os.path.abspath(directory)
        self.cache_dir = os.path.join(self.project_dir, '.cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.all_typeset_data.clear()
        self.reset_history_state()
        folder_name = os.path.basename(self.project_dir.rstrip(os.sep)) or 'project'
        self.current_project_path, note = self._make_project_file_path(folder_name)
        if note:
            self.statusBar().showMessage(note, 6000)
        try:
            if self.project_dir not in self.file_watcher.directories():
                self.file_watcher.addPath(self.project_dir)
        except Exception:
            pass
        self.update_file_list()
        self.save_project(is_auto=True)
        if status_message is None:
            status_message = "New project created and auto-saved."
        self.setWindowTitle(f"Manga OCR & Typeset Tool v14.3.4 - {os.path.basename(self.current_project_path)}")
        self.statusBar().showMessage(status_message, 4000)
    
    def load_folder(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Manga Folder", self.project_dir or "")
        if not dir_path:
            return
    
        dir_path = os.path.abspath(dir_path)
        current_dir = os.path.abspath(self.project_dir) if self.project_dir else None
        if current_dir and os.path.normcase(current_dir) == os.path.normcase(dir_path):
            return
    
        self.save_project(is_auto=True)
    
        if self.project_dir and self.project_dir in self.file_watcher.directories():
            try:
                self.file_watcher.removePath(self.project_dir)
            except Exception:
                pass
    
        if self.pdf_document:
            self.pdf_document.close()
            self.pdf_document = None
        self.current_pdf_page = -1
    
        project_file = self._find_project_file(dir_path)
        if project_file and os.path.isfile(project_file):
            if self._load_project_from_path(project_file, show_dialogs=False):
                self.statusBar().showMessage(f"Loaded project: {os.path.basename(project_file)}", 4000)
                return
            self.statusBar().showMessage("Failed to load existing project; starting new project.", 5000)
    
        self._initialize_new_project(dir_path)
    
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
            # Assign original_pixmap under paint mutex to avoid concurrent painting/destroy races
            qpix_temp = QPixmap(file_path)
            self.paint_mutex.lock()
            try:
                self.original_pixmap = qpix_temp
            finally:
                self.paint_mutex.unlock()
            # Use robust opener to handle truncated/corrupt JPEGs
            self.current_image_pil = self.safe_open_image(file_path)

            # Create QPixmap from PIL image (safer than loading directly from a possibly corrupted file)
            pil_img = self.current_image_pil
            data = pil_img.tobytes('raw', 'RGB')
            qimage = QImage(data, pil_img.width, pil_img.height, pil_img.width * 3, QImage.Format_RGB888)
            # Replace original_pixmap safely while holding the paint mutex
            self.paint_mutex.lock()
            try:
                self.original_pixmap = QPixmap.fromImage(qimage)
            finally:
                self.paint_mutex.unlock()

            key = self.get_current_data_key()
            img_data = self.all_typeset_data.get(key, {'areas': [], 'redo': []})
            self.typeset_areas = img_data['areas']
            self.redo_stack = img_data['redo']
            self.set_selected_area(None, notify=True)
            if hasattr(self, 'image_label') and self.image_label:
                self.image_label.reset_brush_mask()

            self.rebuild_history_for_image(key, self.typeset_areas)
            self.redraw_all_typeset_areas()
            self.update_undo_redo_buttons_state()
            self._refresh_detection_overlay()
            self.refresh_history_views()
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
        # Replace original_pixmap safely while holding the paint mutex
        self.paint_mutex.lock()
        try:
            self.original_pixmap = QPixmap.fromImage(q_image)
        finally:
            self.paint_mutex.unlock()
        self.current_image_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        key = self.get_current_data_key()
        img_data = self.all_typeset_data.get(key, {'areas': [], 'redo': []})
        self.typeset_areas = img_data['areas']
        self.redo_stack = img_data['redo']
        self.set_selected_area(None, notify=True)
        if hasattr(self, 'image_label') and self.image_label:
            self.image_label.reset_brush_mask()

        self.rebuild_history_for_image(key, self.typeset_areas)
        self.redraw_all_typeset_areas()
        self.update_undo_redo_buttons_state()
        self._refresh_detection_overlay()
        self.refresh_history_views()
        self.update_nav_buttons()

    def get_current_data_key(self, path=None, page=-1):
        path_to_use = path if path is not None else self.current_image_path
        page_to_use = page if page != -1 else self.current_pdf_page

        if path_to_use and path_to_use.lower().endswith('.pdf') and page_to_use != -1:
            return f"{path_to_use}::page::{page_to_use}"
        return path_to_use

    def _ensure_workshop_dir(self):
        base_dir = None
        if self.project_dir and os.path.isdir(self.project_dir):
            base_dir = self.project_dir
        elif self.current_image_path:
            base_dir = os.path.dirname(self.current_image_path)
        else:
            base_dir = os.getcwd()
        workshop_dir = os.path.join(base_dir, "workshop")
        os.makedirs(workshop_dir, exist_ok=True)
        return workshop_dir

    def _ensure_inpaint_temp_dir(self):
        """
        Create (if needed) the temp/inpainting directory inside the active project.
        Returns the absolute path to that directory.
        """
        base_dir = self.project_dir or (os.path.dirname(self.current_image_path) if self.current_image_path else os.getcwd())
        temp_root = os.path.join(base_dir, "temp")
        inpaint_dir = os.path.join(temp_root, "inpainting")
        try:
            os.makedirs(inpaint_dir, exist_ok=True)
        except Exception as exc:
            raise RuntimeError(f"Gagal membuat folder inpainting: {exc}") from exc
        return inpaint_dir

    def _write_inpaint_result_copy(self, image_bytes: bytes, output_ext: str) -> str | None:
        """
        Persist an extra copy of the inpaint result directly in the active project directory
        (or alongside the current image if no project is active). The file is suffixed with
        `_cleanup_YYYYmmdd_HHMMSS` so the original asset is untouched.
        """
        try:
            base_dir = self.project_dir or (os.path.dirname(self.current_image_path) if self.current_image_path else os.getcwd())
            output_dir = os.path.join(base_dir, "inpainting")
            os.makedirs(output_dir, exist_ok=True)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            if self.current_image_path:
                base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            else:
                base_name = "inpaint"
            copy_filename = f"{base_name}_cleanup_{timestamp}{output_ext}"
            copy_path = os.path.join(output_dir, copy_filename)

            with open(copy_path, 'wb') as handle:
                handle.write(image_bytes)
            return copy_path
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to store inpaint copy: {exc}", 6000)
            return None

    def _copy_mask_to_project(self, mask_path: str) -> str | None:
        """
        Copy the saved mask PNG into the project-level inpainting folder so the user can inspect
        which pixels were brushed. A timestamped filename is used to avoid collisions.
        """
        if not mask_path or not os.path.exists(mask_path):
            return None
        try:
            base_dir = self.project_dir or (os.path.dirname(self.current_image_path) if self.current_image_path else os.getcwd())
            output_dir = os.path.join(base_dir, "inpainting")
            os.makedirs(output_dir, exist_ok=True)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            if self.current_image_path:
                base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            else:
                base_name = "inpaint"
            dest_path = os.path.join(output_dir, f"{base_name}_mask_{timestamp}.png")
            shutil.copy2(mask_path, dest_path)
            return dest_path
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to copy inpaint mask: {exc}", 6000)
            return None

    def _save_brush_inpaint_artifacts(self, mask_image, image_bytes, output_ext):
        """
        Save the brush mask and resulting inpaint image to the temp/inpainting folder.
        Returns a tuple (mask_path, output_path) if successful.
        """
        try:
            out_dir = self._ensure_inpaint_temp_dir()
        except Exception as exc:
            self.statusBar().showMessage(str(exc), 6000)
            return None, None

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if self.current_image_path:
            base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
        else:
            base_name = "inpaint"

        mask_filename = f"{base_name}_mask_{timestamp}.png"
        cleanup_filename = f"{base_name}_cleanup_{timestamp}{output_ext}"
        mask_path = os.path.join(out_dir, mask_filename)
        cleanup_path = os.path.join(out_dir, cleanup_filename)

        counter = 1
        while os.path.exists(mask_path):
            mask_path = os.path.join(out_dir, f"{base_name}_mask_{timestamp}_{counter}.png")
            counter += 1

        counter = 1
        while os.path.exists(cleanup_path):
            cleanup_path = os.path.join(out_dir, f"{base_name}_cleanup_{timestamp}_{counter}{output_ext}")
            counter += 1

        try:
            mask_to_save = mask_image
            if isinstance(mask_to_save, QImage) and mask_to_save.format() not in (
                QImage.Format_Grayscale8,
                QImage.Format_Alpha8,
            ):
                mask_to_save = mask_to_save.convertToFormat(QImage.Format_Grayscale8)
            if isinstance(mask_to_save, QImage):
                if not mask_to_save.save(mask_path, "PNG"):
                    raise RuntimeError("Mask LaMA gagal disimpan ke disk.")
            else:
                with open(mask_path, 'wb') as mask_file:
                    mask_file.write(mask_to_save)

            with open(cleanup_path, 'wb') as fout:
                fout.write(image_bytes)
        except Exception as exc:
            self.statusBar().showMessage(f"Gagal menyimpan hasil inpainting: {exc}", 6000)
            return None, None

        return mask_path, cleanup_path

    def clear_view(self):
        # Clear pixmaps and data safely while preventing concurrent painting
        self.paint_mutex.lock()
        try:
            self.original_pixmap = None
            self.typeset_pixmap = None
            self.current_image_path = None
            self.current_image_pil = None
            self.typeset_areas.clear()
            self.redo_stack.clear()
        finally:
            self.paint_mutex.unlock()
        self.clear_selected_area()
        if hasattr(self, 'image_label') and self.image_label:
            self.image_label.reset_brush_mask()

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
        # Safely copy the typeset_pixmap under mutex to avoid races with saving/painting
        self.paint_mutex.lock()
        try:
            local_pixmap = self.typeset_pixmap
            if not local_pixmap:
                self.image_label.setPixmap(QPixmap())
                return
            # Work on a copy to avoid holding the mutex during scaling
            pix_copy = local_pixmap.copy()
        finally:
            self.paint_mutex.unlock()

        self.zoom_label.setText(f" Zoom: {self.zoom_factor:.1f}x ")
        scaled_size = pix_copy.size() * self.zoom_factor
        scaled_pixmap = pix_copy.scaled(scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap); self.image_label.adjustSize()
        if hasattr(self.image_label, 'update_brush_cursor'):
            self.image_label.update_brush_cursor()

    def zoom_in(self):
        self.zoom_factor = min(self.zoom_factor + 0.2, 8.0); self.update_display()

    def zoom_out(self):
        self.zoom_factor = max(self.zoom_factor - 0.2, 0.1); self.update_display()

    def reset_view_to_original(self):
        if self.original_pixmap:
            self.typeset_areas.clear(); self.redo_stack.clear()
            self.redraw_all_typeset_areas()
            self.update_undo_redo_buttons_state()

    def _populate_typeset_font_dropdown(self, preferred_display=None, group: str | None = None):
        """Populate the typeset font dropdown. Optionally filter by a font group name.

        preferred_display: preferred font display name to select
        group: if provided, only fonts listed under self.font_groups[group] will be shown
        """
        if not hasattr(self, 'font_dropdown') or not self.font_manager:
            return
        fonts = self.font_manager.list_fonts()
        # If a group is supplied and we have a mapping, filter the fonts.
        if group and getattr(self, 'font_groups', None):
            allowed = set(self.font_groups.get(group, []) )
            # Keep only fonts that exist in the available fonts list
            fonts = [f for f in fonts if f in allowed]

        current_display = self.font_manager.display_name_for_font(getattr(self, 'typeset_font', None))
        target_display = preferred_display or current_display
        with QSignalBlocker(self.font_dropdown):
            self.font_dropdown.clear()
            for name in fonts:
                self.font_dropdown.addItem(name)
                preview_font = self.font_manager.create_qfont(name)
                preview_font.setPointSize(16)
                index = self.font_dropdown.count() - 1
                self.font_dropdown.setItemData(index, preview_font, Qt.FontRole)
        if target_display in fonts:
            with QSignalBlocker(self.font_dropdown):
                self.font_dropdown.setCurrentText(target_display)
        elif fonts:
            with QSignalBlocker(self.font_dropdown):
                self.font_dropdown.setCurrentIndex(0)

    def _typeset_button_stylesheet(self):
        return (
            "QToolButton {"
            " border: 1px solid #1f2b3b;"
            " background-color: #152231;"
            " border-radius: 6px;"
            " padding: 4px;"
            " }"
            " QToolButton:hover {"
            " border-color: #3a9bff;"
            " background-color: #1c2b3d;"
            " }"
            " QToolButton:checked {"
            " border-color: #3a9bff;"
            " background-color: #25426b;"
            " }"
        )

    def _create_tool_toggle(self, icon: QIcon, tooltip: str) -> QToolButton:
        button = QToolButton()
        button.setCheckable(True)
        button.setIcon(icon)
        button.setIconSize(QSize(24, 24))
        button.setCursor(Qt.PointingHandCursor)
        button.setToolTip(tooltip)
        button.setAutoRaise(True)
        button.setMinimumSize(36, 36)
        button.setStyleSheet(self._typeset_button_stylesheet())
        return button

    def _make_style_icon(self, letter: str) -> QIcon:
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        font = QFont('Segoe UI', 18)
        if letter == 'B':
            font.setBold(True)
        elif letter == 'I':
            font.setItalic(True)
        elif letter == 'U':
            font.setUnderline(True)
        painter.setFont(font)
        painter.setPen(QPen(QColor('#f3f6fb')))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, letter)
        painter.end()
        return QIcon(pixmap)

    def _make_outline_icon(self) -> QIcon:
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        outer_pen = QPen(QColor('#f3f6fb'))
        outer_pen.setWidth(2)
        painter.setPen(outer_pen)
        painter.drawEllipse(5, 5, 22, 22)
        painter.setBrush(QBrush(QColor(0, 0, 0, 180)))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(10, 10, 12, 12)
        painter.end()
        return QIcon(pixmap)

    def _make_alignment_icon(self, mode: str) -> QIcon:
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor('#f3f6fb'))
        outline_px = max(1, int(round(SETTINGS.get('typeset', {}).get('outline_width', SETTINGS.get('typeset', {}).get('outline_thickness', 2)))))
        pen.setWidth(outline_px)
        painter.setPen(pen)
        lines = [22, 26, 18]
        y = 8
        for length in lines:
            if mode == 'left':
                start = 6
            elif mode == 'right':
                start = 32 - length - 6
            else:  # center
                start = (32 - length) / 2.0
            painter.drawLine(QPointF(start, y), QPointF(start + length, y))
            y += 10
        painter.end()
        return QIcon(pixmap)

    def _make_orientation_icon(self, mode: str) -> QIcon:
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor('#f3f6fb'))
        outline_px = max(1, int(round(SETTINGS.get('typeset', {}).get('outline_width', SETTINGS.get('typeset', {}).get('outline_thickness', 2)))))
        pen.setWidth(outline_px)

        painter.setPen(pen)
        if mode == 'horizontal':
            for offset in (8, 16, 24):
                painter.drawLine(QPointF(6, offset), QPointF(26, offset))
        else:
            for offset in (8, 16, 24):
                painter.drawLine(QPointF(offset, 6), QPointF(offset, 26))
        painter.end()
        return QIcon(pixmap)

    def _create_initial_typeset_defaults(self):
        display = None
        if self.font_manager and isinstance(self.typeset_font, QFont):
            display = self.font_manager.display_name_for_font(self.typeset_font)
        if not display and self.font_manager:
            display = self.font_manager.default_display
        size_value = self.typeset_font.pointSizeF() or self.typeset_font.pointSize() or 24.0
        return {
            'font_display': display,
            'font_size': float(size_value),
            'line_spacing': float(self.typeset_line_spacing_value),
            'char_spacing': float(self.typeset_char_spacing_value),
            'bold': self.typeset_font.weight() >= QFont.Bold,
            'italic': self.typeset_font.italic(),
            'underline': self.typeset_font.underline(),
            'alignment': self.typeset_alignment,
            'orientation': self.typeset_orientation,
            'outline': bool(self.typeset_outline_enabled),
            'outline_width': float(self.typeset_outline_width),
            'outline_color': self.typeset_outline_color.name() if isinstance(self.typeset_outline_color, QColor) else '#000000',
            'color': self.typeset_color.name(),
        }

    def _collect_current_typeset_defaults(self):
        if not getattr(self, 'font_dropdown', None):
            return self._create_initial_typeset_defaults()
        font_display = self.font_dropdown.currentText() or (self.font_manager.default_display if self.font_manager else 'System Default')
        return {
            'font_display': font_display,
            'font_size': float(self.font_size_spin.value() if getattr(self, 'font_size_spin', None) else 24.0),
            'line_spacing': float(self.typeset_line_spacing_value),
            'char_spacing': float(self.typeset_char_spacing_value),
            'bold': bool(self.bold_toggle.isChecked() if getattr(self, 'bold_toggle', None) else False),
            'italic': bool(self.italic_toggle.isChecked() if getattr(self, 'italic_toggle', None) else False),
            'underline': bool(self.underline_toggle.isChecked() if getattr(self, 'underline_toggle', None) else False),
            'alignment': self.typeset_alignment,
            'orientation': self.typeset_orientation,
            'outline': bool(self.typeset_outline_enabled),
            'outline_width': float(self.typeset_outline_width),
            'outline_color': self.typeset_outline_color.name() if isinstance(self.typeset_outline_color, QColor) else '#000000',
            'color': self.typeset_color.name() if isinstance(self.typeset_color, QColor) else '#000000',
        }

    def _handle_save_typeset_defaults(self):
        self.typeset_defaults = self._collect_current_typeset_defaults()
        status = self.statusBar() if hasattr(self, 'statusBar') else None
        if status:
            status.showMessage("Typeset defaults updated", 2500)

    def _handle_reset_typeset_defaults(self):
        self._apply_typeset_defaults()
        status = self.statusBar() if hasattr(self, 'statusBar') else None
        if status:
            status.showMessage("Defaults restored", 2000)

    def _set_line_spacing_value(self, spacing: float):
        spacing = max(0.6, min(3.0, float(spacing)))
        slider_value = int(round(spacing * 100))
        if getattr(self, 'line_spacing_slider', None):
            with QSignalBlocker(self.line_spacing_slider):
                self.line_spacing_slider.setValue(slider_value)
        self.typeset_line_spacing_value = spacing
        if getattr(self, 'line_spacing_value_label', None):
            self.line_spacing_value_label.setText(f"{spacing:.2f}x")

    def _set_char_spacing_value(self, spacing: float):
        spacing = max(10.0, min(400.0, float(spacing)))
        slider_value = int(round(spacing))
        if getattr(self, 'char_spacing_slider', None):
            with QSignalBlocker(self.char_spacing_slider):
                self.char_spacing_slider.setValue(slider_value)
        self.typeset_char_spacing_value = float(slider_value)
        if getattr(self, 'char_spacing_value_label', None):
            self.char_spacing_value_label.setText(f"{slider_value}%")

    def _apply_typeset_defaults(self):
        if not getattr(self, 'font_dropdown', None):
            return
        defaults = self.typeset_defaults or self._create_initial_typeset_defaults()
        preferred_display = defaults.get('font_display')
        self._populate_typeset_font_dropdown(preferred_display)

        if getattr(self, 'font_size_spin', None):
            with QSignalBlocker(self.font_size_spin):
                self.font_size_spin.setValue(float(defaults.get('font_size', 24.0)))
        if getattr(self, 'bold_toggle', None):
            with QSignalBlocker(self.bold_toggle):
                self.bold_toggle.setChecked(bool(defaults.get('bold', False)))
        if getattr(self, 'italic_toggle', None):
            with QSignalBlocker(self.italic_toggle):
                self.italic_toggle.setChecked(bool(defaults.get('italic', False)))
        if getattr(self, 'underline_toggle', None):
            with QSignalBlocker(self.underline_toggle):
                self.underline_toggle.setChecked(bool(defaults.get('underline', False)))
        self._set_line_spacing_value(defaults.get('line_spacing', 1.1))
        self._set_char_spacing_value(defaults.get('char_spacing', 100.0))

        self.typeset_alignment = defaults.get('alignment', 'center')
        self.typeset_orientation = defaults.get('orientation', 'horizontal')
        self._update_alignment_buttons()
        self._update_orientation_buttons()

        self.typeset_outline_enabled = bool(defaults.get('outline', False))
        if getattr(self, 'outline_toggle', None):
            with QSignalBlocker(self.outline_toggle):
                self.outline_toggle.setChecked(self.typeset_outline_enabled)

        outline_width = defaults.get('outline_width')
        if outline_width is None:
            outline_width = SETTINGS.get('typeset', {}).get('outline_width', SETTINGS.get('typeset', {}).get('outline_thickness', self.typeset_outline_width))
        try:
            outline_width = float(outline_width)
        except Exception:
            outline_width = self.typeset_outline_width
        outline_width = max(0.0, min(outline_width, 12.0))
        self.typeset_outline_width = outline_width
        if getattr(self, 'outline_width_spin', None):
            with QSignalBlocker(self.outline_width_spin):
                self.outline_width_spin.setValue(self.typeset_outline_width)

        outline_color_value = defaults.get('outline_color')
        if outline_color_value is None:
            outline_color_value = SETTINGS.get('typeset', {}).get('outline_color', '#000000')
        outline_color = QColor(outline_color_value) if outline_color_value else QColor('#000000')
        if not outline_color.isValid():
            outline_color = QColor('#000000')
        self.typeset_outline_color = outline_color
        self._update_outline_color_button()
        self._refresh_outline_controls_enabled()

        color_value = defaults.get('color', '#000000')
        color_obj = QColor(color_value)
        if color_obj.isValid():
            self.typeset_color = color_obj
        self._update_color_button()

        self.typeset_font = self._build_current_font()
        self._update_typeset_preview()

    def _build_current_font(self) -> QFont:
        display = None
        if getattr(self, 'font_dropdown', None):
            display = self.font_dropdown.currentText()
        if self.font_manager and display:
            font = self.font_manager.create_qfont(display)
        elif isinstance(self.typeset_font, QFont):
            font = QFont(self.typeset_font)
        else:
            font = QFont('Arial', 14)
        size_value = float(self.font_size_spin.value()) if getattr(self, 'font_size_spin', None) else 24.0
        if size_value <= 0:
            size_value = 12.0
        font.setPointSizeF(size_value)
        if getattr(self, 'bold_toggle', None):
            font.setBold(self.bold_toggle.isChecked())
        if getattr(self, 'italic_toggle', None):
            font.setItalic(self.italic_toggle.isChecked())
        if getattr(self, 'underline_toggle', None):
            font.setUnderline(self.underline_toggle.isChecked())
        font.setLetterSpacing(QFont.PercentageSpacing, self.typeset_char_spacing_value or 100.0)
        return font

    def _on_typeset_font_size_changed(self, value):
        self.typeset_font = self._build_current_font()
        self._update_typeset_preview()

    def _on_typeset_line_spacing_changed(self, slider_value):
        spacing = float(slider_value) / 100.0
        self._set_line_spacing_value(spacing)
        self._update_typeset_preview()

    def _on_typeset_char_spacing_changed(self, slider_value):
        self._set_char_spacing_value(slider_value)
        self.typeset_font = self._build_current_font()
        self._update_typeset_preview()

    def _on_typeset_style_changed(self, *_):
        self.typeset_font = self._build_current_font()
        self._update_typeset_preview()

    def _on_typeset_outline_changed(self, checked):
        self.typeset_outline_enabled = bool(checked)
        self._refresh_outline_controls_enabled()
        self._persist_typeset_preferences()
        self._update_typeset_preview()

    def _on_alignment_button_toggled(self, checked):
        if not checked:
            return
        button = self.sender()
        if not isinstance(button, QToolButton):
            return

    def _on_font_group_changed(self, group_name: str):
        # Called when the font group selector changes. If 'All' selected, pass
        # no group so the full font list is shown.
        if group_name == 'All':
            self._populate_typeset_font_dropdown()
        else:
            self._populate_typeset_font_dropdown(group=group_name)
        # Refresh preview in case font selection affected it
        try:
            self.typeset_font = self._build_current_font()
            self._update_typeset_preview()
        except Exception:
            pass

    def _on_add_font_to_group_clicked(self):
        # Open a simple modal to add a font family to the currently selected group
        current_group = self.font_group_combo.currentText()
        if not current_group or current_group == 'All':
            QMessageBox.information(self, "Select Group", "Please select a specific group to add a font to.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Add Font to {current_group}")
        dialog.setModal(True)
        dlg_layout = QVBoxLayout(dialog)
        dlg_layout.addWidget(QLabel(f"Add a font family name to the group '{current_group}':"))
        font_input = QLineEdit()
        font_input.setPlaceholderText("Type font display name (e.g. 'Badaboom BB') or exact family name")
        dlg_layout.addWidget(font_input)

        # Also provide a dropdown of installed fonts to choose from
        installed_label = QLabel("Or choose from installed fonts:")
        installed_label.setStyleSheet("color: #9cb4d0; font-size: 11px;")
        dlg_layout.addWidget(installed_label)
        installed_combo = QComboBox()
        try:
            installed_fonts = self.font_manager.list_fonts() if getattr(self, 'font_manager', None) else []
            installed_combo.addItem("(none)")
            for f in installed_fonts:
                installed_combo.addItem(f)
        except Exception:
            installed_combo.addItem("(could not list)")
        dlg_layout.addWidget(installed_combo)

        btn_box = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        dlg_layout.addWidget(btn_box)

        if dialog.exec_() != QDialog.Accepted:
            return

        chosen = font_input.text().strip() or (installed_combo.currentText() if installed_combo.currentText() != "(none)" and installed_combo.currentText() != "(could not list)" else '')
        if not chosen:
            QMessageBox.information(self, "No font selected", "No font was provided. Operation cancelled.")
            return

        # Add to mapping and refresh UI
        if not getattr(self, 'font_groups', None):
            self.font_groups = {}
        self.font_groups.setdefault(current_group, [])
        if chosen not in self.font_groups[current_group]:
            self.font_groups[current_group].append(chosen)
        # refresh the font dropdown for the group
        self._populate_typeset_font_dropdown(group=current_group)
        # Persist font groups to settings
        try:
            SETTINGS.setdefault('font_groups', {})
            SETTINGS['font_groups'].update(copy.deepcopy(self.font_groups))
            save_settings(SETTINGS)
        except Exception:
            pass
        QMessageBox.information(self, "Added", f"'{chosen}' added to group '{current_group}'.")

    def _on_add_font_group_clicked(self):
        name, ok = QInputDialog.getText(self, "Add Font Group", "Group name:")
        if not ok or not name.strip():
            return
        grp = name.strip()
        if not getattr(self, 'font_groups', None):
            self.font_groups = {}
        if grp in self.font_groups:
            QMessageBox.information(self, "Exists", f"Group '{grp}' already exists.")
            return
        self.font_groups[grp] = []
        # update combo
        self.font_group_combo.addItem(grp)
        # persist
        try:
            SETTINGS.setdefault('font_groups', {})
            SETTINGS['font_groups'].update(copy.deepcopy(self.font_groups))
            save_settings(SETTINGS)
        except Exception:
            pass
        QMessageBox.information(self, "Created", f"Group '{grp}' created.")

    def _on_remove_font_group_clicked(self):
        grp = self.font_group_combo.currentText()
        if not grp or grp == 'All':
            QMessageBox.information(self, "Select Group", "Please select a specific group to remove.")
            return
        confirm = QMessageBox.question(self, "Remove Group", f"Remove group '{grp}' and all its entries?", QMessageBox.Yes | QMessageBox.No)
        if confirm != QMessageBox.Yes:
            return
        try:
            if getattr(self, 'font_groups', None) and grp in self.font_groups:
                self.font_groups.pop(grp, None)
            # remove from combo
            idx = self.font_group_combo.findText(grp)
            if idx != -1:
                self.font_group_combo.removeItem(idx)
            # persist
            SETTINGS.setdefault('font_groups', {})
            SETTINGS['font_groups'] = copy.deepcopy(self.font_groups)
            save_settings(SETTINGS)
            QMessageBox.information(self, "Removed", f"Group '{grp}' removed.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to remove group: {e}")

    def _on_orientation_button_toggled(self, checked):
        if not checked:
            return
        button = self.sender()
        if not isinstance(button, QToolButton):
            return
        mode = button.property('orientation-mode') or 'horizontal'
        self.typeset_orientation = mode
        self._update_orientation_buttons()
        self._update_typeset_preview()

    def _update_alignment_buttons(self):
        if not all(getattr(self, name, None) for name in ('align_left_button', 'align_center_button', 'align_right_button')):
            return
        mapping = {
            'left': self.align_left_button,
            'center': self.align_center_button,
            'right': self.align_right_button,
        }
        for mode, button in mapping.items():
            with QSignalBlocker(button):
                button.setChecked(self.typeset_alignment == mode)

    def _update_orientation_buttons(self):
        if not all(getattr(self, name, None) for name in ('orientation_horizontal_button', 'orientation_vertical_button')):
            return
        mapping = {
            'horizontal': self.orientation_horizontal_button,
            'vertical': self.orientation_vertical_button,
        }
        for mode, button in mapping.items():
            with QSignalBlocker(button):
                button.setChecked(self.typeset_orientation == mode)

    def _update_color_button(self):
        try:
            if not getattr(self, 'color_button', None):
                return
            color = self.typeset_color if isinstance(self.typeset_color, QColor) else QColor(self.typeset_color)
            if not color.isValid():
                self.color_button.setStyleSheet("")
                return
            luminance = 0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()
            text_color = '#000000' if luminance > 160 else '#f3f6fb'
            self.color_button.setStyleSheet(
                f"QPushButton {{ background-color: {color.name()}; color: {text_color}; border: 1px solid #1f2b3b; border-radius: 6px; padding: 6px 12px; }}"
                " QPushButton:hover { border-color: #3a9bff; }"
            )
        except Exception:
            traceback.print_exc()

    def _update_outline_color_button(self):
        try:
            if not getattr(self, 'outline_color_button', None):
                return
            color = self.typeset_outline_color if isinstance(self.typeset_outline_color, QColor) else QColor(self.typeset_outline_color)
            if not color.isValid():
                self.outline_color_button.setStyleSheet("")
                return
            luminance = 0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()
            text_color = '#000000' if luminance > 160 else '#f3f6fb'
            self.outline_color_button.setStyleSheet(
                f"QPushButton {{ background-color: {color.name()}; color: {text_color}; border: 1px solid #1f2b3b; border-radius: 6px; padding: 6px 12px; }}"
                " QPushButton:hover { border-color: #3a9bff; }"
            )
        except Exception:
            traceback.print_exc()

    def _refresh_outline_controls_enabled(self):
        enabled = bool(self.typeset_outline_enabled)
        if getattr(self, 'outline_color_button', None):
            self.outline_color_button.setEnabled(enabled)
        if getattr(self, 'outline_width_spin', None):
            self.outline_width_spin.setEnabled(enabled)

    def _on_outline_width_changed(self, value):
        try:
            width = float(value)
        except Exception:
            width = self.typeset_outline_width
        self.typeset_outline_width = max(0.0, min(width, 12.0))
        self._persist_typeset_preferences()
        self._update_typeset_preview()

    def _persist_typeset_preferences(self):
        cfg = SETTINGS.setdefault('typeset', {})
        cfg['outline_enabled'] = bool(self.typeset_outline_enabled)
        cfg['outline_width'] = float(self.typeset_outline_width)
        cfg['outline_thickness'] = int(round(max(0.0, self.typeset_outline_width)))
        color = self.typeset_outline_color if isinstance(self.typeset_outline_color, QColor) else QColor(self.typeset_outline_color)
        if not color.isValid():
            color = QColor('#000000')
        cfg['outline_color'] = color.name()
        save_settings(SETTINGS)

    def _update_font_preview_label(self):
        if not getattr(self, 'font_preview_label', None):
            return
        font = QFont(self.typeset_font)
        preview_size = max(12.0, min(font.pointSizeF() or font.pointSize() or 20.0, 28.0))
        font.setPointSizeF(preview_size)
        font.setLetterSpacing(QFont.PercentageSpacing, self.typeset_char_spacing_value)
        self.font_preview_label.setFont(font)
        self.font_preview_label.setText("AaBb123")
        if getattr(self, 'font_dropdown', None):
            self.font_preview_label.setToolTip(self.font_dropdown.currentText())

    def _update_typeset_preview(self):
        try:
            if not getattr(self, 'typeset_preview_label', None):
                return
            self.typeset_font = self._build_current_font()
            self._update_font_preview_label()

            doc = QTextDocument()
            doc.setDocumentMargin(0)
            doc.setDefaultFont(self.typeset_font)

            sample_text = self.preview_sample_text
            if self.typeset_orientation == 'vertical':
                vertical_chars = [ch for ch in self.preview_sample_text if ch.strip()]
                sample_text = '\n'.join(vertical_chars)
            doc.setPlainText(sample_text)

            option = doc.defaultTextOption()
            align_map = {'left': Qt.AlignLeft, 'center': Qt.AlignHCenter, 'right': Qt.AlignRight}
            option.setAlignment(align_map.get(self.typeset_alignment, Qt.AlignHCenter))
            doc.setDefaultTextOption(option)

            cursor = QTextCursor(doc)
            cursor.select(QTextCursor.Document)
            block_format = QTextBlockFormat()
            block_format.setLineHeight(int(self.typeset_line_spacing_value * 100), QTextBlockFormat.ProportionalHeight)
            block_format.setAlignment(option.alignment())
            cursor.setBlockFormat(block_format)
            text_format = QTextCharFormat()
            text_format.setForeground(QBrush(self.typeset_color))
            cursor.mergeCharFormat(text_format)

            doc.setTextWidth(220)
            doc_size = doc.size()
            image_width = max(1, int(math.ceil(doc_size.width())))
            image_height = max(1, int(math.ceil(doc_size.height())))
            image = QImage(image_width, image_height, QImage.Format_ARGB32_Premultiplied)
            image.fill(Qt.transparent)
            painter = QPainter(image)
            doc.drawContents(painter)
            painter.end()

            if self.typeset_outline_enabled and (self.typeset_outline_width or 0) > 0:
                outline_color = self.typeset_outline_color if isinstance(self.typeset_outline_color, QColor) and self.typeset_outline_color.isValid() else self._outline_for_text_color(self.typeset_color)
                image = self._expand_with_outline(image, outline_color, radius=self.typeset_outline_width)

            pixmap = QPixmap.fromImage(image)
            if self.typeset_orientation == 'vertical':
                transform = QTransform()
                transform.rotate(90)
                pixmap = pixmap.transformed(transform, Qt.SmoothTransformation)

            width = max(220, self.typeset_preview_label.width())
            height = max(160, self.typeset_preview_label.height())
            canvas = QPixmap(width, height)
            canvas.fill(Qt.transparent)
            scaled = pixmap.scaled(width - 24, height - 24, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            painter = QPainter(canvas)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.drawPixmap((width - scaled.width()) // 2, (height - scaled.height()) // 2, scaled)
            painter.end()
            self.typeset_preview_label.setPixmap(canvas)
        except Exception:
            traceback.print_exc()

    def on_typeset_font_change(self, display_name):
        if not display_name or not self.font_manager:
            return
        self.typeset_font = self._build_current_font()
        self._update_typeset_preview()

    def import_font(self):
        if not self.font_manager:
            return
        dialog_dir = self.font_manager.font_dir if hasattr(self.font_manager, 'font_dir') else os.getcwd()
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Import Fonts",
            dialog_dir,
            "Font Files (*.ttf *.otf *.ttc *.otc)"
        )
        if not files:
            return

        added = []
        skipped = []
        for path in files:
            display = os.path.splitext(os.path.basename(path))[0]
            try:
                name = self.font_manager.import_font(path)
                added.append(name)
            except FileExistsError:
                skipped.append(f"Font '{display}' already exists.")
            except ValueError as exc:
                skipped.append(f"{display}: {exc}")
            except RuntimeError as exc:
                skipped.append(f"{display}: {exc}")
            except Exception as exc:
                skipped.append(f"{display}: {exc}")

        preferred = added[-1] if added else None
        self._populate_typeset_font_dropdown(preferred)
        if added:
            self.typeset_font = self._build_current_font()
            self._update_typeset_preview()
            QMessageBox.information(self, "Fonts Imported", f"Imported {len(added)} font(s):\n" + ", ".join(added))
        if skipped:
            QMessageBox.warning(self, "Fonts Skipped", "\n".join(skipped))

    def choose_color(self):
        color = QColorDialog.getColor(self.typeset_color, self)
        if color.isValid():
            self.typeset_color = color
            self._update_color_button()
            self._update_typeset_preview()

    def choose_outline_color(self):
        current = self.typeset_outline_color if isinstance(self.typeset_outline_color, QColor) else QColor(self.typeset_outline_color)
        color = QColorDialog.getColor(current if current.isValid() else QColor('#000000'), self)
        if color.isValid():
            self.typeset_outline_color = color
            self._update_outline_color_button()
            self._persist_typeset_preferences()
            self._update_typeset_preview()

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

        mode = self.selection_mode_combo.currentText()
        self.is_processing_selection = True
        try:
            if not self.current_image_pil: return
            unzoomed_rect = self.unzoom_coords(selection_rect)
            if not unzoomed_rect or unzoomed_rect.width() <= 0 or unzoomed_rect.height() <= 0: return

            if "Manual Text" in mode:
                self.start_manual_input(rect=unzoomed_rect)
                return

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
        
        mode = self.selection_mode_combo.currentText()
        result = self.unzoom_coords(scaled_points)
        if not result: return
        unzoomed_polygon, unzoomed_bbox = result

        if "Manual Text" in mode:
            if not unzoomed_bbox or unzoomed_bbox.width() <= 0 or unzoomed_bbox.height() <= 0:
                return
            self.start_manual_input(rect=unzoomed_bbox, polygon=unzoomed_polygon)
            return
        
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
        
    def start_manual_input(self, rect, polygon=None):
        if rect is None:
            return
        current_settings = self.get_current_settings()
        dialog = ManualTextDialog(default_inpaint=current_settings.get('use_inpaint', True), parent=self)
        if dialog.exec_() != QDialog.Accepted:
            self.statusBar().showMessage("Manual text cancelled.", 2500)
            self.image_label.clear_selection()
            return

        manual_text = dialog.get_text().strip()
        if not manual_text:
            self.statusBar().showMessage("Manual text cancelled (empty input).", 2500)
            self.image_label.clear_selection()
            return

        manual_inpaint = dialog.use_inpainting()
        manual_rect = QRect(rect)
        manual_area = self._create_typeset_area(
            manual_rect,
            manual_text,
            current_settings,
            polygon=polygon,
            original_text="Manual Input",
            manual_inpaint=manual_inpaint,
            is_manual=True
        )

        current_key = self.get_current_data_key()
        self.typeset_areas.append(manual_area)
        self.set_selected_area(manual_area, notify=True)
        self.redo_stack.clear()
        if current_key:
            image_record = self.all_typeset_data.setdefault(current_key, {'areas': self.typeset_areas, 'redo': []})
            image_record['areas'] = self.typeset_areas
            image_record['redo'].clear()

        record = self.register_history_entry(current_key, manual_area, "Manual Input", manual_text)
        if record is not None:
            record['manual'] = True

        self.redraw_all_typeset_areas()
        self.update_undo_redo_buttons_state()
        self.refresh_history_views()
        self.statusBar().showMessage("Manual text added.", 3000)
        self.image_label.clear_selection()
        self.update_selection_action_buttons(False)

    def set_transform_preview_active(self, active: bool):
        """Toggle lightweight rendering mode while the user drags or rotates a text area."""
        active = bool(active)
        previous = self.is_transform_preview
        if previous == active:
            return
        self.is_transform_preview = active
        if active:
            if not self._prepare_transform_preview_base():
                self._transform_preview_pixmap = None
        else:
            self._transform_preview_pixmap = None
        try:
            if active or previous:
                self.schedule_typeset_redraw(0)
        except Exception:
            pass

    def redraw_all_typeset_areas(self):
        if not self.original_pixmap: return
        # Protect pixmap assignment and painting from concurrent access
        if hasattr(self, 'deferred_typeset_timer'):
            try:
                self.deferred_typeset_timer.stop()
            except Exception:
                pass
        self.paint_mutex.lock()
        try:
            use_preview = (
                self.is_transform_preview
                and self._transform_preview_pixmap is not None
                and self.selected_typeset_area in self.typeset_areas
            )

            if use_preview:
                base_pixmap = self._transform_preview_pixmap
                if (
                    base_pixmap is None
                    or base_pixmap.isNull()
                    or base_pixmap.size() != self.original_pixmap.size()
                ):
                    if not self._prepare_transform_preview_base():
                        base_pixmap = None
                    else:
                        base_pixmap = self._transform_preview_pixmap
                if base_pixmap is not None:
                    self.typeset_pixmap = base_pixmap.copy()
                    painter = QPainter(self.typeset_pixmap)
                    try:
                        self.draw_single_area(painter, self.selected_typeset_area, self.current_image_pil)
                    finally:
                        try:
                            painter.end()
                        except Exception:
                            pass
                else:
                    use_preview = False

            if not use_preview:
                self.typeset_pixmap = self.original_pixmap.copy()
                painter = QPainter(self.typeset_pixmap)
                try:
                    for area in self.typeset_areas:
                        self.draw_single_area(painter, area, self.current_image_pil)
                finally:
                    try:
                        painter.end()
                    except Exception:
                        pass
        finally:
            self.paint_mutex.unlock()
        self.update_display()

    def _prepare_transform_preview_base(self):
        """Render all areas except the selected one into a cached pixmap for smoother previews."""
        selected = self.selected_typeset_area
        if not self.original_pixmap or not selected:
            self._transform_preview_pixmap = None
            return False
        if selected not in self.typeset_areas:
            self._transform_preview_pixmap = None
            return False

        self.paint_mutex.lock()
        try:
            base_pixmap = self.original_pixmap.copy()
            painter = QPainter(base_pixmap)
            try:
                for area in self.typeset_areas:
                    if area is selected:
                        continue
                    self.draw_single_area(painter, area, self.current_image_pil, for_saving=True)
            finally:
                try:
                    painter.end()
                except Exception:
                    pass
            self._transform_preview_pixmap = base_pixmap
            return True
        except Exception as exc:
            self._transform_preview_pixmap = None
            print(f"Failed to prepare transform preview base: {exc}")
            return False
        finally:
            self.paint_mutex.unlock()

    def schedule_typeset_redraw(self, delay_ms=30):
        try:
            timer = self.deferred_typeset_timer
        except AttributeError:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self.redraw_all_typeset_areas)
            self.deferred_typeset_timer = timer
        try:
            now = time.monotonic()
            if now - getattr(self, '_last_redraw_request', 0.0) < 0.01:
                delay_ms = max(delay_ms, 45)
            self._last_redraw_request = now
            delay = max(1, int(delay_ms))
            if timer.isActive():
                remaining = timer.remainingTime()
                if remaining > 0 and remaining <= delay:
                    return
            timer.start(delay)
        except Exception:
            self.redraw_all_typeset_areas()

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

    def _auto_text_color_for_base(self, base_color: QColor) -> QColor:
        """Return white or black based on luminance threshold (128).

        Use formula: brightness = 0.299*R + 0.587*G + 0.114*B
        """
        try:
            if not isinstance(base_color, QColor) or not base_color.isValid():
                return QColor(0, 0, 0)
            r = base_color.red()
            g = base_color.green()
            b = base_color.blue()
            brightness = 0.299 * r + 0.587 * g + 0.114 * b
            threshold = 128
            try:
                threshold = int(SETTINGS.get('cleanup', {}).get('text_color_threshold', 128))
            except Exception:
                threshold = 128
            if brightness < threshold:
                return QColor(255, 255, 255)
            return QColor(0, 0, 0)
        except Exception:
            return QColor(0, 0, 0)

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

    def draw_single_area(self, painter, area, source_pil_image, for_saving=False):
        try:
            self._draw_single_area_impl(painter, area, source_pil_image, for_saving=for_saving)
        except Exception:
            traceback.print_exc()

    def _draw_single_area_impl(self, painter, area, source_pil_image, for_saving=False):
        settings = self.get_current_settings()
        notes = area.review_notes if isinstance(getattr(area, "review_notes", {}), dict) else {}

        default_inpaint = self._default_cleanup_value('use_inpaint')
        default_background_box = self._default_cleanup_value('use_background_box')

        manual_inpaint_override = notes.get("manual_inpaint")
        if manual_inpaint_override is not None:
            use_inpaint = bool(manual_inpaint_override)
        else:
            use_inpaint = bool(area.get_override('use_inpaint', default_inpaint))
        use_box = bool(area.get_override('use_background_box', default_background_box))

        img_width, img_height = source_pil_image.size

        rect = area.rect
        safe_x = max(0, min(rect.x(), img_width - 1))
        safe_y = max(0, min(rect.y(), img_height - 1))
        safe_width = max(1, min(rect.width(), img_width - safe_x))
        safe_height = max(1, min(rect.height(), img_height - safe_y))
        if safe_width <= 0 or safe_height <= 0:
            return
        area.rect = QRect(safe_x, safe_y, safe_width, safe_height)

        if getattr(area, 'polygon', None):
            clamped_points = []
            for pt in area.polygon:
                clamped_points.append(
                    QPoint(
                        max(0, min(pt.x(), img_width - 1)),
                        max(0, min(pt.y(), img_height - 1))
                    )
                )
            if clamped_points:
                area.polygon = QPolygon(clamped_points)

        skip_heavy_cleanup = bool(self.is_transform_preview and not for_saving)
        if skip_heavy_cleanup:
            use_inpaint = False
            self._draw_preview_area(painter, area, use_box)
            return

        cv_original = cv2.cvtColor(np.array(source_pil_image), cv2.COLOR_RGB2BGR)
        img_height, img_width = cv_original.shape[:2]

        # 1. Buat mask dari bentuk area (polygon atau rectangle)

        base_mask = np.zeros(cv_original.shape[:2], dtype=np.uint8)
        if area.polygon:
            cv_poly_points = np.array([[p.x(), p.y()] for p in area.polygon], dtype=np.int32)
            cv2.fillPoly(base_mask, [cv_poly_points], 255)
        else:
            cv2.rectangle(base_mask,
                        (area.rect.x(), area.rect.y()),
                        (area.rect.right(), area.rect.bottom()),
                        255, -1)

        # 2. Gunakan bubble detector
        if skip_heavy_cleanup:
            bubble_mask = None
            use_inpaint = False
        else:
            bubble_mask = self.find_speech_bubble_mask(cv_original, area.rect, settings, for_saving=for_saving)

        # 3. Gabungkan mask area dengan mask bubble
        combined_mask = cv2.bitwise_and(base_mask, bubble_mask) if bubble_mask is not None else base_mask

        # 4. Tambahkan padding
        padding = settings['inpaint_padding']
        kernel = np.ones((max(1, padding), max(1, padding)), np.uint8)
        final_inpaint_mask = cv2.dilate(combined_mask, kernel, iterations=1)

        # 5. Proses Inpainting
        if use_inpaint:
            inpainted_cv = None

            # Coba advanced inpainting dengan LaMa
            if settings['inpaint_model_key'] and self.inpaint_model:
                try:
                    if (self.inpaint_model is None or
                        self.current_inpaint_model_key != settings['inpaint_model_key']):
                        self.initialize_inpaint_engine()

                    if self.inpaint_model:
                        pil_original = Image.fromarray(cv2.cvtColor(cv_original, cv2.COLOR_BGR2RGB))
                        pil_mask = Image.fromarray((final_inpaint_mask > 0).astype(np.uint8) * 255).convert("L")

                        out_arr = self.inpaint_model(pil_original, pil_mask)
                        if out_arr is None:
                            raise RuntimeError("LaMa inpaint returned None")

                        if isinstance(out_arr, np.ndarray):
                            if out_arr.shape[2] == 3:
                                inpainted_cv = cv2.cvtColor(out_arr, cv2.COLOR_BGR2RGB)
                                inpainted_cv = cv2.cvtColor(inpainted_cv, cv2.COLOR_RGB2BGR)
                            else:
                                inpainted_cv = out_arr
                        else:
                            try:
                                inpainted_cv = cv2.cvtColor(np.array(out_arr.convert("RGB")), cv2.COLOR_RGB2BGR)
                            except Exception:
                                inpainted_cv = None

                except Exception as e:
                    print(f"Advanced inpainting failed: {e}")
                    inpainted_cv = None

            # Fallback ke OpenCV
            if inpainted_cv is None:
                try:
                    algo_map = {"OpenCV-NS": cv2.INPAINT_NS, "OpenCV-Telea": cv2.INPAINT_TELEA}
                    algo = algo_map.get(settings.get('inpaint_model_name', 'OpenCV-NS'), cv2.INPAINT_NS)
                    inpaint_mask_for_cv = (final_inpaint_mask > 0).astype(np.uint8) * 255
                    inpainted_cv = cv2.inpaint(cv_original, inpaint_mask_for_cv, 3, algo)
                except Exception as e:
                    print(f"OpenCV inpainting also failed: {e}")
                    background_color = self.get_background_color(cv_original, area.rect)
                    use_box_global = bool(area.get_override('use_background_box', default_background_box))
                    if use_box_global:
                        painter.save()
                        path = QPainterPath()
                        contours, _ = cv2.findContours(final_inpaint_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in contours:
                            polygon = QPolygonF([QPoint(p[0][0], p[0][1]) for p in cnt])
                            path.addPolygon(polygon)
                        painter.setClipPath(path)
                        painter.fillRect(painter.window(), background_color)
                        painter.restore()
                        return

            # Gambar hasil inpainting
            if inpainted_cv is not None:
                if inpainted_cv.dtype != np.uint8:
                    inpainted_cv = (np.clip(inpainted_cv, 0, 255)).astype(np.uint8)

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
            if use_box:
                painter.fillRect(painter.window(), background_color)
            painter.restore()

        # 6. Render bubble (optional) dan text
        rotation = area.get_rotation() if hasattr(area, 'get_rotation') else float(getattr(area, 'rotation', 0.0))
        center = area.rect.center()

        if getattr(area, 'bubble_enabled', False) and use_box:
            painter.save()
            if abs(rotation) > 0.01:
                painter.translate(center.x(), center.y())
                painter.rotate(rotation)
                painter.translate(-center.x(), -center.y())
            self.draw_area_bubble(painter, area)
            painter.restore()

        # Tentukan warna teks otomatis
        use_auto = bool(SETTINGS.get('cleanup', {}).get('auto_text_color', True))

        # Tetapkan/bersihkan auto text color sesuai setting
        if use_auto:
            if not use_box:
                bg_col = self.get_background_color(cv_original, area.rect)
                area._auto_text_color = self._auto_text_color_for_base(bg_col)
            else:
                bubble_fill = area.get_bubble_fill_color() if getattr(area, 'bubble_enabled', False) else None
                if bubble_fill and bubble_fill.isValid():
                    area._auto_text_color = self._auto_text_color_for_base(bubble_fill)
                else:
                    bg_col = self.get_background_color(cv_original, area.rect)
                    area._auto_text_color = self._auto_text_color_for_base(bg_col)
        else:
            if hasattr(area, '_auto_text_color'):
                delattr(area, '_auto_text_color')

        painter.save()
        if abs(rotation) > 0.01:
            painter.translate(center.x(), center.y())
            painter.rotate(rotation)
            painter.translate(-center.x(), -center.y())
        if not use_box:
            self.draw_area_text_plain(painter, area)
        else:
            self.draw_area_text(painter, area)
        painter.restore()

    def _draw_preview_area(self, painter, area, use_box):
        rotation = area.get_rotation() if hasattr(area, 'get_rotation') else float(getattr(area, 'rotation', 0.0))
        center = area.rect.center()

        painter.save()
        try:
            if getattr(area, 'bubble_enabled', False) and use_box:
                painter.save()
                try:
                    if abs(rotation) > 0.01:
                        painter.translate(center.x(), center.y())
                        painter.rotate(rotation)
                        painter.translate(-center.x(), -center.y())
                    self.draw_area_bubble(painter, area)
                finally:
                    painter.restore()

            painter.save()
            try:
                if abs(rotation) > 0.01:
                    painter.translate(center.x(), center.y())
                    painter.rotate(rotation)
                    painter.translate(-center.x(), -center.y())

                if use_box:
                    self.draw_area_text(painter, area)
                else:
                    self.draw_area_text_plain(painter, area)
            finally:
                painter.restore()
        finally:
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

    def _ideal_outline_color(self, base_color: QColor) -> QColor:
        if not isinstance(base_color, QColor) or not base_color.isValid():
            return QColor(0, 0, 0, 220)
        luminance = 0.299 * base_color.red() + 0.587 * base_color.green() + 0.114 * base_color.blue()
        if luminance > 160:
            return QColor(0, 0, 0, 220)
        return QColor(255, 255, 255, 220)

    def _outline_for_text_color(self, text_color: QColor) -> QColor:
        """Choose outline color based on the text color using luminance rules.

        - If text is pure white, outline black.
        - If text is pure black, outline white.
        - Otherwise compute luminance (0..1). If luminance > 0.5 -> dark outline; else light outline.
        """
        try:
            if not isinstance(text_color, QColor) or not text_color.isValid():
                return QColor(0, 0, 0, 220)
            r = text_color.red() / 255.0
            g = text_color.green() / 255.0
            b = text_color.blue() / 255.0
            # special cases
            if int(text_color.red()) == 255 and int(text_color.green()) == 255 and int(text_color.blue()) == 255:
                return QColor(0, 0, 0, 220)
            if int(text_color.red()) == 0 and int(text_color.green()) == 0 and int(text_color.blue()) == 0:
                return QColor(255, 255, 255, 220)
            # calculate luminance per Rec. 709
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            if luminance > 0.5:
                # bright text => dark outline
                return QColor(0, 0, 0, 220)
            else:
                return QColor(255, 255, 255, 220)
        except Exception:
            return QColor(0, 0, 0, 220)

    def _expand_with_outline(self, image: QImage, outline_color: QColor | None = None, radius: float = 2.0) -> QImage:
        try:
            radius = float(radius)
        except Exception:
            radius = 2.0
        if image.isNull() or radius <= 0.0:
            return image
        base = image
        if image.format() != QImage.Format_ARGB32_Premultiplied:
            base = image.convertToFormat(QImage.Format_ARGB32_Premultiplied)
        outline_color = outline_color or QColor(0, 0, 0, 220)
        if not outline_color.isValid():
            outline_color = QColor(0, 0, 0, 220)
        # Create an image filled with the outline color, then use composition
        # to apply the base image's alpha channel to that colored image.
        # Using CompositionMode_DestinationIn will keep destination pixels
        # only where the source (base) has alpha, effectively applying
        # the alpha mask without relying on alphaChannel()/setAlphaChannel().
        outline = QImage(base.size(), QImage.Format_ARGB32_Premultiplied)
        outline.fill(outline_color.rgba())
        comp = QPainter(outline)
        try:
            comp.setCompositionMode(QPainter.CompositionMode_DestinationIn)
            comp.drawImage(0, 0, base)
        finally:
            comp.end()
        radius_px = max(1, int(math.ceil(radius)))
        expanded = QImage(base.width() + radius_px * 2, base.height() + radius_px * 2, QImage.Format_ARGB32_Premultiplied)
        expanded.fill(Qt.transparent)
        painter = QPainter(expanded)
        offsets = [
            QPoint(dx, dy)
            for dx in range(-radius_px, radius_px + 1)
            for dy in range(-radius_px, radius_px + 1)
            if (dx != 0 or dy != 0) and math.hypot(dx, dy) <= radius + 0.25
        ]
        for offset in offsets:
            painter.drawImage(radius_px + offset.x(), radius_px + offset.y(), outline)
        painter.drawImage(radius_px, radius_px, base)
        painter.end()
        return expanded

    def _render_text_glyph(self, painter: QPainter, char: str, font: QFont, color: QColor | str, position: QPointF, area=None):
        if not char:
            return
        if char.isspace():
            return
        qcolor = QColor(color) if not isinstance(color, QColor) else QColor(color)
        if not qcolor.isValid():
            qcolor = QColor('#000000')
        path = QPainterPath()
        path.addText(position, font, char)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        if area is not None and getattr(area, 'has_text_outline', None) and area.has_text_outline():
            outline_color = area.get_text_outline_color()
            if not outline_color.isValid():
                outline_color = self._outline_for_text_color(qcolor)
            outline_pen = QPen(outline_color)
            width = max(0.2, float(area.get_text_outline_width()))
            outline_pen.setWidthF(width)
            outline_pen.setJoinStyle(Qt.RoundJoin)
            outline_pen.setCapStyle(Qt.RoundCap)
            outline_pen.setCosmetic(True)
            painter.strokePath(path, outline_pen)
        painter.fillPath(path, QBrush(qcolor))

    def draw_area_text_plain(self, painter: QPainter, area):
        """Render text directly on the image without drawing any background boxes or fills.

        This uses QPainter.drawText (via glyph rendering) and respects area._auto_text_color for
        overriding segment colors when present.
        """
        rect = QRectF(area.rect)
        margins = area.get_margins()
        rect = rect.adjusted(margins['left'], margins['top'], -margins['right'], -margins['bottom'])
        if rect.width() <= 0 or rect.height() <= 0:
            return

        # Flatten segments into lines similar to _flatten_segments_to_lines but use the effective color
        lines = []
        current_line = []
        for segment in area.get_segments():
            text_value = segment.get('text', '') or ''
            seg_font = area.segment_to_qfont(segment)
            seg_color = area.segment_to_color(segment)
            use_auto = bool(SETTINGS.get('cleanup', {}).get('auto_text_color', True))
            if use_auto and hasattr(area, '_auto_text_color') and isinstance(area._auto_text_color, QColor):
                seg_color = area._auto_text_color

            # iterate chars
            for ch in text_value:
                if ch == '\n':
                    lines.append(current_line)
                    current_line = []
                    continue
                glyph_font = QFont(seg_font)
                glyph_font.setLetterSpacing(QFont.PercentageSpacing, area.get_char_spacing())
                current_line.append({'char': ch, 'font': glyph_font, 'color': QColor(seg_color)})
        if current_line:
            lines.append(current_line)

        if not lines:
            return

        # Compute line metrics and draw each glyph sequentially
        metrics = self._compute_line_metrics(lines, area)
        total_height = sum(m['height'] for m in metrics)
        y_offset = rect.top() + max(0.0, (rect.height() - total_height) / 2.0)
        baseline = y_offset + (metrics[0]['ascent'] if metrics else 0.0)
        alignment = area.get_alignment()

        for idx, glyphs in enumerate(lines):
            if not glyphs:
                baseline += metrics[idx]['height']
                continue
            total_width = sum(QFontMetrics(g['font']).horizontalAdvance(g['char']) for g in glyphs)
            if alignment == 'left':
                x_start = rect.left()
            elif alignment == 'right':
                x_start = rect.right() - total_width
            else:
                x_start = rect.left() + (rect.width() - total_width) / 2.0

            x = x_start
            for g in glyphs:
                ch = g['char']
                f = g['font']
                color = g['color']
                # Render glyph: we use _render_text_glyph which respects outline if enabled
                self._render_text_glyph(painter, ch, f, color, QPointF(x, baseline), area)
                x += QFontMetrics(f).horizontalAdvance(ch)

            baseline += metrics[idx]['height']

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
            # Allow area to inject an override color (auto-detection) via attribute
            seg_color = area.segment_to_color(segment)
            use_auto = bool(SETTINGS.get('cleanup', {}).get('auto_text_color', True))
            if use_auto and hasattr(area, '_auto_text_color') and isinstance(area._auto_text_color, QColor):
                seg_color = area._auto_text_color

            fmt.setForeground(QBrush(seg_color))
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

        if area and getattr(area, 'has_text_outline', None) and area.has_text_outline():
            # Try to choose a representative text color for the document: prefer the first styled segment color if present
            doc_color = None
            try:
                segs = area.get_segments() or []
                for seg in segs:
                    c = seg.get('color')
                    if c:
                        doc_color = QColor(c) if not isinstance(c, QColor) else c
                        break
            except Exception:
                doc_color = None
            if doc_color is None or not getattr(doc_color, 'isValid', lambda: False)():
                doc_color = area.get_color()
            outline_color = area.get_text_outline_color() if hasattr(area, 'get_text_outline_color') else QColor('#000000')
            if not outline_color.isValid():
                outline_color = self._outline_for_text_color(doc_color)
            radius = max(1, int(math.ceil(area.get_text_outline_width() if hasattr(area, 'get_text_outline_width') else 2.0)))
            image = self._expand_with_outline(image, outline_color, radius=radius)

        pixmap = QPixmap.fromImage(image)
        painter.translate(rect.center())
        if orientation == 'vertical':
            transform = QTransform()
            transform.rotate(90)
            pixmap = pixmap.transformed(transform, Qt.SmoothTransformation)
        painter.translate(-pixmap.width() / 2, -pixmap.height() / 2)
        painter.drawPixmap(0, 0, pixmap)


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
                self._draw_wavy_line(painter, rect, glyphs, baseline, intensity, alignment, area)
            elif effect == 'jagged':
                self._draw_jagged_line(painter, rect, glyphs, baseline, intensity, alignment, area)
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
            self._render_text_glyph(
                painter,
                glyph.get('char', ''),
                glyph.get('font', QFont()),
                glyph.get('color', QColor('#000000')),
                QPointF(-advance / 2.0, 0),
                area
            )
            painter.restore()

            progress += advance

    def _draw_wavy_line(self, painter, rect, glyphs, baseline, intensity, alignment, area):
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
            self._render_text_glyph(
                painter,
                glyph.get('char', ''),
                glyph.get('font', QFont()),
                glyph.get('color', QColor('#000000')),
                QPointF(current_x, baseline + wave_offset),
                area
            )
            current_x += advance

    def _draw_jagged_line(self, painter, rect, glyphs, baseline, intensity, alignment, area):
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
            self._render_text_glyph(
                painter,
                glyph.get('char', ''),
                bold_font,
                glyph.get('color', QColor('#000000')),
                QPointF(0, 0),
                area
            )
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
        manual_polygon = "Manual Text (Pen)" in mode
        pen_mode = (mode == "Pen Tool") or manual_polygon
        brush_mode = (mode == "Brush Inpaint (LaMA)")
        rect_mode = ("Rect" in mode or "Oval" in mode) and not manual_polygon
        transform_mode = (mode == "Transform (Hand)")
        self.image_label.set_transform_mode(transform_mode)
        # Use an explicit pen cursor for pen mode so it's visibly distinct
        if pen_mode:
            if not self.pen_cursor:
                self.pen_cursor = self.create_pen_cursor()
            self.image_label.setCursor(self.pen_cursor)
        elif brush_mode:
            self.image_label.clear_brush_mask()
            self.image_label.update_brush_cursor()
        elif transform_mode:
            self.image_label.setCursor(Qt.OpenHandCursor)
        else:
            self.image_label.setCursor(Qt.CrossCursor if rect_mode else Qt.PointingHandCursor)

        if brush_mode:
            self.selection_confirm_button.setText("Confirm Cleanup")
            self.selection_cancel_button.setText("Reset Brush")
        else:
            self.selection_confirm_button.setText("Confirm")
            self.selection_cancel_button.setText("Cancel")

        self.update_selection_action_buttons(pen_mode or brush_mode)
        if not brush_mode:
            self.image_label.stop_brush_session()
        if not pen_mode:
            self.image_label.polygon_points.clear()
            self.image_label.update()

    def create_pen_cursor(self):
        """Create a small stylized pen/pencil QCursor to differentiate pen mode."""
        # Create a compact pencil cursor ~20x20 with the hotspot at the pencil tip
        size = 20
        pm = QPixmap(size, size)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)

        # pencil shaft
        shaft_color = QColor(80, 60, 40)
        lead_color = QColor(40, 40, 40)
        wood_color = QColor(220, 170, 110)
        metal_color = QColor(180, 180, 180)

        p.setPen(Qt.NoPen)
        # shaft rectangle (angled by drawing a polygon)
        shaft = QPolygon([
            QPoint(3, 5), QPoint(13, 3), QPoint(15, 6), QPoint(5, 8)
        ])
        p.setBrush(shaft_color)
        p.drawPolygon(shaft)

        # wood + lead tip
        tip = QPolygon([
            QPoint(13, 3), QPoint(17, 6), QPoint(15, 6)
        ])
        p.setBrush(wood_color)
        p.drawPolygon(QPolygon([QPoint(13,3), QPoint(16,5), QPoint(15,6)]))
        p.setBrush(lead_color)
        p.drawEllipse(QRect(15,5,2,2))

        # small metal band
        p.setBrush(metal_color)
        p.drawRect(11,4,3,3)

        # outline for clarity
        p.setPen(QPen(QColor(20, 20, 20), 1))
        p.setBrush(Qt.NoBrush)
        p.drawPolygon(shaft)
        p.end()

        # hotspot near the tip (right-most point)
        hotspot_x = 16
        hotspot_y = 6
        return QCursor(pm, hotspot_x, hotspot_y)

    def update_selection_action_buttons(self, visible):
        self.selection_confirm_button.setVisible(visible)
        self.selection_cancel_button.setVisible(visible)
        if not visible:
            self.selection_confirm_button.setEnabled(False)
            return
        if self.selection_mode_combo.currentText() == "Brush Inpaint (LaMA)":
            self.on_brush_mask_updated()
        else:
            self.selection_confirm_button.setEnabled(True)

    def _confirm_selection_action(self):
        mode = self.selection_mode_combo.currentText()
        if mode == "Pen Tool" or "Manual Text (Pen)" in mode:
            self.confirm_pen_selection()
        elif mode == "Brush Inpaint (LaMA)":
            self.confirm_brush_inpaint()

    def _cancel_selection_action(self):
        mode = self.selection_mode_combo.currentText()
        if mode == "Pen Tool" or "Manual Text (Pen)" in mode:
            self.cancel_pen_selection()
        elif mode == "Brush Inpaint (LaMA)":
            self.cancel_brush_inpaint()
        else:
            self.update_selection_action_buttons(False)

    def on_brush_mask_updated(self):
        if self.selection_mode_combo.currentText() != "Brush Inpaint (LaMA)":
            return
        has_mask = False
        if hasattr(self, 'image_label') and self.image_label:
            has_mask = self.image_label.has_brush_mask()
        self.selection_confirm_button.setEnabled(has_mask)

    def _on_brush_size_slider_changed(self, value):
        if getattr(self, 'brush_size_spin', None):
            with QSignalBlocker(self.brush_size_spin):
                self.brush_size_spin.setValue(value)
        if getattr(self, 'image_label', None):
            self.image_label.set_brush_size(value)
            self.image_label.update_brush_cursor()

    def _on_brush_size_spin_changed(self, value):
        if getattr(self, 'brush_size_slider', None):
            with QSignalBlocker(self.brush_size_slider):
                self.brush_size_slider.setValue(value)
        self._on_brush_size_slider_changed(value)

    def confirm_pen_selection(self):
        points = self.image_label.get_polygon_points()
        if len(points) < 3: QMessageBox.warning(self, "Invalid Shape", "Please select at least 3 points."); return
        self.process_polygon_area(points); self.image_label.clear_selection()

    def cancel_pen_selection(self):
        self.image_label.clear_selection()
        self.update_selection_action_buttons(False)

    def confirm_brush_inpaint(self):
        if not getattr(self, 'image_label', None):
            return
        mask_image = self.image_label.export_brush_mask()
        if mask_image is None:
            QMessageBox.warning(self, "Mask Kosong", "Buat mask terlebih dahulu dengan Brush Inpaint sebelum mengirim.")
            return
        mask_for_save = mask_image.copy()
        if self.current_image_pil is None:
            QMessageBox.warning(self, "Tidak Ada Gambar", "Tidak ada gambar aktif untuk diproses.")
            return

        try:
            mask_buffer = QBuffer()
            if not mask_buffer.open(QIODevice.WriteOnly):
                raise RuntimeError("Mask buffer tidak bisa dibuka.")
            if not mask_image.save(mask_buffer, "PNG"):
                raise RuntimeError("Mask tidak bisa dikonversi menjadi PNG.")
            mask_bytes = bytes(mask_buffer.data())
        except Exception as exc:
            QMessageBox.critical(self, "Mask Error", f"Gagal menyiapkan mask untuk dikirim:\n{exc}")
            return

        try:
            image_buffer = io.BytesIO()
            self.current_image_pil.save(image_buffer, format="PNG")
            image_bytes = image_buffer.getvalue()
        except Exception as exc:
            QMessageBox.critical(self, "Image Error", f"Gagal menyiapkan gambar asli:\n{exc}")
            return

        cleanup_cfg = SETTINGS.get('cleanup', {})
        endpoint = cleanup_cfg.get('lama_endpoint') or DEFAULT_BIG_LAMA_ENDPOINT
        model_name = cleanup_cfg.get('lama_model') or DEFAULT_BIG_LAMA_MODEL

        files = {
            'image': ('image.png', image_bytes, 'image/png'),
            'mask': ('mask.png', mask_bytes, 'image/png')
        }
        # lama_cleaner expects many form fields. If they're missing, the server will return 400 (Bad Request)
        # Provide sensible defaults mirroring the frontend defaults so the endpoint accepts the request.
        data = {
            'model': model_name or DEFAULT_BIG_LAMA_MODEL,
            # LDM / HD settings
            'ldmSteps': '25',
            'ldmSampler': 'plms',
            'zitsWireframe': 'true',
            'hdStrategy': 'Crop',
            'hdStrategyCropMargin': '128',
            # Note: server uses the misspelled key 'hdStrategyCropTrigerSize'
            'hdStrategyCropTrigerSize': '512',
            'hdStrategyResizeLimit': '1080',
            # Prompts
            'prompt': '',
            'negativePrompt': '',
            # Cropper
            'useCroper': 'false',
            'croperX': '0',
            'croperY': '0',
            'croperHeight': '0',
            'croperWidth': '0',
            # SD params
            'sdScale': '100',
            'sdMaskBlur': '5',
            'sdStrength': '0.75',
            'sdSteps': '50',
            'sdGuidanceScale': '7.5',
            'sdSampler': 'uni_pc',
            'sdSeed': '42',
            'sdMatchHistograms': 'false',
            # OpenCV inpaint flags
            'cv2Flag': '1',
            'cv2Radius': '5',
            # Paint by example / p2p
            'paintByExampleSteps': '50',
            'paintByExampleGuidanceScale': '7.5',
            'paintByExampleMaskBlur': '5',
            'paintByExampleSeed': '42',
            'paintByExampleMatchHistograms': 'false',
            'p2pSteps': '50',
            'p2pImageGuidanceScale': '1.5',
            'p2pGuidanceScale': '7.5',
            # Controlnet
            'controlnet_conditioning_scale': '0.4',
            'controlnet_method': ''
        }

        prev_enabled = self.selection_confirm_button.isEnabled()
        self.selection_confirm_button.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        response = None
        error = None
        try:
            response = requests.post(endpoint, files=files, data=data, timeout=120)
            response.raise_for_status()
        except Exception as exc:
            error = exc
        finally:
            QApplication.restoreOverrideCursor()

        if error:
            self.selection_confirm_button.setEnabled(prev_enabled)
            QMessageBox.critical(self, "Inpaint Gagal", f"Gagal menghubungi Big-LaMA ({endpoint}):\n{error}")
            return

        content_type = (response.headers.get('Content-Type') or '').lower()
        output_ext = ".png"
        image_bytes = b''

        if 'application/json' in content_type:
            try:
                payload = response.json()
            except Exception as exc:
                self.selection_confirm_button.setEnabled(prev_enabled)
                QMessageBox.critical(self, "Respon Tidak Valid", f"Gagal membaca respon JSON dari server:\n{exc}")
                return
            image_b64 = None
            for key in ('image', 'result', 'data', 'png', 'output'):
                value = payload.get(key)
                if isinstance(value, str) and value:
                    image_b64 = value
                    break
            if not image_b64:
                self.selection_confirm_button.setEnabled(prev_enabled)
                QMessageBox.critical(self, "Respon Tidak Valid", "Respon JSON tidak mengandung field gambar (image/result/data).")
                return
            try:
                image_bytes = base64.b64decode(image_b64)
            except Exception as exc:
                self.selection_confirm_button.setEnabled(prev_enabled)
                QMessageBox.critical(self, "Decode Error", f"Gagal decode data gambar dari server:\n{exc}")
                return
        else:
            image_bytes = response.content or b''
            if 'jpeg' in content_type or 'jpg' in content_type:
                output_ext = ".jpg"

        if not image_bytes:
            self.selection_confirm_button.setEnabled(prev_enabled)
            QMessageBox.critical(self, "Inpaint Gagal", "Server tidak mengembalikan data gambar.")
            return

        saved_mask_path, saved_cleanup_path = self._save_brush_inpaint_artifacts(mask_for_save, image_bytes, output_ext)
        project_copy_path = self._write_inpaint_result_copy(image_bytes, output_ext)
        project_mask_copy_path = self._copy_mask_to_project(saved_mask_path) if saved_mask_path else None
        status_parts = []
        if saved_mask_path:
            status_parts.append(f"Mask: {saved_mask_path}")
        if saved_cleanup_path:
            status_parts.append(f"Temp cleanup: {saved_cleanup_path}")
        if project_copy_path:
            status_parts.append(f"Inpainting copy: {project_copy_path}")
        if project_mask_copy_path:
            status_parts.append(f"Mask copy: {project_mask_copy_path}")
        if status_parts:
            self.statusBar().showMessage(" | ".join(status_parts), 6000)
        else:
            self.statusBar().showMessage("Brush inpainting completed.", 5000)

        # Decide whether to save to workshop or apply in-memory
        save_to_workshop = bool(SETTINGS.get('cleanup', {}).get('lama_save_to_workshop', False))

        if save_to_workshop:
            workshop_dir = self._ensure_workshop_dir()
            original_name = "inpaint"
            if self.current_image_path:
                original_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            base_name = f"{original_name}_cleanup"
            output_path = os.path.join(workshop_dir, base_name + output_ext)
            counter = 1
            while os.path.exists(output_path):
                output_path = os.path.join(workshop_dir, f"{base_name}_{counter}{output_ext}")
                counter += 1

            try:
                with open(output_path, 'wb') as fout:
                    fout.write(image_bytes)
            except Exception as exc:
                self.selection_confirm_button.setEnabled(prev_enabled)
                QMessageBox.critical(self, "Save Error", f"Gagal menyimpan hasil inpaint:\n{exc}")
                return

            self.image_label.clear_brush_mask()
            workshop_msg = f"Hasil LaMA Cleaner tersimpan di:\n{output_path}"
            status_msg = f"Hasil inpainting disimpan di {output_path}"
            if saved_mask_path:
                workshop_msg += f"\nMask: {saved_mask_path}"
                status_msg += f" | Mask: {saved_mask_path}"
            if saved_cleanup_path and os.path.abspath(saved_cleanup_path) != os.path.abspath(output_path):
                workshop_msg += f"\nSalinan temp: {saved_cleanup_path}"
                status_msg += f" | Salinan temp: {saved_cleanup_path}"
            if project_copy_path and os.path.abspath(project_copy_path) != os.path.abspath(output_path):
                workshop_msg += f"\nSalinan proyek: {project_copy_path}"
                status_msg += f" | Salinan proyek: {project_copy_path}"
            if project_mask_copy_path and os.path.abspath(project_mask_copy_path) != os.path.abspath(output_path):
                workshop_msg += f"\nMask proyek: {project_mask_copy_path}"
                status_msg += f" | Mask proyek: {project_mask_copy_path}"
            self.statusBar().showMessage(status_msg, 6000)
            QMessageBox.information(self, "Inpainting Selesai", workshop_msg)
            self.update_selection_action_buttons(True)
        else:
            # Apply the returned image bytes to current image in-memory and refresh display
            try:
                pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                # Update in-memory image and pixmap safely
                self.paint_mutex.lock()
                try:
                    self.current_image_pil = pil_img
                    data = pil_img.tobytes('raw', 'RGB')
                    qimage = QImage(data, pil_img.width, pil_img.height, pil_img.width * 3, QImage.Format_RGB888)
                    self.original_pixmap = QPixmap.fromImage(qimage)
                    # Reset typeset/overlay state since base image changed
                    self.typeset_areas = []
                    self.redo_stack = []
                    self.set_selected_area(None, notify=True)
                    if hasattr(self, 'image_label') and self.image_label:
                        self.image_label.reset_brush_mask()
                finally:
                    self.paint_mutex.unlock()

                # Refresh display and UI
                self.rebuild_history_for_image(self.get_current_data_key(), self.typeset_areas)
                self.redraw_all_typeset_areas()
                self.update_undo_redo_buttons_state()
                self._refresh_detection_overlay()
                self.refresh_history_views()
                self.image_label.clear_brush_mask()
                status_text = "Inpainting berhasil dan diterapkan pada gambar saat ini."
                dialog_text = status_text
                if saved_mask_path:
                    status_text += f" | Mask: {saved_mask_path}"
                    dialog_text += f"\nMask: {saved_mask_path}"
                if saved_cleanup_path:
                    status_text += f" | Salinan: {saved_cleanup_path}"
                    dialog_text += f"\nSalinan hasil: {saved_cleanup_path}"
                if project_copy_path:
                    status_text += f" | Salinan proyek: {project_copy_path}"
                    dialog_text += f"\nSalinan proyek: {project_copy_path}"
                if project_mask_copy_path:
                    status_text += f" | Mask proyek: {project_mask_copy_path}"
                    dialog_text += f"\nMask proyek: {project_mask_copy_path}"
                self.statusBar().showMessage(status_text, 6000)
                QMessageBox.information(self, "Inpainting Selesai", dialog_text)
                self.update_selection_action_buttons(True)
            except Exception as exc:
                self.selection_confirm_button.setEnabled(prev_enabled)
                QMessageBox.critical(self, "Apply Error", f"Gagal menerapkan hasil inpaint ke memori:\n{exc}")
                return

    def cancel_brush_inpaint(self):
        if not getattr(self, 'image_label', None):
            return
        self.image_label.clear_brush_mask()
        self.statusBar().showMessage("Mask inpainting dibersihkan.", 2500)
        self.update_selection_action_buttons(True)

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
            # Non-blocking save: copy QPixmap to QImage under mutex, then save in background
            self.paint_mutex.lock()
            try:
                pix_copy = self.typeset_pixmap.copy()
                qimage = pix_copy.toImage().copy()
            finally:
                self.paint_mutex.unlock()

            # Start background worker to save the QImage
            image_worker = ImageSaveWorker(qimage, filePath)
            image_thread = QThread()
            image_worker.moveToThread(image_thread)

            def _on_img_started():
                try:
                    image_worker.run()
                except Exception as e:
                    try:
                        image_worker.error.emit(str(e))
                    except Exception:
                        pass

            def _img_finished(success, message):
                try:
                    image_thread.quit(); image_thread.wait()
                except Exception:
                    pass
                self.image_save_thread = None
                self.image_save_worker = None
                if success:
                    QMessageBox.information(self, "Success", message)
                else:
                    QMessageBox.critical(self, "Error", message)

            def _img_error(msg):
                self.statusBar().showMessage(f"Image save error: {msg}", 5000)

            image_worker.finished.connect(_img_finished)
            image_worker.error.connect(_img_error)

            self.image_save_worker = image_worker
            self.image_save_thread = image_thread
            image_thread.started.connect(_on_img_started)
            image_thread.start()

    def delete_typeset_area(self, area_to_delete):
        if area_to_delete in self.typeset_areas:
            self.typeset_areas.remove(area_to_delete)
            if self.selected_typeset_area is area_to_delete:
                self.clear_selected_area()
            self.redo_stack.clear()
            self.redo_stack.append(area_to_delete)
            self.redraw_all_typeset_areas()
            self.update_undo_redo_buttons_state()

    def undo_last_action(self):
        if self.typeset_areas:
            undone_area = self.typeset_areas.pop(); self.redo_stack.append(undone_area)
            if self.selected_typeset_area is undone_area:
                self.clear_selected_area()
            self.redraw_all_typeset_areas(); self.update_undo_redo_buttons_state(); self.image_label.clear_selection()

    def redo_last_action(self):
        if self.redo_stack:
            redone_area = self.redo_stack.pop(); self.typeset_areas.append(redone_area)
            self.set_selected_area(redone_area)
            self.redraw_all_typeset_areas(); self.update_undo_redo_buttons_state(); self.image_label.clear_selection()

    def update_undo_redo_buttons_state(self):
        self.undo_button.setEnabled(len(self.typeset_areas) > 0)
        self.redo_button.setEnabled(len(self.redo_stack) > 0)

    def _snapshot_current_image_state(self):
        if not self.current_image_path:
            return
        current_key = self.get_current_data_key()
        self.all_typeset_data[current_key] = {
            'areas': list(self.typeset_areas),
            'redo': list(self.redo_stack),
        }
    
    def _serialize_typeset_map(self):
        serialized = {}
        for key, payload in self.all_typeset_data.items():
            if not isinstance(payload, dict):
                continue
            areas = payload.get('areas') or []
            redo = payload.get('redo') or []
            serialized[key] = {
                'areas': [area.to_payload() if isinstance(area, TypesetArea) else area for area in areas],
                'redo': [area.to_payload() if isinstance(area, TypesetArea) else area for area in redo],
            }
        return serialized
    
    def _collect_project_settings(self):
        try:
            settings = self.get_current_settings() or {}
        except Exception:
            settings = {}
        serialized = {}
        for key, value in settings.items():
            if isinstance(value, QFont):
                serialized[key] = TypesetArea.font_to_dict(value)
            elif isinstance(value, QColor):
                serialized[key] = value.name()
            elif isinstance(value, (QRect, QRectF)):
                serialized[key] = rect_to_dict(value)
            elif isinstance(value, (QPoint, QPointF)):
                serialized[key] = {'x': coerce_int(value.x()), 'y': coerce_int(value.y())}
            elif isinstance(value, (set, tuple)):
                serialized[key] = list(value)
            else:
                serialized[key] = value
        serialized['cleanup'] = {
            'use_background_box': self._default_cleanup_value('use_background_box'),
            'use_inpaint': self._default_cleanup_value('use_inpaint'),
            'apply_mode': self._default_cleanup_value('apply_mode'),
        }
        return serialized
    
    def _build_project_payload(self):
        self._snapshot_current_image_state()
        payload = {
            'schema_version': 2,
            'project_dir': os.path.abspath(self.project_dir) if self.project_dir else None,
            'current_image_path': self.current_image_path,
            'current_pdf_page': int(self.current_pdf_page) if isinstance(self.current_pdf_page, int) else -1,
            'typeset_data': self._serialize_typeset_map(),
            'history_entries': copy.deepcopy(self.history_entries),
            'proofreader_entries': copy.deepcopy(self.proofreader_entries),
            'quality_entries': copy.deepcopy(self.quality_entries),
            'history_counter': int(self.history_counter),
            'typeset_font': TypesetArea.font_to_dict(self.typeset_font),
            'typeset_color': self.typeset_color.name(),
            'typeset_defaults': copy.deepcopy(self.typeset_defaults),
            'settings': self._collect_project_settings(),
            'saved_at': time.time(),
            'app_version': '16.1.0',
        }
        config_block = {'theme': self.current_theme}
        if getattr(self, 'autosave_timer', None):
            config_block['autosave_interval_ms'] = int(self.autosave_timer.interval())
        payload['config'] = config_block
        return payload
    
    
    def _read_project_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as handle:
                data = json.load(handle)
            return data, 'json'
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
        with open(file_path, 'rb') as handle:
            data = pickle.load(handle)
        return data, 'pickle'

    def _migrate_legacy_project(self, legacy_data):
        typeset_font_info = legacy_data.get('font') or {}
        migrated_font = TypesetArea.font_to_dict(TypesetArea.font_from_dict(typeset_font_info)) if isinstance(typeset_font_info, dict) else TypesetArea.font_to_dict(self.typeset_font)
        migrated = {
            'schema_version': 1,
            'project_dir': legacy_data.get('project_dir'),
            'current_image_path': legacy_data.get('current_path'),
            'current_pdf_page': -1,
            'typeset_data': {},
            'history_entries': [],
            'proofreader_entries': [],
            'quality_entries': [],
            'history_counter': 0,
            'typeset_font': migrated_font,
            'typeset_color': legacy_data.get('color', '#000000'),
            'settings': {},
            'config': {
                'theme': self.current_theme,
                'autosave_interval_ms': int(self.autosave_timer.interval()) if getattr(self, 'autosave_timer', None) else None,
            },
            'app_version': 'legacy',
            'saved_at': time.time(),
        }
        for key, payload in (legacy_data.get('all_data') or {}).items():
            areas = payload.get('areas') or []
            redo = payload.get('redo') or []
            migrated['typeset_data'][key] = {
                'areas': [area.to_payload() if isinstance(area, TypesetArea) else area for area in areas],
                'redo': [area.to_payload() if isinstance(area, TypesetArea) else area for area in redo],
            }
        return migrated

    def _deserialize_typeset_map(self, serialized_map, fallback_font, fallback_color):
        result = {}
        warnings = []
        fallback_font = fallback_font or QFont('Arial', 9, QFont.Bold)
        fallback_color = fallback_color or QColor('#000000')
        for key, payload in (serialized_map or {}).items():
            if not isinstance(payload, dict):
                warnings.append(f"Ignored invalid typeset block for {key}.")
                continue
            areas = []
            for area_data in payload.get('areas') or []:
                try:
                    if isinstance(area_data, TypesetArea):
                        area_obj = area_data
                    else:
                        area_obj = TypesetArea.from_payload(area_data, fallback_font=fallback_font, fallback_color=fallback_color)
                    areas.append(area_obj)
                except Exception as exc:
                    warnings.append(f"Failed to load typeset area in {key}: {exc}")
            redo_items = []
            for redo_data in payload.get('redo') or []:
                try:
                    if isinstance(redo_data, TypesetArea):
                        redo_obj = redo_data
                    else:
                        redo_obj = TypesetArea.from_payload(redo_data, fallback_font=fallback_font, fallback_color=fallback_color)
                    redo_items.append(redo_obj)
                except Exception as exc:
                    warnings.append(f"Failed to load redo entry in {key}: {exc}")
            result[key] = {'areas': areas, 'redo': redo_items}
        return result, warnings

    def _sanitize_history_entries(self, history_data, area_lookup, warnings):
        sanitized = []
        max_counter = 0
        for entry in history_data or []:
            if not isinstance(entry, dict):
                warnings.append("Ignored malformed history entry.")
                continue
            hist_id = entry.get('history_id') or entry.get('id')
            if hist_id is None:
                warnings.append("A history entry without identifier was skipped.")
                continue
            hist_id = str(hist_id)
            if hist_id.startswith('H') and hist_id[1:].isdigit():
                numeric = int(hist_id[1:])
                max_counter = max(max_counter, numeric)
            elif hist_id.isdigit():
                numeric = int(hist_id)
                hist_id = f"H{numeric:05d}"
                max_counter = max(max_counter, numeric)
            else:
                warnings.append(f"History id '{hist_id}' has unexpected format.")
            record = dict(entry)
            record['history_id'] = hist_id
            record['id'] = hist_id
            record['timestamp'] = float(record.get('timestamp', time.time()))
            record['original_text'] = record.get('original_text', '')
            record['translated_text'] = record.get('translated_text', '')
            record['translation_style'] = record.get('translation_style', '')
            area_info = area_lookup.get(hist_id)
            if area_info:
                record['image_key'] = area_info['image_key']
                area = area_info['area']
                if record['original_text']:
                    area.original_text = record['original_text']
                if record['translation_style']:
                    area.translation_style = record['translation_style']
                if record['translated_text']:
                    area.update_plain_text(record['translated_text'])
            else:
                if 'image_key' not in record:
                    warnings.append(f"History entry {hist_id} has no matching area.")
            sanitized.append(record)
        return sanitized, max_counter

    def _sanitize_review_entries(self, review_data):
        sanitized = []
        for entry in review_data or []:
            if not isinstance(entry, dict):
                continue
            record = dict(entry)
            hist_id = record.get('history_id') or record.get('id')
            if hist_id is None:
                continue
            record['history_id'] = str(hist_id)
            record['id'] = record['history_id']
            record['timestamp'] = float(record.get('timestamp', time.time()))
            record['original_text'] = record.get('original_text', '')
            record['translated_text'] = record.get('translated_text', '')
            record['translation_style'] = record.get('translation_style', '')
            sanitized.append(record)
        return sanitized

    def _apply_project_payload(self, payload, project_path):
        warnings = []
        project_dir = payload.get('project_dir')
        if project_dir:
            project_dir = os.path.abspath(project_dir)
        if not project_dir or not os.path.isdir(project_dir):
            fallback_dir = os.path.abspath(os.path.dirname(project_path))
            if not project_dir:
                warnings.append("Project directory missing in save data; using project file location.")
            else:
                warnings.append(f"Project directory not found: {project_dir}. Using {fallback_dir} instead.")
            project_dir = fallback_dir
        self.project_dir = project_dir
        self.cache_dir = os.path.join(self.project_dir, '.cache')
        os.makedirs(self.cache_dir, exist_ok=True)

        self.reset_history_state()

        try:
            if self.project_dir not in self.file_watcher.directories():
                self.file_watcher.addPath(self.project_dir)
        except Exception:
            pass

        font_info = payload.get('typeset_font') or {}
        try:
            self.typeset_font = TypesetArea.font_from_dict(font_info)
        except Exception as exc:
            self.typeset_font = QFont('Arial', 9, QFont.Bold)
            warnings.append(f"Failed to load project font: {exc}; default font applied.")
        color_value = payload.get('typeset_color', '#000000')
        color_obj = QColor(color_value)
        if not color_obj.isValid():
            warnings.append(f"Invalid text color '{color_value}', using black.")
            color_obj = QColor('#000000')
        self.typeset_color = color_obj

        defaults_payload = payload.get('typeset_defaults')
        if isinstance(defaults_payload, dict):
            self.typeset_defaults = defaults_payload
        else:
            self.typeset_defaults = self._create_initial_typeset_defaults()
        self._apply_typeset_defaults()

        project_settings_payload = payload.get('settings')
        if isinstance(project_settings_payload, dict):
            cleanup_block = project_settings_payload.get('cleanup')
            if isinstance(cleanup_block, dict):
                apply_mode_value = cleanup_block.get('apply_mode')
                if apply_mode_value in ('global', 'selected') and getattr(self, 'apply_mode_global_radio', None):
                    try:
                        selected_radio = self.apply_mode_selected_radio
                        global_radio = self.apply_mode_global_radio
                        with QSignalBlocker(selected_radio), QSignalBlocker(global_radio):
                            global_radio.setChecked(apply_mode_value == 'global')
                            selected_radio.setChecked(apply_mode_value != 'global')
                    except Exception:
                        pass
                    self._set_global_cleanup_default('apply_mode', apply_mode_value)
                    if getattr(self, 'apply_mode_status_label', None):
                        self.apply_mode_status_label.setText("Mode: Global" if apply_mode_value == 'global' else "Mode: Selected Area")
                if 'use_background_box' in cleanup_block:
                    self._set_global_cleanup_default('use_background_box', bool(cleanup_block['use_background_box']))
                if 'use_inpaint' in cleanup_block:
                    self._set_global_cleanup_default('use_inpaint', bool(cleanup_block['use_inpaint']))
            self._sync_cleanup_controls_from_selection()

        serialized_typeset = payload.get('typeset_data') or payload.get('all_data') or {}
        typeset_map, type_warnings = self._deserialize_typeset_map(serialized_typeset, self.typeset_font, self.typeset_color)
        warnings.extend(type_warnings)
        self.all_typeset_data = typeset_map

        area_lookup = {}
        area_id_max = 0
        for key, record in self.all_typeset_data.items():
            cleaned_areas = []
            for area in record.get('areas', []):
                if not isinstance(area, TypesetArea):
                    continue
                hist_id = getattr(area, 'history_id', None)
                if hist_id:
                    hist_id = str(hist_id)
                    area.history_id = hist_id
                    if hist_id.startswith('H') and hist_id[1:].isdigit():
                        area_id_max = max(area_id_max, int(hist_id[1:]))
                    area_lookup[hist_id] = {'image_key': key, 'area': area}
                cleaned_areas.append(area)
            record['areas'] = cleaned_areas
            redo_clean = []
            for redo_area in record.get('redo', []):
                if isinstance(redo_area, TypesetArea):
                    redo_area.history_id = str(getattr(redo_area, 'history_id', '') or '') or None
                redo_clean.append(redo_area)
            record['redo'] = redo_clean

        history_data = payload.get('history_entries')
        sanitized_history, history_max = self._sanitize_history_entries(history_data, area_lookup, warnings)

        counter_from_payload = payload.get('history_counter')
        if isinstance(counter_from_payload, str) and counter_from_payload.isdigit():
            counter_from_payload = int(counter_from_payload)
        elif not isinstance(counter_from_payload, int):
            counter_from_payload = 0

        self.history_counter = max(counter_from_payload, history_max, area_id_max)
        self.history_entries = sanitized_history
        existing_ids = {entry['history_id'] for entry in self.history_entries}

        for key, record in self.all_typeset_data.items():
            for area in record['areas']:
                hist_id = getattr(area, 'history_id', None)
                if hist_id and hist_id not in existing_ids:
                    if hist_id.startswith('H') and hist_id[1:].isdigit():
                        self.history_counter = max(self.history_counter, int(hist_id[1:]))
                    new_entry = {
                        'id': hist_id,
                        'history_id': hist_id,
                        'image_key': key,
                        'original_text': area.original_text or '',
                        'translated_text': area.text or '',
                        'translation_style': getattr(area, 'translation_style', ''),
                        'timestamp': time.time(),
                    }
                    self.history_entries.append(new_entry)
                    existing_ids.add(hist_id)
                    area_lookup[hist_id] = {'image_key': key, 'area': area}
                if not hist_id:
                    new_id = self.generate_history_id()
                    area.history_id = new_id
                    new_entry = {
                        'id': new_id,
                        'history_id': new_id,
                        'image_key': key,
                        'original_text': area.original_text or '',
                        'translated_text': area.text or '',
                        'translation_style': getattr(area, 'translation_style', ''),
                        'timestamp': time.time(),
                    }
                    self.history_entries.append(new_entry)
                    existing_ids.add(new_id)
                    area_lookup[new_id] = {'image_key': key, 'area': area}

        proof_entries = self._sanitize_review_entries(payload.get('proofreader_entries'))
        quality_entries = self._sanitize_review_entries(payload.get('quality_entries'))
        self.proofreader_entries = proof_entries
        self.quality_entries = quality_entries

        self.history_lookup.clear()
        for hist_id, info in area_lookup.items():
            if hist_id in existing_ids:
                self.history_lookup[hist_id] = info

        saved_image_path = payload.get('current_image_path')
        saved_pdf_page = payload.get('current_pdf_page', -1)
        self.current_pdf_page = int(saved_pdf_page) if isinstance(saved_pdf_page, int) else -1
        self.current_project_path = os.path.abspath(project_path)

        self.update_file_list()
        if saved_image_path and saved_image_path in self.image_files:
            row = self.image_files.index(saved_image_path)
            self.file_list_widget.setCurrentRow(row)
        elif self.image_files:
            self.file_list_widget.setCurrentRow(0)
            if saved_image_path:
                warnings.append(f"Image '{saved_image_path}' not found in folder; opened first file instead.")

        self.refresh_history_views()
        return warnings

    def _load_project_from_path(self, file_path, *, show_dialogs=True):
        warnings = []
        legacy_loaded = False

        # Auto-save current project before switching
        self.save_project(is_auto=True)

        # Hapus watcher lama
        if self.project_dir and self.project_dir in self.file_watcher.directories():
            try:
                self.file_watcher.removePath(self.project_dir)
            except Exception:
                pass

        # Tutup PDF lama jika ada
        if self.pdf_document:
            self.pdf_document.close()
            self.pdf_document = None
        self.current_pdf_page = -1

        try:
            payload, fmt = self._read_project_file(file_path)

            if fmt == 'pickle':
                payload = self._migrate_legacy_project(payload)
                legacy_loaded = True

            warnings = self._apply_project_payload(payload, file_path)

            if legacy_loaded:
                warnings.append("Loaded legacy project format; saving will upgrade it.")

            # Update judul window
            self.setWindowTitle(
                f"Manga OCR & Typeset Tool v14.3.4 - {os.path.basename(self.current_project_path)}"
            )

            # Start autosave only if user enabled it
            try:
                if getattr(self, 'autosave_enabled', True):
                    self.autosave_timer.start()
            except Exception:
                pass

            # Tampilkan warning jika ada
            if warnings and show_dialogs:
                QMessageBox.warning(
                    self,
                    "Project Loaded with Warnings",
                    "\n".join(f"? {w}" for w in warnings)
                )

            # Info success
            if show_dialogs:
                QMessageBox.information(self, "Success", "Project loaded successfully.")
            elif warnings:
                self.statusBar().showMessage("; ".join(warnings), 5000)

            return True

        except Exception as exc:
            if show_dialogs:
                QMessageBox.critical(self, "Error", f"Failed to load project: {exc}")
            else:
                self.statusBar().showMessage(f"Failed to load project: {exc}", 5000)
            return False

    def save_project(self, is_auto=False):
        if not self.project_dir:
            if not is_auto:
                QMessageBox.warning(self, "No Project", "Please load a folder before saving a project.")
            return False
    
        if not self.current_project_path:
            if is_auto:
                self.current_project_path, note = self._make_project_file_path()
                if note:
                    self.statusBar().showMessage(note, 6000)
            else:
                suggested_name = os.path.basename(self.project_dir.rstrip(os.sep)) if self.project_dir else 'project'
                default_path = os.path.join(self.project_dir, f"{suggested_name}.manga_proj") if self.project_dir else ''
                file_path, _ = QFileDialog.getSaveFileName(self, "Save Project", default_path, "Manga Project (*.manga_proj)")
                if not file_path:
                    return False
                if not file_path.lower().endswith('.manga_proj'):
                    file_path += '.manga_proj'
                chosen_path = os.path.abspath(file_path)
                if os.name == 'nt' and len(chosen_path) >= 245:
                    preferred = os.path.splitext(os.path.basename(chosen_path))[0]
                    shortened, note = self._make_project_file_path(preferred, os.path.dirname(chosen_path))
                    if len(os.path.abspath(shortened)) >= 245:
                        shortened, note = self._make_project_file_path(preferred)
                    self.current_project_path = shortened
                    if note:
                        self.statusBar().showMessage(note, 6000)
                else:
                    self.current_project_path = chosen_path
        else:
            if os.name == 'nt' and len(os.path.abspath(self.current_project_path)) >= 245:
                preferred = os.path.splitext(os.path.basename(self.current_project_path))[0]
                base_dir = os.path.dirname(self.current_project_path)
                shortened, note = self._make_project_file_path(preferred, base_dir)
                if len(os.path.abspath(shortened)) >= 245:
                    shortened, note = self._make_project_file_path(preferred)
                if shortened != self.current_project_path:
                    self.current_project_path = shortened
                    if note:
                        self.statusBar().showMessage(note, 6000)

        # Create a quick snapshot under mutex to avoid races, then perform heavy IO in background
        try:
            self.paint_mutex.lock()
            try:
                serialized_typeset = {}
                for k, rec in (self.all_typeset_data or {}).items():
                    try:
                        areas = rec.get('areas', []) or []
                        redo = rec.get('redo', []) or []
                        serialized_typeset[k] = {
                            'areas': [area.to_payload() if isinstance(area, TypesetArea) else area for area in areas],
                            'redo': [r.to_payload() if isinstance(r, TypesetArea) else r for r in redo],
                        }
                    except Exception:
                        serialized_typeset[k] = {'areas': [], 'redo': []}

                snapshot = {
                    'project_dir': self.project_dir,
                    'current_image_path': self.current_image_path,
                    'current_pdf_page': int(self.current_pdf_page) if isinstance(self.current_pdf_page, int) else -1,
                    'typeset_data': serialized_typeset,
                    'history_entries': list(self.history_entries) if getattr(self, 'history_entries', None) is not None else [],
                    'proofreader_entries': list(self.proofreader_entries) if getattr(self, 'proofreader_entries', None) is not None else [],
                    'quality_entries': list(self.quality_entries) if getattr(self, 'quality_entries', None) is not None else [],
                    'history_counter': int(self.history_counter) if getattr(self, 'history_counter', None) is not None else 0,
                    'typeset_font': TypesetArea.font_to_dict(self.typeset_font) if getattr(self, 'typeset_font', None) else None,
                    'typeset_color': self.typeset_color.name() if getattr(self, 'typeset_color', None) else '#000000',
                    'settings': self._collect_project_settings() if hasattr(self, '_collect_project_settings') else {},
                    'app_version': '16.1.0',
                }
            finally:
                self.paint_mutex.unlock()
        except Exception as exc:
            if not is_auto:
                QMessageBox.critical(self, "Error", f"Failed to prepare project data: {exc}")
            return False
        # Ensure target directory exists
        target_dir = os.path.dirname(self.current_project_path) or (self.project_dir or os.getcwd())
        try:
            os.makedirs(target_dir, exist_ok=True)
        except OSError:
            pass

        # Prevent concurrent project saves
        if getattr(self, 'project_save_thread', None) and getattr(self, 'project_save_thread', None).isRunning():
            if not is_auto:
                QMessageBox.information(self, "Save In Progress", "A project save is already in progress.")
            return False

        # Start background worker to write the project file
        worker = ProjectSaveWorker(self.current_project_path, snapshot)
        thread = QThread()
        worker.moveToThread(thread)

        def _on_thread_started():
            try:
                worker.run()
            except Exception as e:
                # emit error via worker signals
                try:
                    worker.error.emit(str(e))
                except Exception:
                    pass

        def _on_finished(success, message):
            # Cleanup thread/worker
            try:
                thread.quit(); thread.wait()
            except Exception:
                pass
            self.project_save_thread = None
            self.project_save_worker = None
            if not is_auto:
                if success:
                    QMessageBox.information(self, "Success", message)
                else:
                    QMessageBox.critical(self, "Error", message)

        def _on_error(msg):
            self.statusBar().showMessage(f"Error saving project: {msg}", 5000)

        worker.finished.connect(_on_finished)
        worker.error.connect(_on_error)

        # store references so we can cancel/inspect
        self.project_save_worker = worker
        self.project_save_thread = thread

        thread.started.connect(_on_thread_started)
        thread.start()
        # Let autosave or UI continue; indicate to user
        if not is_auto:
            self.statusBar().showMessage("Saving project in background...", 3000)
        return True
    
    def auto_save_project(self):
        if QApplication.activeModalWidget() is not None:
            return 
    
        if self.current_project_path and os.path.exists(os.path.dirname(self.current_project_path)):
            if self.save_project(is_auto=True): 
                self.statusBar().showMessage(f"Project auto-saved at {time.strftime('%H:%M:%S')}", 3000)

    def load_project(self):
        default_dir = self.project_dir or ''
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Project", default_dir, "Manga Project (*.manga_proj)")
        if not file_path:
            return
        self._load_project_from_path(file_path)

    def start_inline_edit(self, area):
        if not area:
            return
        try:
            dialog = AdvancedTextEditDialog(area, self.font_manager, parent=self)
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
                try:
                    self.redo_stack.clear()
                except Exception:
                    pass
                self.redraw_all_typeset_areas()
                self.update_undo_redo_buttons_state()
                self.statusBar().showMessage("Text updated", 2000)
        except Exception:
            traceback.print_exc()

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
        about_text = (f"<b>Manga OCR & Typeset Tool v14.3.4</b><br><br>This tool was created to streamline the process of translating manga.<br><br>Powered by Python, PyQt5, and various AI APIs.<br>Enhanced with new features by Gemini.<br><br>Copyright © 2024")
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
        data = self.ai_model_combo.currentData(Qt.UserRole)
        if isinstance(data, tuple) and len(data) == 2:
            return data
        selected_text = self.ai_model_combo.currentText()
        if not selected_text:
            return None, None
        match = re.match(r"\[(.*?)\]\s*(.*)", selected_text)
        if not match:
            return None, None
        provider_name, label = match.groups()
        provider_models = self.AI_PROVIDERS.get(provider_name, {})
        for model_id, info in provider_models.items():
            if info.get('display') == label:
                return provider_name, model_id
        return None, None

    def get_selected_model_info(self):
        index = self.ai_model_combo.currentIndex()
        if index < 0:
            return {}
        model_info = self.ai_model_combo.itemData(index, Qt.UserRole + 1)
        if isinstance(model_info, dict):
            return model_info
        provider, model_id = self.get_selected_model_name()
        return self.AI_PROVIDERS.get(provider, {}).get(model_id, {})

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
        self.last_detection_mode = detection_mode
        self.preview_mode_active = False
        self.set_ui_for_detection(True)

        settings = self.get_current_settings()
        settings['batch_text_detection_enabled'] = (detection_mode == "Text")
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
        self.set_ui_for_detection(False)
        self.overall_progress_bar.setVisible(False)
        if self.get_current_settings()['auto_split_bubbles']:
            self.statusBar().showMessage("Splitting extended items...", 3000)
            QApplication.processEvents()
            for path, detections in self.detected_items_map.items():
                self.detected_items_map[path] = self.split_extended_bubbles(detections)

        if self.last_detection_mode == "Text" and self.detected_items_map:
            self.preview_mode_active = True
            self.cancel_detection_button.setText("Cancel Preview")
            self.cancel_detection_button.setVisible(True)
            self.file_list_widget.setEnabled(True)
            self.prev_button.setEnabled(True)
            self.next_button.setEnabled(True)
        else:
            self.preview_mode_active = False
            self.cancel_detection_button.setVisible(False)
            self.cancel_detection_button.setText("Cancel Detection")

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
        self.preview_mode_active = False
        self.cancel_detection_button.setText("Cancel Detection")
        self.cancel_detection_button.setVisible(False)
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
        if is_detecting:
            self.cancel_detection_button.setText("Cancel Detection")
            self.cancel_detection_button.setVisible(True)
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
        # Stop interactive flows and cancel background save worker first
        self.cancel_interactive_batch()
        try:
            if hasattr(self, 'deferred_typeset_timer'):
                self.deferred_typeset_timer.stop()
        except Exception:
            pass
        try:
            if getattr(self, 'batch_save_worker', None):
                try:
                    self.batch_save_worker.cancel()
                except Exception:
                    pass
        except Exception:
            pass
        # Quit/wait threads if they are present and appear to be running.
        try:
            if getattr(self, 'batch_save_thread', None):
                try:
                    if getattr(self.batch_save_thread, 'isRunning', lambda: False)():
                        self.batch_save_thread.quit(); self.batch_save_thread.wait()
                    else:
                        # still attempt to quit/wait once to be safe
                        self.batch_save_thread.quit(); self.batch_save_thread.wait()
                except RuntimeError:
                    # QThread wrapper was already deleted; ignore
                    pass
                except Exception:
                    pass
        except Exception:
            pass

        # Stop other workers
        for worker_id, pair in list(self.worker_pool.items()):
            # pair may be (thread, worker) or other structure; be defensive
            try:
                thread, worker = pair
            except Exception:
                continue
            try:
                if worker is not None:
                    try:
                        worker.stop()
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                if thread is not None:
                    try:
                        if getattr(thread, 'isRunning', lambda: False)():
                            thread.quit(); thread.wait()
                        else:
                            thread.quit(); thread.wait()
                    except RuntimeError:
                        # QThread wrapper already deleted
                        pass
                    except Exception:
                        pass
            except Exception:
                pass

        if getattr(self, 'batch_processor_thread', None):
            try:
                try:
                    if getattr(self.batch_processor_thread, 'isRunning', lambda: False)():
                        self.batch_processor_thread.quit(); self.batch_processor_thread.wait()
                    else:
                        self.batch_processor_thread.quit(); self.batch_processor_thread.wait()
                except RuntimeError:
                    pass
            except Exception:
                pass
        if getattr(self, 'exchange_rate_thread', None):
            try:
                try:
                    if getattr(self.exchange_rate_thread, 'isRunning', lambda: False)():
                        self.exchange_rate_thread.quit(); self.exchange_rate_thread.wait()
                    else:
                        self.exchange_rate_thread.quit(); self.exchange_rate_thread.wait()
                except RuntimeError:
                    pass
            except Exception:
                pass

        # Acquire paint mutex to ensure no painting is currently active before destroying pixmaps
        try:
            self.paint_mutex.lock()
        except Exception:
            pass
        try:
            # safe to drop pixmaps now
            self.original_pixmap = None
            self.typeset_pixmap = None
        finally:
            try:
                self.paint_mutex.unlock()
            except Exception:
                pass

        self.save_usage_data(); event.accept()
    
    # ===================================================================
    # ======================= OCR & DETECT METHODS ======================
    # ===================================================================

    def detect_text_with_ocr_engine(self, cv_image, settings):
        """Detect text regions and return recognized text polygons."""
        engine = (settings.get('ocr_engine') or 'Tesseract')
        advanced = settings.get('batch_text_detection_enabled', False)

        try:
            raw_results = self._collect_engine_detections(cv_image, settings, engine, advanced)
        except Exception as e:
            print(f"Error during text detection with {engine}: {e}")
            raw_results = []

        if not raw_results:
            return []

        if advanced:
            raw_results = self._tighten_detection_polygons(cv_image, raw_results)

        filtered = self._filter_detection_noise(raw_results, cv_image.shape, advanced=advanced)
        if not filtered:
            return []

        merged = self._merge_text_boxes_to_blocks(filtered, cv_image.shape, strict=advanced)
        if advanced and merged:
            merged = self._tighten_detection_polygons(cv_image, merged)

        final = self._filter_detection_noise(merged, cv_image.shape, advanced=advanced)
        return final

    def _collect_engine_detections(self, cv_image, settings, engine, advanced):
        engine = engine or 'Tesseract'

        if engine == 'DocTR':
            return self._collect_doctr_detections(cv_image, advanced=advanced)
        if engine == 'EasyOCR':
            return self._collect_easyocr_detections(cv_image, advanced=advanced)
        if engine == 'PaddleOCR':
            return self._collect_paddleocr_detections(cv_image, advanced=advanced)
        if engine == 'RapidOCR':
            return self._collect_rapidocr_detections(cv_image, advanced=advanced)
        if engine == 'Manga-OCR':
            return self._collect_manga_detections(cv_image, settings, advanced=advanced)
        if engine == 'AI_OCR':
            regions = self._collect_morphological_regions(cv_image, advanced=advanced)
            results = []
            for _, polygon in regions:
                recognized = self._recognize_polygon(cv_image, polygon, 'AI_OCR', settings)
                results.append((recognized, polygon))
            return results
        if engine == 'Tesseract':
            if advanced:
                return self._collect_tesseract_advanced_detections(cv_image, settings, advanced=True)
            return self._collect_tesseract_native_detections(cv_image, settings.get('ocr_lang') or 'eng')
        return self._collect_easyocr_detections(cv_image, advanced=advanced)

    def _collect_doctr_detections(self, cv_image, advanced=False):
        if not self.doctr_predictor:
            return []

        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        result = self.doctr_predictor([rgb_image])
        items = []
        height, width = cv_image.shape[:2]

        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    line_text = ' '.join(word.value for word in line.words)
                    geometry = line.geometry
                    x1 = int(geometry[0][0] * width)
                    y1 = int(geometry[0][1] * height)
                    x2 = int(geometry[1][0] * width)
                    y2 = int(geometry[1][1] * height)
                    polygon = QPolygon([
                        QPoint(x1, y1),
                        QPoint(x2, y1),
                        QPoint(x2, y2),
                        QPoint(x1, y2),
                    ])
                    items.append((line_text, polygon))

        return items

    def _collect_easyocr_detections(self, cv_image, advanced=False):
        if not self.easyocr_reader:
            return []
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        try:
            ocr_result = self.easyocr_reader.readtext(gray, detail=1)
        except Exception as e:
            print(f"EasyOCR detection error: {e}")
            return []

        items = []
        min_prob = 0.45 if advanced else 0.30
        for bbox, text, prob in ocr_result:
            if advanced and prob < min_prob:
                continue
            polygon = QPolygon([QPoint(int(p[0]), int(p[1])) for p in bbox])
            items.append((text, polygon))
        return items

    def _collect_paddleocr_detections(self, cv_image, advanced=False):
        if not self.paddle_ocr_reader:
            return []
        try:
            ocr_result = self.paddle_ocr_reader.ocr(cv_image, cls=True)
        except Exception as e:
            print(f"PaddleOCR detection error: {e}")
            return []

        items = []
        if ocr_result and ocr_result[0]:
            for line in ocr_result[0]:
                polygon = QPolygon([QPoint(int(p[0]), int(p[1])) for p in line[0]])
                items.append((line[1][0], polygon))
        return items

    def _collect_rapidocr_detections(self, cv_image, advanced=False):
        if not self.rapid_ocr_reader:
            return []
        try:
            ocr_result, _ = self.rapid_ocr_reader(cv_image)
        except Exception as e:
            print(f"RapidOCR detection error: {e}")
            return []

        items = []
        if ocr_result:
            for box_info in ocr_result:
                polygon = QPolygon([QPoint(int(p[0]), int(p[1])) for p in box_info[0]])
                items.append((box_info[1], polygon))
        return items

    def _collect_easy_detection_regions(self, cv_image, advanced=False):
        return self._collect_easyocr_detections(cv_image, advanced=advanced)

    def _collect_morphological_regions(self, cv_image, advanced=False):
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 9)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(thresh, kernel, iterations=1 if not advanced else 2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = gray.shape[:2]
        items = []
        min_area = 120 if advanced else 90
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            if area < max(min_area, 0.00004 * w * h):
                continue
            if ch < 10 or cw < 10:
                continue
            aspect = cw / max(1, ch)
            if advanced and (aspect > 10 or aspect < 0.12):
                continue
            if cw > w * 0.95 and ch > h * 0.5:
                continue
            polygon = QPolygon([
                QPoint(x, y),
                QPoint(x + cw, y),
                QPoint(x + cw, y + ch),
                QPoint(x, y + ch),
            ])
            items.append(('', polygon))
        return items

    def _collect_manga_detections(self, cv_image, settings, advanced=False):
        if not self.manga_ocr_reader:
            return []

        use_easy = settings.get('manga_use_easy_detection', True)
        if use_easy:
            regions = self._collect_easy_detection_regions(cv_image, advanced=advanced)
        else:
            regions = self._collect_morphological_regions(cv_image, advanced=advanced)

        results = []
        for text, polygon in regions:
            recognized = self._recognize_polygon(cv_image, polygon, 'Manga-OCR', settings)
            results.append((recognized or text, polygon))
        return results

    def _collect_tesseract_native_detections(self, cv_image, lang_code):
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        try:
            data = pytesseract.image_to_data(gray, lang=lang_code, config='--oem 1 --psm 3', output_type=pytesseract.Output.DICT)
        except Exception as e:
            print(f"Tesseract detection error: {e}")
            return []

        blocks = {}
        for i in range(len(data['text'])):
            text = (data['text'][i] or '').strip()
            if not text:
                continue
            try:
                conf = float(data['conf'][i])
            except ValueError:
                conf = 0.0
            if conf < 45:
                continue
            block_key = (data.get('page_num', [0])[i], data['block_num'][i])
            rect = QRect(data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            blocks.setdefault(block_key, {'texts': [], 'rects': []})
            blocks[block_key]['texts'].append(text)
            blocks[block_key]['rects'].append(rect)

        results = []
        for info in blocks.values():
            if not info['texts']:
                continue
            combined_text = ' '.join(info['texts'])
            union_rect = info['rects'][0]
            for rect in info['rects'][1:]:
                union_rect = union_rect.united(rect)
            polygon = QPolygon([
                QPoint(union_rect.left(), union_rect.top()),
                QPoint(union_rect.right(), union_rect.top()),
                QPoint(union_rect.right(), union_rect.bottom()),
                QPoint(union_rect.left(), union_rect.bottom()),
            ])
            results.append((combined_text, polygon))
        return results

    def _collect_tesseract_advanced_detections(self, cv_image, settings, advanced=False):
        if settings.get('tesseract_use_easy_detection', True):
            regions = self._collect_easy_detection_regions(cv_image, advanced=advanced)
            results = []
            for _, polygon in regions:
                text = self._recognize_polygon(cv_image, polygon, 'Tesseract', settings)
                results.append((text, polygon))
            return results
        return self._collect_tesseract_native_detections(cv_image, settings.get('ocr_lang') or 'eng')

    def _recognize_polygon(self, cv_image, polygon, engine_name, base_settings):
        rect = polygon.boundingRect()
        h, w = cv_image.shape[:2]
        pad = int(max(rect.width(), rect.height()) * 0.08)
        x1 = max(rect.x() - pad, 0)
        y1 = max(rect.y() - pad, 0)
        x2 = min(rect.x() + rect.width() + pad, w)
        y2 = min(rect.y() + rect.height() + pad, h)
        if x2 - x1 <= 1 or y2 - y1 <= 1:
            return ''
        crop = cv_image[y1:y2, x1:x2].copy()
        local_settings = dict(base_settings)
        local_settings['ocr_engine'] = engine_name
        if engine_name == 'Manga-OCR':
            local_settings['ocr_lang'] = 'ja'
            local_settings['orientation'] = 'Auto-Detect'
        elif engine_name == 'Tesseract':
            local_settings['ocr_lang'] = base_settings.get('ocr_lang') or 'eng'
        text = self.perform_ocr(crop, local_settings)
        return text.strip()

    def _filter_detection_noise(self, items, image_shape, advanced=False):
        if not items:
            return []
        h, w = image_shape[:2]
        min_area_ratio = 0.00004 if advanced else 0.00003
        min_area = max(80, min_area_ratio * w * h)
        max_area_ratio = 0.85 if advanced else 0.9
        filtered = []
        for text, polygon in items:
            cleaned = self._clean_detected_text(text)
            if not cleaned:
                continue
            if len(cleaned) <= 1 and not cleaned.isalnum():
                continue
            if re.fullmatch(r'[\W_]+', cleaned):
                continue
            letters = sum(ch.isalpha() for ch in cleaned)
            digits = sum(ch.isdigit() for ch in cleaned)
            if advanced:
                if letters == 0 and digits == 0 and len(cleaned) <= 3:
                    continue
                if re.fullmatch(r'[!\?\-~•°??????]+', cleaned):
                    continue
                if re.search(r'(.)\1{2,}', cleaned) and len(cleaned) <= 5:
                    continue
            unique_chars = set(cleaned)
            if len(unique_chars) == 1 and cleaned[0] in "!?~…??????#@*/":
                continue
            punctuation = sum(1 for ch in cleaned if not ch.isalnum() and not ch.isspace())
            if advanced and punctuation / max(1, len(cleaned)) > 0.6:
                continue

            rect = polygon.boundingRect()
            area = rect.width() * rect.height()
            if area < min_area:
                continue
            if area > w * h * max_area_ratio:
                continue
            if rect.width() < 6 or rect.height() < 6:
                continue
            aspect_ratio = rect.width() / max(1, rect.height())
            if advanced and (aspect_ratio > 9.0 or aspect_ratio < 0.12):
                continue

            filtered.append((cleaned, self._clamp_polygon(polygon, w, h)))
        return filtered

    def _clean_detected_text(self, text):
        if not text:
            return ''
        cleaned = re.sub(r'\s+', ' ', text)
        return cleaned.strip()

    def _clamp_polygon(self, polygon, width, height):
        clamped_points = []
        for i in range(polygon.count()):
            pt = polygon.point(i)
            x = max(0, min(pt.x(), width - 1))
            y = max(0, min(pt.y(), height - 1))
            clamped_points.append(QPoint(x, y))
        return QPolygon(clamped_points)


    """Helper methods for advanced OCR detection pipeline"""
    def _merge_text_boxes_to_blocks(self, boxes, image_shape, strict=False):
        """Group nearby text boxes into cohesive reading blocks."""
        if not boxes:
            return []
        h, w = image_shape[:2]
        diag = math.hypot(w, h)
        max_gap = diag * (0.018 if strict else 0.04)
        sorted_boxes = [item for item in boxes if item and item[1] is not None]
        sorted_boxes.sort(key=lambda item: item[1].boundingRect().top())

        clusters = []
        for text, polygon in sorted_boxes:
            rect = self._clamp_rect(polygon.boundingRect(), w, h)
            merged = False
            for cluster in clusters:
                if self._rects_should_merge(rect, cluster['rect'], strict, max_gap):
                    cluster['rect'] = cluster['rect'].united(rect)
                    cluster['polygons'].append(polygon)
                    cluster['texts'].append(text)
                    merged = True
                    break
            if not merged:
                clusters.append({'rect': rect, 'polygons': [polygon], 'texts': [text]})

        merged_results = []
        for cluster in clusters:
            combined_text = self._combine_texts(cluster['texts'])
            polygon = self._polygon_from_rect(cluster['rect'])
            merged_results.append((combined_text, polygon))
        return merged_results

    def _rects_should_merge(self, rect_a, rect_b, strict, max_gap):
        if rect_a.intersects(rect_b):
            return True
        distance = self._rect_distance(rect_a, rect_b)
        if distance > max_gap:
            return False
        vertical_overlap = self._axis_overlap_ratio(
            rect_a.top(), rect_a.top() + rect_a.height(),
            rect_b.top(), rect_b.top() + rect_b.height()
        )
        horizontal_overlap = self._axis_overlap_ratio(
            rect_a.left(), rect_a.left() + rect_a.width(),
            rect_b.left(), rect_b.left() + rect_b.width()
        )
        if strict:
            if vertical_overlap >= 0.35 and distance <= max_gap * 0.75:
                return True
            if horizontal_overlap >= 0.55 and distance <= max_gap * 0.75:
                return True
            return False
        if vertical_overlap >= 0.2 or horizontal_overlap >= 0.65:
            return True
        return distance <= max_gap * 0.6

    def _rect_distance(self, rect_a, rect_b):
        ax1 = rect_a.left()
        ax2 = rect_a.right()
        ay1 = rect_a.top()
        ay2 = rect_a.bottom()
        bx1 = rect_b.left()
        bx2 = rect_b.right()
        by1 = rect_b.top()
        by2 = rect_b.bottom()
        dx = max(0, max(bx1 - ax2, ax1 - bx2))
        dy = max(0, max(by1 - ay2, ay1 - by2))
        return math.hypot(dx, dy)

    def _axis_overlap_ratio(self, a_start, a_end, b_start, b_end):
        overlap = max(0.0, min(a_end, b_end) - max(a_start, b_start))
        if overlap <= 0:
            return 0.0
        min_size = max(1.0, min(a_end - a_start, b_end - b_start))
        return overlap / min_size

    def _polygon_from_rect(self, rect):
        x1 = rect.left()
        y1 = rect.top()
        x2 = rect.right()
        y2 = rect.bottom()
        return QPolygon([
            QPoint(x1, y1),
            QPoint(x2, y1),
            QPoint(x2, y2),
            QPoint(x1, y2),
        ])

    def _clamp_rect(self, rect, width, height):
        x = max(0, rect.left())
        y = max(0, rect.top())
        right = min(rect.right(), width - 1)
        bottom = min(rect.bottom(), height - 1)
        if right < x:
            right = x
        if bottom < y:
            bottom = y
        return QRect(x, y, (right - x) + 1, (bottom - y) + 1)

    def _tighten_detection_polygons(self, cv_image, items):
        if not items:
            return []
        h, w = cv_image.shape[:2]
        refined = []
        for text, polygon in items:
            refined_polygon = self._refine_polygon_with_image(cv_image, polygon)
            refined.append((text, self._clamp_polygon(refined_polygon, w, h)))
        return refined

    def _refine_polygon_with_image(self, cv_image, polygon):
        rect = polygon.boundingRect()
        h, w = cv_image.shape[:2]
        rect = self._clamp_rect(rect, w, h)
        if rect.width() <= 2 or rect.height() <= 2:
            return self._polygon_from_rect(rect)

        x, y, width, height = rect.left(), rect.top(), rect.width(), rect.height()
        crop = cv_image[y:y + height, x:x + width]
        if crop.size == 0:
            return self._polygon_from_rect(rect)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        _, thresh_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates = [thresh_inv, thresh]

        best_rect = None
        best_area = None
        for mask in candidates:
            coords = cv2.findNonZero(mask)
            if coords is None:
                continue
            bx, by, bw, bh = cv2.boundingRect(coords)
            area = bw * bh
            if best_rect is None or area < best_area:
                best_rect = (bx, by, bw, bh)
                best_area = area

        if best_rect is None:
            return self._polygon_from_rect(rect)

        bx, by, bw, bh = best_rect
        pad = max(1, int(min(bw, bh) * 0.05))
        bx = max(0, bx - pad)
        by = max(0, by - pad)
        bw = min(width - bx, bw + pad * 2)
        bh = min(height - by, bh + pad * 2)

        refined_rect = QRect(x + bx, y + by, max(1, bw), max(1, bh))
        refined_rect = self._clamp_rect(refined_rect, w, h)
        return self._polygon_from_rect(refined_rect)

    def _combine_texts(self, texts):
        parts = [t.strip() for t in texts if t and t.strip()]
        return ' '.join(parts)
    
    def perform_ocr(self, image_to_process, settings: dict) -> str:
        """
        [DIUBAH] Menjalankan OCR pada gambar yang diberikan berdasarkan pengaturan.
        """
        ocr_engine = settings['ocr_engine']
        orientation = settings.get('orientation', 'Auto-Detect')
        ocr_lang = settings.get('ocr_lang', 'ja')
        raw_text = ""

        # For AI_OCR and MOFRL-GPT we must not alter the crop at all; send the pure raw image.
        if ocr_engine not in ("AI_OCR", "MOFRL-GPT"):
            # Penyesuaian rotasi sesuai orientasi (apply for non-AI engines)
            h, w = image_to_process.shape[:2]
            if orientation == "Vertical" and w > h:
                # rotate so that vertical text becomes horizontal for OCR engines that expect horizontal lines
                image_to_process = cv2.rotate(image_to_process, cv2.ROTATE_90_CLOCKWISE)
            elif orientation == "Horizontal" and h > w:
                # if user selected horizontal but image is taller than wide, rotate to horizontal
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
                # PaddleOCR may expose ocr() or predict() depending on version.
                # Try common call patterns and normalize output.
                result = None
                texts = []
                try:
                    # prefer .ocr if available (older versions)
                    if hasattr(self.paddle_ocr_reader, 'ocr'):
                        result = self.paddle_ocr_reader.ocr(image_to_process, cls=True)
                    elif hasattr(self.paddle_ocr_reader, 'predict'):
                        result = self.paddle_ocr_reader.predict(image_to_process)
                    else:
                        # last-resort: try calling object directly
                        result = self.paddle_ocr_reader(image_to_process)
                except Exception:
                    # fallback to predict with image path or other signature is not attempted here
                    result = None

                # Normalize result to a list of lines
                if not result:
                    raw_text = ""
                else:
                    # result can be: [[(poly, (text, conf)), ...], ...] or similar
                    # Try several common shapes
                    candidate = None
                    if isinstance(result, (list, tuple)) and len(result) > 0:
                        candidate = result[0]

                    if isinstance(candidate, list):
                        for entry in candidate:
                            # entry can be [bbox, (text, prob)] or [ [points], (text, prob) ]
                            try:
                                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                                    text_blob = None
                                    # entry[1] may be (text, prob) or text string
                                    if isinstance(entry[1], (list, tuple)) and len(entry[1]) > 0:
                                        text_blob = entry[1][0]
                                    elif isinstance(entry[1], str):
                                        text_blob = entry[1]
                                    if text_blob:
                                        texts.append(text_blob)
                            except Exception:
                                continue
                    else:
                        # try to walk nested dict/list for 'text' keys
                        try:
                            # common dict-based result contains 'data' or similar
                            for page in result:
                                for line in page:
                                    if isinstance(line, dict):
                                        t = line.get('text') or line.get('transcription')
                                        if t:
                                            texts.append(t)
                                    elif isinstance(line, (list, tuple)) and len(line) >= 2:
                                        sub = line[1]
                                        if isinstance(sub, (list, tuple)) and len(sub) > 0:
                                            texts.append(sub[0])
                        except Exception:
                            pass

                    raw_text = "\n".join([t for t in texts if t])
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

        elif ocr_engine == "AI_OCR":
            provider = settings.get('ocr_ai_provider')
            model_id = settings.get('ocr_ai_model_id')
            model_name = settings.get('ocr_ai_model_name')
            result = self._call_ai_ocr(image_to_process, provider, model_id, model_name)
            return result
        
        elif ocr_engine == "MOFRL-GPT":
            raw_text = self._call_mofrl_ocr(image_to_process, settings)
            return raw_text

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

    def _call_ai_ocr(self, image_bgr, provider_key, model_id, model_name=None):
        if not provider_key or not model_id:
            return "[AI OCR ERROR: No active model configured]"

        provider_cfg = SETTINGS.get('ocr', {}).get(provider_key, {})
        url = (provider_cfg.get('url') or '').strip()
        api_key = (provider_cfg.get('api_key') or '').strip()

        if not url:
            return "[AI OCR ERROR: API URL missing]"
        if not api_key:
            return "[AI OCR ERROR: API key missing]"

        success, buffer = cv2.imencode('.png', image_bgr)
        if not success:
            return "[AI OCR ERROR: Encoding image failed]"

        image_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')

        prompt_text = (
            "Task: OCR. Input: an image. Output: ONLY the text in the image.\n\n"
            "Rules:\n"
            "- No explanations.\n"
            "- No markdown.\n"
            "- No formatting.\n"
            "- Just return the plain text."
        )
        data_url = f"data:image/png;base64,{image_b64}"

        # Prepare several payload variants to account for provider schema differences.
        payload_variants = []

        # Variant A: OpenRouter-style image_url with data URI
        payload_variants.append({
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ]
                }
            ]
        })

        # Variant B: input_image with image_data (some providers expect this)
        payload_variants.append({
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "input_image", "image_data": data_url}
                    ]
                }
            ]
        })

        # Variant C: simple text prompt concatenated with data URI (fallback)
        payload_variants.append({
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": prompt_text + "\n\nImage: " + data_url
                }
            ]
        })

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Ensure temp debug folder
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
        try:
            os.makedirs(temp_dir, exist_ok=True)
        except Exception:
            temp_dir = None

        # Save crop image for debugging
        if temp_dir:
            try:
                # Put debug images under temp/img/aiocr/
                img_dir = os.path.join(temp_dir, 'img', 'aiocr')
                os.makedirs(img_dir, exist_ok=True)
                timestamp = int(time.time())
                crop_path = os.path.join(img_dir, f'aiocr_crop_{timestamp}.png')
                with open(crop_path, 'wb') as f:
                    f.write(buffer.tobytes())
            except Exception:
                crop_path = None
        else:
            crop_path = None

        last_exception = None
        variant_index = 0
        for payload in payload_variants:
            variant_index += 1
            if temp_dir:
                try:
                    ppath = os.path.join(temp_dir, f'aiocr_payload_v{variant_index}.json')
                    with open(ppath, 'w', encoding='utf-8') as pf:
                        json.dump(payload, pf, ensure_ascii=False, indent=2)
                except Exception:
                    pass

            try:
                # provider-specific overrides
                pr_timeout = int(provider_cfg.get('timeout', 45) or 45)
                pr_retries = int(provider_cfg.get('retries', 2) or 2)
                pr_backoff = float(provider_cfg.get('backoff', 1.5) or 1.5)
                response = robust_post(url, headers=headers, json_payload=payload,
                                       timeout=pr_timeout, max_retries=pr_retries, backoff_factor=pr_backoff)
            except requests.RequestException as exc:
                last_exception = exc
                # save response text if any
                if temp_dir:
                    try:
                        errpath = os.path.join(temp_dir, f'aiocr_response_v{variant_index}_error.txt')
                        with open(errpath, 'w', encoding='utf-8') as ef:
                            ef.write(str(exc))
                    except Exception:
                        pass
                # try next variant
                continue

            try:
                data = response.json()
            except ValueError:
                # Save raw response for diagnostics
                if temp_dir:
                    try:
                        rpath = os.path.join(temp_dir, f'aiocr_response_v{variant_index}_raw.txt')
                        with open(rpath, 'w', encoding='utf-8') as rf:
                            rf.write(response.text)
                    except Exception:
                        pass
                # try next variant
                continue

            # Save provider response for debugging
            if temp_dir:
                try:
                    rjson_path = os.path.join(temp_dir, f'aiocr_response_v{variant_index}.json')
                    with open(rjson_path, 'w', encoding='utf-8') as rf:
                        json.dump(data, rf, ensure_ascii=False, indent=2)
                except Exception:
                    pass

            extracted = self._extract_ai_ocr_text(data)
            # if the model explicitly says there's no image, keep trying other variants
            if extracted and 'i cannot see any image' not in extracted.lower():
                # Optionally remove debug files for this run
                try:
                    if SETTINGS.get('cleanup', {}).get('remove_ai_temp_files') and temp_dir:
                        for p in (crop_path, ppath, rpath, rjson_path, errpath):
                            try:
                                if p and os.path.exists(p) and os.path.commonpath([os.path.abspath(p), os.path.abspath(temp_dir)]) == os.path.abspath(temp_dir):
                                    os.remove(p)
                            except Exception:
                                pass
                except Exception:
                    pass
                return extracted

        # If we reach here, all variants failed
        if last_exception:
            return f"[AI OCR REQUEST ERROR: {last_exception}]"
        return "[AI OCR ERROR: Empty or unrecognized response from provider]"
    
    def _call_mofrl_ocr(self, image_bgr, settings):
        """
        MOFRL-GPT: OCR berbasis GPT multimodal (OpenAI/Gemini/OpenRouter)
        Ambil API key dari SETTINGS['apis'][provider]['keys'][0]['value'].
        """
        import base64, cv2, json, requests, traceback

        try:
            translate_cfg = SETTINGS.get('translation', {})
            provider = translate_cfg.get('provider', 'OpenAI').lower()
            model = translate_cfg.get('model', 'gpt-5-nano').lower()

            apis_cfg = SETTINGS.get('apis', {})

            def extract_key(list_obj):
                if not list_obj:
                    return ""
                first = list_obj[0]
                if isinstance(first, dict):
                    return first.get("value", "")
                elif isinstance(first, str):
                    return first
                return ""

            # choose api_url and api_key based on provider
            api_url = ""
            api_key = ""
            if provider.startswith("openai"):
                api_url = "https://api.openai.com/v1/chat/completions"
                api_key = extract_key(apis_cfg.get('openai', {}).get('keys', []))
            elif provider.startswith("gemini"):
                api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
                api_key = extract_key(apis_cfg.get('gemini', {}).get('keys', []))
            elif provider.startswith("openrouter"):
                api_url = "https://openrouter.ai/api/v1/chat/completions"
                api_key = extract_key(apis_cfg.get('openrouter', {}).get('keys', []))
            else:
                return f"[MOFRL ERROR: Provider '{provider}' belum didukung]"

            if not api_key:
                return f"[MOFRL ERROR: API key kosong untuk provider {provider}]"

            # encode crop (raw) and save debug copy
            success, buffer = cv2.imencode('.png', image_bgr)
            if not success:
                return "[MOFRL ERROR: Gagal encode gambar]"
            image_b64 = base64.b64encode(buffer).decode('utf-8')

            prompt = (
                "This is a clear scanned image containing printed text.\n"
                "Your task is to carefully read every visible word and return the exact text as plain text.\n"
                "Keep line breaks and spacing the same.\n"
                "Do not summarize, describe, or reason — just return the literal text you can read."
            )

            # Prepare temp debug folder
            temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
            try:
                os.makedirs(temp_dir, exist_ok=True)
            except Exception:
                temp_dir = None

            if temp_dir:
                try:
                    # Put debug images under temp/img/mofrl/
                    img_dir = os.path.join(temp_dir, 'img', 'mofrl')
                    os.makedirs(img_dir, exist_ok=True)
                    ts = int(time.time())
                    crop_path = os.path.join(img_dir, f'mofrl_crop_{ts}.png')
                    with open(crop_path, 'wb') as cf:
                        cf.write(buffer.tobytes())
                except Exception:
                    pass

            headers = {"Content-Type": "application/json"}
            last_exception = None

            # For OpenAI/OpenRouter try several payload variants (data_uri, input_image, inline text)
            if provider.startswith('openai') or provider.startswith('openrouter'):
                headers["Authorization"] = f"Bearer {api_key}"
                token_field = "max_tokens"
                if model.startswith("gpt-5"):
                    token_field = "max_completion_tokens"

                payload_variants = []
                # Variant A: image_url with data URI
                payload_variants.append({
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                            ]
                        }
                    ],
                    token_field: 2048
                })

                # Variant B: input_image with image_data
                payload_variants.append({
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "input_image", "image_data": f"data:image/png;base64,{image_b64}"}
                            ]
                        }
                    ],
                    token_field: 2048
                })

                # Variant C: prompt + data URI in single content
                payload_variants.append({
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt + "\n\nImage: " + f"data:image/png;base64,{image_b64}"
                        }
                    ],
                    token_field: 2048
                })

                variant_index = 0
                for payload in payload_variants:
                    variant_index += 1
                    # save payload
                    if temp_dir:
                        try:
                            ppath = os.path.join(temp_dir, f'mofrl_payload_v{variant_index}.json')
                            with open(ppath, 'w', encoding='utf-8') as pf:
                                json.dump(payload, pf, ensure_ascii=False, indent=2)
                        except Exception:
                            pass

                    try:
                        # Use apis settings if available for timeout/retries/backoff
                        apis_provider_cfg = apis_cfg.get(provider, {}) or {}
                        pr_timeout = int(apis_provider_cfg.get('timeout', 90) or 90)
                        pr_retries = int(apis_provider_cfg.get('retries', 2) or 2)
                        pr_backoff = float(apis_provider_cfg.get('backoff', 1.5) or 1.5)
                        response = robust_post(api_url, headers=headers, json_payload=payload,
                                               timeout=pr_timeout, max_retries=pr_retries, backoff_factor=pr_backoff)
                    except requests.RequestException as exc:
                        last_exception = exc
                        if temp_dir:
                            try:
                                errpath = os.path.join(temp_dir, f'mofrl_response_v{variant_index}_error.txt')
                                with open(errpath, 'w', encoding='utf-8') as ef:
                                    ef.write(str(exc))
                            except Exception:
                                pass
                        continue

                    try:
                        resp_json = response.json()
                    except Exception:
                        if temp_dir:
                            try:
                                rpath = os.path.join(temp_dir, f'mofrl_response_v{variant_index}_raw.txt')
                                with open(rpath, 'w', encoding='utf-8') as rf:
                                    rf.write(response.text)
                            except Exception:
                                pass
                        continue

                    if temp_dir:
                        try:
                            rjson_path = os.path.join(temp_dir, f'mofrl_response_v{variant_index}.json')
                            with open(rjson_path, 'w', encoding='utf-8') as rf:
                                json.dump(resp_json, rf, ensure_ascii=False, indent=2)
                        except Exception:
                            pass

                    # extract text like AI OCR helper
                    extracted = self._extract_ai_ocr_text(resp_json)
                    if extracted and 'i cannot see any image' not in extracted.lower():
                        # Optionally remove debug files for this run
                        try:
                            if SETTINGS.get('cleanup', {}).get('remove_ai_temp_files') and temp_dir:
                                for p in (crop_path, ppath, rpath, rjson_path, errpath):
                                    try:
                                        if p and os.path.exists(p) and os.path.commonpath([os.path.abspath(p), os.path.abspath(temp_dir)]) == os.path.abspath(temp_dir):
                                            os.remove(p)
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                        return extracted

                if last_exception:
                    return f"[MOFRL ERROR: {last_exception}]"
                return "[MOFRL ERROR: Empty or unrecognized response from provider]"

            elif provider.startswith('gemini'):
                # Gemini expects inline_data and uses API key in query param
                api_url_with_key = api_url + f"?key={api_key}"
                payload = {
                    "contents": [
                        {
                            "parts": [
                                {"text": prompt},
                                {"inline_data": {"mime_type": "image/png", "data": image_b64}}
                            ]
                        }
                    ]
                }

                if temp_dir:
                    try:
                        ppath = os.path.join(temp_dir, 'mofrl_payload_gemini.json')
                        with open(ppath, 'w', encoding='utf-8') as pf:
                            json.dump(payload, pf, ensure_ascii=False, indent=2)
                    except Exception:
                        pass

                try:
                    response = requests.post(api_url_with_key, headers=headers, data=json.dumps(payload), timeout=90)
                    response.raise_for_status()
                except requests.RequestException as exc:
                    return f"[MOFRL ERROR: {exc}]"

                try:
                    resp_json = response.json()
                except Exception:
                    if temp_dir:
                        try:
                            rpath = os.path.join(temp_dir, 'mofrl_response_gemini_raw.txt')
                            with open(rpath, 'w', encoding='utf-8') as rf:
                                rf.write(response.text)
                        except Exception:
                            pass
                    return "[MOFRL ERROR: Invalid JSON from Gemini]"

                if temp_dir:
                    try:
                        rjson_path = os.path.join(temp_dir, 'mofrl_response_gemini.json')
                        with open(rjson_path, 'w', encoding='utf-8') as rf:
                            json.dump(resp_json, rf, ensure_ascii=False, indent=2)
                    except Exception:
                        pass

                # Try to extract content similar to existing code
                result = ""
                # check candidates/content structure
                candidates = resp_json.get('candidates') or []
                if candidates and isinstance(candidates[0], dict) and 'content' in candidates[0]:
                    parts = candidates[0]['content'].get('parts', [])
                    if parts:
                        result = '\n'.join(p.get('text', '') for p in parts if isinstance(p, dict)).strip()

                # fallback
                if not result:
                    result = resp_json.get('output_text') or resp_json.get('text') or ''
                    if isinstance(result, list):
                        result = '\n'.join([r for r in result if isinstance(r, str)])
                    result = (result or '').strip()

                if not result:
                    # Optionally remove debug files
                    try:
                        if SETTINGS.get('cleanup', {}).get('remove_ai_temp_files') and temp_dir:
                            for p in (crop_path, ppath, rpath, rjson_path):
                                try:
                                    if p and os.path.exists(p) and os.path.commonpath([os.path.abspath(p), os.path.abspath(temp_dir)]) == os.path.abspath(temp_dir):
                                        os.remove(p)
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    return "[MOFRL ERROR: hasil kosong]"
                try:
                    if SETTINGS.get('cleanup', {}).get('remove_ai_temp_files') and temp_dir:
                        for p in (crop_path, ppath, rpath, rjson_path):
                            try:
                                if p and os.path.exists(p) and os.path.commonpath([os.path.abspath(p), os.path.abspath(temp_dir)]) == os.path.abspath(temp_dir):
                                    os.remove(p)
                            except Exception:
                                pass
                except Exception:
                    pass
                return result

        except Exception as e:
            traceback.print_exc()
            return f"[MOFRL ERROR: {e}]"

    def _extract_ai_ocr_text(self, response_json):
        if not isinstance(response_json, dict):
            return ""

        choices = response_json.get('choices')
        if isinstance(choices, list) and choices:
            message = choices[0].get('message', {}) if isinstance(choices[0], dict) else {}
            content = message.get('content') if isinstance(message, dict) else None
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = []
                for chunk in content:
                    if isinstance(chunk, dict):
                        text_val = chunk.get('text') or chunk.get('content')
                        if isinstance(text_val, str) and text_val.strip():
                            parts.append(text_val.strip())
                if parts:
                    return '\n'.join(parts).strip()

        # Some providers might return 'message' directly as string
        message = response_json.get('message')
        if isinstance(message, str):
            return message.strip()

        if isinstance(message, dict):
            content = message.get('content')
            if isinstance(content, str):
                return content.strip()

        # Fallback to top-level 'text' or 'output_text'
        for key in ('text', 'output_text'):
            val = response_json.get(key)
            if isinstance(val, str):
                return val.strip()
            if isinstance(val, list):
                parts = [v.strip() for v in val if isinstance(v, str)]
                if parts:
                    return '\n'.join(parts)
        return ""

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

        select_all_btn = QPushButton("Select All Unsaved")
        select_all_btn.clicked.connect(self.select_all_unsaved)
        selection_layout.addWidget(select_all_btn, 1, 0, 1, 2)

        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(self.deselect_all)
        selection_layout.addWidget(deselect_all_btn, 1, 2, 1, 3)
        layout.addLayout(selection_layout)

        self.list_widget = QListWidget()
        self.populate_list()
        layout.addWidget(self.list_widget)

        button_layout = QHBoxLayout()
        self.save_button = QPushButton("Save Selected")
        self.save_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.save_button)
        layout.addLayout(button_layout)

    def populate_list(self):
        self.list_widget.clear()
        for file_path in self.all_files:
            if "_typeset" in file_path.lower():
                continue
            item = QListWidgetItem(os.path.basename(file_path))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)

            is_saved = self.main_app.check_if_saved(file_path)
            if is_saved:
                item.setCheckState(Qt.Unchecked)
                item.setForeground(QColor("gray"))
                item.setText(f"{os.path.basename(file_path)} [SAVED]")
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
                item.setCheckState(Qt.Checked)
                selected_count += 1
                if selected_count >= count:
                    break

    def select_all_unsaved(self):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if not self.main_app.check_if_saved(item.data(Qt.UserRole)):
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)

    def deselect_all(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.Unchecked)

    def get_selected_files(self):
        selected = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.Checked:
                selected.append(item.data(Qt.UserRole))
        return selected


if __name__ == "__main__":
    app = QApplication(sys.argv)

    try:
        import fitz
    except ImportError:
        QMessageBox.critical(None, "Dependency Missing", "PyMuPDF not installed. 'pip install PyMuPDF'.")
        sys.exit()

    if not DEEPL_API_KEY:
        QMessageBox.information(None, "DeepL", "No active DeepL key configured. You can add one in Settings ? API Manager.")
    if not GEMINI_API_KEY:
        QMessageBox.information(None, "Gemini", "No active Gemini key configured. You can add one in Settings ? API Manager.")

    try:
        pytesseract.get_tesseract_version()
    except Exception:
        QMessageBox.warning(None, "Tesseract Not Found", f"Tesseract not found at: {TESSERACT_PATH}\nPlease install it or set the correct path in Settings ? API Manager.")

    if not MangaOcr:
        QMessageBox.warning(None, "Manga-OCR Not Found", "'pip install manga-ocr' to enable the Manga-OCR engine.")
    if not onnxruntime:
        QMessageBox.warning(None, "ONNX Runtime Not Found", "'pip install onnxruntime' to enable some DL detectors.")
    if not paddleocr:
        QMessageBox.warning(None, "PaddleOCR Not Found", "'pip install paddleocr paddlepaddle' to enable the PaddleOCR engine.")
    if not YOLO:
        QMessageBox.warning(None, "Ultralytics Not Found", "'pip install ultralytics' to enable some DL detectors.")

    window = MangaOCRApp()
    window.showMaximized()
    sys.exit(app.exec_())
