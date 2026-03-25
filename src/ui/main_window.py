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
from src.core.workers import *
from src.core.fonts import *
from src.ui.dialogs import *
from src.ui.canvas import *
from src.utils.helpers import *
from src.utils.geometry import *
from src.core.models import EnhancedResult

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

    class ScrollableItemWidget(QScrollArea):
        def __init__(self, text, max_height=100, parent=None):
            super().__init__(parent)
            self.setWidgetResizable(True)
            self.setMaximumHeight(max_height)
            self.setFrameShape(QFrame.NoFrame)
            self.setStyleSheet("background: transparent;")
            # Content
            self.content = QLabel(text)
            self.content.setWordWrap(True)
            self.content.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            self.content.setStyleSheet("background: transparent; color: #ffffff;")
            self.content.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.setWidget(self.content)
            
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
        default_font_groups = {
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
        self.font_groups = copy.deepcopy(default_font_groups)

        settings_font_groups = SETTINGS.get('font_groups')
        if isinstance(settings_font_groups, dict):
            normalized_groups = {}
            for group_name, fonts in settings_font_groups.items():
                if not isinstance(group_name, str):
                    continue
                cleaned_fonts = []
                seen_fonts = set()
                if isinstance(fonts, (list, tuple)):
                    iterator = fonts
                elif isinstance(fonts, set):
                    iterator = sorted(fonts)
                elif isinstance(fonts, str):
                    iterator = [fonts]
                else:
                    iterator = []
                for font_name in iterator:
                    if not isinstance(font_name, str):
                        continue
                    normalized = font_name.strip()
                    if not normalized or normalized in seen_fonts:
                        continue
                    seen_fonts.add(normalized)
                    cleaned_fonts.append(normalized)
                normalized_groups[group_name] = cleaned_fonts
            self.font_groups = normalized_groups
            SETTINGS['font_groups'] = copy.deepcopy(self.font_groups)
        else:
            SETTINGS['font_groups'] = copy.deepcopy(self.font_groups)

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
            'scene': weakref.WeakSet(),
        }

        # --- Scene Management Data ---
        self.scenes = {}
        self.scene_order = []
        self.current_scene_name = None

        font_root = os.path.join(ROOT_DIR, 'src', 'fonts')
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
        style_val = (typeset_cfg.get('outline_style') or 'stroke')
        style_val = style_val.lower() if isinstance(style_val, str) else 'stroke'
        self.typeset_outline_style = style_val if style_val in ('stroke', 'glow') else 'stroke'
        self.preview_sample_text = "Aa Bb Cc"
        self.typeset_defaults = self._create_initial_typeset_defaults()

        self.processing_queue = []
        self.queue_mutex = QMutex()

        # Mutex to protect painting operations that use QPixmap/QImage paint devices
        self.paint_mutex = QMutex()

        self.batch_save_worker = None
        self.batch_save_thread = None

        self.project_dir = None
        self.project_dir = None
        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')
        if not os.path.exists(self.cache_dir):
            try:
                os.makedirs(self.cache_dir)
            except Exception:
                pass
        self.file_watcher = QFileSystemWatcher(self)
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
        autosave_cfg = SETTINGS.get('autosave', {}) if isinstance(SETTINGS.get('autosave'), dict) else {}
        try:
            autosave_interval = int(autosave_cfg.get('interval_ms', 300000))
        except Exception:
            autosave_interval = 300000
        autosave_interval = max(5000, autosave_interval)
        self.autosave_timer.setInterval(autosave_interval)
        self.autosave_timer.timeout.connect(self.auto_save_project)
        # Whether autosave is allowed to run (user preference). Persisted in settings.json.
        self.autosave_enabled = bool(autosave_cfg.get('enabled', True))

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

        # Keyboard Shortcuts for Copy/Paste Typeset
        self.shortcut_paste = QShortcut(QKeySequence("Ctrl+V"), self)
        self.shortcut_paste.activated.connect(self.paste_typeset_area)
        self.shortcut_copy = QShortcut(QKeySequence("Ctrl+C"), self)
        self.shortcut_copy.activated.connect(self.copy_selected_typeset_area)

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
        self._apply_right_panel_styles()

    def setup_right_panel(self):
        # Modernized right-panel layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Top header
        header = QLabel("Tools & Workflows")
        header.setObjectName("panel-title")
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
            (self._create_typeset_tab(), "Typeset"),
            (self._create_cleanup_tab(), "Cleanup"),
            (self._create_history_tab(), "History"),
            (self._create_scene_tab(), "Scenes"),
            (self._create_ai_hardware_tab(), "AI Hardware"),
        ]
        for widget, label in tab_order:
            self.tabs.addTab(widget, label)

        # Tidy tab bar appearance
        tab_bar = self.tabs.tabBar()
        try:
            tab_bar.setExpanding(False)
            tab_bar.setUsesScrollButtons(True)
            tab_bar.setElideMode(Qt.ElideNone)
            tab_bar.setToolTip('Seret tab untuk ubah urutan')
        except Exception:
            pass
        self.tabs.setStyleSheet("""
            QTabWidget::pane { 
                border-top: 1px solid #1f2b3b;
                background: transparent;
                margin-top: -1px;
            }
            QTabBar::tab {
                background: transparent;
                color: #7f8ba7;
                padding: 10px 14px;
                margin: 0px 4px 0px 0px;
                font-size: 13px;
                font-weight: 600;
                border: none;
                border-bottom: 3px solid transparent;
            }
            QTabBar::tab:selected {
                color: #3ba1ff;
                border-bottom: 3px solid #3ba1ff;
            }
            QTabBar::tab:hover:!selected {
                color: #cfd9e6;
                border-bottom: 3px solid #2a3a52;
            }
        """)

        tabs_layout.addWidget(self.tabs)

        # Scrollbar index yang lama dihapus agar desain lebih simpel dan memercayakan ke tab-bar scrolling bawaan.

        # Use a vertical splitter so the user gets 70% tabs and 30% actions proportionally, and can still drag it.
        from PyQt5.QtWidgets import QSplitter
        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.addWidget(tabs_frame)
        
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
        actions_frame.setProperty("panelCard", True)
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
        api_frame = QFrame(); api_frame.setFrameShape(QFrame.NoFrame); api_frame.setProperty("panelCard", True)
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
        
        right_splitter.addWidget(bottom_scroll)
        right_splitter.setStretchFactor(0, 5)
        right_splitter.setStretchFactor(1, 5)
        right_splitter.setSizes([500, 500]) # Fallback ratio

        main_layout.addWidget(right_splitter, 1)

        return main_layout

    def _apply_right_panel_styles(self):
        """Apply a cleaner, modern skin to the Tools & Workflows column."""
        try:
            panel_widget = self.right_panel_scroll.widget()
        except Exception:
            return
        if not panel_widget:
            return

        panel_widget.setStyleSheet("""
            #right-panel {
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0f1824, stop:0.5 #101c2a, stop:1 #0e1622);
                border-left: 1px solid #1f2b3a;
            }
            #right-panel QWidget { color: #e2e8f1; font-size: 10pt; }
            #right-panel QLabel#panel-title {
                color: #f3f7ff;
                letter-spacing: 0.5px;
                font-weight: 800;
            }
            #right-panel QGroupBox {
                background: rgba(255,255,255,0.03);
                border: 1px solid #233242;
                border-radius: 12px;
                margin-top: 10px;
                padding-top: 10px;
            }
            #right-panel QGroupBox::title {
                color: #9fc3f5;
                padding: 4px 6px;
                subcontrol-position: top left;
                left: 8px;
            }
            #right-panel QFrame[panelCard="true"] {
                background: rgba(255,255,255,0.03);
                border: 1px solid #233242;
                border-radius: 12px;
                padding: 10px;
            }
            #right-panel QLabel {
                color: #dfe6ef;
            }
            #right-panel QScrollArea {
                background: transparent;
                border: none;
            }
            #right-panel QPushButton {
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1f6fb5, stop:1 #2980c7);
                color: #e6f2ff;
                border: 1px solid #2d6ea8;
                border-radius: 10px;
                padding: 8px 12px;
                font-weight: 600;
            }
            #right-panel QPushButton:disabled {
                background: #243547;
                color: #7f8a96;
                border-color: #2f3c49;
            }
            #right-panel QPushButton:hover:!disabled {
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2a7fcb, stop:1 #3793de);
            }
            #right-panel QPushButton:pressed:!disabled {
                background: #1b5c92;
            }
            #right-panel QComboBox, #right-panel QLineEdit, #right-panel QSpinBox, #right-panel QDoubleSpinBox {
                background: #0f1a26;
                border: 1px solid #223246;
                border-radius: 6px;
                padding: 6px 8px;
                color: #e2e8f1;
            }
            #right-panel QComboBox::drop-down {
                width: 22px;
                border-left: 1px solid #223246;
            }
            #right-panel QCheckBox, #right-panel QRadioButton {
                color: #dfe6ef;
                spacing: 6px;
            }
        """)

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
        ocr_layout.addWidget(self.ocr_engine_info_label, 2, 0, 1, 2)

        # Force AI OCR Checkbox
        self.force_ai_ocr_checkbox = QCheckBox("Force Use AI OCR (Override Global Setting)")
        self.force_ai_ocr_checkbox.setChecked(bool(SETTINGS.get('force_ai_ocr', False)))
        self.force_ai_ocr_checkbox.setToolTip("If enabled, all OCR operations will use the AI provider selected below, regardless of the main toolbar selection.")
        self.force_ai_ocr_checkbox.stateChanged.connect(lambda val: (
            SETTINGS.update({'force_ai_ocr': bool(self.force_ai_ocr_checkbox.isChecked())}) or save_settings(SETTINGS) or self.ai_ocr_lang_combo.setEnabled(bool(val))
        ))
        ocr_layout.addWidget(self.force_ai_ocr_checkbox, 3, 0, 1, 2)

        # AI OCR Language Input
        ocr_layout.addWidget(QLabel("AI OCR Language:"), 4, 0)
        self.ai_ocr_lang_combo = QComboBox()
        self.ai_ocr_lang_combo.addItems([
            "Japanese", "English", "Korean", "Chinese (Simplified)", "Chinese (Traditional)",
            "Indonesian", "Portuguese (Brazil)", "Spanish", "French", "Russian", "Thai", "Vietnamese",
            "General/Multi-Language"
        ])
        
        # Load saved setting, default to Japanese
        saved_ai_lang = SETTINGS.get('ai_ocr_lang', 'Japanese')
        self.ai_ocr_lang_combo.setCurrentText(saved_ai_lang)
        self.ai_ocr_lang_combo.currentTextChanged.connect(lambda val: SETTINGS.update({'ai_ocr_lang': val}) or save_settings(SETTINGS))
        ocr_layout.addWidget(self.ai_ocr_lang_combo, 4, 1)

        self.translate_combo = self._create_combo_box(ocr_layout, "Translate to:", ["Indonesian", "English"], 5, 0, 1, 2, default="Indonesian")
        # Global orientation selector (kept for compatibility)
        self.orientation_combo = self._create_combo_box(ocr_layout, "Orientation:", ["Auto-Detect", "Horizontal", "Vertical"], 6, 0, 1, 2)
        
        # Per-language orientation overrides
        ocr_layout.addWidget(QLabel("EN Orientation:"), 7, 0)
        self.en_orientation_combo = QComboBox()
        self.en_orientation_combo.addItems(["Auto-Detect", "Horizontal", "Vertical"])
        self.en_orientation_combo.setCurrentText(SETTINGS.get('lang_orientation', {}).get('en', 'Auto-Detect'))
        self.en_orientation_combo.currentTextChanged.connect(lambda val: self._on_lang_orientation_changed('en', val))
        ocr_layout.addWidget(self.en_orientation_combo, 7, 1)

        ocr_layout.addWidget(QLabel("JP Orientation:"), 8, 0)
        self.jp_orientation_combo = QComboBox()
        self.jp_orientation_combo.addItems(["Auto-Detect", "Horizontal", "Vertical"])
        self.jp_orientation_combo.setCurrentText(SETTINGS.get('lang_orientation', {}).get('ja', 'Auto-Detect'))
        self.jp_orientation_combo.currentTextChanged.connect(lambda val: self._on_lang_orientation_changed('ja', val))
        ocr_layout.addWidget(self.jp_orientation_combo, 8, 1)

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
        self.confirm_button = QPushButton("Confirm"); self.confirm_button.clicked.connect(self.confirm_pen_selection); self.confirm_button.setVisible(False)
        self.cancel_button = QPushButton("Cancel"); self.cancel_button.clicked.connect(self.cancel_pen_selection); self.cancel_button.setVisible(False)
        pen_buttons_layout.addWidget(self.confirm_button); pen_buttons_layout.addWidget(self.cancel_button)
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

        gradient_group = QGroupBox("Gradient Coloring")
        gradient_group.setCheckable(True)
        self.gradient_group = gradient_group
        self.gradient_group.toggled.connect(self._on_typeset_gradient_toggled)
        
        grad_layout = QVBoxLayout(gradient_group)
        grad_layout.setContentsMargins(12, 14, 12, 12)
        grad_layout.setSpacing(10)

        angle_row = QHBoxLayout()
        angle_row.addWidget(QLabel("Angle:"))
        self.grad_angle_spin = QDoubleSpinBox()
        self.grad_angle_spin.setRange(0.0, 360.0)
        self.grad_angle_spin.setSingleStep(15.0)
        self.grad_angle_spin.setSuffix(" °")
        self.grad_angle_spin.valueChanged.connect(self._on_typeset_gradient_changed)
        angle_row.addWidget(self.grad_angle_spin)
        grad_layout.addLayout(angle_row)
        
        grad_layout.addWidget(QLabel("Colors:"))
        self.grad_color_list = QListWidget()
        self.grad_color_list.setFixedHeight(80)
        # Apply style to list items for color preview
        grad_layout.addWidget(self.grad_color_list)
        
        btn_row = QHBoxLayout()
        add_c_btn = QPushButton("Add")
        add_c_btn.clicked.connect(self._on_add_gradient_color)
        rem_c_btn = QPushButton("Remove")
        rem_c_btn.clicked.connect(self._on_remove_gradient_color)
        btn_row.addWidget(add_c_btn)
        btn_row.addWidget(rem_c_btn)
        grad_layout.addLayout(btn_row)
        
        layout.addWidget(gradient_group)

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
        self.line_spacing_input = QDoubleSpinBox()
        self.line_spacing_input.setRange(0.6, 3.0)
        self.line_spacing_input.setDecimals(2)
        self.line_spacing_input.setSingleStep(0.05)
        self.line_spacing_input.setToolTip("Adjust the spacing between lines (0.60x - 3.00x).")
        self.line_spacing_input.valueChanged.connect(self._on_typeset_line_spacing_changed)
        line_row.addWidget(self.line_spacing_input, 1)
        self.line_spacing_value_label = QLabel("1.00x")
        self.line_spacing_value_label.setMinimumWidth(60)
        line_row.addWidget(self.line_spacing_value_label)
        spacing_layout.addLayout(line_row, 1, 1)

        spacing_layout.addWidget(QLabel("Character Spacing"), 2, 0)
        char_row = QHBoxLayout()
        char_row.setSpacing(8)
        self.char_spacing_input = QDoubleSpinBox()
        self.char_spacing_input.setRange(10.0, 400.0)
        self.char_spacing_input.setDecimals(0)
        self.char_spacing_input.setSingleStep(1.0)
        self.char_spacing_input.setSuffix(" %")
        self.char_spacing_input.setToolTip("Adjust spacing between characters (percentage).")
        self.char_spacing_input.valueChanged.connect(self._on_typeset_char_spacing_changed)
        char_row.addWidget(self.char_spacing_input, 1)
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

    def populate_ai_models_combo(self, combo: QComboBox):
        combo.clear()
        combo.addItem("Default (Main Setting)", "default")
        
        # Try to populate from self.models if available (custom/local models)
        if hasattr(self, 'models') and self.models:
            for model in self.models:
                if model.get('active', True):
                    name = model.get('name', '')
                    mid = model.get('id', '')
                    if name:
                        combo.addItem(f"Custom: {name}", mid)
        
        # Add Common Cloud Models (Standard) - hardcoded fallbacks or extras
        common_models = [
            ("Google: Gemini 1.5 Pro", "gemini-1.5-pro"),
            ("Google: Gemini 1.5 Flash", "gemini-1.5-flash"),
            ("OpenAI: GPT-4o", "gpt-4o"),
            ("OpenAI: GPT-4o Mini", "gpt-4o-mini"),
        ]
        for name, mid in common_models:
             combo.addItem(name, mid)

    def get_selected_model_name(self):
        """Returns (provider, model_name) from current global SETTINGS."""
        translate_cfg = SETTINGS.get('translation', {})
        provider = translate_cfg.get('provider', 'Google') # Defaulting to Google if missing
        model = translate_cfg.get('model', 'gemini-1.5-flash')
        return provider, model

    def _invoke_ai_review(self, provider, model_name, full_prompt):
        # 1. Logging Setup
        cache_dir = os.path.join(self.base_dir, ".cache")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        log_file = os.path.join(cache_dir, "scene_ai_log.txt")
        
        def log_transaction(stage, content):
            try:
                with open(log_file, "a", encoding="utf-8") as f:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"\n[{timestamp}] [{stage}] [{provider}:{model_name}]\n{content}\n" + "-"*40 + "\n")
            except Exception as e:
                print(f"Log Error: {e}")

        # Log Input
        log_transaction("INPUT", full_prompt)

        # 2. Resolve Provider/Model (Standardize)
        if not provider or provider == 'default': 
            provider, model = self.get_selected_model_name()
            if model_name and model_name != 'default': model = model_name 
            model_name = model
            
        provider = provider.lower()
        response_text = ""
        error_msg = ""

        try:
            # --- GOOGLE / GEMINI ---
            if "google" in provider or "gemini" in provider:
                api_key = SETTINGS.get("api_key_gemini")
                if not api_key:
                    raise Exception("Gemini API Key missing in SETTINGS.")
                
                # Check for cached model instance or create new
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(model_name)
                # Apply standard safety settings if available
                safety = getattr(self, 'translation_safety_settings', None)
                response = model.generate_content(full_prompt, safety_settings=safety)
                response_text = response.text if response else ""

            # --- OPENAI / DEEPSEEK / OPENROUTER ---
            elif "openai" in provider or "deepseek" in provider or "openrouter" in provider:
                # Use shared client or init
                api_key = SETTINGS.get("api_key_openai")
                base_url = SETTINGS.get("openai_base_url", "https://api.openai.com/v1")
                
                # If provider is explicitly openrouter/deepseek via combo string, we might need adjustments
                # But typically main.py handles this via SETTINGS overrides in the Hardware tab
                # We should primarily rely on the global 'openai_client' or re-init logic used in translate_with_openai
                
                client = getattr(self, "openai_client", None)
                if not client or client.api_key != api_key:
                     # Re-init if missing or key changed (simplified)
                     client = openai.OpenAI(api_key=api_key, base_url=base_url)
                
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."}, # Minimal system for Scene
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=0.7 # Standard default
                )
                response_text = resp.choices[0].message.content

            else:
                # Fallback or unknown
                raise Exception(f"Unknown provider: {provider}")

        except Exception as e:
            error_msg = str(e)
            log_transaction("ERROR", error_msg)
            print(f"AI Review Error: {e}")
            return f"[AI ERROR: {e}]"

        # Log Output
        log_transaction("OUTPUT", response_text)
        
        return response_text


    def _create_scene_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 15, 10, 10)
        
        # Scene Selector & Management
        mgmt_layout = QHBoxLayout()
        self.scene_selector = QComboBox()
        self.scene_selector.currentIndexChanged.connect(self.on_scene_selection_changed)
        mgmt_layout.addWidget(QLabel("Scene:"))
        mgmt_layout.addWidget(self.scene_selector, 1)
        
        new_scene_btn = QPushButton("New")
        new_scene_btn.clicked.connect(self.prompt_create_scene)
        mgmt_layout.addWidget(new_scene_btn)
        
        del_scene_btn = QPushButton("Delete")
        del_scene_btn.clicked.connect(self.prompt_delete_scene)
        mgmt_layout.addWidget(del_scene_btn)
        
        layout.addLayout(mgmt_layout)
        
        # Description / Info
        self.scene_info_label = QLabel("Select or create a scene to manage dialogues.")
        self.scene_info_label.setWordWrap(True)
        layout.addWidget(self.scene_info_label)

        # AI Tools for Scene
        ai_tools_layout = QHBoxLayout()
        ai_params_group = QGroupBox("AI Scene Tools")
        ai_group_layout = QHBoxLayout(ai_params_group)
        ai_group_layout.setContentsMargins(5, 5, 5, 5)
        
        self.scene_pf_btn = QPushButton("Proofread Scene")
        self.scene_pf_btn.clicked.connect(lambda: self.process_scene_with_ai("proofreading"))
        ai_group_layout.addWidget(self.scene_pf_btn)
        
        self.scene_qc_btn = QPushButton("Quality Check")
        self.scene_qc_btn.clicked.connect(lambda: self.process_scene_with_ai("quality"))
        ai_group_layout.addWidget(self.scene_qc_btn)
        
        self.scene_natural_btn = QPushButton("Naturalize")
        self.scene_natural_btn.clicked.connect(lambda: self.process_scene_with_ai("naturalization"))
        ai_group_layout.addWidget(self.scene_natural_btn)
        
        ai_tools_layout.addWidget(ai_params_group)

        # Scene Model Selector
        model_layout = QHBoxLayout()
        
        # Checkbox to sync with main AI Hardware tab
        self.use_main_model_checkbox = QCheckBox("Use Main Model (from Hardware Tab)")
        self.use_main_model_checkbox.setChecked(True)
        self.use_main_model_checkbox.setToolTip("If checked, uses the model selected in the AI Hardware configuration.")
        model_layout.addWidget(self.use_main_model_checkbox)

        # Scene-specific combo (hidden if sync is on)
        self.scene_model_combo = QComboBox()
        self.populate_ai_models_combo(self.scene_model_combo)
        self.scene_model_combo.setVisible(False)
        model_layout.addWidget(self.scene_model_combo, 1)
        
        # Toggle visibility logic
        self.use_main_model_checkbox.toggled.connect(lambda checked: self.scene_model_combo.setVisible(not checked))

        ai_group_layout.insertLayout(0, model_layout)

        layout.addLayout(ai_tools_layout)
        
        # Table
        self.scene_table = self._create_result_table()
        self.scene_table.setProperty('result_limit', None)
        self.result_table_registry['scene'].add(self.scene_table)
        layout.addWidget(self.scene_table)
        
        # Apply Actions
        apply_layout = QHBoxLayout()
        self.apply_scene_btn = QPushButton("Apply All to Canvas")
        self.apply_scene_btn.setToolTip("Apply current text in this scene to the actual bubbles on canvas.")
        self.apply_scene_btn.clicked.connect(self.apply_scene_to_canvas)
        apply_layout.addStretch()
        apply_layout.addWidget(self.apply_scene_btn)
        layout.addLayout(apply_layout)
        
        self.refresh_scene_ui_state()
        return tab

    def refresh_scene_ui_state(self):
        # Update selector
        current_idx = -1
        self.scene_selector.blockSignals(True)
        self.scene_selector.clear()
        for i, name in enumerate(self.scene_order):
            self.scene_selector.addItem(name)
            if name == self.current_scene_name:
                current_idx = i
        self.scene_selector.blockSignals(False)
        
        if current_idx >= 0:
            self.scene_selector.setCurrentIndex(current_idx)
            
        has_scene = bool(self.current_scene_name)
        self.scene_table.setVisible(has_scene)
        self.scene_pf_btn.setEnabled(has_scene)
        self.scene_qc_btn.setEnabled(has_scene)
        self.scene_natural_btn.setEnabled(has_scene)
        self.apply_scene_btn.setEnabled(has_scene)
        
        if has_scene:
            count = len(self.scenes.get(self.current_scene_name, []))
            self.scene_info_label.setText(f"Scene: {self.current_scene_name} ({count} items)")
        else:
            self.scene_info_label.setText("No scene selected.")

    def on_scene_selection_changed(self, index):
        if index < 0:
            return
        name = self.scene_selector.itemText(index)
        self.current_scene_name = name
        self.refresh_history_views()
        self.refresh_scene_ui_state()

    def prompt_create_scene(self):
        name, ok = QInputDialog.getText(self, "New Scene", "Scene Name:")
        if ok and name:
            if not self.create_scene(name):
                QMessageBox.warning(self, "Error", "Could not create scene. Name might be empty or duplicate.")
            else:
                self.scene_selector.setCurrentText(name) 
                self.refresh_scene_ui_state() # Force UI refresh

    def prompt_delete_scene(self):
        if not self.current_scene_name:
            return
        confirm = QMessageBox.question(self, "Confirm Delete", f"Delete scene '{self.current_scene_name}'?", QMessageBox.Yes | QMessageBox.No)
        if confirm == QMessageBox.Yes:
            self.delete_scene(self.current_scene_name)
            self.refresh_scene_ui_state() # Update state

    def process_scene_with_ai(self, mode):
        if not self.current_scene_name: return
        entries = self.scenes.get(self.current_scene_name, [])
        if not entries:
            QMessageBox.information(self, "Empty", "Scene is empty.")
            return

        # Determine Model
        if self.use_main_model_checkbox.isChecked():
            # Real-time retrieval from main ai_model_combo
            if hasattr(self, 'ai_model_combo'):
                 full_text = self.ai_model_combo.currentText()
                 if not full_text or not self.ai_model_combo.isEnabled(): 
                      QMessageBox.warning(self, "Model Error", "Please select an AI model in the Hardware/Settings tab first.")
                      return
                 # Parse "Provider: Model" or just "Model"
                 if ":" in full_text:
                     parts = full_text.split(":", 1)
                     provider = parts[0].strip().lower()
                     model_name = parts[1].strip()
                 else:
                     # Fallback assumption if just model name
                     provider = "google" 
                     model_name = full_text
            else:
                 QMessageBox.warning(self, "Error", "Main AI Model configuration not found.")
                 return
        else:
            # Use specific scene combo
            full_text = self.scene_model_combo.currentText()
            if ":" in full_text:
                parts = full_text.split(":", 1)
                provider = parts[0].strip().lower()
                model_name = parts[1].strip()
            else:
                p, m = self.get_selected_model_name()
                provider = p
                model_name = full_text

        if not model_name:
             QMessageBox.warning(self, "No Model", "Select AI Model first.")
             return

        # Context-aware Prompting (Batch Bottom-Up)
        # entries is [Old, ..., New] if relying on insertion order.
        # User wants "Bottom Item" as #1.
        # In UI (as per populate_result_table reversal), Bottom Item is Oldest.
        # So "Text 1" = Oldest. 
        # We iterate entries in chronological order (0..N).
        # Payload construction:
        
        numbered_texts = []
        for i, e in enumerate(entries, 1):
             # Original Text or Translated Text? usually we refine translated text or original?
             # User said: "Hasil OCR + Terjemahan". "Prompt Utama: ... [Text 1]: (Hasil OCR + Terjemahan) ..."
             ocr = e.get('original_text', '')
             trans = e.get('translated_text', '')
             if not trans: trans = "[No Translation]"
             numbered_texts.append(f"[Text {i}]:\nOCR: {ocr}\nCurrent Translation: {trans}")
        
        full_text_block = "\n\n".join(numbered_texts)
        
        prompts = {
            "proofreading": "Fix grammar, typos, and punctuation. Keep the style consistent.",
            "quality": "Check for accuracy, consistency, and missed nuances.",
            "naturalization": "Make dialogues natural, fluent, and localized (e.g. Indonesian context if applicable). Maintain the meaning."
        }
        
        base_prompt = prompts.get(mode, prompts["proofreading"])
        full_prompt = (
            f"You are an expert manga typesetter/translator. Mode: {mode.upper()}.\n"
            f"{base_prompt}\n"
            f"Your task: Process the following batch of texts (Story Order: Text 1 is start).\n"
            f"Requirements:\n"
            f"1. Return the output for EACH text block using the tag [Text X].\n"
            f"2. Output Format:\n"
            f"[Text 1]\n(Your improved text here)\n\n"
            f"[Text 2]\n(Your improved text here)\n\n"
            f"... and so on.\n"
            f"3. Do NOT include explanations, just the tags and the final text.\n"
            f"4. Maintain the count of texts exactly.\n\n"
            f"Input Batch:\n{full_text_block}"
        )

        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage(f"Processing scene ({mode})...")
        response_text = ""
        try:
            response_text = self._invoke_ai_review(provider, model_name, full_prompt)
        finally:
            QApplication.restoreOverrideCursor()
            self.statusBar().clearMessage()

        if not response_text:
            QMessageBox.warning(self, "Error", "AI request failed or empty response.")
            return

        # Parse Output (Tag-based)
        import re
        results = []
        # Fallback to list parse if tags missing? But we instructed tags.
        # We try to extract content between [Text X]
        for i in range(1, len(entries) + 1):
            pattern = fr"\[Text {i}\](.*?)(\[Text {i+1}\]|$)"
            # Handle last item specifically or use generic split
            # Simpler: Split by \[Text \d+\]
            pass
        
        # Regex split
        # We expect [Text 1] ... [Text 2] ...
        # Let's clean headers
        
        # Robust parsing:
        parsed_map = {}
        tokens = re.split(r'\[Text (\d+)\]', response_text, flags=re.IGNORECASE | re.DOTALL)
        # tokens[0] is pre-text (empty), tokens[1] is number, tokens[2] is content, tokens[3] is number, ...
        
        for k in range(1, len(tokens), 2):
            try:
                idx = int(tokens[k])
                content = tokens[k+1].strip()
                parsed_map[idx] = content
            except Exception:
                pass
        
        # Construct result list in order
        final_results = []
        for i in range(1, len(entries) + 1):
            final_results.append(parsed_map.get(i, entries[i-1].get('translated_text', ''))) # Fallback to current if missing

        # Open Review Dialog
        # Current texts for comparison
        current_texts = [e.get('translated_text', '') for e in entries]
        
        dialog = SceneReviewDialog(self, current_texts, final_results)
        if dialog.exec_() == QDialog.Accepted:
            accepted = dialog.accepted_indices # list of (index, new_text) based on chronological 0..N
            count = 0
            for idx, new_txt in accepted:
                if 0 <= idx < len(entries):
                    entries[idx]['translated_text'] = new_txt
                    entries[idx]['ai_model'] = f"{model_name} ({mode})"
                    count += 1
            
            self.refresh_history_views()
            QMessageBox.information(self, "Done", f"Applied changes to {count} items.")

    def apply_scene_to_canvas(self):
        if not self.current_scene_name: return
        entries = self.scenes.get(self.current_scene_name, [])
        count = 0
        for entry in entries:
            hist_id = entry.get('id') or entry.get('history_id')
            if not hist_id: continue
            
            # Check if we should update canvas
            # Use 'apply_history_update' which looks up by history_id
            if self.apply_history_update(hist_id, translated_text=entry.get('translated_text')):
                count += 1
        
        self.redraw_all_typeset_areas()
        QMessageBox.information(self, "Applied", f"Applied {count} updates to canvas.")

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

        table.blockSignals(True)
        table.setRowCount(len(entries))

        # [MODIFIED] Reversed display for Scene (Newest Top, Oldest Bottom)
        display_iter = list(enumerate(entries))
        if source == 'scene':
            display_iter = list(reversed(list(enumerate(entries))))

        for row_idx, (original_idx, entry) in enumerate(display_iter):
            # Row in table is row_idx (0..N)
            # original_idx is index in the source list (0..N)
            
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
            # User Request: Bottom Item is #1. Top is #N.
            # If source is scene, we are iterating in reverse order (Top row is Newest item).
            # So Top Row (row_idx=0) should have Label = N.
            # Bottom Row (row_idx=N) should have Label = 1.
            # original_idx is the index in the chronological list (0=Oldest, N=Newest).
            # So Label = original_idx + 1. 
            # Example: 3 items. List=[Old, Mid, New] -> idx=[0, 1, 2].
            # Display Loop: (2, New), (1, Mid), (0, Old).
            # Row 0: New. idx=0. Total=3. Label -> 3.
            # Row 2: Old. idx=2. Total=3. Label -> 1.
            
            label_num = row_idx + 1
            if source == 'scene':
                # Force Bottom=1 logic (Stack Order: Top=N, Bottom=1)
                label_num = len(entries) - row_idx
                
            number_item = QTableWidgetItem(str(label_num))
            number_item.setTextAlignment(Qt.AlignTop | Qt.AlignCenter)
            table.setItem(row_idx, 0, number_item)

            # NOTE: Use 'row_idx' for table.setItem, not row!
            row = row_idx # remap for existing valid code below
            # Row 0: New. Label -> 3 (2+1). Correct.
            # Row 2: Old. Label -> 1 (0+1). Correct.
            
            label_num = row_idx + 1
            if source == 'scene':
                label_num = original_idx + 1
                
            number_item = QTableWidgetItem(str(label_num))
            number_item.setTextAlignment(Qt.AlignTop | Qt.AlignCenter)
            table.setItem(row_idx, 0, number_item)

            # NOTE: Use 'row_idx' for table.setItem, not row!
            row = row_idx # remap for existing valid code below

            # Column 1: Original (Scrollable)
            if display_original:
                 # We use setCellWidget for scrollable content
                 # But first set a dummy item for sorting if needed (optional)
                 table.setItem(row, 1, QTableWidgetItem("")) 
                 scroll_widget = self.ScrollableItemWidget(display_original, max_height=120)
                 table.setCellWidget(row, 1, scroll_widget)
            else:
                 table.setItem(row, 1, QTableWidgetItem(""))

            translated_text = entry.get('translated_text', '')
            model_label = entry.get('ai_model')
            if model_label:
                display_translated = f"{translated_text}\n(Model: {model_label})" if translated_text else f"(Model: {model_label})"
            else:
                display_translated = translated_text
            
            # Column 2: Translated (Scrollable)
            if display_translated:
                 table.setItem(row, 2, QTableWidgetItem(""))
                 scroll_widget = self.ScrollableItemWidget(display_translated, max_height=120)
                 # Apply staged style if needed
                 if entry.get('staged'):
                     scroll_widget.setStyleSheet("background: #313a3c;")
                 table.setCellWidget(row, 2, scroll_widget)
            else:
                 table.setItem(row, 2, QTableWidgetItem(""))

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
            if source == 'history':
                to_scene_btn = style_button(QPushButton("To Scene", action_widget))
                to_scene_btn.clicked.connect(lambda _, hid=history_id: self.prompt_send_to_scene(hid))
                action_layout.addWidget(to_scene_btn, 0, 0)
                
                edit_button = style_button(QPushButton("Edit", action_widget))
                edit_button.clicked.connect(partial(self.open_result_editor, history_id, source))
                action_layout.addWidget(edit_button, 0, 1)
            else:
                edit_button = style_button(QPushButton("Edit", action_widget))
                edit_button.clicked.connect(partial(self.open_result_editor, history_id, source))
                action_layout.addWidget(edit_button, 0, 0)

            confirm_button = style_button(QPushButton("Confirm", action_widget))
            confirm_button.clicked.connect(partial(self.confirm_result_entry, history_id, source))
            action_layout.addWidget(confirm_button, 0, 1)

            # Delete
            delete_button = style_button(QPushButton("Delete", action_widget))
            delete_button.clicked.connect(partial(self.remove_result_entry, source, history_id))
            action_layout.addWidget(delete_button, 1, 0, 1, 2)

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

    def prompt_send_to_scene(self, history_id):
        if not self.scenes:
             # Offer to create one
             resp = QMessageBox.question(self, "No Scenes", "No scenes exist. Create a new scene?", QMessageBox.Yes | QMessageBox.No)
             if resp == QMessageBox.Yes:
                  self.prompt_create_scene()
                  if not self.scenes: return # Cancelled or failed
             else:
                  return

        entry = self.get_history_entry(history_id)
        if not entry: return

        # Dialog to choose scene
        scenes_list = self.scene_order
        item, ok = QInputDialog.getItem(self, "Add to Scene", "Select Scene:", scenes_list, 0, False)
        if ok and item:
             self.add_entry_to_scene(item, entry)
             self.statusBar().showMessage(f"Added to scene '{item}'.", 3000)

    def prompt_send_to_scene(self, history_id):
        if not self.scenes:
             # Offer to create one
             resp = QMessageBox.question(self, "No Scenes", "No scenes exist. Create a new scene?", QMessageBox.Yes | QMessageBox.No)
             if resp == QMessageBox.Yes:
                  self.prompt_create_scene()
                  if not self.scenes: return # Cancelled or failed
             else:
                  return

        entry = self.get_history_entry(history_id)
        if not entry: return

        # Dialog to choose scene
        scenes_list = self.scene_order
        item, ok = QInputDialog.getItem(self, "Add to Scene", "Select Scene:", scenes_list, 0, False)
        if ok and item:
             self.add_entry_to_scene(item, entry)
             self.statusBar().showMessage(f"Added to scene '{item}'.", 3000)

    def prompt_send_to_scene(self, history_id):
        if not self.scenes:
             # Offer to create one
             resp = QMessageBox.question(self, "No Scenes", "No scenes exist. Create a new scene?", QMessageBox.Yes | QMessageBox.No)
             if resp == QMessageBox.Yes:
                  self.prompt_create_scene()
                  if not self.scenes: return # Cancelled or failed
             else:
                  return

        entry = self.get_history_entry(history_id)
        if not entry: return

        # Dialog to choose scene
        scenes_list = self.scene_order
        item, ok = QInputDialog.getItem(self, "Add to Scene", "Select Scene:", scenes_list, 0, False)
        if ok and item:
             self.add_entry_to_scene(item, entry)
             self.statusBar().showMessage(f"Added to scene '{item}'.", 3000)

    def prompt_send_to_scene(self, history_id):
        if not self.scenes:
             # Offer to create one
             resp = QMessageBox.question(self, "No Scenes", "No scenes exist. Create a new scene?", QMessageBox.Yes | QMessageBox.No)
             if resp == QMessageBox.Yes:
                  self.prompt_create_scene()
                  if not self.scenes: return # Cancelled or failed
             else:
                  return

        entry = self.get_history_entry(history_id)
        if not entry: return

        # Dialog to choose scene
        scenes_list = self.scene_order
        item, ok = QInputDialog.getItem(self, "Add to Scene", "Select Scene:", scenes_list, 0, False)
        if ok and item:
             self.add_entry_to_scene(item, entry)
             self.statusBar().showMessage(f"Added to scene '{item}'.", 3000)

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

    def create_scene(self, name):
        if not name or not isinstance(name, str):
            return False
        name = name.strip()
        if not name:
            return False
        if name in self.scenes:
            return False
        self.scenes[name] = []
        self.scene_order.append(name)
        self.current_scene_name = name
        self.refresh_history_views()
        return True

    def delete_scene(self, name):
        if name not in self.scenes:
            return False
        if name == "Deleted History": # Protect this special scene
            QMessageBox.warning(self, "Cannot Delete", "The 'Deleted History' scene cannot be deleted.")
            return False

        del self.scenes[name]
        if name in self.scene_order:
            self.scene_order.remove(name)
        
        if self.current_scene_name == name:
            self.current_scene_name = self.scene_order[0] if self.scene_order else None
        
        self.refresh_history_views()
        return True

    def add_entry_to_scene(self, scene_name, entry):
        if scene_name not in self.scenes:
            return False
        
        # Clone entry
        new_entry = copy.deepcopy(entry)
        # Ensure it has an ID
        if not new_entry.get('id'):
            new_entry['id'] = self.generate_history_id()
        
        # Avoid duplicates in the scene (based on ID)
        existing_ids = {e.get('id') for e in self.scenes[scene_name]}
        if new_entry.get('id') in existing_ids:
             # Logic choice: update existing or skip? Let's append duplicate if user wants, 
             # but to be safe maybe we should just allow it or warn?
             # For now, let's just append. User can reorder/delete.
             pass

        self.scenes[scene_name].insert(0, new_entry)
        self.refresh_history_views()
        return True

    def move_entry_to_deleted_history(self, history_id):
        # Find the entry in history
        entry = self.get_history_entry(history_id)
        if not entry:
            # Maybe it's in a scene? Or maybe we just can't find it.
            # If we strictly want to preserve DELETED items from canvas, 
            # we should look it up from the area before it's gone?
            # Actually catch: delete_typeset_area calls this.
            return

        target_scene = "Deleted History"
        if target_scene not in self.scenes:
            self.create_scene(target_scene)
        
        # Check if already in deleted history
        deleted_list = self.scenes[target_scene]
        if any(e.get('id') == history_id for e in deleted_list):
            return

        # Clone and add
        new_entry = copy.deepcopy(entry)
        # Mark as deleted from canvas
        new_entry['deleted_from_canvas'] = True
        new_entry['deletion_timestamp'] = time.time()
        
        self.scenes[target_scene].insert(0, new_entry)
        
        # NOTE: Do we remove from main history? 
        # Requirement: "pindahkan secara otomatis" (move automatically).
        # So yes, we should probably remove from the main history view 
        # OR just keep it in history but mark it?
        # User said "Deleted History" category/folder.
        # Let's keep it simple: Add to scene "Deleted History". 
        # Removing from self.history_entries might confuse the "History" tab which logs *everything*.
        # But if the user says "item ... dipindahkan", it implies move.
        # Let's remove from history entries to be safe/clean.
        try:
             self.history_entries.remove(entry)
             if history_id in self.history_lookup:
                 del self.history_lookup[history_id]
        except ValueError:
             pass

        self.refresh_history_views()
    
    def get_scene_entries(self, scene_name):
        return self.scenes.get(scene_name, [])

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
        elif source == 'history':
            # Remove from history list
            self.history_entries = [e for e in self.history_entries if (e.get('history_id') or e.get('id')) != history_id]
            # Also remove from global lookup if exists
            if history_id in self.history_lookup:
                del self.history_lookup[history_id]
            # Note: We do NOT remove from all_typeset_data here automatically, as history might be just a log.
            # But if the user wants it gone, we should probably remove it. 
            # However, safe approach is just removing from history view.
        elif source == 'scene':
            if self.current_scene_name:
                scene_list = self.scenes.get(self.current_scene_name, [])
                self.scenes[self.current_scene_name] = [e for e in scene_list if (e.get('history_id') or e.get('id')) != history_id]
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
            # Filter history by current image
            source_entries = self.history_entries
            if self.current_image_path:
                source_entries = [e for e in source_entries if e.get('image_key') == self.current_image_path]
            entries = self._get_recent_entries(source_entries, limit=None)
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
            'save_image': self.save_image,
            'confirm_pen': self.confirm_pen_via_shortcut,
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
            key = ((event_type or '').lower(), button)
            cb = self._mouse_shortcuts.get(key)
            if cb:
                try:
                    result = cb()
                except TypeError:
                    try:
                        result = cb()
                    except Exception:
                        result = None
                return True if result is None else bool(result)
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
        mouse_sources = list((self._shortcut_callbacks or {}).items())
        # Allow action-backed shortcuts (menu actions) to be triggered by mouse bindings too
        for key, action in (self._action_shortcut_map or {}).items():
            if action is not None:
                mouse_sources.append((key, action.trigger))
        for key, callback in mouse_sources:
            seq = (self.shortcut_sequences.get(key, '') or '').strip()
            if not seq or not seq.upper().startswith('MOUSE:'):
                continue
            parts = seq.split(':')
            if len(parts) >= 3:
                evt = parts[1].lower()
                btn = mouse_name_to_button(parts[2])
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
        model_key = settings.get('inpaint_model_key')

        # Jika pengguna memilih mode OpenCV (atau tidak memilih model LaMa sama sekali),
        # pastikan state lama_cleaner dilepas agar tidak dicoba lagi.
        if not model_key:
            self.inpaint_model = None
            self.current_inpaint_model_key = None
            return

        if model_key == self.current_inpaint_model_key and self.inpaint_model is not None:
            return

        if not self.is_lama_available:
            print("Lama Cleaner not available; falling back to OpenCV inpaint.")
            self.inpaint_model = None
            self.current_inpaint_model_key = None
            return

        model_info = self.dl_models.get(model_key)
        if not model_info or not os.path.exists(model_info['path']):
            print(f"Model file not found: {model_info['path'] if model_info else 'None'}")
            self.inpaint_model = None
            self.current_inpaint_model_key = None
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
                f"- If a Japanese text includes a kanji followed by parentheses like 漢字(かんじ) or word(note), treat the text inside parentheses as a reading or small note — do NOT translate it literally.\n"
                f"- Translate only based on the main kanji meaning, not the content in parentheses.\n"
                f"- Example: 勇者(ゆうしゃ) → translate as 'Hero', not 'Yuusha'.\n"
                f"- Preserve parentheses if they indicate pronunciation notes or clarifications that exist in the original dialogue.\n"
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
                f"- If the Japanese text contains kanji with parentheses — e.g. 漢字(かんじ) or name(note) — treat the content inside parentheses as furigana or reading aid, not as part of the dialogue.\n"
                f"- Translate the main kanji normally, but ignore or omit the reading inside parentheses in the final translation.\n"
                f"- Example: 神様(かみさま) → 'God', not 'Kamisama'.\n"
                f"- If parentheses contain explanatory notes (not reading), keep them translated in parentheses in {target_lang}.\n"
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
            text_outline_style=settings.get('outline_style', getattr(self, 'typeset_outline_style', 'stroke')),
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
        if getattr(self, '_is_refreshing_history', False): return
        self._is_refreshing_history = True
        try:
            sources = [
                ('history', self.history_entries),
                ('proofreader', self.proofreader_entries),
                ('quality', self.quality_entries),
                ('scene', self.scenes.get(self.current_scene_name, []) if self.current_scene_name else []),
            ]

            for source, dataset in sources:
                tables = list(self.result_table_registry.get(source, []))
                if not tables:
                    continue
                dataset = dataset or []

                # Filter history by current image
                if source == 'history' and self.current_image_path:
                    dataset = [e for e in dataset if e.get('image_key') == self.current_image_path]

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
        finally:
            self._is_refreshing_history = False

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

        inpaint_model_text = self.inpaint_model_combo.currentText()
        # Hanya model LaMa yang membutuhkan dependency eksternal. Jika pengguna memilih OpenCV,
        # biarkan kuncinya None supaya engine LaMa tidak lagi dipanggil.
        inpaint_model_key = None
        if "Big-LaMa" in inpaint_model_text:
            inpaint_model_key = 'big_lama'
        elif "Anime" in inpaint_model_text:
            inpaint_model_key = 'anime_inpaint'
            
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
            'inpaint_model_key': inpaint_model_key,
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
            'outline_style': getattr(self, 'typeset_outline_style', 'stroke'),
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
            'outline_style': getattr(self, 'typeset_outline_style', 'stroke'),
            'color': self.typeset_color.name(),
            'gradient_enabled': getattr(self, 'typeset_gradient_enabled', False),
            'gradient_angle': getattr(self, 'typeset_gradient_angle', 0.0),
            'gradient_colors': getattr(self, 'typeset_gradient_colors', ["#FF0000", "#0000FF"]),
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
            'outline_style': getattr(self, 'typeset_outline_style', 'stroke'),
            'color': self.typeset_color.name() if isinstance(self.typeset_color, QColor) else '#000000',
            'gradient_enabled': bool(self.gradient_group.isChecked()) if getattr(self,'gradient_group',None) else False,
            'gradient_angle': float(self.grad_angle_spin.value()) if getattr(self,'grad_angle_spin',None) else 0.0,
            'gradient_colors': [self.grad_color_list.item(i).text() for i in range(self.grad_color_list.count())] if getattr(self,'grad_color_list',None) else ["#FF0000", "#0000FF"],
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
        if getattr(self, 'line_spacing_input', None):
            with QSignalBlocker(self.line_spacing_input):
                self.line_spacing_input.setValue(spacing)
        self.typeset_line_spacing_value = spacing
        if getattr(self, 'line_spacing_value_label', None):
            self.line_spacing_value_label.setText(f"{spacing:.2f}x")

    def _set_char_spacing_value(self, spacing: float):
        spacing = max(10.0, min(400.0, float(spacing)))
        if getattr(self, 'char_spacing_input', None):
            with QSignalBlocker(self.char_spacing_input):
                self.char_spacing_input.setValue(spacing)
        self.typeset_char_spacing_value = float(spacing)
        if getattr(self, 'char_spacing_value_label', None):
            self.char_spacing_value_label.setText(f"{spacing:.0f}%")

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
        style_val = (defaults.get('outline_style') or 'stroke')
        if isinstance(style_val, str):
            style_val = style_val.lower()
        self.typeset_outline_style = style_val if style_val in ('stroke', 'glow') else 'stroke'
        self._update_outline_color_button()
        self._refresh_outline_controls_enabled()

        color_value = defaults.get('color', '#000000')
        color_obj = QColor(color_value)
        if color_obj.isValid():
            self.typeset_color = color_obj
        self._update_color_button()
        
        # Gradient defaults
        self.typeset_gradient_enabled = bool(defaults.get('gradient_enabled', False))
        self.typeset_gradient_angle = float(defaults.get('gradient_angle', 0.0))
        self.typeset_gradient_colors = list(defaults.get('gradient_colors', ["#FF0000", "#0000FF"]))
        if getattr(self, 'gradient_group', None):
            with QSignalBlocker(self.gradient_group):
                self.gradient_group.setChecked(self.typeset_gradient_enabled)
        if getattr(self, 'grad_angle_spin', None):
            with QSignalBlocker(self.grad_angle_spin):
                self.grad_angle_spin.setValue(self.typeset_gradient_angle)
        if getattr(self, 'grad_color_list', None):
             self._update_gradient_list_ui(self.typeset_gradient_colors)

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

    def _on_typeset_line_spacing_changed(self, value):
        self._set_line_spacing_value(value)
        self._update_typeset_preview()

    def _on_typeset_char_spacing_changed(self, value):
        self._set_char_spacing_value(value)
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

    def _on_typeset_gradient_toggled(self, checked):
        self._on_typeset_gradient_changed()

    def _on_typeset_gradient_changed(self, *_):
        self._persist_typeset_preferences()
        self._update_typeset_preview()
        
        # Apply to selected area if any
        if self.selected_typeset_area:
            with QSignalBlocker(self.selected_typeset_area):
                self.selected_typeset_area.gradient_enabled = self.typeset_gradient_enabled
                self.selected_typeset_area.gradient_colors = list(self.typeset_gradient_colors)
                self.selected_typeset_area.gradient_angle = self.typeset_gradient_angle
            self.redraw_all_typeset_areas()

    def _on_add_gradient_color(self):
        c = QColorDialog.getColor(Qt.white, self, "Add Gradient Color")
        if c.isValid():
            count = self.grad_color_list.count()
            current_colors = []
            for i in range(count):
                current_colors.append(self.grad_color_list.item(i).text())
            current_colors.append(c.name())
            self._update_gradient_list_ui(current_colors)
            self._on_typeset_gradient_changed()

    def _on_remove_gradient_color(self):
        row = self.grad_color_list.currentRow()
        if row >= 0:
            count = self.grad_color_list.count()
            if count <= 2:
                QMessageBox.warning(self, "Limit", "Gradient must have at least 2 colors.")
                return
            self.grad_color_list.takeItem(row)
            self._on_typeset_gradient_changed()

    def _update_gradient_list_ui(self, colors):
        self.grad_color_list.clear()
        for hex_c in colors:
            item = QListWidgetItem(hex_c)
            bg = QColor(hex_c)
            fg = QColor(0,0,0) if bg.lightness() > 128 else QColor(255,255,255)
            item.setBackground(bg)
            item.setForeground(fg)
            self.grad_color_list.addItem(item)

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
        cfg['outline_style'] = getattr(self, 'typeset_outline_style', 'stroke')
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
                style = getattr(self, 'typeset_outline_style', 'stroke')
                if style == 'glow':
                    image = self._expand_with_outline(image, outline_color, radius=self.typeset_outline_width * 1.4, opacity=0.6)
                else:
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
        self.update_pen_tool_buttons_visibility(False)

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
        def _clamp_rect(raw_rect):
            if raw_rect is None:
                return QRect()
            safe_x = max(0, min(raw_rect.x(), img_width - 1))
            safe_y = max(0, min(raw_rect.y(), img_height - 1))
            safe_width = max(1, min(raw_rect.width(), img_width - safe_x))
            safe_height = max(1, min(raw_rect.height(), img_height - safe_y))
            return QRect(safe_x, safe_y, safe_width, safe_height)

        text_rect = _clamp_rect(area.get_text_rect())
        cleanup_rect = _clamp_rect(area.get_cleanup_rect())
        if text_rect.width() <= 0 or text_rect.height() <= 0:
            return
        area.rect = text_rect
        area.set_cleanup_rect(cleanup_rect)

        def _clamp_polygon(poly):
            if not poly:
                return None
            clamped_points = []
            for pt in poly:
                clamped_points.append(
                    QPoint(
                        max(0, min(pt.x(), img_width - 1)),
                        max(0, min(pt.y(), img_height - 1))
                    )
                )
            return QPolygon(clamped_points) if clamped_points else None

        text_polygon = _clamp_polygon(getattr(area, 'polygon', None))
        cleanup_polygon = _clamp_polygon(area.get_cleanup_polygon() or getattr(area, 'polygon', None))
        area.polygon = text_polygon
        area.set_cleanup_polygon(cleanup_polygon)

        skip_heavy_cleanup = bool(self.is_transform_preview and not for_saving)
        if skip_heavy_cleanup:
            use_inpaint = False
            self._draw_preview_area(painter, area, use_box)
            return

        cleanup_rect = area.get_cleanup_rect()
        cleanup_polygon = area.get_cleanup_polygon()
        text_rect = area.get_text_rect()

        cv_original = cv2.cvtColor(np.array(source_pil_image), cv2.COLOR_RGB2BGR)
        img_height, img_width = cv_original.shape[:2]

        # 1. Buat mask dari bentuk area (polygon atau rectangle)

        base_mask = np.zeros(cv_original.shape[:2], dtype=np.uint8)
        if cleanup_polygon:
            cv_poly_points = np.array([[p.x(), p.y()] for p in cleanup_polygon], dtype=np.int32)
            cv2.fillPoly(base_mask, [cv_poly_points], 255)
        else:
            cv2.rectangle(base_mask,
                        (cleanup_rect.x(), cleanup_rect.y()),
                        (cleanup_rect.right(), cleanup_rect.bottom()),
                        255, -1)

        # 2. Gunakan bubble detector
        if skip_heavy_cleanup:
            bubble_mask = None
            use_inpaint = False
        else:
            bubble_mask = self.find_speech_bubble_mask(cv_original, cleanup_rect, settings, for_saving=for_saving)

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
            model_key = settings.get('inpaint_model_key')
            if model_key:
                try:
                    if (self.inpaint_model is None or
                        self.current_inpaint_model_key != model_key):
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
                    background_color = self.get_background_color(cv_original, cleanup_rect)
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
            background_color = self.get_background_color(cv_original, cleanup_rect)
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
        text_center = text_rect.center()
        cleanup_center = cleanup_rect.center()

        if getattr(area, 'bubble_enabled', False) and use_box:
            painter.save()
            if abs(rotation) > 0.01:
                painter.translate(cleanup_center.x(), cleanup_center.y())
                painter.rotate(rotation)
                painter.translate(-cleanup_center.x(), -cleanup_center.y())
            self.draw_area_bubble(painter, area)
            painter.restore()

        # Tentukan warna teks otomatis
        use_auto = bool(SETTINGS.get('cleanup', {}).get('auto_text_color', True))

        # Tetapkan/bersihkan auto text color sesuai setting
        if use_auto:
            if not use_box:
                bg_col = self.get_background_color(cv_original, text_rect)
                area._auto_text_color = self._auto_text_color_for_base(bg_col)
            else:
                bubble_fill = area.get_bubble_fill_color() if getattr(area, 'bubble_enabled', False) else None
                if bubble_fill and bubble_fill.isValid():
                    area._auto_text_color = self._auto_text_color_for_base(bubble_fill)
                else:
                    bg_col = self.get_background_color(cv_original, text_rect)
                    area._auto_text_color = self._auto_text_color_for_base(bg_col)
        else:
            if hasattr(area, '_auto_text_color'):
                delattr(area, '_auto_text_color')

        painter.save()
        if abs(rotation) > 0.01:
            painter.translate(text_center.x(), text_center.y())
            painter.rotate(rotation)
            painter.translate(-text_center.x(), -text_center.y())
        if not use_box:
            constrain_text = bool(SETTINGS.get('cleanup', {}).get('constrain_text', False))
            if constrain_text:
                self.draw_area_text(painter, area)
            else:
                self.draw_area_text_plain(painter, area)
        else:
            self.draw_area_text(painter, area)
        painter.restore()

    def _draw_preview_area(self, painter, area, use_box):
        rotation = area.get_rotation() if hasattr(area, 'get_rotation') else float(getattr(area, 'rotation', 0.0))
        text_rect = area.get_text_rect()
        cleanup_rect = area.get_cleanup_rect()
        text_center = text_rect.center()
        cleanup_center = cleanup_rect.center()

        painter.save()
        try:
            if getattr(area, 'bubble_enabled', False) and use_box:
                painter.save()
                try:
                    if abs(rotation) > 0.01:
                        painter.translate(cleanup_center.x(), cleanup_center.y())
                        painter.rotate(rotation)
                        painter.translate(-cleanup_center.x(), -cleanup_center.y())
                    self.draw_area_bubble(painter, area)
                finally:
                    painter.restore()

            painter.save()
            try:
                if abs(rotation) > 0.01:
                    painter.translate(text_center.x(), text_center.y())
                    painter.rotate(rotation)
                    painter.translate(-text_center.x(), -text_center.y())

                if use_box:
                    self.draw_area_text(painter, area)
                else:
                    constrain_text = bool(SETTINGS.get('cleanup', {}).get('constrain_text', False))
                    if constrain_text:
                        self.draw_area_text(painter, area)
                    else:
                        self.draw_area_text_plain(painter, area)
            finally:
                painter.restore()
        finally:
            painter.restore()

    def draw_area_bubble(self, painter, area):
        path = QPainterPath()
        cleanup_polygon = area.get_cleanup_polygon() if hasattr(area, 'get_cleanup_polygon') else None
        if cleanup_polygon:
            path.addPolygon(QPolygonF(cleanup_polygon))
        else:
            rect = QRectF(area.get_cleanup_rect() if hasattr(area, 'get_cleanup_rect') else area.rect)
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

    def _expand_with_outline(self, image: QImage, outline_color: QColor | None = None, radius: float = 2.0, opacity: float = 1.0) -> QImage:
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
        painter.setOpacity(max(0.1, min(1.0, float(opacity))))
        offsets = [
            QPoint(dx, dy)
            for dx in range(-radius_px, radius_px + 1)
            for dy in range(-radius_px, radius_px + 1)
            if (dx != 0 or dy != 0) and math.hypot(dx, dy) <= radius + 0.25
        ]
        for offset in offsets:
            painter.drawImage(radius_px + offset.x(), radius_px + offset.y(), outline)
        painter.setOpacity(1.0)
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
            style = area.get_text_outline_style()
            if style == 'glow':
                outline_color.setAlpha(max(80, min(255, int(outline_color.alpha() * 0.7))))
                outline_pen.setColor(outline_color)
                outline_pen.setWidthF(width * 1.6)
            else:
                outline_pen.setWidthF(width)
            outline_pen.setJoinStyle(Qt.RoundJoin)
            outline_pen.setCapStyle(Qt.RoundCap)
            outline_pen.setCosmetic(True)
            painter.strokePath(path, outline_pen)
            
        brush = QBrush(qcolor)
        if area and getattr(area, 'gradient_enabled', False):
            grad_colors = getattr(area, 'gradient_colors', None)
            if grad_colors and len(grad_colors) >= 2:
                import math
                rect = getattr(area, 'rect', None)
                if rect and hasattr(rect, 'center'):
                     angle = getattr(area, 'gradient_angle', 0.0)
                     rad = math.radians(angle)
                     cx, cy = rect.center().x(), rect.center().y()
                     # Use diagonal as diameter to ensure simplified coverage
                     r = math.hypot(rect.width(), rect.height()) / 1.5
                     
                     # Calculate start and end points relative to center
                     dx = math.cos(rad) * r
                     dy = math.sin(rad) * r
                     
                     start = QPointF(cx - dx, cy - dy)
                     end = QPointF(cx + dx, cy + dy)
                     
                     gradient = QLinearGradient(start, end)
                     step = 1.0 / (len(grad_colors) - 1)
                     for i, c in enumerate(grad_colors):
                         gradient.setColorAt(min(1.0, i * step), QColor(c))
                     brush = QBrush(gradient)

        painter.fillPath(path, brush)

    def draw_area_text_plain(self, painter: QPainter, area):
        """Render text directly on the image without drawing any background boxes or fills.

        This uses QPainter.drawText (via glyph rendering) and respects area._auto_text_color for
        overriding segment colors when present.
        """
        rect = QRectF(area.get_text_rect() if hasattr(area, 'get_text_rect') else area.rect)
        margins = area.get_margins()
        rect = rect.adjusted(margins['left'], margins['top'], -margins['right'], -margins['bottom'])
        if rect.width() <= 0 or rect.height() <= 0:
            return
        if area and getattr(area, 'has_text_outline', None) and area.has_text_outline():
            try:
                if area.get_text_outline_style() == 'glow':
                    self._draw_rich_text_document(painter, rect, area, area.get_orientation())
                    return
            except Exception:
                pass

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
        rect = QRectF(area.get_text_rect() if hasattr(area, 'get_text_rect') else area.rect)
        margins = area.get_margins()
        rect = rect.adjusted(margins['left'], margins['top'], -margins['right'], -margins['bottom'])
        if rect.width() <= 0 or rect.height() <= 0:
            return

        effect = area.get_effect().lower()
        orientation = area.get_orientation()

        if orientation == 'vertical' and effect != 'none':
            effect = 'none'

        if effect == 'none':
            # Check for gradient. If enabled, force glyph rendering path (draw_area_text_plain)
            # which supports custom QPens/Brushes for gradients/outlines that QTextDocument does not.
            if getattr(area, 'gradient_enabled', False):
                 self.draw_area_text_plain(painter, area)
            else:
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
            try:
                style = area.get_text_outline_style()
            except Exception:
                style = 'stroke'
            if style == 'glow':
                image = self._expand_with_outline(image, outline_color, radius=radius * 1.4, opacity=0.6)
            else:
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
        rect_mode = ("Rect" in mode or "Oval" in mode) and not manual_polygon
        transform_mode = (mode == "Transform (Hand)")
        self.image_label.set_transform_mode(transform_mode)
        # Use an explicit pen cursor for pen mode so it's visibly distinct
        if pen_mode:
            if not self.pen_cursor:
                self.pen_cursor = self.create_pen_cursor()
            self.image_label.setCursor(self.pen_cursor)
        elif transform_mode:
            self.image_label.setCursor(Qt.OpenHandCursor)
        else:
            self.image_label.setCursor(Qt.CrossCursor if rect_mode else Qt.PointingHandCursor)
        self.update_pen_tool_buttons_visibility(pen_mode)
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

    def update_pen_tool_buttons_visibility(self, visible):
        self.confirm_button.setVisible(visible); self.cancel_button.setVisible(visible)

    def confirm_pen_selection(self, warn_if_invalid: bool = True):
        points = self.image_label.get_polygon_points()
        if len(points) < 3:
            if warn_if_invalid:
                QMessageBox.warning(self, "Invalid Shape", "Please select at least 3 points.")
            else:
                self.statusBar().showMessage("No pen selection to confirm.", 2000)
            return False
        self.process_polygon_area(points)
        self.image_label.clear_selection()
        self.update_pen_tool_buttons_visibility(False)
        return True

    def confirm_pen_via_shortcut(self):
        """Shortcut-friendly wrapper that avoids modal warnings when nothing is drawn."""
        return self.confirm_pen_selection(warn_if_invalid=False)

    def cancel_pen_selection(self):
        self.image_label.clear_selection()
        self.update_pen_tool_buttons_visibility(False)

    def save_image(self):
        if not self.typeset_pixmap: QMessageBox.warning(self, "No Image", "There is no image to save."); return

        # Get settings
        gen_cfg = SETTINGS.get('general', {})
        def_fmt = gen_cfg.get('save_format', 'PNG').upper()
        quality = int(gen_cfg.get('save_quality', 95))
        
        ext_map = {'PNG': '.png', 'JPG': '.jpg', 'JPEG': '.jpg', 'WEBP': '.webp'}
        def_ext = ext_map.get(def_fmt, '.png')
        
        filters = "PNG Image (*.png);;JPEG Image (*.jpg);;WebP Image (*.webp)"
        
        # Select filter based on setting
        initial_filter = ""
        if 'PNG' in def_fmt: initial_filter = "PNG Image (*.png)"
        elif 'JPG' in def_fmt or 'JPEG' in def_fmt: initial_filter = "JPEG Image (*.jpg)"
        elif 'WEBP' in def_fmt: initial_filter = "WebP Image (*.webp)"

        if self.pdf_document:
            original_filename = os.path.basename(self.current_image_path)
            name, _ = os.path.splitext(original_filename)
            save_suggestion = os.path.join(os.path.dirname(self.current_image_path), f"{name}_page_{self.current_pdf_page + 1}_typeset{def_ext}")
        else:
            original_filename = os.path.basename(self.current_image_path)
            name, _ = os.path.splitext(original_filename)
            save_suggestion = os.path.join(os.path.dirname(self.current_image_path), f"{name}_typeset{def_ext}")

        filePath, _ = QFileDialog.getSaveFileName(self, "Save Typeset Image", save_suggestion, filters, initial_filter)
        if filePath:
            # Non-blocking save: copy QPixmap to QImage under mutex, then save in background
            self.paint_mutex.lock()
            try:
                pix_copy = self.typeset_pixmap.copy()
                qimage = pix_copy.toImage().copy()
            finally:
                self.paint_mutex.unlock()

            # Start background worker to save the QImage
            # We let the worker infer format from extension, but pass quality preference
            image_worker = ImageSaveWorker(qimage, filePath, quality=quality)
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
            # Sync to Deleted History Scene
            history_id = getattr(area_to_delete, 'history_id', None)
            if history_id:
                self.move_entry_to_deleted_history(history_id)
            
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
            'scenes': copy.deepcopy(self.scenes),
            'scene_order': copy.deepcopy(self.scene_order),
            'current_scene_name': self.current_scene_name,
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

        # Load Scenes
        loaded_scenes = payload.get('scenes', {})
        loaded_scene_order = payload.get('scene_order', [])
        loaded_current_scene = payload.get('current_scene_name')

        if isinstance(loaded_scenes, dict):
            # Sanitize scenes
            clean_scenes = {}
            for s_name, s_entries in loaded_scenes.items():
                if isinstance(s_name, str) and isinstance(s_entries, list):
                    clean_scenes[s_name] = self._sanitize_review_entries(s_entries)
            self.scenes = clean_scenes
        else:
            self.scenes = {}

        if isinstance(loaded_scene_order, list):
            self.scene_order = [s for s in loaded_scene_order if isinstance(s, str) and s in self.scenes]
            # Ensure all scenes are in order list
            for s_name in self.scenes:
                if s_name not in self.scene_order:
                    self.scene_order.append(s_name)
        else:
            self.scene_order = list(self.scenes.keys())

        if isinstance(loaded_current_scene, str) and loaded_current_scene in self.scenes:
            self.current_scene_name = loaded_current_scene
        elif self.scene_order:
            self.current_scene_name = self.scene_order[0]
        else:
            self.current_scene_name = None

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
            dialog = AdvancedTextEditDialog(parent=self, area=area, font_manager=self.font_manager)
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
                area.text_outline = bool(result.get('text_outline', area.has_text_outline()))
                area.text_outline_width = float(result.get('text_outline_width', area.get_text_outline_width()))
                color_val = result.get('text_outline_color', area.get_text_outline_color())
                if isinstance(color_val, QColor):
                    color_val = color_val.name()
                area.text_outline_color = color_val
                area.text_outline_style = result.get('text_outline_style', area.get_text_outline_style())

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

        # Get settings
        gen_cfg = SETTINGS.get('general', {})
        save_fmt = gen_cfg.get('save_format', 'PNG')
        save_qual = int(gen_cfg.get('save_quality', -1))

        self.overall_progress_bar.setVisible(True); self.overall_progress_bar.setValue(0)
        self.statusBar().showMessage("Starting batch save...")

        self.batch_save_thread = QThread()
        self.batch_save_worker = BatchSaveWorker(self, files_to_save, fmt=save_fmt, quality=save_qual)
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
        # Check for common extensions
        for check_ext in ['.png', '.jpg', '.jpeg', '.webp']:
            if os.path.exists(f"{path_part}_typeset{check_ext}"):
                return True
        return False
    
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

    # ===================================================================
    # ======================= COPY / PASTE FEATURE ======================
    # ===================================================================
    def copy_selected_typeset_area(self):
        """Salin area typeset yang dipilih ke clipboard."""
        if not self.selected_typeset_area:
            return
        
        try:
            payload = self.selected_typeset_area.to_payload()
            container = {
                'type': 'manga_ocr_typeset',
                'data': payload
            }
            json_text = json.dumps(container, ensure_ascii=False)
            QApplication.clipboard().setText(json_text)
            self.statusBar().showMessage("Typeset area copied to clipboard.", 2000)
        except Exception as e:
            self.statusBar().showMessage(f"Failed to copy: {e}", 3000)

    def paste_typeset_area(self):
        """Tempel area typeset dari clipboard."""
        clipboard = QApplication.clipboard()
        text = clipboard.text()
        if not text:
            return

        try:
            container = json.loads(text)
            if not isinstance(container, dict) or container.get('type') != 'manga_ocr_typeset':
                # Not our data, ignore or handle as plain text if needed
                return
            
            data = container.get('data')
            if not data:
                return

            # Buat area baru dari data
            new_area = TypesetArea.from_payload(data)
            
            # Geser sedikit agar tidak menumpuk persis di atas yang asli
            current_rect = new_area.rect
            new_rect = current_rect.translated(20, 20)
            
            # Clamp ke image size
            if self.original_pixmap:
                w, h = self.original_pixmap.width(), self.original_pixmap.height()
                if new_rect.left() > w or new_rect.top() > h:
                     new_rect.moveTo(20, 20)
                if new_rect.right() > w: new_rect.setWidth(max(10, w - new_rect.x()))
                if new_rect.bottom() > h: new_rect.setHeight(max(10, h - new_rect.y()))
            
            new_area.rect = new_rect
            # Juga update polygon/cleanup jika ada
            if new_area.polygon:
                new_area.polygon.translate(20, 20)
            if new_area.cleanup_rect:
                new_area.cleanup_rect.translate(20, 20)
            if new_area.cleanup_polygon:
                new_area.cleanup_polygon.translate(20, 20)

            # Assign new ID to avoid conflict if history tracking uses it
            new_area.history_id = None

            # Tambahkan ke list
            self.typeset_areas.append(new_area)
            self.set_selected_area(new_area)
            self.redraw_all_typeset_areas()
            self.statusBar().showMessage("Typeset area pasted.", 2000)

        except json.JSONDecodeError:
            pass # Bukan JSON valid
        except Exception as e:
            self.statusBar().showMessage(f"Failed to paste: {e}", 3000)
            traceback.print_exc()

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
                if re.fullmatch(r'[!\?\-•°??????]+', cleaned):
                    continue
                repeated = re.search(r'(.)\1{2,}', cleaned)
                if repeated and len(cleaned) <= 5:
                    if repeated.group(1) != '~':
                        continue
            unique_chars = set(cleaned)
            if len(unique_chars) == 1 and cleaned[0] in "!?…??????#@*/":
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
        # [MODIFIED] Check 'force_ai_ocr' override
        if SETTINGS.get('force_ai_ocr', False):
            ocr_engine = "AI_OCR"
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

    def _get_ai_ocr_prompt(self, lang):
        if lang == "Japanese":
            return (
                "Task: Optical Character Recognition (OCR) for Japanese text.\n"
                "Input: an image.\n"
                "Output: ONLY the recognized text in natural reading order.\n\n"
                "Rules:\n"
                "- Do NOT explain or add any commentary.\n"
                "- Do NOT output markdown or formatting symbols.\n"
                "- Keep line breaks if they appear in the original image.\n"
                "- Preserve punctuation (。, 、, …, !, ? etc.).\n"
                "- When a small note or furigana is written next to a kanji, output it in parentheses after the kanji.\n"
                "  Example: 漢字 + note → 漢字(note)\n"
                "- If the note appears *before* the kanji (vertically aligned text), treat it the same way: 漢字(note).\n"
                "- If the note is unrelated annotation or translation note, also wrap it in parentheses.\n"
                "- Do NOT merge notes and kanji into a single block like [note][kanji].\n"
                "- Do NOT drop ellipses (…)\n"
                "- Just return the plain text with correct kanji-note pairing."
            )
        elif lang == "English":
            return (
                "Task: Optical Character Recognition (OCR) for English text.\n"
                "Input: an image.\n"
                "Output: ONLY the recognized text in natural reading order.\n\n"
                "Rules:\n"
                "- Do NOT explain or add any commentary.\n"
                "- Do NOT output markdown or formatting symbols.\n"
                "- Maintain original line breaks.\n"
                "- Preserve punctuation.\n"
                "- Return ONLY the plain text."
            )
        elif lang == "Korean":
            return (
                "Task: Optical Character Recognition (OCR) for Korean text (Hangul).\n"
                "Input: an image.\n"
                "Output: ONLY the recognized text in natural reading order.\n\n"
                "Rules:\n"
                "- Do NOT explain or add any commentary.\n"
                "- Do NOT output markdown or formatting symbols.\n"
                "- Maintain original line breaks.\n"
                "- Preserve punctuation.\n"
                "- Return ONLY the plain text."
            )
        elif lang == "Chinese":
            return (
                "Task: Optical Character Recognition (OCR) for Chinese text.\n"
                "Input: an image.\n"
                "Output: ONLY the recognized text in natural reading order.\n\n"
                "Rules:\n"
                "- Do NOT explain or add any commentary.\n"
                "- Do NOT output markdown or formatting symbols.\n"
                "- Maintain original line breaks.\n"
                "- Preserve punctuation.\n"
                "- Return ONLY the plain text."
            )
        else:
             return (
                "Task: Optical Character Recognition (OCR).\n"
                "Input: an image.\n"
                "Output: ONLY the recognized text in natural reading order.\n\n"
                "Rules:\n"
                "- Do NOT explain or add any commentary.\n"
                "- Output the text exactly as seen in the image.\n"
                "- Preserve punctuation and line breaks.\n"
                "- Return ONLY the plain text."
            )

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
        ai_lang = SETTINGS.get('ai_ocr_lang', 'Japanese')
        prompt_text = self._get_ai_ocr_prompt(ai_lang)

        data_url = f"data:image/png;base64,{image_b64}"
        
        # [CACHE SYSTEM]
        cache_key = hashlib.md5((image_b64 + prompt_text + model_id).encode('utf-8')).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"aiocr_{cache_key}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    print(f"[CACHE HIT] Returning cached OCR result for {cache_key}")
                    return cached_data.get('text', '')
            except Exception:
                pass

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
                # [CACHE SAVE]
                try:
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump({'text': extracted, 'model': model_id, 'timestamp': time.time()}, f)
                except Exception:
                    pass

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

            ai_lang = SETTINGS.get('ai_ocr_lang', 'Japanese')
            prompt = self._get_ai_ocr_prompt(ai_lang)
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
