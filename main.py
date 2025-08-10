import os
import sys
import numpy as np
import cv2
import pytesseract
import requests
import easyocr
import pickle
import google.generativeai as genai
import time
import json
import hashlib
import re
import fitz # from PyMuPDF
import configparser # --- BARU v7.8 ---
from datetime import date
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QFileDialog, QTextEdit, QScrollArea, QComboBox, QMessageBox,
    QProgressBar, QShortcut, QListWidget, QListWidgetItem, QColorDialog, QFontDialog,
    QLineEdit, QAction, QDialog, QCheckBox, QStatusBar, QDoubleSpinBox
)
from PyQt5.QtGui import (
    QPixmap, QPainter, QPen, QColor, QFont, QKeySequence, QPolygon, QPainterPath, QPolygonF, QImage
)
from PyQt5.QtCore import Qt, QRect, QPoint, pyqtSignal, QTimer, QThread, QObject

# --- BARU v7.8: Fungsi untuk menangani config.ini ---
def load_or_create_config():
    """
    Memuat konfigurasi dari config.ini. Jika tidak ada, buat file default
    dan keluar dari aplikasi agar pengguna bisa mengisinya.
    """
    config_path = 'config.ini'
    config = configparser.ConfigParser()

    if not os.path.exists(config_path):
        print(f"'{config_path}' not found. Creating a default file.")
        config['API'] = {
            'DEEPL_KEY': 'your_deepl_key_here',
            'GEMINI_KEY': 'your_gemini_key_here'
        }
        config['PATHS'] = {
            'TESSERACT_PATH': r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        }
        with open(config_path, 'w') as configfile:
            config.write(configfile)
        
        QMessageBox.information(
            None, 
            "Configuration File Created", 
            f"'{config_path}' was not found.\nA default file has been created.\n\nPlease open it, enter your API keys, verify the Tesseract path, and then restart the application."
        )
        sys.exit() # Keluar agar pengguna bisa mengisi config

    config.read(config_path)
    return config

# --- Muat Konfigurasi Saat Startup ---
try:
    config = load_or_create_config()
    DEEPL_API_KEY = config.get('API', 'DEEPL_KEY')
    GEMINI_API_KEY = config.get('API', 'GEMINI_KEY')
    TESSERACT_PATH = config.get('PATHS', 'TESSERACT_PATH')
except (configparser.Error, KeyError) as e:
    # Tampilkan error jika file config rusak atau key-nya hilang
    QMessageBox.critical(
        None, 
        "Configuration Error", 
        f"Failed to read 'config.ini'. Please ensure it is correctly formatted with [API] and [PATHS] sections.\nError: {e}"
    )
    sys.exit()

# --- Konfigurasi Batasan API ---
GEMINI_RPM_LIMIT = 15   # Requests Per Minute
GEMINI_RPD_LIMIT = 1000 # Requests Per Day

# Konfigurasi Gemini API
if GEMINI_API_KEY and "your_gemini_key_here" not in GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"Gagal mengkonfigurasi Gemini API: {e}")

# Konfigurasi Manga-OCR
try:
    from manga_ocr import MangaOcr
except ImportError:
    MangaOcr = None
    print("Peringatan: Pustaka 'manga-ocr' tidak ditemukan. Opsi engine Manga-OCR akan dinonaktifkan.")

# Konfigurasi Tesseract
try:
    if TESSERACT_PATH and os.path.exists(TESSERACT_PATH):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    else:
        # Tidak perlu error kritis di sini, akan diperiksa saat aplikasi dimulai
        print("Tesseract path from config.ini is invalid or not set.")
except Exception:
    print("Could not set Tesseract path. Please check config.ini.")

# --------------------------------------------------------------------

class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)

class OCRWorkerSignals(WorkerSignals):
    ocr_complete = pyqtSignal(str, object)
    batch_result_ready = pyqtSignal(list)
    api_limit_reached = pyqtSignal()

class TranslateWorkerSignals(WorkerSignals):
    translation_complete = pyqtSignal(str, object)
    api_limit_reached = pyqtSignal()

class OCRWorker(QObject):
    def __init__(self, main_app, image_to_process, settings, context_data, is_batch=False):
        super().__init__()
        self.main_app = main_app
        self.image_to_process = image_to_process
        self.settings = settings
        self.context_data = context_data
        self.is_batch = is_batch
        self.signals = OCRWorkerSignals()

    def run(self):
        try:
            if self.is_batch:
                self.run_batch_processing()
            else:
                self.run_single_ocr()
        except Exception as e:
            self.signals.error.emit(f"OCR Worker Error: {e}")
        finally:
            self.signals.finished.emit()

    def run_single_ocr(self):
        self.signals.progress.emit(20, "Preprocessing image...")
        preprocessed_image, _ = self.main_app.preprocess_for_ocr(self.image_to_process, self.settings['orientation'])
        self.signals.progress.emit(50, f"Running OCR ({self.settings['ocr_engine']})...")
        raw_text = self.perform_ocr(preprocessed_image)
        self.signals.progress.emit(90, "OCR complete.")
        self.context_data['original_ocr_text'] = raw_text
        self.signals.ocr_complete.emit(raw_text, self.context_data)

    def run_batch_processing(self):
        self.signals.progress.emit(5, "Batch mode: Detecting text bubbles...")
        cv_image = cv2.cvtColor(np.array(self.image_to_process), cv2.COLOR_RGB2BGR)
        if not self.main_app.easyocr_reader:
            self.signals.error.emit("EasyOCR not initialized. It's required for batch mode bubble detection.")
            return
        results = self.main_app.easyocr_reader.readtext(cv_image, detail=1, paragraph=True)
        total_bubbles = len(results)
        if total_bubbles == 0:
            self.signals.progress.emit(100, "No text bubbles found.")
            self.signals.batch_result_ready.emit([])
            return
        processed_areas = []
        for i, (bbox, text, conf) in enumerate(results):
            progress_val = int(10 + (i / total_bubbles) * 80)
            self.signals.progress.emit(progress_val, f"Processing bubble {i+1}/{total_bubbles}...")
            try:
                x_min, y_min = map(int, bbox[0])
                x_max, y_max = map(int, bbox[2])
                typeset_rect = QRect(QPoint(x_min, y_min), QPoint(x_max, y_max))
                cropped_img = self.image_to_process.crop((x_min, y_min, x_max, y_max))
                cropped_cv_img = cv2.cvtColor(np.array(cropped_img), cv2.COLOR_RGB2BGR)
                preprocessed_image, _ = self.main_app.preprocess_for_ocr(cropped_cv_img, self.settings['orientation'])
                cache_key = self.main_app.get_cache_key(preprocessed_image)
                cached_result = self.main_app.check_cache(cache_key)
                if cached_result:
                    translated_text = cached_result['translated_text']
                else:
                    raw_text = self.perform_ocr(preprocessed_image)
                    processed_text = self.main_app.clean_and_join_text(raw_text)
                    text_for_translation = processed_text
                    if self.settings['use_ai'] and processed_text:
                        corrected_text = self.main_app.correct_text_with_gemini(processed_text)
                        if corrected_text is not None:
                            text_for_translation = corrected_text
                        else:
                            self.signals.api_limit_reached.emit()
                    if not text_for_translation:
                        translated_text = ""
                    else:
                        translated_text = self.main_app.translate_text(text_for_translation, self.settings['target_lang'])
                    self.main_app.write_to_cache(cache_key, {'original_text': text_for_translation, 'translated_text': translated_text})
                if translated_text:
                    new_area = TypesetArea(typeset_rect, translated_text, self.settings['font'], self.settings['color'])
                    processed_areas.append(new_area)
            except Exception as e:
                print(f"Error processing bubble {i+1}: {e}")
                continue
        self.signals.progress.emit(95, "Finalizing batch results...")
        self.signals.batch_result_ready.emit(processed_areas)

    def perform_ocr(self, image_to_process):
        ocr_engine = self.settings['ocr_engine']
        orientation = self.settings['orientation']
        raw_text = ""
        if ocr_engine == "Manga-OCR":
            if not self.main_app.manga_ocr_reader:
                raise Exception("Manga-OCR is not initialized.")
            pil_img = Image.fromarray(cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB))
            raw_text = self.main_app.manga_ocr_reader(pil_img)
        elif ocr_engine == "EasyOCR":
            if not self.main_app.easyocr_reader:
                raise Exception("EasyOCR is not initialized.")
            gray = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2GRAY)
            vertical = (orientation == "Vertical")
            results = self.main_app.easyocr_reader.readtext(gray, detail=0, paragraph=True, vertical=vertical)
            raw_text = "\n".join(results)
        else:
            gray = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2GRAY)
            lang_map = {"Japanese": "jpn", "English": "eng", "Chinese": "chi_sim", "Korean": "kor"}
            tess_lang = lang_map.get(self.settings['ocr_lang'], "eng")
            psm = 5 if orientation == "Vertical" else 6
            custom_config = f'--oem 1 --psm {psm} -l {tess_lang}'
            raw_text = pytesseract.image_to_string(gray, config=custom_config).strip()
        return raw_text

class TranslateWorker(QObject):
    def __init__(self, main_app, text_to_process, settings, context_data):
        super().__init__()
        self.main_app = main_app
        self.text_to_process = text_to_process
        self.settings = settings
        self.context_data = context_data
        self.signals = TranslateWorkerSignals()

    def run(self):
        try:
            self.signals.progress.emit(10, "Cleaning OCR text...")
            processed_text = self.main_app.clean_and_join_text(self.text_to_process)
            text_for_translation = processed_text
            if self.settings['use_ai'] and processed_text:
                self.signals.progress.emit(40, "Correcting with AI...")
                corrected_text = self.main_app.correct_text_with_gemini(processed_text)
                if corrected_text is not None:
                    text_for_translation = corrected_text
                else:
                    self.signals.api_limit_reached.emit()
            if not text_for_translation:
                self.signals.translation_complete.emit("", self.context_data)
                return
            self.signals.progress.emit(70, "Translating text...")
            translated_text = self.main_app.translate_text(text_for_translation, self.settings['target_lang'])
            preprocessed_image_cv = self.context_data.get('preprocessed_image_cv')
            if preprocessed_image_cv is not None:
                cache_key = self.main_app.get_cache_key(preprocessed_image_cv)
                self.main_app.write_to_cache(cache_key, {'original_text': text_for_translation, 'translated_text': translated_text})
            self.signals.translation_complete.emit(translated_text, self.context_data)
        except Exception as e:
            self.signals.error.emit(f"Translate Worker Error: {e}")
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

    def get_selection_mode(self):
        return self.main_window.selection_mode_combo.currentText()

    def get_polygon_points(self):
        return self.polygon_points

    def mouseDoubleClickEvent(self, event):
        if not self.main_window.original_pixmap: return
        unzoomed_pos = self.main_window.unzoom_coords(event.pos(), as_point=True)
        if unzoomed_pos:
            for area in reversed(self.main_window.typeset_areas):
                if area.rect.contains(unzoomed_pos):
                    self.areaDoubleClicked.emit(area)
                    return

    def mousePressEvent(self, event):
        if not self.main_window.original_pixmap: return
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
        mode = self.get_selection_mode()
        if mode == "Rectangle":
            if self.dragging:
                self.selection_end = event.pos()
                self.update()
        elif mode == "Pen Tool":
            self.current_mouse_pos = event.pos()
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
        QMainWindow, QMenuBar, QMenu, QDialog { background-color: #2e2e2e; }
        QMenuBar::item:selected, QMenu::item:selected { background-color: #007acc; }
        QWidget { color: #f0f0f0; background-color: #2e2e2e; font-size: 10pt; }
        QLabel { padding: 2px; background-color: transparent; }
        QLabel[text*="<h3>"] { color: #00aaff; font-size: 12pt; font-weight: bold; margin-top: 10px; }
        QPushButton { background-color: #007acc; color: white; padding: 8px; border: 1px solid #005f9e; border-radius: 4px; margin: 2px; }
        QPushButton:hover { background-color: #008ae6; }
        QPushButton:pressed { background-color: #005f9e; }
        QPushButton:disabled { background-color: #555555; border-color: #444; }
        QTextEdit, QComboBox, QListWidget, QDoubleSpinBox, QLineEdit { background-color: #3c3c3c; color: #f0f0f0; border: 1px solid #555; padding: 5px; border-radius: 4px; }
        QScrollArea { border: 1px solid #444; }
        QProgressBar { border: 1px solid #555; border-radius: 4px; text-align: center; color: #f0f0f0; }
        QProgressBar::chunk { background-color: #007acc; border-radius: 3px; }
        QStatusBar { background-color: #2e2e2e; color: #f0f0f0; }
    """
    LIGHT_THEME_STYLESHEET = """
        QMainWindow, QMenuBar, QMenu, QDialog { background-color: #f0f0f0; }
        QMenuBar::item:selected, QMenu::item:selected { background-color: #0078d7; color: white; }
        QWidget { color: #000000; background-color: #f0f0f0; font-size: 10pt; }
        QLabel { padding: 2px; background-color: transparent; }
        QLabel[text*="<h3>"] { color: #005a9e; font-size: 12pt; font-weight: bold; margin-top: 10px; }
        QPushButton { background-color: #0078d7; color: white; padding: 8px; border: 1px solid #005a9e; border-radius: 4px; margin: 2px; }
        QPushButton:hover { background-color: #106ebe; }
        QPushButton:pressed { background-color: #005a9e; }
        QPushButton:disabled { background-color: #dcdcdc; border-color: #c0c0c0; color: #808080; }
        QTextEdit, QComboBox, QListWidget, QDoubleSpinBox, QLineEdit { background-color: #ffffff; color: #000000; border: 1px solid #c0c0c0; padding: 5px; border-radius: 4px; }
        QScrollArea { border: 1px solid #c0c0c0; }
        QProgressBar { border: 1px solid #c0c0c0; border-radius: 4px; text-align: center; color: #000000; background-color: #e6e6e6;}
        QProgressBar::chunk { background-color: #0078d7; border-radius: 3px; }
        QStatusBar { background-color: #f0f0f0; color: #000000; }
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Manga OCR & Typeset Tool v7.8")
        self.setGeometry(50, 50, 1600, 900)
        self.image_files = []
        self.current_image_index = -1
        self.current_image_pil = None
        self.original_pixmap = None
        self.typeset_pixmap = None
        self.zoom_factor = 1.0
        self.typeset_areas = []
        self.redo_stack = []
        self.easyocr_reader = None
        self.manga_ocr_reader = None
        self.current_project_path = None
        self.current_theme = 'dark'
        self.typeset_font = QFont("Arial", 12, QFont.Bold)
        self.typeset_color = QColor(Qt.black)
        self.inline_editor = None
        self.editing_area = None
        self.worker_thread = None
        self.current_worker = None
        self.project_dir = None
        self.cache_dir = None
        
        # PDF State Variables
        self.pdf_document = None
        self.current_pdf_page = -1
        
        self.usage_file_path = os.path.join(os.path.expanduser("~"), "manga_ocr_usage_v7.dat")
        self.usage_data = {'date': str(date.today()), 'daily_count': 0, 'minute_count': 0, 'last_call_timestamp': 0}
        self.api_limit_timer = QTimer(self)
        self.api_limit_timer.setInterval(1000)
        self.api_limit_timer.timeout.connect(self.periodic_limit_check)
        self.autosave_timer = QTimer(self)
        self.autosave_timer.setInterval(300000)
        self.autosave_timer.timeout.connect(self.auto_save_project)
        self.init_ui()
        self.setup_styles()
        self.setup_shortcuts()
        self.initialize_ocr_engines()
        self.load_usage_data()
        self.check_limits_and_update_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        self.setup_menu_bar()
        self.setStatusBar(QStatusBar(self))
        left_panel = QVBoxLayout()
        left_panel.addWidget(QLabel("Image Files"))
        self.file_list_widget = QListWidget()
        self.file_list_widget.currentItemChanged.connect(self.on_file_selected)
        left_panel.addWidget(self.file_list_widget)
        load_folder_button = QPushButton("Load Folder")
        load_folder_button.clicked.connect(self.load_folder)
        left_panel.addWidget(load_folder_button)
        center_panel = QVBoxLayout()
        self.image_label = SelectableImageLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.areaDoubleClicked.connect(self.start_inline_edit)
        image_scroll = QScrollArea()
        image_scroll.setWidget(self.image_label)
        image_scroll.setWidgetResizable(True)
        center_panel.addWidget(image_scroll)
        nav_zoom_layout = QHBoxLayout()
        self.prev_button = QPushButton("<< Previous")
        self.prev_button.clicked.connect(self.load_prev_image)
        nav_zoom_layout.addWidget(self.prev_button)
        nav_zoom_layout.addStretch()
        self.zoom_out_button = QPushButton("Zoom Out (-)")
        self.zoom_out_button.clicked.connect(self.zoom_out)
        nav_zoom_layout.addWidget(self.zoom_out_button)
        self.zoom_label = QLabel(f" Zoom: {self.zoom_factor:.1f}x ")
        nav_zoom_layout.addWidget(self.zoom_label)
        self.zoom_in_button = QPushButton("Zoom In (+)")
        self.zoom_in_button.clicked.connect(self.zoom_in)
        nav_zoom_layout.addWidget(self.zoom_in_button)
        nav_zoom_layout.addStretch()
        self.next_button = QPushButton("Next >>")
        self.next_button.clicked.connect(self.load_next_image)
        nav_zoom_layout.addWidget(self.next_button)
        center_panel.addLayout(nav_zoom_layout)
        right_panel = QVBoxLayout()
        self.setup_right_panel(right_panel)
        main_layout.addLayout(left_panel, 15)
        main_layout.addLayout(center_panel, 55)
        main_layout.addLayout(right_panel, 30)

    def setup_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('&File')
        save_project_action = QAction('Save Project', self); save_project_action.setShortcut('Ctrl+S'); save_project_action.triggered.connect(self.save_project); file_menu.addAction(save_project_action)
        load_project_action = QAction('Load Project', self); load_project_action.setShortcut('Ctrl+O'); load_project_action.triggered.connect(self.load_project); file_menu.addAction(load_project_action)
        
        file_menu.addSeparator()
        export_pdf_action = QAction('Export Folder to PDF...', self)
        export_pdf_action.triggered.connect(self.export_to_pdf)
        file_menu.addAction(export_pdf_action)
        
        view_menu = menu_bar.addMenu('&View')
        toggle_theme_action = QAction('Toggle Light/Dark Mode', self); toggle_theme_action.triggered.connect(self.toggle_theme); view_menu.addAction(toggle_theme_action)
        help_menu = menu_bar.addMenu('&Help / Usage')
        about_action = QAction('About & API Usage', self); about_action.triggered.connect(self.show_about_dialog); help_menu.addAction(about_action)

    def setup_right_panel(self, right_panel):
        right_panel.addWidget(QLabel("<h3>Processing Options</h3>"))
        ocr_engines = ["EasyOCR", "Tesseract"];
        if MangaOcr: ocr_engines.insert(0, "Manga-OCR")
        self.ocr_engine_combo = self._create_combo_box(right_panel, "OCR Engine:", ocr_engines)
        self.orientation_combo = self._create_combo_box(right_panel, "Text Orientation:", ["Auto-Detect", "Horizontal", "Vertical"])
        self.lang_combo = self._create_combo_box(right_panel, "OCR Language (Tesseract):", ["Japanese", "English", "Chinese", "Korean"], "Japanese")
        self.translate_combo = self._create_combo_box(right_panel, "Translate to:", ["Indonesian", "English", "Japanese", "Chinese", "Korean"], "Indonesian")
        self.ai_correction_checkbox = QCheckBox("Use AI to Correct Text"); self.ai_correction_checkbox.setChecked(True); self.ai_correction_checkbox.setToolTip("Enable/Disable AI-powered text correction using Gemini API."); right_panel.addWidget(self.ai_correction_checkbox)
        self.review_checkbox = QCheckBox("Review Before Translating"); self.review_checkbox.setChecked(True); right_panel.addWidget(self.review_checkbox)
        self.ocr_engine_combo.currentTextChanged.connect(self.on_ocr_engine_changed); self.on_ocr_engine_changed(self.ocr_engine_combo.currentText())
        right_panel.addWidget(QLabel("<h3>Selection Tools</h3>"))
        self.selection_mode_combo = self._create_combo_box(right_panel, "Selection Mode:", ["Rectangle", "Pen Tool"]); self.selection_mode_combo.currentTextChanged.connect(self.selection_mode_changed)
        pen_buttons_layout = QHBoxLayout()
        self.confirm_button = QPushButton("Confirm Selection"); self.confirm_button.clicked.connect(self.confirm_pen_selection); self.confirm_button.setVisible(False); pen_buttons_layout.addWidget(self.confirm_button)
        self.cancel_button = QPushButton("Cancel"); self.cancel_button.clicked.connect(self.cancel_pen_selection); self.cancel_button.setVisible(False); pen_buttons_layout.addWidget(self.cancel_button)
        right_panel.addLayout(pen_buttons_layout)
        right_panel.addWidget(QLabel("<h3>Typesetting Options</h3>"))
        typeset_options_layout = QHBoxLayout()
        self.font_button = QPushButton("Choose Font"); self.font_button.clicked.connect(self.choose_font); typeset_options_layout.addWidget(self.font_button)
        self.color_button = QPushButton("Choose Color"); self.color_button.clicked.connect(self.choose_color); typeset_options_layout.addWidget(self.color_button)
        right_panel.addLayout(typeset_options_layout)
        self.vertical_typeset_checkbox = QCheckBox("Typeset Vertically"); right_panel.addWidget(self.vertical_typeset_checkbox)
        right_panel.addWidget(QLabel("<h3>Actions</h3>"))
        self.batch_process_button = QPushButton("Batch Process Current Image"); self.batch_process_button.clicked.connect(self.start_batch_processing); right_panel.addWidget(self.batch_process_button)
        undo_redo_layout = QHBoxLayout()
        self.undo_button = QPushButton("Undo (Ctrl+Z)"); self.undo_button.clicked.connect(self.undo_last_action); self.undo_button.setEnabled(False); undo_redo_layout.addWidget(self.undo_button)
        self.redo_button = QPushButton("Redo (Ctrl+Y)"); self.redo_button.clicked.connect(self.redo_last_action); self.redo_button.setEnabled(False); undo_redo_layout.addWidget(self.redo_button)
        right_panel.addLayout(undo_redo_layout)
        self.reset_button = QPushButton("Reset View"); self.reset_button.clicked.connect(self.reset_view_to_original); right_panel.addWidget(self.reset_button)
        self.save_button = QPushButton("Save Image"); self.save_button.clicked.connect(self.save_image); right_panel.addWidget(self.save_button)
        right_panel.addStretch()
        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False); right_panel.addWidget(self.progress_bar)
        right_panel.addWidget(QLabel("Extracted & Translated Text:")); self.text_output = QTextEdit(); right_panel.addWidget(self.text_output)

    def _create_combo_box(self, parent_layout, label_text, items, default=None):
        h_layout = QHBoxLayout(); h_layout.addWidget(QLabel(label_text)); combo = QComboBox(); combo.addItems(items)
        if default: combo.setCurrentText(default)
        h_layout.addWidget(combo); parent_layout.addLayout(h_layout); return combo

    def setup_styles(self):
        self.setStyleSheet(self.DARK_THEME_STYLESHEET)

    def setup_shortcuts(self):
        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self); self.undo_shortcut.activated.connect(self.undo_last_action)
        self.redo_shortcut = QShortcut(QKeySequence("Ctrl+Y"), self); self.redo_shortcut.activated.connect(self.redo_last_action)

    def initialize_ocr_engines(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        gpu_available = False
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            if not gpu_available: print("CUDA not available, OCR engines will run on CPU.")
        except ImportError:
            print("PyTorch not found, OCR engines will run on CPU.")
        try:
            self.statusBar().showMessage("Initializing EasyOCR...")
            lang_list = ['en']
            self.easyocr_reader = easyocr.Reader(lang_list, gpu=gpu_available)
            self.statusBar().showMessage("EasyOCR reader initialized.", 3000)
        except Exception as e:
            QMessageBox.critical(self, "EasyOCR Error", f"Could not initialize EasyOCR.\nIt is required for batch processing.\nError: {e}")
        if MangaOcr:
            try:
                self.statusBar().showMessage("Initializing Manga-OCR...")
                self.manga_ocr_reader = MangaOcr()
                self.statusBar().showMessage("Manga-OCR reader initialized.", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Manga-OCR Error", f"Could not initialize Manga-OCR.\nThis engine will be disabled.\nError: {e}")
                self.manga_ocr_reader = None
        QApplication.restoreOverrideCursor()
        self.statusBar().showMessage("Ready", 3000)

    def on_ocr_engine_changed(self, engine):
        self.lang_combo.setEnabled(engine == "Tesseract")
        self.orientation_combo.setEnabled(engine != "Manga-OCR")

    def clean_and_join_text(self, raw_text):
        return ' '.join(raw_text.split())

    def correct_text_with_gemini(self, text_to_correct):
        if not text_to_correct.strip(): return ""
        if not GEMINI_API_KEY or "your_gemini_key_here" in GEMINI_API_KEY:
            print("Gemini API Key not configured. Skipping AI correction.")
            return text_to_correct
        if not self.check_and_increment_usage():
            return None
        try:
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            prompt = (f"Correct the grammar, spelling, and capitalization of the following text from an OCR scan of a manga. The text might contain errors like incorrect spacing, mixed case, or wrong characters. Do not add any extra explanations, introductions, or quotation marks around the result. Just return the corrected text. Text to correct: \"{text_to_correct}\"")
            response = model.generate_content(prompt)
            return response.text.strip() if response.parts else text_to_correct
        except Exception as e:
            print(f"Error calling Gemini API: {e}"); return text_to_correct

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

    def start_ocr_worker(self, image_to_process, context_data):
        if self.worker_thread and self.worker_thread.isRunning():
            QMessageBox.warning(self, "Busy", "A process is already running. Please wait."); return
        self.progress_bar.setVisible(True); self.progress_bar.setValue(0); self.progress_bar.setFormat("Starting OCR...")
        settings = self.get_current_settings()
        self.worker_thread = QThread()
        self.current_worker = OCRWorker(self, image_to_process, settings, context_data)
        self.current_worker.moveToThread(self.worker_thread)
        self.current_worker.signals.ocr_complete.connect(self.on_ocr_complete)
        self.current_worker.signals.progress.connect(self.update_progress)
        self.current_worker.signals.error.connect(self.on_worker_error)
        self.current_worker.signals.finished.connect(self.on_worker_finished)
        self.worker_thread.started.connect(self.current_worker.run)
        self.worker_thread.start()

    def on_ocr_complete(self, ocr_text, context_data):
        self.progress_bar.setFormat("OCR Finished. Awaiting next step...")
        if self.review_checkbox.isChecked():
            dialog = ReviewDialog(ocr_text, self)
            if dialog.exec_():
                QTimer.singleShot(0, lambda: self.start_translate_worker(dialog.get_text(), context_data))
            else:
                self.progress_bar.setVisible(False); self.image_label.clear_selection()
        else:
            QTimer.singleShot(0, lambda: self.start_translate_worker(ocr_text, context_data))

    def start_translate_worker(self, text_to_process, context_data):
        if self.worker_thread and self.worker_thread.isRunning():
            QMessageBox.warning(self, "Busy", "A process is already running. Please wait."); return
        self.progress_bar.setVisible(True); self.progress_bar.setValue(0); self.progress_bar.setFormat("Starting Translation...")
        settings = self.get_current_settings()
        self.worker_thread = QThread()
        self.current_worker = TranslateWorker(self, text_to_process, settings, context_data)
        self.current_worker.moveToThread(self.worker_thread)
        self.current_worker.signals.translation_complete.connect(self.on_translation_complete)
        self.current_worker.signals.progress.connect(self.update_progress)
        self.current_worker.signals.error.connect(self.on_worker_error)
        self.current_worker.signals.finished.connect(self.on_worker_finished)
        self.current_worker.signals.api_limit_reached.connect(self.on_api_limit_reached)
        self.worker_thread.started.connect(self.current_worker.run)
        self.worker_thread.start()

    def on_translation_complete(self, translated_text, context_data):
        typeset_rect = context_data.get('typeset_rect'); polygon = context_data.get('polygon')
        new_area = TypesetArea(typeset_rect, translated_text, self.typeset_font, self.typeset_color, polygon)
        self.typeset_areas.append(new_area); self.redo_stack.clear()
        self.redraw_all_typeset_areas(); self.update_undo_redo_buttons_state()
        current_text = self.text_output.toPlainText(); separator = "\n" + "="*40 + "\n"
        new_entry = f"Translation ({self.translate_combo.currentText()}):\n{translated_text}"
        self.text_output.setPlainText(f"{current_text}{separator}{new_entry}" if current_text else new_entry)
        self.text_output.verticalScrollBar().setValue(self.text_output.verticalScrollBar().maximum())

    def start_batch_processing(self):
        if not self.current_image_pil:
            QMessageBox.warning(self, "No Image", "Please load an image first."); return
        if self.worker_thread and self.worker_thread.isRunning():
            QMessageBox.warning(self, "Busy", "A process is already running. Please wait."); return
        self.progress_bar.setVisible(True); self.progress_bar.setValue(0); self.progress_bar.setFormat("Starting Batch Mode...")
        settings = self.get_current_settings()
        self.worker_thread = QThread()
        self.current_worker = OCRWorker(self, self.current_image_pil, settings, {}, is_batch=True)
        self.current_worker.moveToThread(self.worker_thread)
        self.current_worker.signals.batch_result_ready.connect(self.on_batch_result_ready)
        self.current_worker.signals.progress.connect(self.update_progress)
        self.current_worker.signals.error.connect(self.on_worker_error)
        self.current_worker.signals.finished.connect(self.on_worker_finished)
        self.current_worker.signals.api_limit_reached.connect(self.on_api_limit_reached)
        self.worker_thread.started.connect(self.current_worker.run)
        self.worker_thread.start()

    def update_progress(self, value, text):
        self.progress_bar.setValue(value); self.progress_bar.setFormat(text)

    def on_batch_result_ready(self, new_areas):
        if not new_areas:
            self.statusBar().showMessage("Batch processing finished, but no text areas were added.", 5000); return
        self.typeset_areas.extend(new_areas); self.redo_stack.clear()
        self.redraw_all_typeset_areas(); self.update_undo_redo_buttons_state()
        self.statusBar().showMessage(f"Batch processing complete. Added {len(new_areas)} typeset areas.", 5000)

    def on_worker_error(self, error_msg):
        QMessageBox.critical(self, "Processing Error", f"An error occurred in the worker thread:\n{error_msg}")
        self.progress_bar.setVisible(False); self.image_label.clear_selection()

    def on_worker_finished(self):
        if self.worker_thread: self.worker_thread.quit(); self.worker_thread.wait()
        self.worker_thread = None; self.current_worker = None
        self.image_label.clear_selection(); self.progress_bar.setVisible(False)

    def on_api_limit_reached(self):
        QMessageBox.warning(self, "API Limit Reached", 
                              "Gemini API limit has been reached. "
                              "Proceeding without AI correction for this request.")
        self.check_limits_and_update_ui()

    def get_current_settings(self):
        return {'ocr_engine': self.ocr_engine_combo.currentText(), 'ocr_lang': self.lang_combo.currentText(), 'orientation': self.orientation_combo.currentText(), 'target_lang': self.translate_combo.currentText(), 'use_ai': self.ai_correction_checkbox.isChecked(), 'font': self.typeset_font, 'color': self.typeset_color}

    def translate_text(self, text, target_lang):
        if not text.strip(): return ""
        if not DEEPL_API_KEY or "your_deepl_key_here" in DEEPL_API_KEY: return "[DEEPL API KEY NOT CONFIGURED]"
        try:
            lang_map = {"Indonesian": "ID", "English": "EN-US", "Japanese": "JA", "Chinese": "ZH", "Korean": "KO"}
            url = "https://api-free.deepl.com/v2/translate"
            params = {"auth_key": DEEPL_API_KEY, "text": text, "target_lang": lang_map.get(target_lang, "ID")}
            response = requests.post(url, data=params, timeout=20); response.raise_for_status()
            return response.json()["translations"][0]["text"]
        except requests.exceptions.RequestException as e: return f"[Translation Network Error: {e}]"
        except Exception as e: return f"[Translation Error: {e}]"

    def load_usage_data(self):
        try:
            if os.path.exists(self.usage_file_path):
                with open(self.usage_file_path, 'rb') as f: self.usage_data = pickle.load(f)
                if self.usage_data.get('date') != str(date.today()):
                    self.usage_data = {'date': str(date.today()), 'daily_count': 0, 'minute_count': 0, 'last_call_timestamp': 0}
                    self.save_usage_data()
            else: self.save_usage_data()
        except Exception as e:
            print(f"Could not load or create usage data file: {e}")
            self.usage_data = {'date': str(date.today()), 'daily_count': 0, 'minute_count': 0, 'last_call_timestamp': 0}

    def save_usage_data(self):
        try:
            with open(self.usage_file_path, 'wb') as f: pickle.dump(self.usage_data, f)
        except Exception as e: print(f"Could not save usage data: {e}")

    def check_and_increment_usage(self):
        self.load_usage_data(); current_time = time.time()
        if self.usage_data['daily_count'] >= GEMINI_RPD_LIMIT:
            print("Daily API limit reached."); return False
        time_since_last_call = current_time - self.usage_data['last_call_timestamp']
        if time_since_last_call > 60: self.usage_data['minute_count'] = 0
        if self.usage_data['minute_count'] >= GEMINI_RPM_LIMIT:
            print(f"Per-minute API limit reached. Try again in {60 - time_since_last_call:.0f} seconds."); return False
        self.usage_data['daily_count'] += 1; self.usage_data['minute_count'] += 1; self.usage_data['last_call_timestamp'] = current_time
        self.save_usage_data(); print(f"Gemini API call count: {self.usage_data['minute_count']}(RPM), {self.usage_data['daily_count']}(RPD)"); return True

    def check_limits_and_update_ui(self):
        daily_limit_reached = self.usage_data.get('daily_count', 0) >= GEMINI_RPD_LIMIT
        time_since_last_call = time.time() - self.usage_data.get('last_call_timestamp', 0)
        minute_limit_reached = False
        if time_since_last_call <= 60:
            if self.usage_data.get('minute_count', 0) >= GEMINI_RPM_LIMIT: minute_limit_reached = True
        if daily_limit_reached:
            self.ai_correction_checkbox.setChecked(False); self.ai_correction_checkbox.setEnabled(False); self.ai_correction_checkbox.setToolTip("AI Correction disabled: Daily API limit reached.")
            if self.api_limit_timer.isActive(): self.api_limit_timer.stop()
        elif minute_limit_reached:
            self.ai_correction_checkbox.setChecked(False); self.ai_correction_checkbox.setEnabled(False); remaining_time = 61 - int(time_since_last_call)
            self.ai_correction_checkbox.setToolTip(f"AI Correction disabled: Per-minute limit reached. Available in {remaining_time}s.")
            if not self.api_limit_timer.isActive(): self.api_limit_timer.start()
        else:
            self.ai_correction_checkbox.setEnabled(True); self.ai_correction_checkbox.setToolTip("Enable/Disable AI-powered text correction using Gemini API.")
            if self.api_limit_timer.isActive(): self.api_limit_timer.stop()

    def periodic_limit_check(self):
        if not self.ai_correction_checkbox.isEnabled():
            if self.usage_data.get('daily_count', 0) >= GEMINI_RPD_LIMIT: self.api_limit_timer.stop(); return
            time_since_last_call = time.time() - self.usage_data.get('last_call_timestamp', 0)
            if time_since_last_call > 60:
                print("Per-minute API cooldown finished. Re-enabling AI feature.")
                self.ai_correction_checkbox.setEnabled(True); self.ai_correction_checkbox.setChecked(True)
                self.ai_correction_checkbox.setToolTip("Enable/Disable AI-powered text correction using Gemini API."); self.api_limit_timer.stop()
            else:
                remaining_time = 61 - int(time_since_last_call)
                self.ai_correction_checkbox.setToolTip(f"AI Correction disabled: Per-minute limit reached. Available in {remaining_time}s.")

    def load_folder(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Manga Folder")
        if dir_path:
            if self.pdf_document:
                self.pdf_document.close()
                self.pdf_document = None
            self.current_pdf_page = -1
            
            self.project_dir = dir_path; self.cache_dir = os.path.join(self.project_dir, ".cache"); os.makedirs(self.cache_dir, exist_ok=True)
            self.image_files = []; self.file_list_widget.clear()
            
            supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.pdf')
            
            def natural_sort_key(s):
                return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
            
            sorted_files = sorted(os.listdir(dir_path), key=natural_sort_key)

            for fname in sorted_files:
                if fname.lower().endswith(supported_formats): 
                    self.image_files.append(os.path.join(dir_path, fname))
            
            for fpath in self.image_files: 
                self.file_list_widget.addItem(QListWidgetItem(os.path.basename(fpath)))
            
            if self.image_files: 
                self.file_list_widget.setCurrentRow(0)

    def on_file_selected(self, current_item, previous_item):
        if not current_item: return
        row = self.file_list_widget.row(current_item)
        if row != self.current_image_index: 
            self.load_item_at_index(row)

    def load_item_at_index(self, index):
        if 0 <= index < len(self.image_files):
            self.current_image_index = index
            file_path = self.image_files[index]
            
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
            self.typeset_areas.clear(); self.redo_stack.clear()
            self.typeset_pixmap = self.original_pixmap.copy()
            self.image_label.clear_selection(); self.update_display()
            self.update_undo_redo_buttons_state(); self.text_output.clear()
        except Exception as e:
            QMessageBox.critical(self, "Error Loading Image", f"Could not load image: {file_path}\nError: {e}")
            self.original_pixmap = None; self.typeset_pixmap = None; self.update_display()

    def load_pdf(self, file_path):
        try:
            if self.pdf_document:
                self.pdf_document.close()
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
        
        self.current_pdf_page = page_number
        page = self.pdf_document.load_page(page_number)
        pix = page.get_pixmap(dpi=150)
        q_image = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888)
        self.original_pixmap = QPixmap.fromImage(q_image)
        self.current_image_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        self.typeset_areas.clear(); self.redo_stack.clear()
        self.typeset_pixmap = self.original_pixmap.copy()
        self.image_label.clear_selection(); self.update_display()
        self.update_undo_redo_buttons_state(); self.text_output.clear()
        self.update_nav_buttons()

    def load_next_image(self):
        if self.pdf_document:
            if self.current_pdf_page < self.pdf_document.page_count - 1:
                self.load_pdf_page(self.current_pdf_page + 1)
        elif self.current_image_index < len(self.image_files) - 1:
            self.file_list_widget.setCurrentRow(self.current_image_index + 1)

    def load_prev_image(self):
        if self.pdf_document:
            if self.current_pdf_page > 0:
                self.load_pdf_page(self.current_pdf_page - 1)
        elif self.current_image_index > 0:
            self.file_list_widget.setCurrentRow(self.current_image_index - 1)

    def update_nav_buttons(self):
        if self.pdf_document:
            self.prev_button.setEnabled(self.current_pdf_page > 0)
            self.next_button.setEnabled(self.current_pdf_page < self.pdf_document.page_count - 1)
            self.statusBar().showMessage(f"PDF Page {self.current_pdf_page + 1} / {self.pdf_document.page_count}")
        else:
            self.prev_button.setEnabled(self.current_image_index > 0)
            self.next_button.setEnabled(self.current_image_index < len(self.image_files) - 1)
            if self.current_image_index != -1:
                self.statusBar().showMessage(f"Image {self.current_image_index + 1} / {len(self.image_files)}")

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
            self.typeset_pixmap = self.original_pixmap.copy()
            self.zoom_factor = 1.0; self.image_label.clear_selection()
            self.update_display(); self.update_undo_redo_buttons_state()

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
        if not self.current_image_pil: return
        unzoomed_rect = self.unzoom_coords(selection_rect)
        if not unzoomed_rect or unzoomed_rect.width() <= 0 or unzoomed_rect.height() <= 0: return
        cropped_img = self.current_image_pil.crop((unzoomed_rect.x(), unzoomed_rect.y(), unzoomed_rect.right(), unzoomed_rect.bottom()))
        cropped_cv_img = cv2.cvtColor(np.array(cropped_img), cv2.COLOR_RGB2BGR)
        context = {'typeset_rect': unzoomed_rect, 'preprocessed_image_cv': cropped_cv_img}
        self.start_ocr_worker(cropped_cv_img, context)

    def process_polygon_area(self, scaled_points):
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
        context = {'typeset_rect': unzoomed_bbox, 'polygon': unzoomed_polygon, 'preprocessed_image_cv': img_for_ocr}
        self.start_ocr_worker(img_for_ocr, context)

    def redraw_all_typeset_areas(self):
        if not self.original_pixmap: return
        self.typeset_pixmap = self.original_pixmap.copy()
        painter = QPainter(self.typeset_pixmap)
        for area in self.typeset_areas:
            self.draw_single_area(painter, area)
        painter.end()
        self.update_display()

    def draw_single_area(self, painter, area):
        cv_original = cv2.cvtColor(np.array(self.current_image_pil), cv2.COLOR_RGB2BGR)
        mask = np.zeros(cv_original.shape[:2], dtype=np.uint8)
        if area.polygon:
            cv_poly_points = np.array([[p.x(), p.y()] for p in area.polygon], dtype=np.int32)
            cv2.fillPoly(mask, [cv_poly_points], 255)
        else:
            cv2.rectangle(mask, (area.rect.x(), area.rect.y()), (area.rect.right(), area.rect.bottom()), 255, -1)
        inpainted_cv = cv2.inpaint(cv_original, mask, 3, cv2.INPAINT_TELEA)
        inpainted_pil = Image.fromarray(cv2.cvtColor(inpainted_cv, cv2.COLOR_BGR2RGB))
        data = inpainted_pil.tobytes('raw', 'RGB')
        q_image = QImage(data, inpainted_pil.width, inpainted_pil.height, QImage.Format_RGB888)
        painter.drawImage(area.rect, q_image, area.rect)
        painter.save()
        if area.polygon:
            path = QPainterPath(); path.addPolygon(QPolygonF(area.polygon)); painter.setClipPath(path)
        is_vertical = self.vertical_typeset_checkbox.isChecked()
        self.draw_text_with_options(painter, area.rect, area.text, area.get_font(), area.get_color(), is_vertical)
        painter.restore()

    def draw_text_with_options(self, painter, rect, text, font, color, is_vertical=False):
        if is_vertical:
            self.draw_rotated_vertical_text(painter, rect, text, font, color); return
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
            original_filename = os.path.basename(self.image_files[self.current_image_index])
            name, _ = os.path.splitext(original_filename)
            save_suggestion = os.path.join(os.path.dirname(self.image_files[self.current_image_index]), f"{name}_page_{self.current_pdf_page + 1}_typeset.png")
        else:
            original_filename = os.path.basename(self.image_files[self.current_image_index])
            name, _ = os.path.splitext(original_filename)
            save_suggestion = os.path.join(os.path.dirname(self.image_files[self.current_image_index]), f"{name}_typeset.png")

        filePath, _ = QFileDialog.getSaveFileName(self, "Save Typeset Image", save_suggestion, "PNG Image (*.png);;JPEG Image (*.jpg)")
        if filePath:
            if not self.typeset_pixmap.save(filePath): QMessageBox.critical(self, "Error", "Failed to save the image.")
            else: QMessageBox.information(self, "Success", f"Image saved to:\n{filePath}")

    def undo_last_action(self):
        if self.typeset_areas:
            undone_area = self.typeset_areas.pop(); self.redo_stack.append(undone_area)
            self.redraw_all_typeset_areas(); self.update_undo_redo_buttons_state(); self.image_label.clear_selection()

    def redo_last_action(self):
        if self.redo_stack:
            redone_area = self.redo_stack.pop(); self.typeset_areas.append(redone_area)
            self.redraw_all_typeset_areas(); self.update_undo_redo_buttons_state(); self.image_label.clear_selection()

    def update_undo_redo_buttons_state(self):
        self.undo_button.setEnabled(len(self.typeset_areas) > 0); self.redo_button.setEnabled(len(self.redo_stack) > 0)

    def save_project(self):
        if not self.image_files: QMessageBox.warning(self, "Nothing to Save", "Please load a folder first."); return False
        if not self.current_project_path:
            filePath, _ = QFileDialog.getSaveFileName(self, "Save Project", "", "Manga Project (*.manga_proj)")
            if not filePath: return False
            self.current_project_path = filePath
        project_data = {'image_files': self.image_files, 'current_index': self.current_image_index, 'typeset_areas': self.typeset_areas, 'redo_stack': self.redo_stack, 'text_output': self.text_output.toPlainText(), 'font': TypesetArea.font_to_dict(self.typeset_font), 'color': self.typeset_color.name()}
        try:
            with open(self.current_project_path, 'wb') as f: pickle.dump(project_data, f)
            self.setWindowTitle(f"Manga OCR & Typeset Tool v7.8 - {os.path.basename(self.current_project_path)}"); self.autosave_timer.start(); return True
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
            self.project_dir = os.path.dirname(filePath); self.cache_dir = os.path.join(self.project_dir, ".cache"); os.makedirs(self.cache_dir, exist_ok=True)
            self.image_files = project_data.get('image_files', []); self.typeset_areas = project_data.get('typeset_areas', []); self.redo_stack = project_data.get('redo_stack', [])
            font_info = project_data.get('font', {'family': 'Arial', 'pointSize': 12, 'weight': QFont.Bold, 'italic': False})
            self.typeset_font = QFont(); self.typeset_font.setFamily(font_info['family']); self.typeset_font.setPointSize(font_info['pointSize']); self.typeset_font.setWeight(font_info['weight']); self.typeset_font.setItalic(font_info['italic'])
            self.typeset_color = QColor(project_data.get('color', '#000000'))
            self.file_list_widget.clear()
            for fpath in self.image_files: self.file_list_widget.addItem(QListWidgetItem(os.path.basename(fpath)))
            index_to_load = project_data.get('current_index', -1)
            if index_to_load != -1:
                self.load_item_at_index(index_to_load)
                self.typeset_areas = project_data.get('typeset_areas', []); self.redo_stack = project_data.get('redo_stack', [])
                self.text_output.setPlainText(project_data.get('text_output', '')); self.redraw_all_typeset_areas(); self.update_undo_redo_buttons_state()
                self.file_list_widget.setCurrentRow(index_to_load)
            self.current_project_path = filePath; self.setWindowTitle(f"Manga OCR & Typeset Tool v7.8 - {os.path.basename(self.current_project_path)}"); self.autosave_timer.start()
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
        if self.current_theme == 'dark': self.current_theme = 'light'; self.setStyleSheet(self.LIGHT_THEME_STYLESHEET)
        else: self.current_theme = 'dark'; self.setStyleSheet(self.DARK_THEME_STYLESHEET)

    def show_about_dialog(self):
        self.load_usage_data()
        daily_count = self.usage_data.get('daily_count', 0); minute_count = self.usage_data.get('minute_count', 0)
        time_since_last_call = time.time() - self.usage_data.get('last_call_timestamp', 0)
        current_rpm = 0 if time_since_last_call > 60 else minute_count
        about_text = (f"<b>Manga OCR & Typeset Tool v7.8</b><br><br>This tool was created to streamline the process of translating manga.<br><br>Powered by Python, PyQt5, Manga-OCR, EasyOCR, Tesseract, and Gemini API.<br>Enhanced with new features by Gemini.<br><br>---<br><br><b>Gemini API Usage (gemini-1.5-flash-latest):</b><br> - Usage This Minute: <b>{current_rpm} / {GEMINI_RPM_LIMIT}</b> RPM<br> - Usage Today: <b>{daily_count} / {GEMINI_RPD_LIMIT}</b> RPD<br><br><i>The AI feature will be disabled automatically if a limit is reached. The RPM count resets 60 seconds after the first call in a new minute, and the RPD count resets daily.</i><br><br>Copyright © 2024")
        QMessageBox.about(self, "About & API Usage", about_text)

    def get_cache_key(self, cv_image):
        image_bytes = cv2.imencode('.png', cv_image)[1].tobytes()
        return hashlib.md5(image_bytes).hexdigest()

    def check_cache(self, key):
        if not self.cache_dir or self.current_image_index < 0: return None
        current_item_name = os.path.basename(self.image_files[self.current_image_index])
        
        if self.pdf_document:
            current_item_name += f"_page_{self.current_pdf_page}"

        cache_file = os.path.join(self.cache_dir, f"{current_item_name}.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                try: return json.load(f).get(key)
                except json.JSONDecodeError: return None
        return None

    def write_to_cache(self, key, data):
        if not self.cache_dir or self.current_image_index < 0: return
        current_item_name = os.path.basename(self.image_files[self.current_image_index])
        
        if self.pdf_document:
            current_item_name += f"_page_{self.current_pdf_page}"

        cache_file = os.path.join(self.cache_dir, f"{current_item_name}.json")
        cache_data = {}
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                try: cache_data = json.load(f)
                except json.JSONDecodeError: cache_data = {}
        cache_data[key] = data
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=4, ensure_ascii=False)

    def export_to_pdf(self):
        if not self.project_dir:
            QMessageBox.warning(self, "No Folder Loaded", "Please load a folder containing images first.")
            return

        image_files_to_export = [f for f in self.image_files if f.lower().endswith('.png')]

        if not image_files_to_export:
            QMessageBox.warning(self, "No PNG Files Found", "The current folder contains no .png files to export.")
            return

        folder_name = os.path.basename(self.project_dir)
        save_suggestion = os.path.join(self.project_dir, f"{folder_name}.pdf")
        
        pdf_path, _ = QFileDialog.getSaveFileName(self, "Save PDF As", save_suggestion, "PDF Files (*.pdf)")

        if not pdf_path:
            return

        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', os.path.basename(s))]
        
        image_files_to_export.sort(key=natural_sort_key)

        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage("Exporting to PDF... Please wait.")
        
        try:
            images_pil = []
            for i, f in enumerate(image_files_to_export):
                self.progress_bar.setVisible(True)
                self.update_progress(int((i/len(image_files_to_export))*100), f"Converting {os.path.basename(f)}...")
                img = Image.open(f).convert("RGB")
                images_pil.append(img)

            if images_pil:
                self.update_progress(100, "Saving PDF...")
                images_pil[0].save(
                    pdf_path, "PDF", resolution=100.0, save_all=True, append_images=images_pil[1:]
                )
                QMessageBox.information(self, "Success", f"Successfully exported {len(images_pil)} images to:\n{pdf_path}")
            else:
                raise Exception("No images could be processed.")

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"An error occurred while exporting to PDF:\n{e}")
        finally:
            QApplication.restoreOverrideCursor()
            self.progress_bar.setVisible(False)
            self.statusBar().showMessage("Ready", 3000)

    def wheelEvent(self, event):
        if self.pdf_document and not (self.worker_thread and self.worker_thread.isRunning()):
            if event.angleDelta().y() < 0:
                self.load_next_image()
            elif event.angleDelta().y() > 0:
                self.load_prev_image()
        
        super().wheelEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    try:
        import fitz
    except ImportError:
        QMessageBox.critical(None, "Dependency Missing", "PyMuPDF is not installed. Please run 'pip install PyMuPDF'.\nPDF features will be disabled.")
        sys.exit()
        
    # --- PERUBAHAN v7.8: Pengecekan config dipindahkan ke sini ---
    if not DEEPL_API_KEY or "your_deepl_key_here" in DEEPL_API_KEY:
        QMessageBox.warning(None, "DeepL API Key Missing", "Please provide your valid DeepL API key in 'config.ini' to enable translation.")
    if not GEMINI_API_KEY or "your_gemini_key_here" in GEMINI_API_KEY:
        QMessageBox.warning(None, "Gemini API Key Missing", "Please provide your valid Gemini API key in 'config.ini' to enable AI text correction.")
    
    try:
        tesseract_version = pytesseract.get_tesseract_version()
        print(f"Tesseract version {tesseract_version} found.")
    except Exception:
        QMessageBox.warning(None, "Tesseract Not Found", f"Tesseract was not found at the path specified in config.ini:\n{TESSERACT_PATH}\nPlease install it or correct the path. The 'Tesseract' engine will be disabled.")
    
    if not MangaOcr:
        QMessageBox.warning(None, "Manga-OCR Not Found", "The 'manga-ocr' library is not installed. Please run 'pip install manga-ocr'.\nThe 'Manga-OCR' engine will be disabled.")
    
    window = MangaOCRApp()
    window.show()
    sys.exit(app.exec_())
