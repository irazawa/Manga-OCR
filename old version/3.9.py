# Manga OCR & Typeset Tool v3.9
# Ditingkatkan oleh Gemini
#
# --- FITUR BARU DI v3.9 ---
# 1. Perbaikan Model AI: Memperbarui nama model Gemini ke 'gemini-1.5-flash-latest'
#    untuk mengatasi error 404 dan memastikan kompatibilitas dengan API terbaru.
#
# --- FITUR SEBELUMNYA ---
# - Integrasi AI untuk Koreksi Teks
# - Zoom Tetap & Fungsi Redo
# - Pratinjau & Edit OCR
# - Light & Dark Mode
# - Project Management
#
# --- INSTRUKSI INSTALASI ---
# 1. Pastikan Python terinstal.
# 2. Instal library yang diperlukan:
#    pip install PyQt5 numpy opencv-python pytesseract requests Pillow easyocr torch torchvision torchaudio google-generativeai
#
# 3. Instal Tesseract OCR Engine.
#
# 4. Konfigurasi Path & API Key di bawah ini.
# --------------------------------------------------------------------

import os
import sys
import numpy as np
import cv2
import pytesseract
import requests
import easyocr
import pickle
import google.generativeai as genai
from PIL import Image, ImageQt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QFileDialog, QTextEdit, QScrollArea, QComboBox, QMessageBox,
    QProgressBar, QShortcut, QListWidget, QListWidgetItem, QColorDialog, QFontDialog,
    QLineEdit, QAction, QDialog, QCheckBox
)
from PyQt5.QtGui import (
    QPixmap, QPainter, QPen, QColor, QFont, QKeySequence, QPolygon, QPainterPath, QPolygonF
)
from PyQt5.QtCore import Qt, QRect, QPoint, QSize, pyqtSignal

# --- KONFIGURASI PENTING ---
DEEPL_API_KEY = "2107028d-26f0-4604-a625-80d62f226311:fx" # GANTI DENGAN API KEY DEEPL ANDA
GEMINI_API_KEY = "AIzaSyCRJNB4SukhoZvOaZNCO9p1bLCNMnHABTM" # GANTI DENGAN API KEY GEMINI ANDA

# Konfigurasi Gemini API
if GEMINI_API_KEY and "GANTI_DENGAN" not in GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"Gagal mengkonfigurasi Gemini API: {e}")

try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except Exception:
    print("Tesseract path not found at default location. Please check configuration if needed.")

# --------------------------------------------------------------------

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
        self.translate_button = QPushButton("Translate")
        self.translate_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.translate_button)
        
        layout.addLayout(button_layout)

    def get_text(self):
        return self.text_edit.toPlainText()

class TypesetArea:
    """Menyimpan semua informasi tentang satu area typesetting."""
    def __init__(self, rect, text, font, color, polygon=None):
        self.rect = rect
        self.text = text
        self.font_info = self.font_to_dict(font)
        self.color_info = color.name()
        self.polygon = polygon

    @staticmethod
    def font_to_dict(font):
        return {
            'family': font.family(),
            'pointSize': font.pointSize(),
            'weight': font.weight(),
            'italic': font.italic()
        }

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
        QTextEdit, QComboBox, QListWidget { background-color: #3c3c3c; color: #f0f0f0; border: 1px solid #555; padding: 5px; border-radius: 4px; }
        QScrollArea { border: 1px solid #444; }
        QProgressBar { border: 1px solid #555; border-radius: 4px; text-align: center; color: #f0f0f0; }
        QProgressBar::chunk { background-color: #007acc; border-radius: 3px; }
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
        QTextEdit, QComboBox, QListWidget { background-color: #ffffff; color: #000000; border: 1px solid #c0c0c0; padding: 5px; border-radius: 4px; }
        QScrollArea { border: 1px solid #c0c0c0; }
        QProgressBar { border: 1px solid #c0c0c0; border-radius: 4px; text-align: center; color: #000000; background-color: #e6e6e6;}
        QProgressBar::chunk { background-color: #0078d7; border-radius: 3px; }
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Manga OCR & Typeset Tool v3.9")
        self.setGeometry(50, 50, 1600, 900)

        # State
        self.image_files = []
        self.current_image_index = -1
        self.current_image_pil = None
        self.original_pixmap = None
        self.typeset_pixmap = None
        self.zoom_factor = 1.0
        self.typeset_areas = []
        self.redo_stack = []
        self.easyocr_reader = None
        self.current_project_path = None
        self.current_theme = 'dark'
        self.typeset_font = QFont("Arial", 12, QFont.Bold)
        self.typeset_color = QColor(Qt.black)
        self.inline_editor = None
        self.editing_area = None

        self.init_ui()
        self.setup_styles()
        self.setup_shortcuts()
        self.update_easyocr_reader()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        self.setup_menu_bar()
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
        save_project_action = QAction('Save Project', self)
        save_project_action.setShortcut('Ctrl+S')
        save_project_action.triggered.connect(self.save_project)
        file_menu.addAction(save_project_action)
        load_project_action = QAction('Load Project', self)
        load_project_action.setShortcut('Ctrl+O')
        load_project_action.triggered.connect(self.load_project)
        file_menu.addAction(load_project_action)
        view_menu = menu_bar.addMenu('&View')
        toggle_theme_action = QAction('Toggle Light/Dark Mode', self)
        toggle_theme_action.triggered.connect(self.toggle_theme)
        view_menu.addAction(toggle_theme_action)
        help_menu = menu_bar.addMenu('&Help')
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

    def setup_right_panel(self, right_panel):
        right_panel.addWidget(QLabel("<h3>Processing Options</h3>"))
        self.ocr_engine_combo = self._create_combo_box(right_panel, "OCR Engine:", ["EasyOCR", "Tesseract"])
        self.lang_combo = self._create_combo_box(right_panel, "OCR Language:", ["Japanese", "English", "Chinese", "Korean"], "English")
        self.orientation_combo = self._create_combo_box(right_panel, "Text Orientation:", ["Vertical", "Horizontal"], "Vertical")
        self.translate_combo = self._create_combo_box(right_panel, "Translate to:", ["Indonesian", "English", "Japanese", "Chinese", "Korean"], "Indonesian")
        
        self.review_checkbox = QCheckBox("Review Before Translating")
        self.review_checkbox.setChecked(True)
        right_panel.addWidget(self.review_checkbox)
        
        self.ocr_engine_combo.currentTextChanged.connect(self.on_ocr_engine_changed)
        self.on_ocr_engine_changed("EasyOCR")
        
        right_panel.addWidget(QLabel("<h3>Selection Tools</h3>"))
        self.selection_mode_combo = self._create_combo_box(right_panel, "Selection Mode:", ["Rectangle", "Pen Tool"])
        self.selection_mode_combo.currentTextChanged.connect(self.selection_mode_changed)
        pen_buttons_layout = QHBoxLayout()
        self.confirm_button = QPushButton("Confirm Selection")
        self.confirm_button.clicked.connect(self.confirm_pen_selection)
        self.confirm_button.setVisible(False)
        pen_buttons_layout.addWidget(self.confirm_button)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_pen_selection)
        self.cancel_button.setVisible(False)
        pen_buttons_layout.addWidget(self.cancel_button)
        right_panel.addLayout(pen_buttons_layout)
        right_panel.addWidget(QLabel("<h3>Typesetting Options</h3>"))
        typeset_options_layout = QHBoxLayout()
        self.font_button = QPushButton("Choose Font")
        self.font_button.clicked.connect(self.choose_font)
        typeset_options_layout.addWidget(self.font_button)
        self.color_button = QPushButton("Choose Color")
        self.color_button.clicked.connect(self.choose_color)
        typeset_options_layout.addWidget(self.color_button)
        right_panel.addLayout(typeset_options_layout)
        self.style_transfer_button = QPushButton("Mimic Font Style (Concept)")
        self.style_transfer_button.clicked.connect(self.show_style_transfer_info)
        right_panel.addWidget(self.style_transfer_button)
        right_panel.addWidget(QLabel("<h3>Actions</h3>"))
        
        undo_redo_layout = QHBoxLayout()
        self.undo_button = QPushButton("Undo (Ctrl+Z)")
        self.undo_button.clicked.connect(self.undo_last_action)
        self.undo_button.setEnabled(False)
        undo_redo_layout.addWidget(self.undo_button)

        self.redo_button = QPushButton("Redo (Ctrl+Y)")
        self.redo_button.clicked.connect(self.redo_last_action)
        self.redo_button.setEnabled(False)
        undo_redo_layout.addWidget(self.redo_button)
        right_panel.addLayout(undo_redo_layout)

        self.reset_button = QPushButton("Reset View")
        self.reset_button.clicked.connect(self.reset_view_to_original)
        right_panel.addWidget(self.reset_button)
        self.save_button = QPushButton("Save Image")
        self.save_button.clicked.connect(self.save_image)
        right_panel.addWidget(self.save_button)
        right_panel.addStretch()
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        right_panel.addWidget(self.progress_bar)
        right_panel.addWidget(QLabel("Extracted & Translated Text:"))
        self.text_output = QTextEdit()
        right_panel.addWidget(self.text_output)

    def _create_combo_box(self, parent_layout, label_text, items, default=None):
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel(label_text))
        combo = QComboBox()
        combo.addItems(items)
        if default: combo.setCurrentText(default)
        h_layout.addWidget(combo)
        parent_layout.addLayout(h_layout)
        return combo

    def setup_styles(self):
        self.setStyleSheet(self.DARK_THEME_STYLESHEET)

    def setup_shortcuts(self):
        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.activated.connect(self.undo_last_action)
        self.redo_shortcut = QShortcut(QKeySequence("Ctrl+Y"), self)
        self.redo_shortcut.activated.connect(self.redo_last_action)

    def update_easyocr_reader(self):
        if self.ocr_engine_combo.currentText() != "EasyOCR":
            return
        
        lang_list = ['ja', 'en'] # Tambahkan bahasa yang relevan

        try:
            print(f"Initializing EasyOCR for language(s): {lang_list}...")
            QApplication.setOverrideCursor(Qt.WaitCursor)
            gpu_available = True
            try:
                import torch
                if not torch.cuda.is_available():
                    gpu_available = False
            except ImportError:
                gpu_available = False
            
            self.easyocr_reader = easyocr.Reader(lang_list, gpu=gpu_available)
            print("EasyOCR reader initialized successfully.")
        except Exception as e:
            QMessageBox.critical(self, "EasyOCR Error", f"Could not initialize EasyOCR.\n"
                                         f"Please check your installation.\nError: {e}")
            self.easyocr_reader = None
        finally:
            QApplication.restoreOverrideCursor()

    def on_ocr_engine_changed(self, engine):
        if engine == "EasyOCR":
            self.orientation_combo.setEnabled(False)
            self.lang_combo.setEnabled(False) # EasyOCR menangani bahasa dari reader
            self.update_easyocr_reader()
        else: # Tesseract
            self.orientation_combo.setEnabled(True)
            self.lang_combo.setEnabled(True)

    def clean_and_join_text(self, raw_text):
        return ' '.join(raw_text.split())

    def correct_text_with_gemini(self, text_to_correct):
        """Memperbaiki teks menggunakan Gemini API."""
        if not text_to_correct.strip():
            return "" # Kembalikan string kosong jika tidak ada input
        
        if not GEMINI_API_KEY or "GANTI_DENGAN" in GEMINI_API_KEY:
            print("Gemini API Key tidak dikonfigurasi. Melewatkan perbaikan teks dengan AI.")
            return text_to_correct

        try:
            # --- PERBAIKAN: Menggunakan model yang lebih baru dan valid ---
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            prompt = (
                "Correct the grammar, spelling, and capitalization of the following text from an OCR scan. "
                "The text might contain errors like incorrect spacing, mixed case, or wrong characters. "
                "Do not add any extra explanations, introductions, or quotation marks around the result. Just return the corrected text. "
                "For example, if the input is 'tonicHT; BloOD Will Fly; Flesh Will Be Cut; AND Lives Will Be LOsT; AS Our CONTESTANTS EnGAGE In MORTAL Battlel', "
                "the output should be 'Tonight, blood will fly, flesh will be cut, and lives will be lost, as our contestants engage in mortal battle!'.\n\n"
                f"Text to correct: \"{text_to_correct}\""
            )
            response = model.generate_content(prompt)
            if response.parts:
                return response.text.strip()
            else:
                print("Peringatan: Respons dari Gemini kosong.")
                return text_to_correct
        except Exception as e:
            print(f"Error saat memanggil Gemini API: {e}")
            return text_to_correct

    def run_ocr_and_translate(self, image_to_process, typesetting_rect, polygon=None):
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(10)
            if isinstance(image_to_process, Image.Image):
                cv_image = cv2.cvtColor(np.array(image_to_process), cv2.COLOR_RGB2BGR)
            else:
                cv_image = image_to_process
            self.progress_bar.setValue(30)
            QApplication.processEvents()
            ocr_engine = self.ocr_engine_combo.currentText()
            raw_text = ""
            if ocr_engine == "EasyOCR":
                if not self.easyocr_reader:
                    raise Exception("EasyOCR is not initialized.")
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                results = self.easyocr_reader.readtext(gray, detail=1, paragraph=True)
                raw_text = "\n".join([res[1] for res in results])
            else: # Tesseract
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                if np.mean(gray) > 240: ocr_image = gray
                else: _, ocr_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                lang_map = {"Japanese": "jpn_vert", "English": "eng", "Chinese": "chi_sim_vert", "Korean": "kor_vert"}
                tess_lang = lang_map.get(self.lang_combo.currentText(), "eng")
                orientation = self.orientation_combo.currentText()
                if orientation == "Horizontal": tess_lang = tess_lang.replace('_vert', '')
                psm = '5' if orientation == "Vertical" else '6'
                custom_config = f'--oem 1 --psm {psm} -l {tess_lang}'
                raw_text = pytesseract.image_to_string(ocr_image, config=custom_config).strip()
            
            processed_text = self.clean_and_join_text(raw_text)
            
            # --- PANGGILAN AI UNTUK KOREKSI ---
            self.progress_bar.setFormat("Correcting text with AI...")
            self.progress_bar.setValue(50)
            QApplication.processEvents()
            
            print(f"Teks sebelum perbaikan AI: {processed_text}")
            corrected_text_by_ai = self.correct_text_with_gemini(processed_text)
            print(f"Teks setelah perbaikan AI: {corrected_text_by_ai}")
            
            text_for_translation = corrected_text_by_ai
            # --- AKHIR PANGGILAN AI ---
            
            if self.review_checkbox.isChecked():
                dialog = ReviewDialog(text_for_translation, self)
                if dialog.exec_() == QDialog.Accepted:
                    text_for_translation = dialog.get_text()
                else:
                    self.progress_bar.setVisible(False)
                    self.image_label.clear_selection()
                    return

            self.progress_bar.setFormat("Translating...")
            self.progress_bar.setValue(70)
            QApplication.processEvents()
            
            target_lang = self.translate_combo.currentText()
            translated_text = self.translate_text(text_for_translation, target_lang)
            
            self.progress_bar.setValue(90)
            current_text = self.text_output.toPlainText()
            separator = "\n" + "="*40 + "\n"
            new_entry = f"Original (AI Corrected):\n{text_for_translation}\n\nTranslation ({target_lang}):\n{translated_text}"
            self.text_output.setPlainText(f"{current_text}{separator}{new_entry}" if current_text else new_entry)
            self.text_output.verticalScrollBar().setValue(self.text_output.verticalScrollBar().maximum())
            
            new_area = TypesetArea(typesetting_rect, translated_text, self.typeset_font, self.typeset_color, polygon)
            self.typeset_areas.append(new_area)
            
            self.redo_stack.clear()
            self.redraw_all_typeset_areas()
            self.update_undo_redo_buttons_state()
            self.progress_bar.setValue(100)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during processing:\n{str(e)}")
            self.image_label.clear_selection()
        finally:
            self.progress_bar.setVisible(False)
            self.progress_bar.setFormat("%p%")


    def translate_text(self, text, target_lang):
        if not text.strip(): return "[No text to translate]"
        if not DEEPL_API_KEY or "GANTI_DENGAN" in DEEPL_API_KEY:
            return "[DEEPL API KEY NOT CONFIGURED]"
        try:
            lang_map = {"Indonesian": "ID", "English": "EN-US", "Japanese": "JA", "Chinese": "ZH", "Korean": "KO"}
            url = "https://api-free.deepl.com/v2/translate"
            params = {"auth_key": DEEPL_API_KEY, "text": text, "target_lang": lang_map.get(target_lang, "ID")}
            response = requests.post(url, data=params, timeout=20)
            response.raise_for_status()
            return response.json()["translations"][0]["text"]
        except requests.exceptions.RequestException as e:
            return f"[Translation Network Error: {e}]"
        except Exception as e:
            return f"[Translation Error: {e}]"

    def load_folder(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Manga Folder")
        if dir_path:
            self.image_files = []
            self.file_list_widget.clear()
            supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
            for fname in sorted(os.listdir(dir_path)):
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
            self.load_image_at_index(row)

    def load_image_at_index(self, index):
        if 0 <= index < len(self.image_files):
            self.current_image_index = index
            file_path = self.image_files[index]
            try:
                self.current_image_pil = Image.open(file_path).convert('RGB')
                self.original_pixmap = QPixmap(file_path)
                self.typeset_areas.clear()
                self.redo_stack.clear()
                self.typeset_pixmap = self.original_pixmap.copy()
                self.image_label.clear_selection()
                self.update_display()
                self.update_undo_redo_buttons_state()
                self.text_output.clear()
            except Exception as e:
                QMessageBox.critical(self, "Error Loading Image", f"Could not load image: {file_path}\nError: {e}")
                self.original_pixmap = None
                self.typeset_pixmap = None
                self.update_display()
            self.update_nav_buttons()

    def load_next_image(self):
        if self.current_image_index < len(self.image_files) - 1:
            self.file_list_widget.setCurrentRow(self.current_image_index + 1)

    def load_prev_image(self):
        if self.current_image_index > 0:
            self.file_list_widget.setCurrentRow(self.current_image_index - 1)

    def update_nav_buttons(self):
        self.prev_button.setEnabled(self.current_image_index > 0)
        self.next_button.setEnabled(self.current_image_index < len(self.image_files) - 1)

    def update_display(self):
        if not self.typeset_pixmap:
            self.image_label.setPixmap(QPixmap())
            return
        self.zoom_label.setText(f" Zoom: {self.zoom_factor:.1f}x ")
        scaled_size = self.typeset_pixmap.size() * self.zoom_factor
        scaled_pixmap = self.typeset_pixmap.scaled(scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.adjustSize()

    def zoom_in(self):
        self.zoom_factor = min(self.zoom_factor + 0.2, 8.0)
        self.update_display()

    def zoom_out(self):
        self.zoom_factor = max(self.zoom_factor - 0.2, 0.1)
        self.update_display()

    def reset_view_to_original(self):
        if self.original_pixmap:
            self.typeset_areas.clear()
            self.redo_stack.clear()
            self.typeset_pixmap = self.original_pixmap.copy()
            self.zoom_factor = 1.0
            self.image_label.clear_selection()
            self.update_display()
            self.update_undo_redo_buttons_state()

    def choose_font(self):
        font, ok = QFontDialog.getFont(self.typeset_font, self)
        if ok:
            self.typeset_font = font

    def choose_color(self):
        color = QColorDialog.getColor(self.typeset_color, self)
        if color.isValid():
            self.typeset_color = color

    def unzoom_coords(self, selection_obj, as_point=False):
        pixmap = self.image_label.pixmap()
        if not pixmap or pixmap.isNull(): return None
        label_size = self.image_label.size()
        pixmap_size = pixmap.size()
        offset_x = max(0, (label_size.width() - pixmap_size.width()) // 2)
        offset_y = max(0, (label_size.height() - pixmap_size.height()) // 2)
        if as_point and isinstance(selection_obj, QPoint):
            unzoomed_x = int((selection_obj.x() - offset_x) / self.zoom_factor)
            unzoomed_y = int((selection_obj.y() - offset_y) / self.zoom_factor)
            return QPoint(unzoomed_x, unzoomed_y)
        if isinstance(selection_obj, QRect):
            unzoomed_x = int((selection_obj.x() - offset_x) / self.zoom_factor)
            unzoomed_y = int((selection_obj.y() - offset_y) / self.zoom_factor)
            unzoomed_width = int(selection_obj.width() / self.zoom_factor)
            unzoomed_height = int(selection_obj.height() / self.zoom_factor)
            return QRect(unzoomed_x, unzoomed_y, unzoomed_width, unzoomed_height)
        elif isinstance(selection_obj, list):
            unzoomed_points = []
            for p in selection_obj:
                unzoomed_x = int((p.x() - offset_x) / self.zoom_factor)
                unzoomed_y = int((p.y() - offset_y) / self.zoom_factor)
                unzoomed_points.append(QPoint(unzoomed_x, unzoomed_y))
            polygon = QPolygon(unzoomed_points)
            return polygon, polygon.boundingRect()
        return None

    def process_rect_area(self, selection_rect):
        if not self.current_image_pil: return
        unzoomed_rect = self.unzoom_coords(selection_rect)
        if not unzoomed_rect or unzoomed_rect.width() <= 0 or unzoomed_rect.height() <= 0: return
        cropped_img = self.current_image_pil.crop((
            unzoomed_rect.x(), unzoomed_rect.y(),
            unzoomed_rect.right(), unzoomed_rect.bottom()
        ))
        self.run_ocr_and_translate(cropped_img, unzoomed_rect)

    def process_polygon_area(self, scaled_points):
        if not self.current_image_pil: return
        result = self.unzoom_coords(scaled_points)
        if not result: return
        unzoomed_polygon, unzoomed_bbox = result
        if not unzoomed_bbox or unzoomed_bbox.width() <= 0 or unzoomed_bbox.height() <= 0: return
        cropped_pil_img = self.current_image_pil.crop((
            unzoomed_bbox.x(), unzoomed_bbox.y(),
            unzoomed_bbox.right(), unzoomed_bbox.bottom()
        ))
        cropped_cv_img = cv2.cvtColor(np.array(cropped_pil_img), cv2.COLOR_RGB2BGR)
        mask = np.zeros(cropped_cv_img.shape[:2], dtype=np.uint8)
        relative_poly_points = [QPoint(p.x() - unzoomed_bbox.x(), p.y() - unzoomed_bbox.y()) for p in unzoomed_polygon]
        cv_poly_points = np.array([[p.x(), p.y()] for p in relative_poly_points], dtype=np.int32)
        cv2.fillPoly(mask, [cv_poly_points], 255)
        white_bg = np.full(cropped_cv_img.shape, 255, dtype=np.uint8)
        fg = cv2.bitwise_and(cropped_cv_img, cropped_cv_img, mask=mask)
        bg = cv2.bitwise_and(white_bg, white_bg, mask=cv2.bitwise_not(mask))
        img_for_ocr = cv2.add(fg, bg)
        self.run_ocr_and_translate(img_for_ocr, unzoomed_bbox, polygon=unzoomed_polygon)

    def redraw_all_typeset_areas(self):
        if not self.original_pixmap: return
        self.typeset_pixmap = self.original_pixmap.copy()
        painter = QPainter(self.typeset_pixmap)
        for area in self.typeset_areas:
            self.draw_single_area(painter, area)
        painter.end()
        self.update_display()

    def draw_single_area(self, painter, area):
        original_crop_pil = self.current_image_pil.crop((area.rect.x(), area.rect.y(), area.rect.right(), area.rect.bottom()))
        avg_color = np.array(original_crop_pil).mean(axis=(0, 1))
        bg_color = QColor(int(avg_color[0]), int(avg_color[1]), int(avg_color[2]))
        if area.polygon:
            painter.save()
            path = QPainterPath()
            path.addPolygon(QPolygonF(area.polygon))
            painter.setClipPath(path)
            painter.fillRect(area.rect, bg_color)
            self.draw_text_with_options(painter, area.rect, area.text, area.get_font(), area.get_color())
            painter.restore()
        else:
            painter.fillRect(area.rect, bg_color)
            self.draw_text_with_options(painter, area.rect, area.text, area.get_font(), area.get_color())

    def draw_text_with_options(self, painter, rect, text, font, color):
        margin_x = int(rect.width() * 0.05)
        margin_y = int(rect.height() * 0.05)
        inner_rect = rect.adjusted(margin_x, margin_y, -margin_x, -margin_y)
        if inner_rect.width() <= 0 or inner_rect.height() <= 0:
            return
        current_font = QFont(font)
        font_size = inner_rect.height()
        current_font.setPointSizeF(font_size)
        min_font_size = 6.0
        while current_font.pointSizeF() > min_font_size:
            painter.setFont(current_font)
            bounding_rect = painter.boundingRect(inner_rect, Qt.TextWordWrap, text)
            if bounding_rect.height() <= inner_rect.height() and bounding_rect.width() <= inner_rect.width():
                break
            current_font.setPointSizeF(current_font.pointSizeF() - 1)
        outline_color = Qt.white if color.value() < 128 else Qt.black
        painter.setPen(outline_color)
        for dx, dy in [(-1,-1), (1,-1), (-1,1), (1,1), (-2,0), (2,0), (0,-2), (0,2)]:
            painter.drawText(inner_rect.translated(dx, dy), Qt.AlignCenter | Qt.TextWordWrap, text)
        painter.setPen(color)
        painter.drawText(inner_rect, Qt.AlignCenter | Qt.TextWordWrap, text)

    def selection_mode_changed(self, mode):
        self.image_label.clear_selection()
        if mode == "Rectangle":
            self.image_label.setCursor(Qt.CrossCursor)
        else:
            self.image_label.setCursor(Qt.PointingHandCursor)

    def update_pen_tool_buttons_visibility(self, visible):
        self.confirm_button.setVisible(visible)
        self.cancel_button.setVisible(visible)

    def confirm_pen_selection(self):
        points = self.image_label.get_polygon_points()
        if len(points) < 3:
            QMessageBox.warning(self, "Invalid Shape", "Please select at least 3 points.")
            return
        self.process_polygon_area(points)
        self.image_label.clear_selection()

    def cancel_pen_selection(self):
        self.image_label.clear_selection()

    def save_image(self):
        if not self.typeset_pixmap:
            QMessageBox.warning(self, "No Image", "There is no image to save.")
            return
        original_filename = os.path.basename(self.image_files[self.current_image_index])
        name, _ = os.path.splitext(original_filename)
        save_suggestion = os.path.join(os.path.dirname(self.image_files[self.current_image_index]), f"{name}_typeset.png")
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Typeset Image", save_suggestion, "PNG Image (*.png);;JPEG Image (*.jpg)")
        if filePath:
            if not self.typeset_pixmap.save(filePath):
                QMessageBox.critical(self, "Error", "Failed to save the image.")
            else:
                QMessageBox.information(self, "Success", f"Image saved to:\n{filePath}")

    def undo_last_action(self):
        if self.typeset_areas:
            undone_area = self.typeset_areas.pop()
            self.redo_stack.append(undone_area)
            self.redraw_all_typeset_areas()
            self.update_undo_redo_buttons_state()
            self.image_label.clear_selection()

    def redo_last_action(self):
        if self.redo_stack:
            redone_area = self.redo_stack.pop()
            self.typeset_areas.append(redone_area)
            self.redraw_all_typeset_areas()
            self.update_undo_redo_buttons_state()
            self.image_label.clear_selection()

    def update_undo_redo_buttons_state(self):
        self.undo_button.setEnabled(len(self.typeset_areas) > 0)
        self.redo_button.setEnabled(len(self.redo_stack) > 0)

    def save_project(self):
        if not self.image_files:
            QMessageBox.warning(self, "Nothing to Save", "Please load a folder first.")
            return
        if not self.current_project_path:
            filePath, _ = QFileDialog.getSaveFileName(self, "Save Project", "", "Manga Project (*.manga_proj)")
            if not filePath:
                return
            self.current_project_path = filePath
        project_data = {
            'image_files': self.image_files,
            'current_index': self.current_image_index,
            'typeset_areas': self.typeset_areas,
            'text_output': self.text_output.toPlainText(),
            'font': TypesetArea.font_to_dict(self.typeset_font),
            'color': self.typeset_color.name(),
        }
        try:
            with open(self.current_project_path, 'wb') as f:
                pickle.dump(project_data, f)
            self.setWindowTitle(f"Manga OCR & Typeset Tool v3.9 - {os.path.basename(self.current_project_path)}")
            QMessageBox.information(self, "Success", f"Project saved to\n{self.current_project_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save project: {e}")

    def load_project(self):
        filePath, _ = QFileDialog.getOpenFileName(self, "Load Project", "", "Manga Project (*.manga_proj)")
        if not filePath:
            return
        try:
            with open(filePath, 'rb') as f:
                project_data = pickle.load(f)
            self.image_files = project_data.get('image_files', [])
            self.typeset_areas = project_data.get('typeset_areas', [])
            font_info = project_data.get('font', {'family': 'Arial', 'pointSize': 12, 'weight': QFont.Bold, 'italic': False})
            self.typeset_font = QFont()
            self.typeset_font.setFamily(font_info['family'])
            self.typeset_font.setPointSize(font_info['pointSize'])
            self.typeset_font.setWeight(font_info['weight'])
            self.typeset_font.setItalic(font_info['italic'])
            self.typeset_color = QColor(project_data.get('color', '#000000'))
            self.file_list_widget.clear()
            for fpath in self.image_files:
                self.file_list_widget.addItem(QListWidgetItem(os.path.basename(fpath)))
            index_to_load = project_data.get('current_index', -1)
            if index_to_load != -1:
                self.load_image_at_index(index_to_load)
                self.typeset_areas = project_data.get('typeset_areas', [])
                self.text_output.setPlainText(project_data.get('text_output', ''))
                self.redraw_all_typeset_areas()
                self.update_undo_redo_buttons_state()
                self.file_list_widget.setCurrentRow(index_to_load)
            self.current_project_path = filePath
            self.setWindowTitle(f"Manga OCR & Typeset Tool v3.9 - {os.path.basename(self.current_project_path)}")
            QMessageBox.information(self, "Success", "Project loaded successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load project: {e}")

    def show_style_transfer_info(self):
        QMessageBox.information(self, "Conceptual Feature: Style Transfer",
            "This is a placeholder for a future advanced feature.")

    def start_inline_edit(self, area):
        if self.inline_editor:
            self.finish_inline_edit()
        self.editing_area = area
        self.inline_editor = QLineEdit(self.image_label)
        zoomed_rect = self.zoom_coords(area.rect)
        self.inline_editor.setGeometry(zoomed_rect)
        self.inline_editor.setText(area.text)
        self.inline_editor.setFont(area.get_font())
        if self.current_theme == 'dark':
            self.inline_editor.setStyleSheet("background-color: rgba(255, 255, 255, 0.9); color: black; border: 2px solid #007acc;")
        else:
            self.inline_editor.setStyleSheet("background-color: rgba(50, 50, 50, 0.9); color: white; border: 2px solid #0078d7;")
        self.inline_editor.editingFinished.connect(self.finish_inline_edit)
        self.inline_editor.show()
        self.inline_editor.setFocus()

    def finish_inline_edit(self):
        if not self.inline_editor: return
        self.editing_area.text = self.inline_editor.text()
        self.redo_stack.clear() # Hapus redo stack saat ada editan baru
        self.redraw_all_typeset_areas()
        self.update_undo_redo_buttons_state()
        self.inline_editor.deleteLater()
        self.inline_editor = None
        self.editing_area = None

    def zoom_coords(self, unzoomed_rect):
        pixmap = self.image_label.pixmap()
        if not pixmap or pixmap.isNull(): return QRect()
        label_size = self.image_label.size()
        pixmap_size = pixmap.size()
        offset_x = max(0, (label_size.width() - pixmap_size.width()) // 2)
        offset_y = max(0, (label_size.height() - pixmap_size.height()) // 2)
        zoomed_x = int(unzoomed_rect.x() * self.zoom_factor + offset_x)
        zoomed_y = int(unzoomed_rect.y() * self.zoom_factor + offset_y)
        zoomed_w = int(unzoomed_rect.width() * self.zoom_factor)
        zoomed_h = int(unzoomed_rect.height() * self.zoom_factor)
        return QRect(zoomed_x, zoomed_y, zoomed_w, zoomed_h)

    def toggle_theme(self):
        if self.current_theme == 'dark':
            self.current_theme = 'light'
            self.setStyleSheet(self.LIGHT_THEME_STYLESHEET)
        else:
            self.current_theme = 'dark'
            self.setStyleSheet(self.DARK_THEME_STYLESHEET)
    
    def show_about_dialog(self):
        QMessageBox.about(self, "About Manga OCR & Typeset Tool",
            "<b>Manga OCR & Typeset Tool v3.9</b><br><br>"
            "This tool was created to streamline the process of translating manga.<br><br>"
            "Powered by Python, PyQt5, EasyOCR, Tesseract, and Gemini API.<br>"
            "Enhanced with new features by Gemini.<br><br>"
            "Copyright © 2024")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    try:
        tesseract_version = pytesseract.get_tesseract_version()
        print(f"Tesseract version {tesseract_version} found.")
    except Exception:
        print("Tesseract not found. Tesseract functionality will be disabled.")
    
    if not DEEPL_API_KEY or "GANTI_DENGAN" in DEEPL_API_KEY:
        QMessageBox.warning(None, "DeepL API Key Missing", "Please provide your valid DeepL API key in the script to enable translation.")
    
    if not GEMINI_API_KEY or "GANTI_DENGAN" in GEMINI_API_KEY:
        QMessageBox.warning(None, "Gemini API Key Missing", "Please provide your valid Gemini API key in the script to enable AI text correction.")
        
    window = MangaOCRApp()
    window.show()
    sys.exit(app.exec_())
