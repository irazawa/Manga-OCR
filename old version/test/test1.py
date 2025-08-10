import os
import sys
import numpy as np
import cv2
import pytesseract
import requests
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
    QTextEdit, QScrollArea, QComboBox,
    QMessageBox, QProgressBar, QShortcut
)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QKeySequence, QPolygon
from PyQt5.QtCore import Qt, QRect, QPoint

# --- KONFIGURASI ---
# Ganti dengan API Key DeepL Anda yang valid
DEEPL_API_KEY = "2107028d-26f0-4604-a625-80d62f226311:fx" 
# Sesuaikan path ke Tesseract jika diinstal di lokasi lain
# Contoh untuk Windows:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Contoh untuk macOS/Linux (biasanya tidak perlu jika ada di PATH):
# pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract' 

# Pastikan TESSDATA_PREFIX juga diatur jika perlu
# os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

class SelectableImageLabel(QLabel):
    """Label gambar yang mendukung seleksi area dengan mouse (Rectangle & Pen Tool)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.setMouseTracking(True)
        
        # Atribut untuk seleksi
        self.selection_start = None
        self.selection_end = None
        self.selection_rect = QRect()
        self.dragging = False
        self.polygon_points = []
        self.current_mouse_pos = None

        self.setCursor(Qt.CrossCursor) # Kursor default untuk mode Rectangle

    def get_selection_mode(self):
        """Mendapatkan mode seleksi saat ini dari main window."""
        return self.main_window.selection_mode_combo.currentText()

    def get_polygon_points(self):
        """Mengembalikan titik-titik poligon yang telah digambar."""
        return self.polygon_points

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
                if self.selection_rect.width() > 10 and self.selection_rect.height() > 10:
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
            # Gambar titik-titik vertex
            for point in self.polygon_points:
                painter.drawEllipse(point, 3, 3) 
            
            # Gambar garis antar titik
            if len(self.polygon_points) > 1:
                painter.drawPolyline(QPolygon(self.polygon_points))
            
            # Gambar garis preview ke kursor
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
    """Jendela utama aplikasi OCR dan Typesetting."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Manga OCR & Typeset Tool")
        self.setGeometry(100, 100, 1400, 900)
        self.current_image = None
        self.image_path = ""
        self.original_pixmap = None
        self.typeset_pixmap = None
        self.zoom_factor = 1.0
        self.history_stack = []

        self.init_ui()
        self.setup_styles()
        self.setup_shortcuts()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Panel Kiri (Gambar & Kontrol)
        left_panel = QVBoxLayout()
        self.image_label = SelectableImageLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        
        image_scroll = QScrollArea()
        image_scroll.setWidget(self.image_label)
        image_scroll.setWidgetResizable(True)
        left_panel.addWidget(image_scroll)

        # Kontrol utama di bawah gambar
        main_control_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        main_control_layout.addWidget(self.load_button)
        
        self.undo_button = QPushButton("Undo (Ctrl+Z)")
        self.undo_button.clicked.connect(self.undo_last_action)
        self.undo_button.setEnabled(False)
        main_control_layout.addWidget(self.undo_button)

        self.reset_button = QPushButton("Reset View")
        self.reset_button.clicked.connect(self.reset_view_to_original)
        main_control_layout.addWidget(self.reset_button)
        
        self.save_button = QPushButton("Save Image")
        self.save_button.clicked.connect(self.save_image)
        main_control_layout.addWidget(self.save_button)
        left_panel.addLayout(main_control_layout)

        # Kontrol Seleksi (BARU)
        selection_tools_layout = QHBoxLayout()
        selection_tools_layout.addWidget(QLabel("Selection Mode:"))
        self.selection_mode_combo = QComboBox()
        self.selection_mode_combo.addItems(["Rectangle", "Pen Tool"])
        self.selection_mode_combo.currentTextChanged.connect(self.selection_mode_changed)
        selection_tools_layout.addWidget(self.selection_mode_combo)
        
        self.confirm_button = QPushButton("Confirm Selection")
        self.confirm_button.clicked.connect(self.confirm_pen_selection)
        self.confirm_button.setVisible(False)
        selection_tools_layout.addWidget(self.confirm_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_pen_selection)
        self.cancel_button.setVisible(False)
        selection_tools_layout.addWidget(self.cancel_button)
        selection_tools_layout.addStretch()
        left_panel.addLayout(selection_tools_layout)

        # Kontrol Zoom
        zoom_control_layout = QHBoxLayout()
        zoom_control_layout.addStretch()
        self.zoom_out_button = QPushButton("Zoom Out (-)")
        self.zoom_out_button.clicked.connect(self.zoom_out)
        zoom_control_layout.addWidget(self.zoom_out_button)
        self.zoom_label = QLabel(f" Zoom: {self.zoom_factor:.1f}x ")
        zoom_control_layout.addWidget(self.zoom_label)
        self.zoom_in_button = QPushButton("Zoom In (+)")
        self.zoom_in_button.clicked.connect(self.zoom_in)
        zoom_control_layout.addWidget(self.zoom_in_button)
        zoom_control_layout.addStretch()
        left_panel.addLayout(zoom_control_layout)

        # Panel Kanan (Opsi & Hasil)
        right_panel = QVBoxLayout()
        self.setup_right_panel(right_panel)

        main_layout.addLayout(left_panel, 70)
        main_layout.addLayout(right_panel, 30)

    def setup_right_panel(self, right_panel):
        options_layout = QVBoxLayout()
        controls = {
            "Mode:": ["Translate Only", "Translate & Typeset"],
            "OCR Language:": ["Japanese", "English", "Chinese", "Korean"],
            "Text Orientation:": ["Vertical", "Horizontal"],
            "Translate to:": ["Indonesian", "English", "Japanese", "Chinese", "Korean"]
        }
        self.mode_combo = self._create_combo_box(options_layout, "Mode:", controls["Mode:"])
        self.lang_combo = self._create_combo_box(options_layout, "OCR Language:", controls["OCR Language:"], "Japanese")
        self.orientation_combo = self._create_combo_box(options_layout, "Text Orientation:", controls["Text Orientation:"], "Vertical")
        self.translate_combo = self._create_combo_box(options_layout, "Translate to:", controls["Translate to:"], "Indonesian")
        right_panel.addLayout(options_layout)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        right_panel.addWidget(self.progress_bar)
        right_panel.addWidget(QLabel("Extracted & Translated Text:"))
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(False)
        right_panel.addWidget(self.text_output)

    def _create_combo_box(self, layout, label_text, items, default=None):
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel(label_text))
        combo = QComboBox()
        combo.addItems(items)
        if default: combo.setCurrentText(default)
        h_layout.addWidget(combo)
        layout.addLayout(h_layout)
        return combo

    def setup_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #2e2e2e; color: #f0f0f0; }
            QLabel { padding: 5px; font-size: 10pt; }
            QPushButton {
                background-color: #007acc; color: white;
                padding: 8px; border: none; border-radius: 4px; margin: 2px;
                font-size: 10pt;
            }
            QPushButton:hover { background-color: #005f9e; }
            QPushButton:disabled { background-color: #555555; }
            QTextEdit, QComboBox {
                background-color: #3c3c3c; color: #f0f0f0;
                border: 1px solid #555; padding: 5px; border-radius: 4px;
                font-size: 10pt;
            }
            QScrollArea { border: 1px solid #444; }
            QProgressBar {
                border: 1px solid #555; border-radius: 4px; text-align: center; color: #f0f0f0;
            }
            QProgressBar::chunk { background-color: #007acc; }
        """)

    def setup_shortcuts(self):
        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.activated.connect(self.undo_last_action)

    def selection_mode_changed(self, mode):
        self.image_label.clear_selection()
        if mode == "Rectangle":
            self.image_label.setCursor(Qt.CrossCursor)
        else: # Pen Tool
            self.image_label.setCursor(Qt.PointingHandCursor)
    
    def update_pen_tool_buttons_visibility(self, visible):
        self.confirm_button.setVisible(visible)
        self.cancel_button.setVisible(visible)

    def confirm_pen_selection(self):
        points = self.image_label.get_polygon_points()
        if len(points) < 3:
            QMessageBox.warning(self, "Invalid Shape", "Please select at least 3 points to form a polygon.")
            return
        self.process_polygon_area(points)
        self.image_label.clear_selection()

    def cancel_pen_selection(self):
        self.image_label.clear_selection()
    
    def unzoom_coords(self, selection_obj):
        pixmap = self.image_label.pixmap()
        if not pixmap or pixmap.isNull(): return None
        
        label_size = self.image_label.size()
        pixmap_size = pixmap.size()
        offset_x = max(0, (label_size.width() - pixmap_size.width()) // 2)
        offset_y = max(0, (label_size.height() - pixmap_size.height()) // 2)

        if isinstance(selection_obj, QRect):
            pixmap_relative_x = selection_obj.x() - offset_x
            pixmap_relative_y = selection_obj.y() - offset_y
            unzoomed_x = int(pixmap_relative_x / self.zoom_factor)
            unzoomed_y = int(pixmap_relative_y / self.zoom_factor)
            unzoomed_width = int(selection_obj.width() / self.zoom_factor)
            unzoomed_height = int(selection_obj.height() / self.zoom_factor)
            return QRect(unzoomed_x, unzoomed_y, unzoomed_width, unzoomed_height)
        
        elif isinstance(selection_obj, list): # List of QPoints
            unzoomed_points = []
            for p in selection_obj:
                pixmap_relative_x = p.x() - offset_x
                pixmap_relative_y = p.y() - offset_y
                unzoomed_x = int(pixmap_relative_x / self.zoom_factor)
                unzoomed_y = int(pixmap_relative_y / self.zoom_factor)
                unzoomed_points.append(QPoint(unzoomed_x, unzoomed_y))
            
            polygon = QPolygon(unzoomed_points)
            return polygon, polygon.boundingRect()
        return None

    def process_rect_area(self, selection_rect):
        if not self.current_image: return
        unzoomed_rect = self.unzoom_coords(selection_rect)
        if not unzoomed_rect or unzoomed_rect.width() <= 0 or unzoomed_rect.height() <= 0: return

        cropped_img = self.current_image.crop((
            unzoomed_rect.x(), unzoomed_rect.y(),
            unzoomed_rect.right(), unzoomed_rect.bottom()
        ))
        self.run_ocr_and_translate(cropped_img, unzoomed_rect)

    def process_polygon_area(self, scaled_points):
        if not self.current_image: return
        result = self.unzoom_coords(scaled_points)
        if not result: return
        unzoomed_polygon, unzoomed_bbox = result
        
        if not unzoomed_bbox or unzoomed_bbox.width() <= 0 or unzoomed_bbox.height() <= 0: return

        cropped_pil_img = self.current_image.crop((
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
        
        self.run_ocr_and_translate(img_for_ocr, unzoomed_bbox)

    def run_ocr_and_translate(self, image_to_process, typesetting_rect):
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(10)
            
            if isinstance(image_to_process, Image.Image):
                cv_image = cv2.cvtColor(np.array(image_to_process), cv2.COLOR_RGB2BGR)
            else:
                cv_image = image_to_process

            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            self.progress_bar.setValue(30)
            
            if np.mean(gray) > 240: # Cek jika background sudah putih (dari Pen Tool)
                ocr_image = gray
            else:
                _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                if cv2.countNonZero(thresholded) < thresholded.size / 2:
                    thresholded = cv2.bitwise_not(thresholded)
                kernel = np.ones((1,1), np.uint8)
                ocr_image = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=1)
            
            self.progress_bar.setValue(50)

            lang_map = {"Japanese": "jpn_vert", "English": "eng", "Chinese": "chi_sim_vert", "Korean": "kor_vert"}
            tess_lang = lang_map.get(self.lang_combo.currentText(), "eng")
            orientation = self.orientation_combo.currentText()
            if orientation == "Horizontal":
                tess_lang = tess_lang.replace('_vert', '')
            
            psm = '5' if orientation == "Vertical" else '6'
            custom_config = f'--oem 1 --psm {psm} -l {tess_lang}'
            text = pytesseract.image_to_string(ocr_image, config=custom_config).strip()
            self.progress_bar.setValue(70)

            target_lang = self.translate_combo.currentText()
            translated_text = self.translate_text(text, target_lang)
            self.progress_bar.setValue(90)
            
            current_text = self.text_output.toPlainText()
            separator = "\n" + "="*40 + "\n"
            new_entry = f"Original ({self.lang_combo.currentText()}):\n{text}\n\nTranslation ({target_lang}):\n{translated_text}"
            self.text_output.setPlainText(f"{current_text}{separator}{new_entry}" if current_text else new_entry)
            self.text_output.verticalScrollBar().setValue(self.text_output.verticalScrollBar().maximum())
            
            if self.mode_combo.currentText() == "Translate & Typeset":
                # Ambil crop asli dari gambar untuk mendapatkan warna background yang benar
                original_crop_pil = self.current_image.crop((
                    typesetting_rect.x(), typesetting_rect.y(),
                    typesetting_rect.right(), typesetting_rect.bottom()
                ))
                original_crop_cv = cv2.cvtColor(np.array(original_crop_pil), cv2.COLOR_RGB2BGR)
                self.typeset_text_on_image(typesetting_rect, translated_text, original_crop_cv)
                self.update_display()
            
            self.progress_bar.setValue(100)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred:\n{str(e)}")
            self.image_label.clear_selection()
        finally:
            self.progress_bar.setVisible(False)
    
    def load_image(self, *args):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Manga Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*)", options=options)
        if file_path:
            self.image_path = file_path
            try:
                self.current_image = Image.open(file_path).convert('RGB')
                self.original_pixmap = QPixmap(file_path)
                self.reset_view_to_original()
                self.text_output.clear()
            except Exception as e:
                QMessageBox.critical(self, "Error Loading Image", f"Could not load the image file.\nError: {e}")


    def reset_view_to_original(self, *args):
        if self.original_pixmap:
            self.history_stack.clear()
            self.history_stack.append(self.original_pixmap.copy())
            self.typeset_pixmap = self.history_stack[-1].copy()
            self.zoom_factor = 1.0
            self.image_label.clear_selection()
            self.update_display()
            self.update_undo_button_state()

    def update_display(self):
        if not self.typeset_pixmap: return
        self.zoom_label.setText(f" Zoom: {self.zoom_factor:.1f}x ")
        scaled_size = self.typeset_pixmap.size() * self.zoom_factor
        scaled_pixmap = self.typeset_pixmap.scaled(scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.adjustSize()

    def zoom_in(self):
        self.zoom_factor = min(self.zoom_factor + 0.2, 5.0)
        self.update_display()

    def zoom_out(self):
        self.zoom_factor = max(self.zoom_factor - 0.2, 0.2)
        self.update_display()

    def translate_text(self, text, target_lang):
        if not text.strip(): return "[No text to translate]"
        if not DEEPL_API_KEY or "MASUKKAN_API_KEY" in DEEPL_API_KEY:
            return "[DEEPL API KEY NOT CONFIGURED]"
        try:
            lang_map = {"Indonesian": "ID", "English": "EN-US", "Japanese": "JA", "Chinese": "ZH", "Korean": "KO"}
            url = "https://api-free.deepl.com/v2/translate"
            params = {"auth_key": DEEPL_API_KEY, "text": text, "target_lang": lang_map.get(target_lang, "ID")}
            response = requests.post(url, data=params, timeout=15)
            response.raise_for_status()
            return response.json()["translations"][0]["text"]
        except requests.exceptions.RequestException as e:
            print(f"Translation network error: {e}")
            return f"[Translation Network Error]"
        except Exception as e:
            print(f"Translation error: {e}")
            # Cek jika error terkait otentikasi
            if "403" in str(e):
                return "[Translation Error: Invalid API Key]"
            return f"[Translation Error: {str(e)}]"

    def typeset_text_on_image(self, rect, text, cropped_cv_image):
        new_pixmap = self.typeset_pixmap.copy()
        painter = QPainter(new_pixmap)
        
        # Tentukan warna background dari area crop asli
        gray_crop = cv2.cvtColor(cropped_cv_image, cv2.COLOR_BGR2GRAY)
        avg_brightness = int(np.mean(gray_crop))
        bg_color = QColor(avg_brightness, avg_brightness, avg_brightness)
        
        # Tentukan warna teks berdasarkan kontras dengan background
        text_color = QColor(Qt.white) if avg_brightness < 128 else QColor(Qt.black)
        
        # Isi area seleksi dengan warna background yang di-rata-ratakan
        painter.fillRect(rect, bg_color)

        font = QFont("Arial", 10, QFont.Bold)
        margin_x = int(rect.width() * 0.05)
        margin_y = int(rect.height() * 0.05)
        inner_rect = rect.adjusted(margin_x, margin_y, -margin_x, -margin_y)
        
        if inner_rect.width() <= 0 or inner_rect.height() <= 0:
            painter.end()
            return
        
        # Logika untuk menyesuaikan ukuran font secara dinamis
        font_size = inner_rect.height()
        font.setPointSizeF(font_size)
        
        min_font_size = 6.0
        while font.pointSizeF() > min_font_size:
            painter.setFont(font)
            bounding_rect = painter.boundingRect(inner_rect, Qt.TextWordWrap, text)
            if bounding_rect.height() <= inner_rect.height() and bounding_rect.width() <= inner_rect.width():
                break
            font.setPointSizeF(font.pointSizeF() - 1)
        
        painter.setPen(text_color)
        painter.drawText(inner_rect, Qt.AlignCenter | Qt.TextWordWrap, text)
        painter.end()
        
        self.history_stack.append(new_pixmap)
        self.typeset_pixmap = self.history_stack[-1].copy()
        self.update_undo_button_state()

    def save_image(self):
        if not self.typeset_pixmap:
            QMessageBox.warning(self, "No Image", "There is no image to save.")
            return
        options = QFileDialog.Options()
        # Ambil nama file asli untuk sugesti nama file simpan
        original_filename = os.path.basename(self.image_path)
        name, ext = os.path.splitext(original_filename)
        save_suggestion = f"{name}_typeset.png"

        filePath, _ = QFileDialog.getSaveFileName(self, "Save Typeset Image", save_suggestion, "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg)", options=options)
        if filePath:
            if not self.typeset_pixmap.save(filePath):
                QMessageBox.critical(self, "Error", "Failed to save the image.")
            else:
                QMessageBox.information(self, "Success", f"Image saved successfully to:\n{filePath}")

    def undo_last_action(self):
        if len(self.history_stack) > 1:
            self.history_stack.pop()
            self.typeset_pixmap = self.history_stack[-1].copy()
            self.update_display()
            self.update_undo_button_state()
            self.image_label.clear_selection()

    def update_undo_button_state(self):
        self.undo_button.setEnabled(len(self.history_stack) > 1)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Pengecekan Tesseract
    try:
        tesseract_version = pytesseract.get_tesseract_version()
        print(f"Tesseract version {tesseract_version} found.")
    except Exception as e:
        QMessageBox.critical(None, "Tesseract Error", f"Tesseract not found or configured incorrectly.\n"
                                                      f"Please check the 'pytesseract.pytesseract.tesseract_cmd' path in the script.\n\nError: {e}")
        sys.exit(1)
    
    # Pengecekan API Key
    if not DEEPL_API_KEY or "MASUKKAN_API_KEY" in DEEPL_API_KEY:
         QMessageBox.warning(None, "API Key Missing", "Please replace 'MASUKKAN_API_KEY_DEEPL_ANDA_DI_SINI' in the script with your valid DeepL API key.")

    window = MangaOCRApp()
    window.show()
    sys.exit(app.exec_())
