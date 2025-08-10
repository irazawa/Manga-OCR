import os
import sys
import numpy as np
import cv2
import pytesseract
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, 
    QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, 
    QTextEdit, QScrollArea, QComboBox, QSlider,
    QMessageBox, QProgressBar
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal

# Konfigurasi Tesseract OCR
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'
except Exception as e:
    print(f"Error configuring Tesseract: {e}")

class OCRThread(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, image, lang, threshold):
        super().__init__()
        self.image = image
        self.lang = lang
        self.threshold = threshold

    def run(self):
        try:
            self.progress.emit(10)
            
            # Konversi ke OpenCV format
            cv_image = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
            self.progress.emit(20)
            
            # Konversi ke grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            self.progress.emit(30)
            
            # Thresholding
            _, thresholded = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
            self.progress.emit(50)
            
            # Denoising
            denoised = cv2.fastNlMeansDenoising(thresholded, None, 10, 7, 21)
            self.progress.emit(70)
            
            # Ekstrak teks
            custom_config = f'--oem 3 --psm 6 -l {self.lang}'
            text = pytesseract.image_to_string(denoised, config=custom_config)
            self.progress.emit(90)
            
            self.finished.emit(text)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.progress.emit(100)

class MangaOCRApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Manga OCR Reader")
        self.setGeometry(100, 100, 1000, 700)
        self.current_image = None
        self.image_path = ""
        self.ocr_thread = None
        
        self.init_ui()
        self.setup_styles()
        
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Layout utama
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        
        # Panel kiri (gambar)
        left_panel = QVBoxLayout()
        
        # Area gambar dengan scroll
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(500, 500)
        
        image_scroll = QScrollArea()
        image_scroll.setWidget(self.image_label)
        image_scroll.setWidgetResizable(True)
        left_panel.addWidget(image_scroll)
        
        # Tombol load gambar
        self.load_button = QPushButton("Load Manga Image")
        self.load_button.clicked.connect(self.load_image)
        left_panel.addWidget(self.load_button)
        
        # Panel kanan (kontrol dan hasil)
        right_panel = QVBoxLayout()
        
        # Kontrol bahasa
        lang_layout = QHBoxLayout()
        lang_layout.addWidget(QLabel("OCR Language:"))
        
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["Japanese", "English", "Chinese", "Korean"])
        self.lang_combo.setCurrentText("Japanese")
        lang_layout.addWidget(self.lang_combo)
        
        right_panel.addLayout(lang_layout)
        
        # Kontrol threshold
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Threshold:"))
        
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(150)
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)
        threshold_layout.addWidget(self.threshold_slider)
        
        self.threshold_label = QLabel("150")
        threshold_layout.addWidget(self.threshold_label)
        
        right_panel.addLayout(threshold_layout)
        
        # Tombol proses
        self.process_button = QPushButton("Extract Text")
        self.process_button.clicked.connect(self.process_image)
        self.process_button.setEnabled(False)
        right_panel.addWidget(self.process_button)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        right_panel.addWidget(self.progress_bar)
        
        # Area hasil teks
        right_panel.addWidget(QLabel("Extracted Text:"))
        
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        right_panel.addWidget(self.text_output)
        
        # Gabungkan panel
        main_layout.addLayout(left_panel, 60)
        main_layout.addLayout(right_panel, 40)
    
    def setup_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QLabel {
                padding: 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
            QTextEdit, QComboBox, QSlider {
                background-color: white;
                border: 1px solid #ddd;
                padding: 5px;
                border-radius: 4px;
            }
            QScrollArea {
                border: 1px solid #ddd;
            }
            QProgressBar {
                border: 1px solid #ddd;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
    
    def update_threshold_label(self):
        self.threshold_label.setText(str(self.threshold_slider.value()))
    
    def load_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Manga Image", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*)", 
            options=options
        )
        
        if file_path:
            self.image_path = file_path
            self.current_image = Image.open(file_path)
            
            # Tampilkan gambar
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
            
            self.process_button.setEnabled(True)
            self.text_output.clear()
    
    def process_image(self):
        if not self.current_image:
            return
        
        # Dapatkan parameter
        lang_map = {
            "Japanese": "jpn",
            "English": "eng",
            "Chinese": "chi_sim",
            "Korean": "kor"
        }
        selected_lang = lang_map[self.lang_combo.currentText()]
        threshold = self.threshold_slider.value()
        
        # Nonaktifkan UI selama proses
        self.set_ui_enabled(False)
        self.progress_bar.setVisible(True)
        
        # Buat thread OCR
        self.ocr_thread = OCRThread(self.current_image, selected_lang, threshold)
        self.ocr_thread.finished.connect(self.on_ocr_finished)
        self.ocr_thread.error.connect(self.on_ocr_error)
        self.ocr_thread.progress.connect(self.progress_bar.setValue)
        self.ocr_thread.start()
    
    def set_ui_enabled(self, enabled):
        self.load_button.setEnabled(enabled)
        self.process_button.setEnabled(enabled)
        self.lang_combo.setEnabled(enabled)
        self.threshold_slider.setEnabled(enabled)
    
    def on_ocr_finished(self, text):
        self.text_output.setPlainText(text)
        self.set_ui_enabled(True)
        self.progress_bar.setVisible(False)
        
        # Tampilkan gambar yang diproses
        self.show_processed_image()
    
    def on_ocr_error(self, error_msg):
        QMessageBox.critical(self, "OCR Error", f"Error during OCR processing:\n{error_msg}")
        self.set_ui_enabled(True)
        self.progress_bar.setVisible(False)
    
    def show_processed_image(self):
        if not self.current_image:
            return
        
        try:
            # Proses image untuk ditampilkan
            cv_image = cv2.cvtColor(np.array(self.current_image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            _, thresholded = cv2.threshold(gray, self.threshold_slider.value(), 255, cv2.THRESH_BINARY)
            
            # Konversi ke QPixmap
            height, width = thresholded.shape
            bytes_per_line = width
            q_img = QImage(thresholded.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_img)
            
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
        except Exception as e:
            print(f"Error showing processed image: {e}")
    
    def closeEvent(self, event):
        if self.ocr_thread and self.ocr_thread.isRunning():
            self.ocr_thread.terminate()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Cek Tesseract
    try:
        pytesseract.get_tesseract_version()
    except Exception as e:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText("Tesseract OCR not found!")
        msg.setInformativeText(str(e))
        msg.exec_()
        sys.exit(1)
    
    window = MangaOCRApp()
    window.show()
    sys.exit(app.exec_())