import os
import re
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QListWidget, QMessageBox
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QUrl

# PDF viewer pakai WebEngine
from PyQt5.QtWebEngineWidgets import QWebEngineView


def extract_number(filename):
    """Ambil angka dari awal nama file."""
    match = re.match(r"(\d+)", filename)
    if match:
        return int(match.group(1))
    return float('inf')  # Kalau nggak ada angka di depannya, taruh di akhir


def custom_sort(files):
    """Sorting custom: cek angka depan."""
    return sorted(files, key=lambda f: extract_number(f))


def konversi_gambar_ke_pdf(folder_path, output_pdf_name, progress_callback=None):
    """Konversi semua gambar ke satu PDF."""
    files = [f for f in os.listdir(folder_path)
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    files = custom_sort(files)

    if not files:
        raise FileNotFoundError("Tidak ada gambar ditemukan.")

    output_pdf_path = os.path.join(folder_path, output_pdf_name)
    c = canvas.Canvas(output_pdf_path, pagesize=letter)
    width, height = letter

    total = len(files)
    for i, image_file in enumerate(files, 1):
        image_path = os.path.join(folder_path, image_file)
        img = Image.open(image_path)
        img_width, img_height = img.size
        aspect = img_height / float(img_width)
        new_width = width
        new_height = new_width * aspect
        if new_height > height:
            new_height = height
            new_width = new_height / aspect
        c.drawImage(image_path, 0, (height - new_height) / 2,
                    width=new_width, height=new_height)
        c.showPage()
        if progress_callback:
            progress_callback(int(i / total * 100))

    c.save()
    return output_pdf_path


class PDFConverterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image to PDF Converter - Modern UI")
        self.setGeometry(100, 100, 900, 600)
        self.folder_path = ""
        self.pdf_path = ""

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        title = QLabel("🖼️ Konversi Gambar ke PDF")
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        btn_layout = QHBoxLayout()
        self.btn_select = QPushButton("Pilih Folder")
        self.btn_select.clicked.connect(self.select_folder)
        self.btn_select.setStyleSheet(self.button_style())

        self.btn_convert = QPushButton("Konversi ke PDF")
        self.btn_convert.clicked.connect(self.convert_pdf)
        self.btn_convert.setStyleSheet(self.button_style())

        btn_layout.addWidget(self.btn_select)
        btn_layout.addWidget(self.btn_convert)
        main_layout.addLayout(btn_layout)

        self.list_files = QListWidget()
        main_layout.addWidget(self.list_files)

        self.progress = QProgressBar()
        main_layout.addWidget(self.progress)

        self.pdf_view = QWebEngineView()
        main_layout.addWidget(self.pdf_view, stretch=1)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def button_style(self):
        return """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 8px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Pilih Folder Gambar")
        if folder:
            self.folder_path = folder
            files = [f for f in os.listdir(folder)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            files = custom_sort(files)
            self.list_files.clear()
            self.list_files.addItems(files)

    def convert_pdf(self):
        if not self.folder_path:
            QMessageBox.warning(self, "Error", "Pilih folder gambar terlebih dahulu!")
            return
        try:
            pdf_name = "output.pdf"
            self.pdf_path = konversi_gambar_ke_pdf(
                self.folder_path, pdf_name,
                progress_callback=self.progress.setValue
            )
            self.pdf_view.load(QUrl.fromLocalFile(self.pdf_path))
            QMessageBox.information(self, "Selesai", f"PDF berhasil dibuat: {self.pdf_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = PDFConverterApp()
    window.show()
    sys.exit(app.exec_())
