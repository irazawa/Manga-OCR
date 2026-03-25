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

from src.ui.main_window import MangaOCRApp
from src.core.fonts import set_global_font_manager
from src.core.config import *

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
