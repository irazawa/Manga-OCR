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
