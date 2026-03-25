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

MOUSE_BUTTON_NAME_MAP = {
    Qt.LeftButton: 'Left',
    Qt.RightButton: 'Right',
    Qt.MiddleButton: 'Middle',
    Qt.BackButton: 'Back',
    Qt.ForwardButton: 'Forward',
    Qt.TaskButton: 'Task',
}

def mouse_button_to_name(button: Qt.MouseButton) -> str:
    """Return a stable token for a mouse/stylus button so it can round-trip in settings."""
    if button in (Qt.BackButton, Qt.XButton1):
        return 'Back'
    if button in (Qt.ForwardButton, Qt.XButton2):
        return 'Forward'
    for qt_button, name in MOUSE_BUTTON_NAME_MAP.items():
        if button == qt_button:
            return name
    # Qt exposes extra buttons as bit flags; persist the numeric value for anything unknown.
    value = int(button)
    if value == int(Qt.XButton1):
        return 'X1'
    if value == int(Qt.XButton2):
        return 'X2'
    return f"Button{value}" if value else "Unknown"

def mouse_name_to_button(name: str):
    """Parse a persisted mouse button token back into a Qt.MouseButton value."""
    if not name:
        return None
    normalized = name.strip().lower()
    lookup = {
        'left': Qt.LeftButton,
        'l': Qt.LeftButton,
        'right': Qt.RightButton,
        'r': Qt.RightButton,
        'middle': Qt.MiddleButton,
        'm': Qt.MiddleButton,
        'mid': Qt.MiddleButton,
        'wheel': Qt.MiddleButton,
        'back': Qt.BackButton,
        'backward': Qt.BackButton,
        'x1': Qt.XButton1,
        'forward': Qt.ForwardButton,
        'x2': Qt.XButton2,
        'task': Qt.TaskButton,
    }
    if normalized in lookup:
        return lookup[normalized]
    if normalized.startswith('button'):
        try:
            value = int(normalized.replace('button', ''))
            return Qt.MouseButton(value)
        except Exception:
            return None
    return None
