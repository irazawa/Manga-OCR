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

from src.utils.geometry import mouse_button_to_name, mouse_name_to_button

class ShortcutCaptureEdit(QWidget):
    """Shortcut field that can capture keyboard or mouse buttons (incl. extra/back/pen buttons)."""
    sequence_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sequence = ''
        self._capturing = False
        self._ignore_mouse_event = False
        self.setFocusPolicy(Qt.StrongFocus)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.display = QLineEdit(self)
        self.display.setReadOnly(True)
        self.display.setPlaceholderText("Capture keyboard or mouse (back/forward/middle/pen)")
        self.display.installEventFilter(self)
        layout.addWidget(self.display, 1)

        self.capture_btn = QToolButton(self)
        self.capture_btn.setText("Capture")
        self.capture_btn.clicked.connect(self.start_capture)
        layout.addWidget(self.capture_btn)

    def eventFilter(self, obj, event):
        if obj is self.display and event.type() in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease):
            # Start capture when the field itself is clicked.
            if not self._capturing:
                self.start_capture()
            return True
        return super().eventFilter(obj, event)

    def start_capture(self):
        if self._capturing:
            return
        self._capturing = True
        self._ignore_mouse_event = True  # ignore the click that started capture
        self.display.setText("Listening...")
        try:
            self.setFocus()
        except Exception:
            pass
        try:
            self.grabKeyboard()
            self.grabMouse()
        except Exception:
            pass

    def stop_capture(self):
        if not self._capturing:
            return
        self._capturing = False
        self._ignore_mouse_event = False
        try:
            self.releaseKeyboard()
            self.releaseMouse()
        except Exception:
            pass
        if not self._sequence:
            self.display.clear()

    def mousePressEvent(self, event):
        if not self._capturing:
            self.start_capture()
            return
        if self._ignore_mouse_event:
            # swallow the click that triggered capture
            self._ignore_mouse_event = False
            return
        self._record_mouse_sequence('press', event.button())

    def mouseReleaseEvent(self, event):
        if not self._capturing:
            return super().mouseReleaseEvent(event)
        if self._ignore_mouse_event:
            self._ignore_mouse_event = False
            return
        self._record_mouse_sequence('release', event.button())

    def mouseDoubleClickEvent(self, event):
        if not self._capturing:
            return super().mouseDoubleClickEvent(event)
        if self._ignore_mouse_event:
            self._ignore_mouse_event = False
            return
        self._record_mouse_sequence('double', event.button())

    def keyPressEvent(self, event):
        if not self._capturing:
            return super().keyPressEvent(event)
        key = event.key()
        if key in (Qt.Key_Control, Qt.Key_Shift, Qt.Key_Alt, Qt.Key_Meta, Qt.Key_unknown):
            return
        if key == Qt.Key_Escape:
            self.clear_sequence()
            self.stop_capture()
            return
        sequence = QKeySequence(event.modifiers() | key).toString(QKeySequence.PortableText)
        self.set_sequence(sequence)
        self.stop_capture()

    def focusOutEvent(self, event):
        self.stop_capture()
        return super().focusOutEvent(event)

    def _record_mouse_sequence(self, event_type: str, button: Qt.MouseButton):
        name = mouse_button_to_name(button)
        if not name or name == "Unknown":
            name = f"Button{int(button)}"
        self.set_sequence(f"MOUSE:{event_type}:{name}")
        self.stop_capture()

    def set_sequence(self, sequence: str):
        self._sequence = sequence.strip() if sequence else ''
        self._update_display()
        self.sequence_changed.emit(self._sequence)

    def clear_sequence(self):
        self._sequence = ''
        self.display.clear()
        self.sequence_changed.emit('')

    def clear(self):
        self.clear_sequence()

    def sequence(self) -> str:
        return self._sequence

    def _update_display(self):
        if not self._sequence:
            self.display.clear()
            return
        if self._sequence.upper().startswith('MOUSE:'):
            parts = self._sequence.split(':')
            if len(parts) >= 3:
                evt = parts[1].lower()
                btn = parts[2]
                btn_label = btn
                friendly = mouse_button_to_name(mouse_name_to_button(btn) or Qt.NoButton)
                if friendly and friendly != "Unknown":
                    btn_label = friendly
                evt_label = evt.capitalize()
                self.display.setText(f"Mouse {evt_label} · {btn_label}")
                return
        seq_obj = QKeySequence(self._sequence)
        rendered = seq_obj.toString(QKeySequence.NativeText)
        self.display.setText(rendered if rendered else self._sequence)

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
