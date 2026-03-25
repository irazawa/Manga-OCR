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
    QKeySequenceEdit, QStackedWidget
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
from src.ui.widgets import *
from src.ui.panels import *
from src.utils.helpers import *
from src.core.fonts import *
from src.core.models import EnhancedResult

class APIManagerDialog(QDialog):
    """Modal wrapper that reuses APIManagerPanel and persists changes."""

    TRANSLATION_PROVIDERS = APIManagerPanel.TRANSLATION_PROVIDERS
    OCR_PROVIDERS = APIManagerPanel.OCR_PROVIDERS

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('API Manager')
        self.setModal(True)
        self.resize(860, 520)

        layout = QVBoxLayout(self)
        self.panel = APIManagerPanel(SETTINGS, self)
        layout.addWidget(self.panel)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel, self)
        save_btn = buttons.button(QDialogButtonBox.Save)
        save_btn.setText("Save")
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_save(self):
        exported = self.panel.export_settings()
        if exported is None:
            warnings = "\n".join(self.panel.validation_messages())
            details = f"\n\n{warnings}" if warnings else ""
            QMessageBox.warning(self, "Validation", f"Please fix the highlighted issues before saving.{details}")
            return

        SETTINGS.setdefault('apis', {})
        SETTINGS.setdefault('ocr', {})
        SETTINGS.setdefault('tesseract', {})
        SETTINGS['apis'] = copy.deepcopy(exported.get('apis', {}))
        SETTINGS['ocr'] = copy.deepcopy(exported.get('ocr', {}))
        SETTINGS['tesseract'] = copy.deepcopy(exported.get('tesseract', {}))
        save_settings(SETTINGS)

        try:
            refresh_api_clients()
        except Exception:
            pass

        parent = self.parent()
        if parent and hasattr(parent, 'populate_ocr_languages'):
            try:
                parent.populate_ocr_languages()
            except Exception:
                pass

        QMessageBox.information(self, "Success", "API settings updated successfully.")
        self.accept()

class SettingsDialog(QDialog):
    def __init__(self, parent=None, autosave_enabled: bool = True, autosave_interval_ms: int = 300000):
        super().__init__(parent)
        self.setWindowTitle('Preferences')
        self.setModal(True)
        self.resize(360, 140)

        layout = QVBoxLayout(self)

        # Autosave enable
        self.autosave_checkbox = QCheckBox('Enable autosave', self)
        self.autosave_checkbox.setChecked(bool(autosave_enabled))
        layout.addWidget(self.autosave_checkbox)

        # Interval (seconds)
        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel('Autosave interval (seconds):'))
        self.interval_spin = QSpinBox(self)
        self.interval_spin.setRange(5, 3600)  # 5s..1h
        # store/display in seconds for user convenience
        self.interval_spin.setValue(max(5, int(autosave_interval_ms / 1000)))
        interval_layout.addWidget(self.interval_spin)
        layout.addLayout(interval_layout)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_values(self):
        return {
            'enabled': bool(self.autosave_checkbox.isChecked()),
            'interval_ms': int(self.interval_spin.value() * 1000)
        }


class ModelEditDialog(QDialog):
    def __init__(self, parent=None, name: str = "", model_id: str = "", description: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Model Settings")
        self.setModal(True)
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name_edit = QLineEdit(name)
        self.name_edit.setPlaceholderText("Display name")
        self.id_edit = QLineEdit(model_id)
        self.id_edit.setPlaceholderText("provider/model-id")
        self.description_edit = QLineEdit(description)
        self.description_edit.setPlaceholderText("Optional description")
        form.addRow("Model Name", self.name_edit)
        form.addRow("Model ID", self.id_edit)
        form.addRow("Description", self.description_edit)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_values(self):
        return {
            "name": self.name_edit.text().strip(),
            "id": self.id_edit.text().strip(),
            "description": self.description_edit.text().strip()
        }

class OpenRouterSettingsDialog(QDialog):
    """Backward-compatible dialog wrapper around OpenRouterSettingsPanel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("OpenRouter Translate Settings")
        self.setModal(True)
        self.resize(640, 480)

        layout = QVBoxLayout(self)
        self.panel = OpenRouterSettingsPanel(SETTINGS, self)
        layout.addWidget(self.panel)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self):
        self.panel.export_settings()
        self.accept()

    def get_settings(self):
        return self.panel.get_settings()

class SettingsCenterDialog(QDialog):
    """Unified settings dialog — modern sidebar navigation layout."""

    # Nav item definitions: (label, icon-emoji)
    _NAV_ITEMS = [
        ("General",     "⚙"),
        ("Cleanup",     "🧹"),
        ("Translation", "🌐"),
        ("Shortcuts",   "⌨"),
        ("API Keys",    "🔑"),
    ]

    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setObjectName("SettingsCenterDialog")
        # Fit dialog to the available screen — cap at 85 % height, 75 % width
        try:
            screen = QApplication.primaryScreen().availableGeometry()
            max_h = int(screen.height() * 0.85)
            max_w = int(screen.width() * 0.75)
            dlg_w = min(940, max_w)
            dlg_h = min(660, max_h)
            self.resize(dlg_w, dlg_h)
            self.setMaximumHeight(max_h)
        except Exception:
            self.resize(860, 580)
            self.setMaximumHeight(700)

        self._initial_autosave_enabled, self._initial_autosave_interval = self._current_autosave_state()
        self._initial_cleanup = copy.deepcopy(SETTINGS.get('cleanup', {}))
        self._initial_api = {
            'apis': copy.deepcopy(SETTINGS.get('apis', {})),
            'ocr': copy.deepcopy(SETTINGS.get('ocr', {})),
            'tesseract': copy.deepcopy(SETTINGS.get('tesseract', {})),
        }
        self._initial_translate = copy.deepcopy(SETTINGS.get('translate', {}).get('openrouter', {}))
        self._initial_shortcuts = copy.deepcopy(SETTINGS.get('shortcuts', {}))
        self.shortcut_editors = {}

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Splitter: left sidebar nav + right content ─────────────────
        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setObjectName("settings-splitter")
        splitter.setHandleWidth(1)

        # Left nav panel
        nav_panel = QWidget()
        nav_panel.setObjectName("settings-nav-panel")
        nav_panel.setFixedWidth(200)
        nav_vbox = QVBoxLayout(nav_panel)
        nav_vbox.setContentsMargins(0, 0, 0, 0)
        nav_vbox.setSpacing(0)

        header_lbl = QLabel("  ⚙  Settings")
        header_lbl.setObjectName("settings-nav-header")
        header_lbl.setFixedHeight(58)
        nav_vbox.addWidget(header_lbl)

        self._nav_list = QListWidget()
        self._nav_list.setObjectName("settings-nav-list")
        self._nav_list.setFocusPolicy(Qt.NoFocus)
        self._nav_list.setSpacing(2)
        for label, icon in self._NAV_ITEMS:
            item = QListWidgetItem(f"    {icon}   {label}")
            item.setSizeHint(QSize(180, 46))
            self._nav_list.addItem(item)
        nav_vbox.addWidget(self._nav_list, 1)

        footer_lbl = QLabel("  Manga Tool v14")
        footer_lbl.setObjectName("settings-nav-footer")
        footer_lbl.setFixedHeight(34)
        nav_vbox.addWidget(footer_lbl)

        splitter.addWidget(nav_panel)

        # Right: stacked content + button bar
        right_panel = QWidget()
        right_panel.setObjectName("settings-right-panel")
        right_vbox = QVBoxLayout(right_panel)
        right_vbox.setContentsMargins(0, 0, 0, 0)
        right_vbox.setSpacing(0)

        self._pages = QStackedWidget()
        self._pages.setObjectName("settings-pages")

        self.general_tab = self._create_general_tab()
        self.cleanup_tab = self._create_cleanup_tab()
        self.translation_tab = self._create_translation_tab()
        self.shortcuts_tab = self._create_shortcuts_tab()
        self.api_tab = self._create_api_tab()

        for page in (self.general_tab, self.cleanup_tab,
                     self.translation_tab, self.shortcuts_tab, self.api_tab):
            self._pages.addWidget(page)

        right_vbox.addWidget(self._pages, 1)

        # Button bar
        btn_bar = QWidget()
        btn_bar.setObjectName("settings-btn-bar")
        btn_bar_hbox = QHBoxLayout(btn_bar)
        btn_bar_hbox.setContentsMargins(24, 10, 24, 14)
        btn_bar_hbox.setSpacing(10)
        btn_bar_hbox.addStretch(1)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setObjectName("settings-cancel-btn")
        self._cancel_btn.setFixedWidth(100)
        self._cancel_btn.clicked.connect(self.reject)

        self._save_btn = QPushButton("Save Changes")
        self._save_btn.setObjectName("settings-save-btn")
        self._save_btn.setFixedWidth(130)
        self._save_btn.clicked.connect(self._on_save)

        btn_bar_hbox.addWidget(self._cancel_btn)
        btn_bar_hbox.addWidget(self._save_btn)
        right_vbox.addWidget(btn_bar)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        root.addWidget(splitter, 1)

        # Wire nav selection → page switch
        self._nav_list.currentRowChanged.connect(self._on_nav_changed)
        self._nav_list.setCurrentRow(0)

        self._apply_settings_styles()

    def _on_nav_changed(self, index):
        """Switch stacked page when user clicks nav item."""
        if 0 <= index < self._pages.count():
            self._pages.setCurrentIndex(index)

    def _make_page_scroll(self):
        """Helper: returns (scroll_area, inner_layout) for a settings page."""
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget()
        inner.setObjectName("settings-page-inner")
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(28, 24, 28, 24)
        inner_layout.setSpacing(20)
        scroll.setWidget(inner)
        page_layout.addWidget(scroll)
        return page, inner_layout

    def _make_page_header(self, title: str, subtitle: str = ""):
        """Returns a styled header widget for a page."""
        w = QWidget()
        w.setObjectName("settings-page-header")
        vl = QVBoxLayout(w)
        vl.setContentsMargins(0, 0, 0, 4)
        vl.setSpacing(2)
        t = QLabel(title)
        t.setObjectName("settings-page-title")
        vl.addWidget(t)
        if subtitle:
            s = QLabel(subtitle)
            s.setObjectName("settings-page-subtitle")
            s.setWordWrap(True)
            vl.addWidget(s)
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setObjectName("settings-sep")
        vl.addWidget(sep)
        return w

    def _make_option_row(self, label: str, desc: str, widget):
        """Returns a horizontal row: label+desc on left, widget on right."""
        row = QWidget()
        row.setObjectName("settings-option-row")
        hl = QHBoxLayout(row)
        hl.setContentsMargins(16, 12, 16, 12)
        hl.setSpacing(16)
        text_col = QVBoxLayout()
        text_col.setSpacing(2)
        lbl = QLabel(label)
        lbl.setObjectName("settings-option-label")
        text_col.addWidget(lbl)
        if desc:
            dlbl = QLabel(desc)
            dlbl.setObjectName("settings-option-desc")
            dlbl.setWordWrap(True)
            text_col.addWidget(dlbl)
        hl.addLayout(text_col, 1)
        hl.addWidget(widget)
        return row

    def _create_general_tab(self):
        page, layout = self._make_page_scroll()
        layout.addWidget(self._make_page_header(
            "General", "Application-wide preferences."))

        # ── Autosave card ────────────────────────────────────────────────
        as_card = QGroupBox("💾  Autosave")
        as_card.setObjectName("settings-card")
        as_vbox = QVBoxLayout(as_card)
        as_vbox.setSpacing(4)
        as_vbox.setContentsMargins(0, 8, 0, 8)

        self.autosave_checkbox = QCheckBox("Enable autosave")
        as_vbox.addWidget(self._make_option_row(
            "Autosave",
            "Periodically save a backup of your current project.",
            self.autosave_checkbox))

        self.autosave_interval_spin = QSpinBox()
        self.autosave_interval_spin.setRange(5, 3600)
        self.autosave_interval_spin.setSingleStep(5)
        self.autosave_interval_spin.setSuffix(" s")
        self.autosave_interval_spin.setFixedWidth(90)
        as_vbox.addWidget(self._make_option_row(
            "Interval",
            "How often to auto-save (seconds).",
            self.autosave_interval_spin))

        layout.addWidget(as_card)

        # ── Output card ──────────────────────────────────────────────────
        out_card = QGroupBox("🖼  Output")
        out_card.setObjectName("settings-card")
        out_vbox = QVBoxLayout(out_card)
        out_vbox.setSpacing(4)
        out_vbox.setContentsMargins(0, 8, 0, 8)

        self.save_format_combo = QComboBox()
        self.save_format_combo.addItems(["PNG", "WEBP", "JPG"])
        self.save_format_combo.setFixedWidth(110)
        current_fmt = SETTINGS.get('general', {}).get('save_format', 'PNG').upper()
        if current_fmt not in ["PNG", "WEBP", "JPG"]: current_fmt = "PNG"
        self.save_format_combo.setCurrentText(current_fmt)
        out_vbox.addWidget(self._make_option_row(
            "Image Format", "File format for exported/saved pages.",
            self.save_format_combo))

        self.save_quality_spin = QSpinBox()
        self.save_quality_spin.setRange(10, 100)
        self.save_quality_spin.setSuffix("%")
        self.save_quality_spin.setFixedWidth(90)
        self.save_quality_spin.setValue(int(SETTINGS.get('general', {}).get('save_quality', 95)))
        out_vbox.addWidget(self._make_option_row(
            "Quality", "Compression quality for WEBP/JPG exports.",
            self.save_quality_spin))

        layout.addWidget(out_card)
        layout.addStretch(1)

        self.autosave_checkbox.setChecked(self._initial_autosave_enabled)
        self.autosave_interval_spin.setValue(max(5, int(self._initial_autosave_interval / 1000)))

        return page

    def _create_cleanup_tab(self):
        page, layout = self._make_page_scroll()
        layout.addWidget(self._make_page_header(
            "Cleanup", "Defaults applied to new text areas during cleanup."))

        card = QGroupBox("🧹  Text Defaults")
        card.setObjectName("settings-card")
        card_vbox = QVBoxLayout(card)
        card_vbox.setSpacing(4)
        card_vbox.setContentsMargins(0, 8, 0, 8)

        cleanup_cfg = SETTINGS.get('cleanup', {})

        self.auto_text_color_checkbox = QCheckBox()
        self.auto_text_color_checkbox.setChecked(bool(cleanup_cfg.get('auto_text_color', True)))
        card_vbox.addWidget(self._make_option_row(
            "Auto text color",
            "Automatically pick a contrasting text color for each bubble.",
            self.auto_text_color_checkbox))

        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(0, 255)
        self.threshold_spin.setFixedWidth(90)
        self.threshold_spin.setToolTip("Higher values prefer brighter text.")
        self.threshold_spin.setValue(int(cleanup_cfg.get('text_color_threshold', 128)))
        card_vbox.addWidget(self._make_option_row(
            "Color threshold",
            "Luminance threshold for auto text color inversion (0–255).",
            self.threshold_spin))

        self.use_background_box_checkbox = QCheckBox()
        self.use_background_box_checkbox.setChecked(bool(cleanup_cfg.get('use_background_box', True)))
        card_vbox.addWidget(self._make_option_row(
            "Background box",
            "Draw a filled background box behind new translated text by default.",
            self.use_background_box_checkbox))

        self.constrain_text_checkbox = QCheckBox()
        self.constrain_text_checkbox.setToolTip("Text will wrap to box width even when the background is hidden.")
        self.constrain_text_checkbox.setChecked(bool(cleanup_cfg.get('constrain_text', False)))
        card_vbox.addWidget(self._make_option_row(
            "Constrain text",
            "Wrap text to box width even when the background box is off.",
            self.constrain_text_checkbox))

        self.remove_ai_temp_checkbox = QCheckBox()
        self.remove_ai_temp_checkbox.setToolTip("Delete temp/ debug files after each successful AI OCR/MOFRL run.")
        self.remove_ai_temp_checkbox.setChecked(bool(cleanup_cfg.get('remove_ai_temp_files', False)))
        card_vbox.addWidget(self._make_option_row(
            "Remove AI temp files",
            "Delete temporary debug files after AI OCR & MOFRL runs.",
            self.remove_ai_temp_checkbox))

        layout.addWidget(card)
        layout.addStretch(1)
        return page

    def _create_translation_tab(self):
        page = QWidget()
        vbox = QVBoxLayout(page)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)
        # Header
        hdr = QWidget()
        hdr.setObjectName("settings-page-header-bar")
        hdr_l = QVBoxLayout(hdr)
        hdr_l.setContentsMargins(28, 14, 28, 6)
        title = QLabel("Translation")
        title.setObjectName("settings-page-title")
        sub = QLabel("Configure OpenRouter translation API and models.")
        sub.setObjectName("settings-page-subtitle")
        sub.setWordWrap(True)
        sep = QFrame(); sep.setFrameShape(QFrame.HLine); sep.setObjectName("settings-sep")
        hdr_l.addWidget(title)
        hdr_l.addWidget(sub)
        hdr_l.addWidget(sep)
        vbox.addWidget(hdr)
        # Wrap panel in scroll area so it never exceeds dialog height
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        self.openrouter_panel = OpenRouterSettingsPanel(SETTINGS, scroll)
        scroll.setWidget(self.openrouter_panel)
        vbox.addWidget(scroll, 1)
        return page

    def _create_shortcuts_tab(self):
        page = QWidget()
        vbox = QVBoxLayout(page)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)
        # Header
        hdr = QWidget()
        hdr.setObjectName("settings-page-header-bar")
        hdr_l = QVBoxLayout(hdr)
        hdr_l.setContentsMargins(28, 18, 28, 8)
        title = QLabel("Shortcuts")
        title.setObjectName("settings-page-title")
        sub = QLabel("Assign keyboard or mouse shortcuts. Leave blank to disable. "
                     "Click a field then press a key/mouse button to capture.")
        sub.setObjectName("settings-page-subtitle")
        sub.setWordWrap(True)
        sep = QFrame(); sep.setFrameShape(QFrame.HLine); sep.setObjectName("settings-sep")
        hdr_l.addWidget(title); hdr_l.addWidget(sub); hdr_l.addWidget(sep)
        vbox.addWidget(hdr)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(28, 16, 28, 24)
        container_layout.setSpacing(18)

        category_order = []
        grouped = {}
        for key, label, category in SHORTCUT_DEFINITIONS:
            grouped.setdefault(category, []).append((key, label))
            if category not in category_order:
                category_order.append(category)

        user_shortcuts = SETTINGS.get('shortcuts', {}) or {}
        for category in category_order:
            entries = grouped.get(category, [])
            if not entries:
                continue
            group_box = QGroupBox(category)
            group_box.setObjectName("settings-card")
            form = QFormLayout(group_box)
            form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            form.setHorizontalSpacing(18)
            form.setVerticalSpacing(10)

            for key, label in entries:
                editor = ShortcutCaptureEdit()
                seq = user_shortcuts.get(key)
                if seq is None:
                    seq = DEFAULT_SHORTCUTS.get(key, '')
                if seq:
                    editor.set_sequence(seq)
                else:
                    editor.clear_sequence()

                row_widget = QWidget()
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(6)
                row_layout.addWidget(editor, 1)

                clear_btn = QToolButton()
                clear_btn.setText("✕")
                clear_btn.setToolTip("Clear shortcut")
                clear_btn.clicked.connect(editor.clear_sequence)
                row_layout.addWidget(clear_btn)

                default_btn = QToolButton()
                default_btn.setText("↺")
                default_btn.setToolTip("Restore default")

                def _reset_editor(target_editor=editor, target_key=key):
                    default_seq = DEFAULT_SHORTCUTS.get(target_key, '')
                    target_editor.set_sequence(default_seq)

                default_btn.clicked.connect(_reset_editor)
                row_layout.addWidget(default_btn)

                form.addRow(label, row_widget)
                self.shortcut_editors[key] = editor

            container_layout.addWidget(group_box)

        container_layout.addStretch(1)
        scroll.setWidget(container)
        vbox.addWidget(scroll, 1)
        return page

    def _create_api_tab(self):
        page = QWidget()
        vbox = QVBoxLayout(page)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)
        # Header
        hdr = QWidget()
        hdr.setObjectName("settings-page-header-bar")
        hdr_l = QVBoxLayout(hdr)
        hdr_l.setContentsMargins(28, 14, 28, 6)
        title = QLabel("API Keys")
        title.setObjectName("settings-page-title")
        sub = QLabel("Manage translation and OCR API keys, providers, and Tesseract path.")
        sub.setObjectName("settings-page-subtitle")
        sub.setWordWrap(True)
        sep = QFrame(); sep.setFrameShape(QFrame.HLine); sep.setObjectName("settings-sep")
        hdr_l.addWidget(title); hdr_l.addWidget(sub); hdr_l.addWidget(sep)
        vbox.addWidget(hdr)
        # Wrap panel in scroll area so it never exceeds dialog height
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        self.api_panel = APIManagerPanel(SETTINGS, scroll)
        scroll.setWidget(self.api_panel)
        vbox.addWidget(scroll, 1)
        return page

    def _apply_settings_styles(self):
        """Premium dark settings skin — GitHub Dark palette."""
        self.setStyleSheet("""
            /* ── Dialog root ─────────────────────────────────────────── */
            QDialog#SettingsCenterDialog {
                background: #0d1117;
                color: #e6edf3;
                font-family: 'Segoe UI', 'Inter', sans-serif;
                font-size: 10pt;
            }

            /* ── Left sidebar ────────────────────────────────────────── */
            #settings-nav-panel {
                background: #161b22;
                border-right: 1px solid #30363d;
            }
            #settings-nav-header {
                background: #161b22;
                color: #e6edf3;
                font-size: 13pt;
                font-weight: 700;
                border-bottom: 1px solid #30363d;
                padding-left: 4px;
            }
            #settings-nav-footer {
                background: #161b22;
                color: #484f58;
                font-size: 8.5pt;
                border-top: 1px solid #30363d;
                padding-left: 4px;
            }
            #settings-nav-list {
                background: #161b22;
                border: none;
                outline: none;
                color: #c9d1d9;
                font-size: 10pt;
            }
            #settings-nav-list::item {
                border-radius: 8px;
                margin: 2px 8px;
                padding: 2px 0;
            }
            #settings-nav-list::item:selected {
                background: #1f6feb;
                color: #ffffff;
                font-weight: 600;
            }
            #settings-nav-list::item:hover:!selected {
                background: #21262d;
                color: #e6edf3;
            }

            /* ── Right panel / pages ─────────────────────────────────── */
            #settings-right-panel, #settings-pages, #settings-page-inner {
                background: #0d1117;
            }
            #settings-page-header-bar {
                background: #0d1117;
            }
            #settings-page-title {
                font-size: 15pt;
                font-weight: 700;
                color: #e6edf3;
            }
            #settings-page-subtitle {
                font-size: 9.5pt;
                color: #8b949e;
            }
            #settings-sep {
                color: #30363d;
                background: #30363d;
                max-height: 1px;
                border: none;
                margin-top: 6px;
            }

            /* ── Cards (GroupBox) ───────────────────────────────────── */
            QGroupBox {
                background: #161b22;
                border: 1px solid #30363d;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                color: #79c0ff;
                font-weight: 600;
                font-size: 9.5pt;
                padding: 0 10px;
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
            }

            /* ── Option rows ────────────────────────────────────────── */
            #settings-option-row {
                background: transparent;
                border-bottom: 1px solid #21262d;
            }
            #settings-option-row:last-child {
                border-bottom: none;
            }
            #settings-option-label {
                font-size: 10pt;
                font-weight: 600;
                color: #e6edf3;
            }
            #settings-option-desc {
                font-size: 8.5pt;
                color: #8b949e;
            }

            /* ── Inputs ─────────────────────────────────────────────── */
            QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QKeySequenceEdit {
                background: #21262d;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 6px 10px;
                color: #e6edf3;
                selection-background-color: #1f6feb;
            }
            QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border-color: #1f6feb;
            }
            QComboBox::drop-down {
                width: 20px;
                border-left: 1px solid #30363d;
            }
            QComboBox QAbstractItemView {
                background: #21262d;
                border: 1px solid #30363d;
                selection-background-color: #1f6feb;
                color: #e6edf3;
            }

            /* ── Checkboxes ─────────────────────────────────────────── */
            QCheckBox {
                spacing: 8px;
                color: #e6edf3;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #30363d;
                border-radius: 4px;
                background: #21262d;
            }
            QCheckBox::indicator:checked {
                background: #1f6feb;
                border-color: #1f6feb;
                image: none;
            }
            QCheckBox::indicator:hover {
                border-color: #388bfd;
            }

            /* ── Buttons ────────────────────────────────────────────── */
            QPushButton, QToolButton {
                background: #21262d;
                color: #c9d1d9;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 6px 14px;
                font-weight: 600;
            }
            QPushButton:hover:!disabled, QToolButton:hover:!disabled {
                background: #30363d;
                border-color: #8b949e;
                color: #e6edf3;
            }
            QPushButton:pressed:!disabled, QToolButton:pressed:!disabled {
                background: #161b22;
            }
            QPushButton:disabled, QToolButton:disabled {
                background: #161b22;
                color: #484f58;
                border-color: #21262d;
            }

            /* Primary Save button */
            #settings-save-btn {
                background: #1f6feb;
                color: #ffffff;
                border: 1px solid #388bfd;
                border-radius: 6px;
                font-weight: 700;
                font-size: 10pt;
            }
            #settings-save-btn:hover {
                background: #388bfd;
                border-color: #58a6ff;
            }
            #settings-save-btn:pressed {
                background: #1158c7;
            }

            /* Ghost Cancel button */
            #settings-cancel-btn {
                background: transparent;
                color: #8b949e;
                border: 1px solid #30363d;
            }
            #settings-cancel-btn:hover {
                background: #21262d;
                color: #e6edf3;
            }

            /* ── Bottom button bar ──────────────────────────────────── */
            #settings-btn-bar {
                background: #161b22;
                border-top: 1px solid #30363d;
            }

            /* ── Misc ───────────────────────────────────────────────── */
            QScrollArea, QScrollBar {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical {
                width: 8px;
                background: transparent;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #30363d;
                border-radius: 4px;
                min-height: 24px;
            }
            QScrollBar::handle:vertical:hover {
                background: #484f58;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
            QLabel {
                color: #e6edf3;
            }
            QRadioButton {
                color: #e6edf3;
                spacing: 8px;
            }
            QTabWidget::pane {
                border: 1px solid #30363d;
                background: #0d1117;
                border-radius: 6px;
            }
            QTabBar::tab {
                background: #21262d;
                color: #8b949e;
                border: 1px solid #30363d;
                border-radius: 4px;
                padding: 6px 14px;
                margin: 2px;
            }
            QTabBar::tab:selected {
                background: #1f6feb;
                color: #ffffff;
                font-weight: 700;
                border-color: #1f6feb;
            }
            QTabBar::tab:hover:!selected {
                background: #30363d;
                color: #e6edf3;
            }
            QTableWidget {
                background: #161b22;
                border: 1px solid #30363d;
                gridline-color: #21262d;
                color: #e6edf3;
                border-radius: 6px;
            }
            QHeaderView::section {
                background: #21262d;
                color: #8b949e;
                border: none;
                border-bottom: 1px solid #30363d;
                padding: 6px;
                font-weight: 600;
            }
            QListWidget {
                background: #161b22;
                border: 1px solid #30363d;
                color: #e6edf3;
                border-radius: 6px;
            }
            QListWidget::item:selected {
                background: #1f6feb;
                color: #fff;
            }
            QGroupBox#settings-card {
                background: #161b22;
                border: 1px solid #30363d;
                border-radius: 10px;
            }
            QSplitter::handle {
                background: #30363d;
            }
        """)


    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def set_active_tab(self, key: str):
        if not key:
            return
        idx_map = {
            'general': 0, 'cleanup': 1, 'translation': 2,
            'shortcuts': 3, 'shortcut': 3, 'api': 4, 'apis': 4, 'keys': 4,
        }
        idx = idx_map.get(str(key).lower())
        if idx is not None:
            self._nav_list.setCurrentRow(idx)

    def _current_autosave_state(self):
        enabled = False
        interval_ms = 300000
        try:
            timer = getattr(self.main_window, 'autosave_timer', None)
            if timer is not None:
                interval_ms = timer.interval()
                enabled = timer.isActive()
        except Exception:
            pass
        if hasattr(self.main_window, 'autosave_enabled'):
            enabled = bool(getattr(self.main_window, 'autosave_enabled'))
        return enabled, max(5000, int(interval_ms))

    def _apply_autosave_settings(self, enabled, interval_ms):
        try:
            timer = getattr(self.main_window, 'autosave_timer', None)
            if timer is None:
                timer = QTimer(self.main_window)
                timer.timeout.connect(self.main_window.auto_save_project)
                self.main_window.autosave_timer = timer
            try:
                interval_ms = int(interval_ms)
            except Exception:
                interval_ms = 300000
            interval_ms = max(5000, interval_ms)
            timer.setInterval(interval_ms)
            self.main_window.autosave_enabled = bool(enabled)
            if enabled:
                timer.start()
            else:
                timer.stop()
            autosave_cfg = SETTINGS.setdefault('autosave', {})
            autosave_cfg['enabled'] = bool(enabled)
            autosave_cfg['interval_ms'] = interval_ms
            save_settings(SETTINGS)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Save handler
    # ------------------------------------------------------------------
    def _on_save(self):
        api_export = self.api_panel.export_settings()
        if api_export is None:
            warnings = "\n".join(self.api_panel.validation_messages())
            details = f"\n\n{warnings}" if warnings else ""
            QMessageBox.warning(self, "API Settings", f"Please fix the highlighted API settings issues before saving.{details}")
            self.tabs.setCurrentWidget(self.api_tab)
            return

        openrouter_settings = self.openrouter_panel.export_settings()

        autosave_enabled = bool(self.autosave_checkbox.isChecked())
        autosave_interval_ms = int(self.autosave_interval_spin.value() * 1000)
        self._apply_autosave_settings(autosave_enabled, autosave_interval_ms)

        gen_cfg = SETTINGS.setdefault('general', {})
        gen_cfg['save_format'] = self.save_format_combo.currentText()
        gen_cfg['save_quality'] = self.save_quality_spin.value()

        cleanup_cfg = SETTINGS.setdefault('cleanup', {})
        prev_auto_color = bool(cleanup_cfg.get('auto_text_color', True))
        prev_threshold = int(cleanup_cfg.get('text_color_threshold', 128))
        prev_use_box = bool(cleanup_cfg.get('use_background_box', True))

        cleanup_cfg['auto_text_color'] = bool(self.auto_text_color_checkbox.isChecked())
        cleanup_cfg['text_color_threshold'] = int(self.threshold_spin.value())
        use_box_value = bool(self.use_background_box_checkbox.isChecked())
        if hasattr(self.main_window, '_set_global_cleanup_default'):
            try:
                self.main_window._set_global_cleanup_default('use_background_box', use_box_value, persist=False)
            except Exception:
                cleanup_cfg['use_background_box'] = use_box_value
        cleanup_cfg['use_background_box'] = use_box_value
        cleanup_cfg['constrain_text'] = bool(self.constrain_text_checkbox.isChecked())
        # Persist new AI temp cleanup option
        try:
            cleanup_cfg['remove_ai_temp_files'] = bool(self.remove_ai_temp_checkbox.isChecked())
        except Exception:
            cleanup_cfg['remove_ai_temp_files'] = bool(cleanup_cfg.get('remove_ai_temp_files', False))

        shortcut_settings = {}
        for key, editor in self.shortcut_editors.items():
            sequence = (editor.sequence() or '').strip()
            default_seq = DEFAULT_SHORTCUTS.get(key, '')
            if not sequence:
                # Only persist blanks when overriding a default binding
                if default_seq:
                    shortcut_settings[key] = ''
            elif sequence != default_seq:
                shortcut_settings[key] = sequence

        shortcuts_changed = shortcut_settings != self._initial_shortcuts

        SETTINGS.setdefault('translate', {})
        SETTINGS['translate']['openrouter'] = copy.deepcopy(openrouter_settings)
        SETTINGS['apis'] = copy.deepcopy(api_export.get('apis', {}))
        SETTINGS['ocr'] = copy.deepcopy(api_export.get('ocr', {}))
        SETTINGS['tesseract'] = copy.deepcopy(api_export.get('tesseract', {}))
        SETTINGS['shortcuts'] = shortcut_settings

        save_settings(SETTINGS)

        try:
            refresh_api_clients()
        except Exception:
            pass

        if hasattr(self.main_window, 'reload_shortcuts'):
            try:
                self.main_window.reload_shortcuts()
            except Exception:
                pass

        if hasattr(self.main_window, 'populate_ocr_languages'):
            try:
                self.main_window.populate_ocr_languages()
            except Exception:
                pass

        if hasattr(self.main_window, 'populate_ai_models'):
            try:
                self.main_window.populate_ai_models()
            except Exception:
                pass

        if (cleanup_cfg.get('auto_text_color') != prev_auto_color) or (cleanup_cfg.get('text_color_threshold') != prev_threshold):
            try:
                self.main_window.redraw_all_typeset_areas()
            except Exception:
                pass

        status_parts = []
        if (cleanup_cfg.get('auto_text_color') != prev_auto_color) or (cleanup_cfg.get('text_color_threshold') != prev_threshold) or (cleanup_cfg.get('use_background_box') != prev_use_box):
            status_parts.append("Cleanup defaults updated")
        if openrouter_settings != self._initial_translate:
            status_parts.append("Translation settings updated")
        if any(api_export.get(key, {}) != self._initial_api.get(key, {}) for key in ('apis', 'ocr', 'tesseract')):
            status_parts.append("API settings updated")
        if (autosave_enabled != self._initial_autosave_enabled) or (abs(autosave_interval_ms - self._initial_autosave_interval) > 1):
            status_parts.append("Autosave preferences updated")
        if shortcuts_changed:
            status_parts.append("Shortcuts updated")

        if status_parts and hasattr(self.main_window, 'statusBar'):
            try:
                self.main_window.statusBar().showMessage(" · ".join(status_parts), 4000)
            except Exception:
                pass

        self.accept()

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


class AdvancedTextEditDialog(QDialog):
    EFFECT_OPTIONS = [
        ("None", "none"),
        ("Curved", "curved"),
        ("Wavy", "wavy"),
        ("Jagged Shout", "jagged"),
    ]
    ALIGN_OPTIONS = [
        ("Center", Qt.AlignHCenter),
        ("Left", Qt.AlignLeft),
        ("Right", Qt.AlignRight),
        ("Justify", Qt.AlignJustify),
    ]
    EMOJI_PRESETS = [
        ("Heart", "❤"),
        ("Heart1", "♥︎"),
        ("Heart2", "♡"),
        ("Heart3", "❤"),
        ("Heart3", "ㅤ♡ㅤ"),
        ("Sparkle", "✨"),
        ("Star", "★"),
        ("Music", "♪"),
        ("Shock", "⁉"),
        ("Sweat", "💦"),
        ("Smile", "😊"),
        ("Angry", "😠"),
        ("Glow", "glow"),
    ]

    GRADIENT_DIRECTIONS = [
        ("Custom", -1),
        ("Left -> Right", 0.0),
        ("Top -> Bottom", 90.0),
        ("Right -> Left", 180.0),
        ("Bottom -> Top", 270.0),
        ("TL -> BR", 45.0),
        ("TR -> BL", 135.0),
        ("BR -> TL", 315.0),
        ("BL -> TR", 225.0),
    ]

    def __init__(self, parent=None, area=None, font_manager=None):
        super().__init__(parent)
        self.area = area
        self.font_manager = font_manager
        self.result = None
        self.setWindowTitle("Advanced Text Editor")
        self.setModal(True)
        # Restore previous size if available, else use default responsive size
        saved_size = SETTINGS.get('ui', {}).get('advanced_editor_size')
        if saved_size and len(saved_size) == 2 and saved_size[0] > 900:
            try:
                self.resize(int(saved_size[0]), int(saved_size[1]))
            except Exception:
                self._apply_default_size()
        else:
            self._apply_default_size()


        self.setObjectName("AdvancedTextEditDialog")
        self.setStyleSheet("""
            QDialog#AdvancedTextEditDialog {
                background: #0f1624;
                color: #e8eef7;
                font-size: 10.5pt;
            }
            QGroupBox {
                background: rgba(255,255,255,0.03);
                border: 1px solid #1f2b36;
                border-radius: 12px;
                margin-top: 12px;
                padding-top: 14px;
            }
            QGroupBox::title {
                color: #9fc3f5;
                padding: 0 8px;
                font-weight: 600;
            }
            QLabel { color: #e8eef7; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit {
                background: #0c121d;
                border: 1px solid #1f2b36;
                border-radius: 8px;
                padding: 6px 8px;
            }
            QPushButton, QToolButton {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #1f6fb5, stop:1 #2c8ae6);
                color: #e8eef7;
                border: 1px solid #2b6aa1;
                border-radius: 8px;
                padding: 7px 12px;
                font-weight: 600;
            }
            QPushButton:hover, QToolButton:hover { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #2a7fcb, stop:1 #3b9af3); }
            QPushButton:disabled, QToolButton:disabled {
                background: #182131;
                color: #7f8a96;
                border-color: #1f2b36;
            }
            QTextEdit {
                border-radius: 12px;
                padding: 10px;
                line-height: 1.4;
            }
            #ate-hero {
                border: 1px solid #1f2b36;
                border-radius: 14px;
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #132036, stop:1 #0c1729);
                padding: 10px;
            }
            #ate-title { font-size: 16px; font-weight: 700; color: #eaf3ff; }
            #ate-subtitle { color: #8fa6c5; }
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 14, 16, 14)
        main_layout.setSpacing(10)

        hero = QFrame()
        hero.setObjectName("ate-hero")
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(14, 12, 14, 12)
        hero_layout.setSpacing(4)
        title = QLabel("Advanced Text Editor")
        title.setObjectName("ate-title")
        subtitle = QLabel("Fine-tune text formatting. Select a range to style only that portion or press Ctrl+A to target the whole bubble.")
        subtitle.setObjectName("ate-subtitle")
        subtitle.setWordWrap(True)
        hero_layout.addWidget(title)
        hero_layout.addWidget(subtitle)
        main_layout.addWidget(hero)

        toolbar_layout = QHBoxLayout()
        # Font group selector for the advanced editor (if the main application provided groups)
        self.font_group_combo = QComboBox()
        self.font_group_combo.setMinimumWidth(160)
        self.font_group_combo.addItem("All")
        toolbar_layout.addWidget(self.font_group_combo)
        self.font_combo = QComboBox()
        self.font_combo.setMinimumWidth(220)
        toolbar_layout.addWidget(self.font_combo)

        self.font_preview = QLabel("AaBb123")
        self.font_preview.setFixedWidth(140)
        self.font_preview.setAlignment(Qt.AlignCenter)
        self.font_preview.setStyleSheet("border: 1px solid #1f2b3b; border-radius: 6px; padding: 4px;")
        toolbar_layout.addWidget(self.font_preview)

        self.font_size_spin = QDoubleSpinBox(); self.font_size_spin.setRange(4.0, 220.0); self.font_size_spin.setDecimals(1); self.font_size_spin.setSingleStep(1.0); self.font_size_spin.setSuffix(" pt")
        toolbar_layout.addWidget(self.font_size_spin)

        self.bold_button = QToolButton(); self.bold_button.setText("B"); self.bold_button.setCheckable(True); self.bold_button.setToolTip("Toggle bold")
        toolbar_layout.addWidget(self.bold_button)

        self.italic_button = QToolButton(); self.italic_button.setText("I"); self.italic_button.setCheckable(True); self.italic_button.setToolTip("Toggle italic")
        toolbar_layout.addWidget(self.italic_button)

        self.underline_button = QToolButton(); self.underline_button.setText("U"); self.underline_button.setCheckable(True); self.underline_button.setToolTip("Toggle underline")
        toolbar_layout.addWidget(self.underline_button)

        self.color_button = QToolButton(); self.color_button.setText("Color"); self.color_button.setToolTip("Change text color")
        toolbar_layout.addWidget(self.color_button)

        toolbar_layout.addSpacing(10)
        self.ai_translate_btn = QPushButton("AI Translate")
        self.ai_translate_btn.setToolTip("Translate currently selected text (or all text) using active AI model")
        self.ai_translate_btn.clicked.connect(self._on_ai_translate_clicked)
        toolbar_layout.addWidget(self.ai_translate_btn)

        # After creating widgets, populate group combo if possible and hook handlers
        try:
            main_win = self.parent()
            if getattr(main_win, 'font_groups', None):
                # Clear default 'All' then add groups
                self.font_group_combo.clear(); self.font_group_combo.addItem('All')
                for g in main_win.font_groups.keys():
                    self.font_group_combo.addItem(g)
            # connect group changes to a local handler
            self.font_group_combo.currentTextChanged.connect(self._on_dialog_font_group_changed)
        except Exception:
            pass

        # Populate the font combo initially (respecting any selected group)
        try:
            sel = self.font_group_combo.currentText()
            if sel == 'All':
                self._populate_dialog_fonts(None)
            else:
                self._populate_dialog_fonts(sel)
        except Exception:
            pass

        self.emoji_button = QToolButton(); self.emoji_button.setText("Emotes"); self.emoji_button.setToolTip("Insert emoticons or symbols")
        self.emoji_menu = QMenu(self)
        for label, symbol in self.EMOJI_PRESETS:
            action = self.emoji_menu.addAction(f"{label} {symbol}")
            action.triggered.connect(lambda checked=False, s=symbol: self._insert_emoji(s))
        self.emoji_button.setMenu(self.emoji_menu); self.emoji_button.setPopupMode(QToolButton.InstantPopup)
        toolbar_layout.addWidget(self.emoji_button)

        toolbar_layout.addStretch()

        # Scrollable container so the bottom buttons stay reachable on smaller screens
        content_scroll = QScrollArea()
        content_scroll.setWidgetResizable(True)
        content_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(10)

        content_layout.addLayout(toolbar_layout)

        self.text_edit = QTextEdit(); self.text_edit.setAcceptRichText(True); self.text_edit.setMinimumHeight(240)
        content_layout.addWidget(self.text_edit, 1)
        self._outline_color = self.area.get_text_outline_color() if hasattr(self.area, 'get_text_outline_color') else QColor('#000000')
        self._last_text_cursor = QTextCursor(self.text_edit.document())
        self.text_edit.cursorPositionChanged.connect(self._remember_cursor_state)
        self.text_edit.cursorPositionChanged.connect(self._sync_toolbar_from_cursor)

        layout_group = QGroupBox("Layout & Effects")
        layout_grid = QGridLayout(layout_group)

        self.orientation_combo = QComboBox(); self.orientation_combo.addItems(["Horizontal", "Vertical"])
        layout_grid.addWidget(QLabel("Orientation:"), 0, 0); layout_grid.addWidget(self.orientation_combo, 0, 1)

        self.effect_combo = QComboBox()
        for label, value in self.EFFECT_OPTIONS:
            self.effect_combo.addItem(label, value)
        layout_grid.addWidget(QLabel("Effect:"), 0, 2); layout_grid.addWidget(self.effect_combo, 0, 3)

        self.effect_intensity_spin = QDoubleSpinBox(); self.effect_intensity_spin.setRange(0.0, 300.0); self.effect_intensity_spin.setDecimals(1); self.effect_intensity_spin.setSingleStep(5.0)
        layout_grid.addWidget(QLabel("Effect Strength:"), 1, 0); layout_grid.addWidget(self.effect_intensity_spin, 1, 1)

        self.alignment_combo = QComboBox()
        for label, _ in self.ALIGN_OPTIONS:
            self.alignment_combo.addItem(label)
        layout_grid.addWidget(QLabel("Alignment:"), 1, 2); layout_grid.addWidget(self.alignment_combo, 1, 3)

        self.line_spacing_spin = QDoubleSpinBox(); self.line_spacing_spin.setRange(0.6, 3.0); self.line_spacing_spin.setSingleStep(0.1); self.line_spacing_spin.setValue(1.0)
        layout_grid.addWidget(QLabel("Line Spacing:"), 2, 0); layout_grid.addWidget(self.line_spacing_spin, 2, 1)

        self.char_spacing_spin = QDoubleSpinBox(); self.char_spacing_spin.setRange(50.0, 400.0); self.char_spacing_spin.setSingleStep(5.0); self.char_spacing_spin.setSuffix(" %")
        layout_grid.addWidget(QLabel("Character Spacing:"), 2, 2); layout_grid.addWidget(self.char_spacing_spin, 2, 3)

        self.bubble_checkbox = QCheckBox("Render bubble (white fill, black outline)")
        layout_grid.addWidget(self.bubble_checkbox, 3, 0, 1, 4)

        content_layout.addWidget(layout_group)

        outline_group = QGroupBox("Text Outline & Glow")
        outline_layout = QGridLayout(outline_group)
        self.text_outline_checkbox = QCheckBox("Enable outline / glow")
        outline_layout.addWidget(self.text_outline_checkbox, 0, 0, 1, 4)

        self.outline_style_combo = QComboBox()
        self.outline_style_combo.addItems(["Stroke", "Glow"])
        outline_layout.addWidget(QLabel("Style:"), 1, 0)
        outline_layout.addWidget(self.outline_style_combo, 1, 1)

        self.outline_width_spin = QDoubleSpinBox()
        self.outline_width_spin.setRange(0.0, 30.0)
        self.outline_width_spin.setSingleStep(0.5)
        self.outline_width_spin.setDecimals(1)
        outline_layout.addWidget(QLabel("Width / Glow Radius:"), 1, 2)
        outline_layout.addWidget(self.outline_width_spin, 1, 3)

        self.outline_color_button = QPushButton("Outline Color")
        outline_layout.addWidget(QLabel("Color:"), 2, 0)
        outline_layout.addWidget(self.outline_color_button, 2, 1, 1, 3)

        glow_hint = QLabel("Tip: choose Glow for soft halos, Stroke for crisp comic outlines.")
        glow_hint.setStyleSheet("color: #9bb3cf; font-size: 10.2pt;")
        glow_hint.setWordWrap(True)
        outline_layout.addWidget(glow_hint, 3, 0, 1, 4)

        content_layout.addWidget(outline_group)

        # Gradient Group
        gradient_group = QGroupBox("Gradient Fill")
        gradient_layout = QGridLayout(gradient_group)
        self.gradient_enabled_checkbox = QCheckBox("Enable Gradient")
        gradient_layout.addWidget(self.gradient_enabled_checkbox, 0, 0, 1, 4)

        self.gradient_color1_btn = QPushButton("Start Color")
        self.gradient_color2_btn = QPushButton("End Color")
        gradient_layout.addWidget(QLabel("Colors:"), 1, 0)
        gradient_layout.addWidget(self.gradient_color1_btn, 1, 1)
        gradient_layout.addWidget(self.gradient_color2_btn, 1, 2)

        self.gradient_angle_spin = QDoubleSpinBox()
        self.gradient_angle_spin.setRange(0.0, 360.0)
        self.gradient_angle_spin.setSingleStep(15.0)
        self.gradient_angle_spin.setSuffix(" °")
        self.gradient_angle_spin.setSuffix(" °")
        
        self.gradient_direction_combo = QComboBox()
        for label, _ in self.GRADIENT_DIRECTIONS:
            self.gradient_direction_combo.addItem(label)
            
        gradient_layout.addWidget(QLabel("Angle:"), 1, 3)
        gradient_layout.addWidget(self.gradient_angle_spin, 1, 4)
        gradient_layout.addWidget(QLabel("Direction:"), 1, 5)
        gradient_layout.addWidget(self.gradient_direction_combo, 1, 6)
        
        content_layout.addWidget(gradient_group)

        margin_group = QGroupBox("Inner Margins (px)")
        margin_grid = QGridLayout(margin_group)
        self.margin_top_spin = QSpinBox(); self.margin_top_spin.setRange(0, 400)
        self.margin_right_spin = QSpinBox(); self.margin_right_spin.setRange(0, 400)
        self.margin_bottom_spin = QSpinBox(); self.margin_bottom_spin.setRange(0, 400)
        self.margin_left_spin = QSpinBox(); self.margin_left_spin.setRange(0, 400)
        margin_grid.addWidget(QLabel("Top:"), 0, 0); margin_grid.addWidget(self.margin_top_spin, 0, 1)
        margin_grid.addWidget(QLabel("Right:"), 0, 2); margin_grid.addWidget(self.margin_right_spin, 0, 3)
        margin_grid.addWidget(QLabel("Bottom:"), 1, 0); margin_grid.addWidget(self.margin_bottom_spin, 1, 1)
        margin_grid.addWidget(QLabel("Left:"), 1, 2); margin_grid.addWidget(self.margin_left_spin, 1, 3)
        content_layout.addWidget(margin_group)

        bezier_group = QGroupBox("Bezier Control Points (0.0 - 1.0)")
        bezier_layout = QGridLayout(bezier_group)
        self.cp1x_spin = self._create_bezier_spin(); self.cp1y_spin = self._create_bezier_spin()
        self.cp2x_spin = self._create_bezier_spin(); self.cp2y_spin = self._create_bezier_spin()
        bezier_layout.addWidget(QLabel("Control 1 X:"), 0, 0); bezier_layout.addWidget(self.cp1x_spin, 0, 1)
        bezier_layout.addWidget(QLabel("Control 1 Y:"), 0, 2); bezier_layout.addWidget(self.cp1y_spin, 0, 3)
        bezier_layout.addWidget(QLabel("Control 2 X:"), 1, 0); bezier_layout.addWidget(self.cp2x_spin, 1, 1)
        bezier_layout.addWidget(QLabel("Control 2 Y:"), 1, 2); bezier_layout.addWidget(self.cp2y_spin, 1, 3)
        content_layout.addWidget(bezier_group)
        content_layout.addStretch(1)

        content_scroll.setWidget(content_widget)
        main_layout.addWidget(content_scroll, 1)

        button_box = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Save)
        button_box.button(QDialogButtonBox.Save).setText("Apply")
        button_box.accepted.connect(self._handle_accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

        # Revert to global defaults button
        self.revert_button = QPushButton("Revert to Global Defaults")
        self.revert_button.setToolTip("Clear per-area overrides and use global defaults for this area.")
        def _on_revert_clicked():
            notes = self.area.review_notes if isinstance(getattr(self.area, 'review_notes', {}), dict) else {}
            for legacy in ('manual_inpaint', 'manual'):
                if legacy in notes:
                    notes.pop(legacy, None)
            self.area.review_notes = notes
            if hasattr(self.area, 'clear_override'):
                self.area.clear_override('use_inpaint')
                self.area.clear_override('use_background_box')
            parent = self.parent()
            if parent and hasattr(parent, 'redraw_all_typeset_areas'):
                try:
                    parent.redraw_all_typeset_areas()
                except Exception:
                    pass
            if parent and getattr(parent, 'selected_typeset_area', None) is self.area:
                try:
                    parent._sync_cleanup_controls_from_selection()
                except Exception:
                    pass
            QMessageBox.information(self, "Reverted", "This area's overrides have been cleared and global defaults will be used.")
        self.revert_button.clicked.connect(_on_revert_clicked)
        main_layout.addWidget(self.revert_button)

        self.gradient_enabled_checkbox.toggled.connect(self._update_gradient_ui_state)
        self.gradient_color1_btn.clicked.connect(lambda: self._choose_gradient_color(0))
        self.gradient_color2_btn.clicked.connect(lambda: self._choose_gradient_color(1))
        self.gradient_direction_combo.currentIndexChanged.connect(self._on_gradient_direction_changed)
        self.gradient_angle_spin.valueChanged.connect(self._on_gradient_angle_changed)
        
        self.gradient_colors_store = ["#FF0000", "#0000FF"]

        self.font_combo.currentTextChanged.connect(self._change_font_family)
        self.font_size_spin.valueChanged.connect(self._change_font_size)
        self.bold_button.toggled.connect(self._toggle_bold)
        self.italic_button.toggled.connect(self._toggle_italic)
        self.underline_button.toggled.connect(self._toggle_underline)
        self.color_button.clicked.connect(self._choose_color)
        self.text_edit.cursorPositionChanged.connect(self._sync_toolbar_from_cursor)
        self.alignment_combo.currentIndexChanged.connect(self._apply_alignment)
        self.line_spacing_spin.valueChanged.connect(self._apply_line_spacing)
        self.char_spacing_spin.valueChanged.connect(self._apply_char_spacing)
        self.text_outline_checkbox.toggled.connect(self._update_outline_controls_enabled)
        self.outline_style_combo.currentIndexChanged.connect(self._update_outline_controls_enabled)
        self.outline_width_spin.valueChanged.connect(self._update_outline_controls_enabled)
        self.outline_width_spin.valueChanged.connect(self._update_outline_controls_enabled)
        self.outline_color_button.clicked.connect(self._choose_outline_color)

        self._populate_font_combo()
        self._load_area_into_editor()
        self._sync_toolbar_from_cursor()

    def _update_gradient_ui_state(self):
        enabled = self.gradient_enabled_checkbox.isChecked()
        self.gradient_color1_btn.setEnabled(enabled)
        self.gradient_color2_btn.setEnabled(enabled)
        self.gradient_angle_spin.setEnabled(enabled)
        self.gradient_direction_combo.setEnabled(enabled)

    def _on_gradient_direction_changed(self):
        idx = self.gradient_direction_combo.currentIndex()
        if idx < 0: return
        label, angle = self.GRADIENT_DIRECTIONS[idx]
        if angle != -1:
            with QSignalBlocker(self.gradient_angle_spin):
                self.gradient_angle_spin.setValue(angle)

    def _on_gradient_angle_changed(self):
        val = self.gradient_angle_spin.value()
        # Find close match
        best_idx = 0 # Custom
        for i, (label, angle) in enumerate(self.GRADIENT_DIRECTIONS):
            if angle != -1 and abs(angle - val) < 1.0:
                best_idx = i
                break
            # Check for 360/0 equivalence
            if angle == 0.0 and abs(360.0 - val) < 1.0:
                best_idx = i
                break
        
        with QSignalBlocker(self.gradient_direction_combo):
            self.gradient_direction_combo.setCurrentIndex(best_idx)

    def _choose_gradient_color(self, index):
        current_hex = self.gradient_colors_store[index] if 0 <= index < len(self.gradient_colors_store) else "#000000"
        color = QColorDialog.getColor(QColor(current_hex), self, "Select Gradient Color")
        if color.isValid():
            if index < len(self.gradient_colors_store):
                self.gradient_colors_store[index] = color.name()
            else:
                self.gradient_colors_store.append(color.name())
            self._update_gradient_buttons_style()

    def _update_gradient_buttons_style(self):
        c1 = self.gradient_colors_store[0]
        c2 = self.gradient_colors_store[1] if len(self.gradient_colors_store) > 1 else c1
        self.gradient_color1_btn.setStyleSheet(f"background-color: {c1}; color: {'#000000' if QColor(c1).lightness() > 128 else '#ffffff'}; border: 1px solid #555;")
        self.gradient_color2_btn.setStyleSheet(f"background-color: {c2}; color: {'#000000' if QColor(c2).lightness() > 128 else '#ffffff'}; border: 1px solid #555;")

    def _apply_default_size(self):
        try:
            screen = QApplication.primaryScreen()
            if screen:
                geo = screen.availableGeometry()
                # Lebarkan hingga 85% layar
                self.resize(int(geo.width() * 0.85), int(geo.height() * 0.85))
            else:
                self.resize(1200, 800)
        except Exception:
            self.resize(1200, 800)

    def closeEvent(self, event):
        try:
            s = self.size()
            SETTINGS.setdefault('ui', {})['advanced_editor_size'] = (s.width(), s.height())
        except Exception:
            pass
        super().closeEvent(event)

    def _create_bezier_spin(self):
        spin = QDoubleSpinBox()
        spin.setRange(-1.0, 2.0)
        spin.setDecimals(3)
        spin.setSingleStep(0.05)
        return spin

    def _remember_cursor_state(self):
        try:
            cursor = self.text_edit.textCursor()
            if self._is_cursor_valid(cursor):
                self._last_text_cursor = QTextCursor(cursor)
            else:
                self._last_text_cursor = QTextCursor(self.text_edit.document())
        except Exception:
            traceback.print_exc()

    def _is_cursor_valid(self, cursor: QTextCursor | None) -> bool:
        try:
            if cursor is None or cursor.isNull():
                return False
            doc = self.text_edit.document()
            if cursor.document() is not doc:
                return False
            pos = cursor.position()
            anchor = cursor.anchor()
            length = doc.characterCount()
            if pos < 0 or anchor < 0 or pos > length or anchor > length:
                return False
            return True
        except Exception:
            return False

    def _populate_font_combo(self):
        if not self.font_manager:
            return
        fonts = self.font_manager.list_fonts()
        with QSignalBlocker(self.font_combo):
            self.font_combo.clear()
            for name in fonts:
                self.font_combo.addItem(name)
                preview_font = self.font_manager.create_qfont(name)
                preview_font.setPointSize(16)
                idx = self.font_combo.count() - 1
                self.font_combo.setItemData(idx, preview_font, Qt.FontRole)

    def _update_font_preview(self, display_name):
        if not self.font_manager or not display_name:
            return
        base_font = self.text_edit.currentCharFormat().font()
        if not base_font.family():
            base_font = self.area.get_font()
        preview_font = self.font_manager.create_qfont(display_name, base_font=base_font)
        preview_font.setPointSizeF(self.font_size_spin.value())
        preview_font.setWeight(base_font.weight())
        preview_font.setItalic(base_font.italic())
        self.font_preview.setFont(preview_font)
        self.font_preview.setToolTip(display_name)

    def _load_area_into_editor(self):
        self.text_edit.clear()
        cursor = QTextCursor(self.text_edit.document())
        cursor.movePosition(QTextCursor.Start)

        segments = self.area.get_segments()
        if not segments:
            segments = [{'text': self.area.text or '', 'font': self.area.font_to_dict(self.area.get_font()), 'color': self.area.get_color().name(), 'underline': False}]

        for segment in segments:
            text_value = segment.get('text', '')
            if not text_value:
                continue
            fmt = QTextCharFormat()
            seg_font = self.area.segment_to_qfont(segment)
            fmt.setFont(seg_font)
            fmt.setForeground(QBrush(self.area.segment_to_color(segment)))
            if segment.get('underline', seg_font.underline()):
                fmt.setFontUnderline(True)

            parts = text_value.split('\n')
            for idx, part in enumerate(parts):
                cursor.insertText(part, fmt)
                if idx < len(parts) - 1:
                    cursor.insertBlock()

        orientation = self.area.get_orientation()
        with QSignalBlocker(self.orientation_combo):
            self.orientation_combo.setCurrentIndex(1 if orientation == 'vertical' else 0)

        current_effect = self.area.get_effect()
        effect_idx = next((i for i, (_, val) in enumerate(self.EFFECT_OPTIONS) if val == current_effect), 0)
        with QSignalBlocker(self.effect_combo):
            self.effect_combo.setCurrentIndex(effect_idx)

        # [FIX] Load Margins
        margins = self.area.get_margins()
        if margins:
            self.margin_top_spin.setValue(int(margins.get('top', 0)))
            self.margin_right_spin.setValue(int(margins.get('right', 0)))
            self.margin_bottom_spin.setValue(int(margins.get('bottom', 0)))
            self.margin_left_spin.setValue(int(margins.get('left', 0)))

        # [FIX] Load Alignment
        align_val = self.area.get_alignment()
        if align_val:
            # Match "left", "center", "right"
            idx_align = next((i for i, (lbl, _) in enumerate(self.ALIGN_OPTIONS) if lbl.lower().startswith(align_val.lower())), 1)
            with QSignalBlocker(self.alignment_combo):
                self.alignment_combo.setCurrentIndex(idx_align)
        
        # [FIX] Load Spacing
        self.line_spacing_spin.setValue(float(self.area.get_line_spacing()))
        self.char_spacing_spin.setValue(float(self.area.get_char_spacing()))

        # [FIX] Load Text Outline
        has_outline = bool(self.area.has_text_outline() if hasattr(self.area, 'has_text_outline') else False)
        self.text_outline_checkbox.setChecked(has_outline)
        
        outline_width = float(self.area.get_text_outline_width() if hasattr(self.area, 'get_text_outline_width') else 2.0)
        self.outline_width_spin.setValue(outline_width)
        
        outline_style = self.area.get_text_outline_style() if hasattr(self.area, 'get_text_outline_style') else 'stroke'
        style_idx = 1 if outline_style == 'glow' else 0
        with QSignalBlocker(self.outline_style_combo):
            self.outline_style_combo.setCurrentIndex(style_idx)
            
        outline_color_val = self.area.get_text_outline_color() if hasattr(self.area, 'get_text_outline_color') else '#000000'
        self._outline_color = QColor(outline_color_val) if isinstance(outline_color_val, (str, QColor)) else QColor('#000000')
        self._update_outline_color_button_ui(self._outline_color)
        self._update_outline_controls_enabled()

        self.effect_intensity_spin.setValue(self.area.get_effect_intensity())
        self.bubble_checkbox.setChecked(bool(getattr(self.area, 'bubble_enabled', False)))

        bezier = self.area.get_bezier_points()
        if len(bezier) >= 2:
            self.cp1x_spin.setValue(bezier[0].get('x', 0.25)); self.cp1y_spin.setValue(bezier[0].get('y', 0.2))
            self.cp2x_spin.setValue(bezier[1].get('x', 0.75)); self.cp2y_spin.setValue(bezier[1].get('y', 0.2))

        # [NEW] Load Gradient State
        self.gradient_enabled_checkbox.setChecked(bool(self.area.get_extra('gradient_enabled')))
        
        grad_colors = self.area.get_extra('gradient_colors')
        if isinstance(grad_colors, list) and len(grad_colors) >= 2:
            self.gradient_colors_store = list(grad_colors)
        else:
            self.gradient_colors_store = ["#FF0000", "#0000FF"]
        self._update_gradient_buttons_style()
        
        self.gradient_angle_spin.setValue(float(self.area.get_extra('gradient_angle') or 0.0))
        
        direction_str = self.area.get_extra('gradient_direction')
        if direction_str:
            self.gradient_direction_combo.setCurrentText(direction_str)


    def _on_ai_translate_clicked(self):
        parent = self.parent()
        if not parent: return

        # Get text to translate: selection or all text
        cursor = self.text_edit.textCursor()
        selected_text = cursor.selectedText()
        if not selected_text:
            text_to_translate = self.text_edit.toPlainText()
            is_selection = False
        else:
            text_to_translate = selected_text
            is_selection = True
        
        if not text_to_translate.strip():
            return

        provider, model_name = parent.get_selected_model_name()
        if not model_name:
            QMessageBox.warning(self, "No Model", "Please select an AI model in the main window first.")
            return

        self.ai_translate_btn.setEnabled(False)
        self.ai_translate_btn.setText("Translating...")
        QApplication.processEvents()

        try:
            # We don't have style context easily available, so pass empty style
            # Unless we want to try to infer it from the area, but let's keep it simple
            translated = parent.translate_with_ai(text_to_translate, {}, provider, model_name, {})
            
            # Replace text
            if is_selection:
                cursor.insertText(str(translated))
            else:
                self.text_edit.setPlainText(str(translated))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Translation failed: {str(e)}")
        finally:
            self.ai_translate_btn.setEnabled(True)
            self.ai_translate_btn.setText("AI Translate")

        margins = self.area.get_margins()
        self.margin_top_spin.setValue(int(margins.get('top', 0)))
        self.margin_right_spin.setValue(int(margins.get('right', 0)))
        self.margin_bottom_spin.setValue(int(margins.get('bottom', 0)))
        self.margin_left_spin.setValue(int(margins.get('left', 0)))

        align_value = self.area.get_alignment()
        align_idx = next((i for i, (label, _) in enumerate(self.ALIGN_OPTIONS) if label.lower().startswith(align_value)), 0)
        with QSignalBlocker(self.alignment_combo):
            self.alignment_combo.setCurrentIndex(align_idx)
        self._apply_alignment()

        self.line_spacing_spin.setValue(self.area.get_line_spacing())
        self.char_spacing_spin.setValue(self.area.get_char_spacing())
        self._apply_line_spacing()
        self._apply_char_spacing()
        self.text_outline_checkbox.setChecked(bool(self.area.has_text_outline() if hasattr(self.area, 'has_text_outline') else False))
        style_value = 'stroke'
        try:
            style_value = self.area.get_text_outline_style()
        except Exception:
            style_value = 'stroke'
        style_idx = 0 if style_value == 'stroke' else 1
        with QSignalBlocker(self.outline_style_combo):
            self.outline_style_combo.setCurrentIndex(style_idx)
        with QSignalBlocker(self.outline_width_spin):
            self.outline_width_spin.setValue(float(self.area.get_text_outline_width() if hasattr(self.area, 'get_text_outline_width') else 2.0))
        self._outline_color = self.area.get_text_outline_color() if hasattr(self.area, 'get_text_outline_color') else QColor('#000000')
        self._update_outline_color_button_ui(self._outline_color)
        self._update_outline_color_button_ui(self._outline_color)
        self._update_outline_controls_enabled()
        
        # Load Gradient
        g_enabled = getattr(self.area, 'gradient_enabled', False)
        g_colors = getattr(self.area, 'gradient_colors', None) or ["#ff0000", "#0000ff"]
        g_angle = getattr(self.area, 'gradient_angle', 0.0)
        
        with QSignalBlocker(self.gradient_enabled_checkbox):
            self.gradient_enabled_checkbox.setChecked(bool(g_enabled))
        self.gradient_colors_store = list(g_colors)
        if len(self.gradient_colors_store) < 2: self.gradient_colors_store = ["#ff0000", "#0000ff"]
        with QSignalBlocker(self.gradient_angle_spin):
            self.gradient_angle_spin.setValue(float(g_angle))
            
        # Sync combo
        self._on_gradient_angle_changed()
            
        self._update_gradient_buttons_style()
        self._update_gradient_ui_state()

        base_font = self.area.get_font()
        display_name = None
        if self.font_manager:
            display_name = self.font_manager.display_name_for_font(base_font)
        if not display_name:
            display_name = base_font.family()
        with QSignalBlocker(self.font_combo):
            if display_name:
                self.font_combo.setCurrentText(display_name)
        with QSignalBlocker(self.font_size_spin):
            self.font_size_spin.setValue(base_font.pointSizeF() or base_font.pointSize())
        self._update_font_preview(display_name)
        self._last_text_cursor = QTextCursor(self.text_edit.document())

    def _insert_emoji(self, text):
        try:
            if not text:
                return
            cursor = self.text_edit.textCursor()
            if not self._is_cursor_valid(cursor):
                cursor = QTextCursor(self.text_edit.document())
                cursor.movePosition(QTextCursor.End)
            cursor.insertText(text)
            self.text_edit.setTextCursor(cursor)
            self._last_text_cursor = QTextCursor(cursor)
        except Exception:
            traceback.print_exc()

    def _populate_dialog_fonts(self, group: str | None = None):
        """Populate the dialog font combo, optionally filtering by a group name provided by the main window."""
        try:
            fonts = self.font_manager.list_fonts() if self.font_manager else []
            if group and getattr(self.parent(), 'font_groups', None):
                allowed = set(self.parent().font_groups.get(group, []))
                fonts = [f for f in fonts if f in allowed]
            with QSignalBlocker(self.font_combo):
                self.font_combo.clear()
                for name in fonts:
                    self.font_combo.addItem(name)
                    preview_font = self.font_manager.create_qfont(name) if self.font_manager else QFont(name)
                    preview_font.setPointSize(14)
                    idx = self.font_combo.count() - 1
                    self.font_combo.setItemData(idx, preview_font, Qt.FontRole)
        except Exception:
            traceback.print_exc()

    def _on_dialog_font_group_changed(self, group_name: str):
        if group_name == 'All':
            self._populate_dialog_fonts(None)
        else:
            self._populate_dialog_fonts(group=group_name)

    def _merge_char_format(self, fmt):
        if fmt is None:
            return
        try:
            cursor = self.text_edit.textCursor()
            if cursor is None or cursor.isNull():
                return
            if not cursor.hasSelection():
                stored = getattr(self, '_last_text_cursor', None)
                if self._is_cursor_valid(stored) and stored.hasSelection():
                    cursor = QTextCursor(stored)
                    self.text_edit.setTextCursor(cursor)
                else:
                    cursor.select(QTextCursor.WordUnderCursor)
            self.text_edit.setTextCursor(cursor)
            cursor.mergeCharFormat(fmt)
            self.text_edit.mergeCurrentCharFormat(fmt)
            self.text_edit.ensureCursorVisible()
            self._last_text_cursor = QTextCursor(self.text_edit.textCursor())
        except Exception:
            traceback.print_exc()

    def _change_font_family(self, display_name):
        try:
            if not display_name:
                return
            cursor_font = self.text_edit.currentCharFormat().font()
            if not cursor_font.family():
                cursor_font = self.area.get_font()
            if self.font_manager:
                new_font = self.font_manager.create_qfont(display_name, base_font=cursor_font)
            else:
                new_font = QFont(cursor_font)
                new_font.setFamily(display_name)
            new_font.setPointSizeF(self.font_size_spin.value())
            new_font.setLetterSpacing(cursor_font.letterSpacingType(), cursor_font.letterSpacing())
            new_font.setWeight(cursor_font.weight())
            new_font.setItalic(cursor_font.italic())
            new_font.setUnderline(cursor_font.underline())
            fmt = QTextCharFormat()
            fmt.setFont(new_font)
            self._merge_char_format(fmt)
            self._update_font_preview(display_name)
            self._last_text_cursor = QTextCursor(self.text_edit.textCursor())
        except Exception:
            traceback.print_exc()

    def _change_font_size(self, value):
        try:
            fmt = QTextCharFormat()
            fmt.setFontPointSize(float(value))
            self._merge_char_format(fmt)
            current = self.font_combo.currentText()
            if current:
                self._update_font_preview(current)
            self._last_text_cursor = QTextCursor(self.text_edit.textCursor())
        except Exception:
            traceback.print_exc()

    def _toggle_bold(self, checked):
        try:
            fmt = QTextCharFormat()
            fmt.setFontWeight(QFont.Bold if checked else QFont.Normal)
            self._merge_char_format(fmt)
            self._last_text_cursor = QTextCursor(self.text_edit.textCursor())
        except Exception:
            traceback.print_exc()

    def _toggle_italic(self, checked):
        try:
            fmt = QTextCharFormat()
            fmt.setFontItalic(checked)
            self._merge_char_format(fmt)
            self._last_text_cursor = QTextCursor(self.text_edit.textCursor())
        except Exception:
            traceback.print_exc()

    def _toggle_underline(self, checked):
        try:
            fmt = QTextCharFormat()
            fmt.setFontUnderline(checked)
            self._merge_char_format(fmt)
            self._last_text_cursor = QTextCursor(self.text_edit.textCursor())
        except Exception:
            traceback.print_exc()

    def _choose_color(self):
        try:
            # User Feedback: ensure gradient logic is handled. 
            # If gradient is enabled, we use the first gradient color as initial, or cursor color
            initial = self._current_color_from_cursor()
            if self.gradient_enabled_checkbox.isChecked() and self.gradient_colors_store:
                 initial = QColor(self.gradient_colors_store[0])
            
            color = QColorDialog.getColor(initial, self, "Select Text Color")
            if color.isValid():
                # If user picks a solid color, disable gradient to avoid confusion/overrides
                if self.gradient_enabled_checkbox.isChecked():
                    self.gradient_enabled_checkbox.setChecked(False)
                    
                fmt = QTextCharFormat()
                fmt.setForeground(QBrush(color))
                self._merge_char_format(fmt)
                self._update_color_button(color)
                self._last_text_cursor = QTextCursor(self.text_edit.textCursor())
        except Exception:
            traceback.print_exc()

    def _current_color_from_cursor(self):
        fmt = self.text_edit.currentCharFormat()
        brush = fmt.foreground()
        return brush.color() if brush.style() != Qt.NoBrush else self.area.get_color()

    def _update_color_button(self, color):
        try:
            if color.isValid():
                self.color_button.setStyleSheet(f"background-color: {color.name()}; border: 1px solid #222; color: white;")
            else:
                self.color_button.setStyleSheet("")
        except Exception:
            traceback.print_exc()

    def _update_outline_color_button_ui(self, color: QColor | None = None):
        if not getattr(self, 'outline_color_button', None):
            return
        if color is None:
            color = self._outline_color if isinstance(getattr(self, '_outline_color', None), QColor) else QColor('#000000')
        if not color.isValid():
            color = QColor('#000000')
        self._outline_color = color
        try:
            luminance = 0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()
            text_color = '#000000' if luminance > 160 else '#f3f6fb'
            self.outline_color_button.setStyleSheet(
                f"QPushButton {{ background-color: {color.name()}; color: {text_color}; border: 1px solid #1f2b36; border-radius: 8px; padding: 6px 10px; }}"
                " QPushButton:hover { border-color: #3a9bff; }"
            )
        except Exception:
            traceback.print_exc()

    def _choose_outline_color(self):
        try:
            color = QColorDialog.getColor(self._outline_color, self, "Select Outline/Glow Color")
            if color.isValid():
                self._outline_color = color
                self._update_outline_color_button_ui(color)
        except Exception:
            traceback.print_exc()

    def _update_outline_controls_enabled(self):
        enabled = self.text_outline_checkbox.isChecked()
        self.outline_style_combo.setEnabled(enabled)
        self.outline_width_spin.setEnabled(enabled)
        self.outline_color_button.setEnabled(enabled)
        # Give quick visual cue when glow is selected by dimming the button text slightly
        if enabled and self.outline_style_combo.currentText().lower().startswith('glow'):
            self.outline_color_button.setText("Glow Color")
        else:
            self.outline_color_button.setText("Outline Color")

    def _apply_alignment(self):
        try:
            doc = self.text_edit.document()
            original = self.text_edit.textCursor()
            cursor = QTextCursor(doc)
            cursor.select(QTextCursor.Document)
            block_format = QTextBlockFormat()
            align_label, align_flag = self.ALIGN_OPTIONS[self.alignment_combo.currentIndex()]
            block_format.setAlignment(align_flag)
            block_format.setLineHeight(int(self.line_spacing_spin.value() * 100), QTextBlockFormat.ProportionalHeight)
            cursor.setBlockFormat(block_format)
            self.text_edit.setTextCursor(original)
        except Exception:
            traceback.print_exc()

    def _apply_line_spacing(self):
        try:
            doc = self.text_edit.document()
            original = self.text_edit.textCursor()
            cursor = QTextCursor(doc)
            cursor.select(QTextCursor.Document)
            block_format = QTextBlockFormat()
            block_format.setLineHeight(int(self.line_spacing_spin.value() * 100), QTextBlockFormat.ProportionalHeight)
            _, align_flag = self.ALIGN_OPTIONS[self.alignment_combo.currentIndex()]
            block_format.setAlignment(align_flag)
            cursor.setBlockFormat(block_format)
            self.text_edit.setTextCursor(original)
        except Exception:
            traceback.print_exc()

    def _apply_char_spacing(self):
        try:
            spacing_value = self.char_spacing_spin.value()
            original = self.text_edit.textCursor()
            cursor = QTextCursor(self.text_edit.document())
            cursor.beginEditBlock()
            cursor.select(QTextCursor.Document)
            fmt = QTextCharFormat()
            base_font = self.text_edit.currentCharFormat().font()
            if not base_font.family():
                base_font = self.area.get_font()
            font = QFont(base_font)
            font.setLetterSpacing(QFont.PercentageSpacing, spacing_value)
            fmt.setFont(font)
            cursor.mergeCharFormat(fmt)
            cursor.endEditBlock()
            self.text_edit.setTextCursor(original)
        except Exception:
            traceback.print_exc()

    def _sync_toolbar_from_cursor(self):
        fmt = self.text_edit.currentCharFormat()
        current_font = fmt.font()
        if not current_font.family():
            current_font = self.area.get_font()
        display = None
        if self.font_manager:
            display = self.font_manager.display_name_for_font(current_font)
        if not display:
            display = current_font.family()
        if display:
            with QSignalBlocker(self.font_combo):
                self.font_combo.setCurrentText(display)
        point = fmt.fontPointSize() or current_font.pointSizeF() or current_font.pointSize()
        with QSignalBlocker(self.font_size_spin): self.font_size_spin.setValue(point)
        with QSignalBlocker(self.bold_button): self.bold_button.setChecked(fmt.fontWeight() >= QFont.Bold)
        with QSignalBlocker(self.italic_button): self.italic_button.setChecked(fmt.fontItalic())
        with QSignalBlocker(self.underline_button): self.underline_button.setChecked(fmt.fontUnderline())
        self._update_color_button(self._current_color_from_cursor())
        self._update_color_button(self._current_color_from_cursor())
        if display:
            self._update_font_preview(display)
        
        # Adaptive background
        text_col = self._current_color_from_cursor()
        bg_col = "#0c121d" # dark default
        text_col_q = QColor(text_col) if isinstance(text_col, str) or isinstance(text_col, QColor) else None
        if text_col_q and text_col_q.isValid() and text_col_q.lightness() < 128:
            bg_col = "#f0f0f0" # light background for dark text
        
        col_str = text_col if text_col else '#e8eef7'
        if isinstance(col_str, QColor):
            col_str = col_str.name()
            
        self.text_edit.setStyleSheet(f"QTextEdit {{ background: {bg_col}; border-radius: 12px; padding: 10px; color: {col_str}; }}")

    def _extract_segments(self):
        from src.ui.canvas import TypesetArea
        segments = []
        try:
            doc = self.text_edit.document()
            block = doc.begin()
            while block != doc.end():
                it = block.begin()
                while not it.atEnd():
                    fragment = it.fragment()
                    if fragment.isValid():
                        text = fragment.text()
                        if text:
                            fmt = fragment.charFormat()
                            font = fmt.font()
                            segments.append({
                                'text': text,
                                'font': TypesetArea.font_to_dict(font),
                                'color': fmt.foreground().color().name() if fmt.foreground().color().isValid() else self.area.get_color().name(),
                                'underline': fmt.fontUnderline(),
                            })
                    it += 1
                block = block.next()
                if block != doc.end():
                    segments.append({'text': '\n', 'font': TypesetArea.font_to_dict(self.area.get_font()), 'color': self.area.get_color().name(), 'underline': False})
        except Exception:
            traceback.print_exc()
        return segments

    def _handle_accept(self):
        try:
            segments = self._extract_segments()
            plain_text = ''.join(seg.get('text', '') for seg in segments)
            orientation = 'vertical' if self.orientation_combo.currentIndex() == 1 else 'horizontal'
            effect = self.effect_combo.currentData()
            bezier_data = [
                {'x': self.cp1x_spin.value(), 'y': self.cp1y_spin.value()},
                {'x': self.cp2x_spin.value(), 'y': self.cp2y_spin.value()},
            ]
            margins = {
                'top': self.margin_top_spin.value(),
                'right': self.margin_right_spin.value(),
                'bottom': self.margin_bottom_spin.value(),
                'left': self.margin_left_spin.value(),
            }
            align_label, _ = self.ALIGN_OPTIONS[self.alignment_combo.currentIndex()]
            self.result = {
                'segments': segments,
                'plain_text': plain_text,
                'orientation': orientation,
                'effect': effect,
                'effect_intensity': self.effect_intensity_spin.value(),
                'bezier_points': bezier_data,
                'bubble_enabled': self.bubble_checkbox.isChecked(),
                'alignment': align_label.lower(),
                'line_spacing': self.line_spacing_spin.value(),
                'char_spacing': self.char_spacing_spin.value(),
                'margins': margins,
                'text_outline': self.text_outline_checkbox.isChecked(),
                'text_outline_width': self.outline_width_spin.value(),
                'text_outline_color': self._outline_color.name() if isinstance(self._outline_color, QColor) else str(self._outline_color),
                'text_outline_style': self.outline_style_combo.currentText().lower(),
                'gradient_enabled': self.gradient_enabled_checkbox.isChecked(),
                'gradient_colors': self.gradient_colors_store,
                'gradient_angle': self.gradient_angle_spin.value(),
                'gradient_direction': self.gradient_direction_combo.currentText(),
            }
            self.accept()
            self.accept()
            self._last_text_cursor = QTextCursor(self.text_edit.document())
        except Exception as e:
            traceback.print_exc()
            QMessageBox.warning(self, "Apply Failed", f"Failed to apply text changes: {str(e)}")

    def get_result(self):
        return self.result



class SceneReviewDialog(QDialog):
    def __init__(self, parent, original_items, ai_proposals):
        super().__init__(parent)
        self.setWindowTitle("Review AI Changes")
        self.resize(900, 600)
        self.accepted_indices = []
        
        layout = QVBoxLayout(self)
        
        # Info
        layout.addWidget(QLabel("Review the proposed changes below. Uncheck items you want to skip."))
        
        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["#", "Original / Current", "AI Proposal", "Apply"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        
        # Populate
        # original_items is [Text 1 (Old), Text 2, ..., Text N (New)]
        # ai_proposals is matching list
        row_count = min(len(original_items), len(ai_proposals))
        self.table.setRowCount(row_count)
        
        for i in range(row_count):
            orig = original_items[i]
            prop = ai_proposals[i]
            
            # Col 0: Index
            self.table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            
            # Col 1: Original
            orig_item = QTableWidgetItem(orig)
            orig_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.table.setItem(i, 1, orig_item)
            
            # Col 2: Proposal
            prop_item = QTableWidgetItem(prop)
            prop_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable)
            self.table.setItem(i, 2, prop_item)
            
            # Col 3: Checkbox
            chk_widget = QWidget()
            chk_layout = QHBoxLayout(chk_widget)
            chk_layout.setContentsMargins(0,0,0,0)
            chk_layout.setAlignment(Qt.AlignCenter)
            chk = QCheckBox()
            chk.setChecked(True)
            chk_layout.addWidget(chk)
            self.table.setCellWidget(i, 3, chk_widget)
            
        layout.addWidget(self.table)
        
        # Buttons
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.button(QDialogButtonBox.Ok).setText("Apply Selected")
        btn_box.accepted.connect(self.accept_changes)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)
        
    def accept_changes(self):
        self.accepted_indices = []
        for i in range(self.table.rowCount()):
            widget = self.table.cellWidget(i, 3)
            if widget:
                chk = widget.findChild(QCheckBox)
                if chk and chk.isChecked():
                    # Get potentially edited text
                    new_text = self.table.item(i, 2).text()
                    self.accepted_indices.append((i, new_text))
        self.accept()

class HistoryEditDialog(QDialog):
    def __init__(self, entry, styles, allow_original=True, allow_style=True, parent=None):
        super().__init__(parent)
        self.entry = entry  # Dictionary copy of the history/area data
        self.result = None
        self.setWindowTitle("Advanced Text Edit")
        self.setModal(True)
        self.resize(700, 600)

        main_layout = QVBoxLayout(self)
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # ----------------------------------------------------------------
        # Tab 1: Text & Translation
        # ----------------------------------------------------------------
        self.text_tab = QWidget()
        text_layout = QVBoxLayout(self.text_tab)
        
        info_label = QLabel("Adjust the text below.")
        text_layout.addWidget(info_label)

        original_label = QLabel("Original OCR")
        text_layout.addWidget(original_label)
        self.original_edit = QTextEdit()
        self.original_edit.setPlainText(entry.get('original_text', ''))
        self.original_edit.setMinimumHeight(100)
        if not allow_original:
            self.original_edit.setReadOnly(True)
            self.original_edit.setStyleSheet("background-color: #111824; color: #7f8ba7;")
        text_layout.addWidget(self.original_edit)

        translated_label = QLabel("Translated Text")
        text_layout.addWidget(translated_label)
        self.translated_edit = QTextEdit()
        self.translated_edit.setPlainText(entry.get('translated_text', ''))
        self.translated_edit.setMinimumHeight(120)
        text_layout.addWidget(self.translated_edit)

        translate_btn_layout = QHBoxLayout()
        translate_btn_layout.addStretch()
        self.translate_button = QPushButton("Translate")
        self.translate_button.setToolTip("Translate the OCR text using the active translation provider")
        translate_btn_layout.addWidget(self.translate_button)
        text_layout.addLayout(translate_btn_layout)
        self.translate_button.clicked.connect(self._on_translate_clicked)

        # AI Model Selector
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("AI Model:"))
        self.ai_model_combo = QComboBox()
        # Populate
        parent = self.parent()
        if parent:
             current_provider, current_model = parent.get_selected_model_name()
             if current_model:
                 self.ai_model_combo.addItem(f"{current_provider}: {current_model}")
             
             known_models = ["Gemini: gemini-1.5-flash", "Gemini: gemini-1.5-pro", "OpenAI: gpt-4o", "OpenAI: gpt-4o-mini"]
             for m in known_models:
                 if self.ai_model_combo.findText(m) == -1:
                     self.ai_model_combo.addItem(m)
        else:
             self.ai_model_combo.addItem("Gemini: gemini-1.5-flash")
        
        model_layout.addWidget(self.ai_model_combo)
        text_layout.addLayout(model_layout)

        if allow_style:
            style_layout = QHBoxLayout()
            style_label = QLabel("Translation Style")
            style_layout.addWidget(style_label)
            self.style_combo = QComboBox()
            self.style_combo.addItems(styles or [])
            current_style = entry.get('translation_style', '')
            if current_style and current_style in (styles or []):
                self.style_combo.setCurrentText(current_style)
            elif styles:
                self.style_combo.setCurrentIndex(0)
            style_layout.addWidget(self.style_combo)
            style_layout.addStretch()
            text_layout.addLayout(style_layout)
        else:
            self.style_combo = None

        self.tab_widget.addTab(self.text_tab, "Text")
        
        # Typography tab removed as requested


        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        button_box.button(QDialogButtonBox.Ok).setText("Apply")
        button_box.accepted.connect(self.handle_accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

    def _build_typography_tab(self):
        # We need access to FontManager. Currently GLOBAL_FONT_MANAGER or self.parent().font_manager
        parent = self.parent()
        self.font_manager = getattr(parent, 'font_manager', None)

        layout = QScrollArea()
        layout.setWidgetResizable(True)
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setSpacing(15)
        layout.setWidget(container)

        self.overrides = self.entry.get('overrides', {}).copy()
        # Helper to get value falling back to app defaults if not in overrides
        # But here we are editing specific area, so we should show what's currently active on the area
        # The 'entry' passed in is just a dict derived from history.
        # Ideally we want the actual TypesetArea object or enough data to reconstruct state.
        # Assuming 'entry' has keys like 'font_family', 'font_size', etc. 
        # If the entry comes from 'history', it might be sparse.
        # The caller 'open_result_editor' passes a dict from history. History doesn't store full rich typography usually.
        # Wait, if `HistoryEditDialog` is "Advanced Text Edit", we typically want to edit the active TypesetArea.
        # If this dialog is invoked from the ResultTable (history), it might not have the TypesetArea reference.
        # However, the user request implies changing these settings for the text on screen. 
        # Let's assume we can init fields from the `entry` if keys exist, or defaults.
        
        # --- Appearance: Font & Color ---
        g_font = QGroupBox("Font & Color")
        g_font_layout = QGridLayout(g_font)
        
        g_font_layout.addWidget(QLabel("Family:"), 0, 0)
        self.font_combo = QFontComboBox() 
        # Ideally use custom font manager population if possible, matching main app
        if self.font_manager:
            self.font_combo.clear()
            for f in self.font_manager.list_fonts():
                self.font_combo.addItem(f)
        current_font = self.entry.get('font_family') or "Arial"
        self.font_combo.setCurrentText(current_font)
        g_font_layout.addWidget(self.font_combo, 0, 1)

        g_font_layout.addWidget(QLabel("Size:"), 0, 2)
        self.size_spin = QDoubleSpinBox()
        self.size_spin.setRange(4, 300); self.size_spin.setValue(float(self.entry.get('font_size', 14)))
        g_font_layout.addWidget(self.size_spin, 0, 3)

        g_font_layout.addWidget(QLabel("Color:"), 1, 0)
        self.color_btn = QPushButton("Pick Color")
        self.color_val = self.entry.get('text_color', '#000000')
        self.color_btn.setStyleSheet(f"background-color: {self.color_val}; color: {'#000' if self._is_light(self.color_val) else '#fff'}")
        self.color_btn.clicked.connect(self._pick_color)
        g_font_layout.addWidget(self.color_btn, 1, 1)

        vbox.addWidget(g_font)

        # --- Gradient ---
        g_grad = QGroupBox("Gradient Coloring")
        g_grad.setCheckable(True)
        self.grad_enabled = bool(self.entry.get('gradient_enabled', False))
        g_grad.setChecked(self.grad_enabled)
        g_grad_layout = QVBoxLayout(g_grad)

        # Angle
        angle_row = QHBoxLayout()
        angle_row.addWidget(QLabel("Angle (deg):"))
        self.grad_angle_spin = QDoubleSpinBox()
        self.grad_angle_spin.setRange(0, 360)
        self.grad_angle_spin.setSingleStep(15)
        self.grad_angle_spin.setValue(float(self.entry.get('gradient_angle', 0.0)))
        angle_row.addWidget(self.grad_angle_spin)
        g_grad_layout.addLayout(angle_row)

        # Colors List
        self.grad_colors = list(self.entry.get('gradient_colors', ["#FF0000", "#0000FF"]))
        if not isinstance(self.grad_colors, list) or len(self.grad_colors) < 2:
            self.grad_colors = ["#FF0000", "#0000FF"]
        
        self.grad_list = QListWidget()
        self.grad_list.setFixedHeight(100)
        self._refresh_grad_list()
        g_grad_layout.addWidget(self.grad_list)

        grad_btns = QHBoxLayout()
        add_c_btn = QPushButton("Add Color")
        add_c_btn.clicked.connect(self._add_grad_color)
        rem_c_btn = QPushButton("Remove Color")
        rem_c_btn.clicked.connect(self._remove_grad_color)
        grad_btns.addWidget(add_c_btn)
        grad_btns.addWidget(rem_c_btn)
        g_grad_layout.addLayout(grad_btns)
        
        vbox.addWidget(g_grad)
        self.grad_group = g_grad

        # --- Layout ---
        g_layout = QGroupBox("Layout")
        l_layout = QFormLayout(g_layout)
        self.align_combo = QComboBox()
        self.align_combo.addItems(["left", "center", "right", "justify"])
        self.align_combo.setCurrentText(self.entry.get('alignment', 'center'))
        l_layout.addRow("Alignment:", self.align_combo)
        
        self.line_spacing_spin = QDoubleSpinBox(); self.line_spacing_spin.setRange(0.5, 5.0); self.line_spacing_spin.setSingleStep(0.1)
        self.line_spacing_spin.setValue(float(self.entry.get('line_spacing', 1.0)))
        l_layout.addRow("Line Spacing:", self.line_spacing_spin)

        vbox.addWidget(g_layout)

        # Add scroll area to tab
        layout_in_tab = QVBoxLayout(self.min_typography_tab)
        layout_in_tab.addWidget(layout)

    def _is_light(self, color_str):
        c = QColor(color_str)
        return c.lightness() > 128

    def _pick_color(self):
        c = QColorDialog.getColor(QColor(self.color_val), self, "Pick Text Color")
        if c.isValid():
            self.color_val = c.name()
            self.color_btn.setStyleSheet(f"background-color: {self.color_val}; color: {'#000' if self._is_light(self.color_val) else '#fff'}")

    def _refresh_grad_list(self):
        self.grad_list.clear()
        for c in self.grad_colors:
            item = QListWidgetItem(c)
            item.setBackground(QColor(c))
            item.setForeground(QColor('#000' if self._is_light(c) else '#fff'))
            self.grad_list.addItem(item)

    def _add_grad_color(self):
        c = QColorDialog.getColor(Qt.white, self, "Add Gradient Color")
        if c.isValid():
            self.grad_colors.append(c.name())
            self._refresh_grad_list()

    def _remove_grad_color(self):
        row = self.grad_list.currentRow()
        if row >= 0 and len(self.grad_colors) > 2:
            self.grad_colors.pop(row)
            self._refresh_grad_list()
        elif len(self.grad_colors) <= 2:
            QMessageBox.warning(self, "Limit", "Gradient must have at least 2 colors.")

    def handle_accept(self):
        self.result = {
            'original_text': self.original_edit.toPlainText(),
            'translated_text': self.translated_edit.toPlainText(),
            'translation_style': self.style_combo.currentText() if self.style_combo else (self.entry.get('translation_style') or ''),
        }
        self.accept()

    def _on_translate_clicked(self):
        parent = self.parent()
        if not parent:
            return

        ocr_text = self.original_edit.toPlainText() or ''
        if not ocr_text.strip():
            return

        # Determine provider/model from local combo
        combo_text = self.ai_model_combo.currentText()
        if ":" in combo_text:
            provider, model_name = [x.strip() for x in combo_text.split(":", 1)]
        else:
             # Fallback
             provider, model_name = parent.get_selected_model_name()

        if not model_name:
            QMessageBox.warning(self, "No Model", "Please select an AI model.")
            return

        self.translate_button.setEnabled(False)
        self.translate_button.setText("Translating...")
        QApplication.processEvents()

        try:
            # Need to get styles if any
            local_settings = SETTINGS.copy()
            if self.style_combo:
                style_name = self.style_combo.currentText()
                if style_name:
                    local_settings['translation_style'] = style_name
            
            # Call parent method with SETTINGS dict
            translated = parent.translate_with_ai(ocr_text, {}, provider, model_name, local_settings)
            self.translated_edit.setPlainText(str(translated))
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Translation failed: {str(e)}")
        finally:
            self.translate_button.setEnabled(True)
            self.translate_button.setText("Translate")

    def get_result(self):
        return self.result



class ManualTextDialog(QDialog):
    def __init__(self, default_inpaint=True, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manual Text Input")
        self.setModal(True)
        self.resize(420, 340)

        main_layout = QVBoxLayout(self)

        instructions = QLabel("Type the text you want to place inside the selected area.")
        instructions.setWordWrap(True)
        main_layout.addWidget(instructions)

        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Enter manual text here...")
        self.text_edit.setMinimumHeight(160)
        main_layout.addWidget(self.text_edit)

        self.inpaint_checkbox = QCheckBox("Apply inpainting before adding text")
        self.inpaint_checkbox.setChecked(default_inpaint)
        main_layout.addWidget(self.inpaint_checkbox)

        button_box = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        button_box.button(QDialogButtonBox.Ok).setText("Apply")
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

    def get_text(self):
        return self.text_edit.toPlainText()

    def use_inpainting(self):
        return self.inpaint_checkbox.isChecked()

class BatchSaveDialog(QDialog):
    save_requested = pyqtSignal(list)

    def __init__(self, all_files, parent=None):
        super().__init__(parent)
        self.main_app = parent
        self.all_files = all_files
        self.setWindowTitle("Batch Save Images")
        self.setMinimumSize(600, 700)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        selection_layout = QGridLayout()
        presets = [5, 10, 15, 20, 25]
        for i, num in enumerate(presets):
            btn = QPushButton(f"Select Next {num} Unsaved")
            btn.clicked.connect(lambda _, n=num: self.select_next_unsaved(n))
            selection_layout.addWidget(btn, 0, i)

        select_all_btn = QPushButton("Select All Unsaved")
        select_all_btn.clicked.connect(self.select_all_unsaved)
        selection_layout.addWidget(select_all_btn, 1, 0, 1, 2)

        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(self.deselect_all)
        selection_layout.addWidget(deselect_all_btn, 1, 2, 1, 3)
        layout.addLayout(selection_layout)

        self.list_widget = QListWidget()
        self.populate_list()
        layout.addWidget(self.list_widget)

        button_layout = QHBoxLayout()
        self.save_button = QPushButton("Save Selected")
        self.save_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.save_button)
        layout.addLayout(button_layout)

    def populate_list(self):
        self.list_widget.clear()
        for file_path in self.all_files:
            if "_typeset" in file_path.lower():
                continue
            item = QListWidgetItem(os.path.basename(file_path))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)

            is_saved = self.main_app.check_if_saved(file_path)
            if is_saved:
                item.setCheckState(Qt.Unchecked)
                item.setForeground(QColor("gray"))
                item.setText(f"{os.path.basename(file_path)} [SAVED]")
            else:
                item.setCheckState(Qt.Unchecked)

            item.setData(Qt.UserRole, file_path)
            self.list_widget.addItem(item)

    def select_next_unsaved(self, count):
        self.deselect_all()
        selected_count = 0
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            file_path = item.data(Qt.UserRole)
            if not self.main_app.check_if_saved(file_path):
                item.setCheckState(Qt.Checked)
                selected_count += 1
                if selected_count >= count:
                    break

    def select_all_unsaved(self):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if not self.main_app.check_if_saved(item.data(Qt.UserRole)):
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)

    def deselect_all(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.Unchecked)

    def get_selected_files(self):
        selected = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.Checked:
                selected.append(item.data(Qt.UserRole))
        return selected
