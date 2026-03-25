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

from src.core.config import *

class APIManagerPanel(QWidget):
    """Panel widget to manage translation and AI OCR API settings."""

    TRANSLATION_PROVIDERS = ['gemini', 'openai', 'deepl', 'google']
    OCR_PROVIDERS = {
        'openrouter': "OpenRouter",
        'other': "Other"
    }

    def __init__(self, initial_settings=None, parent=None):
        super().__init__(parent)
        base_settings = initial_settings or SETTINGS
        self.temp_settings = copy.deepcopy(base_settings)

        self.translation_provider_widgets = {}
        self.ocr_provider_widgets = {}

        main_layout = QVBoxLayout(self)
        self.tab_widget = QTabWidget(self)
        main_layout.addWidget(self.tab_widget)

        self.translation_tab = QWidget()
        self._build_translation_tab()
        self.tab_widget.addTab(self.translation_tab, "Translation APIs")

        self.ocr_tab = QWidget()
        self._build_ocr_tab()
        self.tab_widget.addTab(self.ocr_tab, "AI OCR")

        # Tesseract section below tabs
        tess_group = QGroupBox('Tesseract OCR Path')
        tess_layout = QHBoxLayout(tess_group)
        self.tess_path_edit = QLineEdit(self.temp_settings.get('tesseract', {}).get('path', ''))
        self.tess_path_edit.setPlaceholderText('Path to tesseract executable')
        browse_btn = QPushButton('Browse')
        browse_btn.clicked.connect(self._browse_tesseract)
        tess_layout.addWidget(self.tess_path_edit)
        tess_layout.addWidget(browse_btn)
        tess_hint = QLabel("Set the executable path Tesseract OCR should use.")
        tess_hint.setStyleSheet("color: #9cb4d0; font-size: 11px;")
        tess_layout.addWidget(tess_hint)
        main_layout.addWidget(tess_group)

        self._load_from_settings()

    # ------------------------------------------------------------------
    # UI Builders
    # ------------------------------------------------------------------
    def _build_translation_tab(self):
        layout = QVBoxLayout(self.translation_tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        description = QLabel("Manage translation API keys and select the active key per provider.")
        description.setWordWrap(True)
        description.setStyleSheet("color: #9cb4d0;")
        layout.addWidget(description)

        for provider in self.TRANSLATION_PROVIDERS:
            group = QGroupBox(provider.capitalize())
            gl = QHBoxLayout(group)
            list_widget = QListWidget()
            list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
            list_widget.setMinimumHeight(60)
            gl.addWidget(list_widget, 1)

            button_col = QVBoxLayout()
            add_btn = QPushButton('Add Key')
            add_btn.setToolTip('Add a new API key for this provider')
            rem_btn = QPushButton('Remove Key')
            rem_btn.setToolTip('Remove the selected API key')
            set_btn = QPushButton('Set Active')
            set_btn.setToolTip('Mark the selected key as active')
            button_col.addWidget(add_btn)
            button_col.addWidget(rem_btn)
            button_col.addWidget(set_btn)
            button_col.addStretch()
            gl.addLayout(button_col)

            self.translation_provider_widgets[provider] = {
                'list': list_widget,
                'add': add_btn,
                'remove': rem_btn,
                'set': set_btn
            }

            add_btn.clicked.connect(partial(self._on_add_translation_key, provider))
            rem_btn.clicked.connect(partial(self._on_remove_translation_key, provider))
            set_btn.clicked.connect(partial(self._on_set_translation_active, provider))

            layout.addWidget(group)

        layout.addStretch()

    def _build_ocr_tab(self):
        layout = QVBoxLayout(self.ocr_tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        description = QLabel("Configure AI-powered OCR providers. Set API URL, API key, and manage available models.")
        description.setWordWrap(True)
        description.setStyleSheet("color: #9cb4d0;")
        layout.addWidget(description)

        for provider_key, display_name in self.OCR_PROVIDERS.items():
            group = QGroupBox(display_name)
            group_layout = QVBoxLayout(group)
            group_layout.setSpacing(8)

            info_layout = QGridLayout()
            info_layout.setColumnStretch(1, 1)

            url_edit = QLineEdit()
            url_edit.setPlaceholderText('API endpoint URL')
            url_edit.setToolTip('Enter the full endpoint URL for the OCR provider')
            api_key_edit = QLineEdit()
            api_key_edit.setPlaceholderText('API key')
            api_key_edit.setEchoMode(QLineEdit.Password)
            api_key_edit.setToolTip('Enter the API key for this provider')
            # Small toggle to reveal/hide API key for convenience
            show_key_btn = QPushButton('Show')
            show_key_btn.setCheckable(True)
            show_key_btn.setToolTip('Show/Hide API key')
            def _toggle_key_visibility(ckb, edit=api_key_edit, btn=show_key_btn):
                try:
                    if ckb.isChecked():
                        edit.setEchoMode(QLineEdit.Normal)
                        btn.setText('Hide')
                    else:
                        edit.setEchoMode(QLineEdit.Password)
                        btn.setText('Show')
                except Exception:
                    pass
            show_key_btn.toggled.connect(_toggle_key_visibility)

            info_layout.addWidget(QLabel("API URL"), 0, 0)
            info_layout.addWidget(url_edit, 0, 1)
            info_layout.addWidget(QLabel("API Key"), 1, 0)
            # place API key edit and show/hide button in a small horizontal layout
            key_widget = QWidget()
            key_layout = QHBoxLayout(key_widget)
            key_layout.setContentsMargins(0, 0, 0, 0)
            key_layout.addWidget(api_key_edit)
            key_layout.addWidget(show_key_btn)
            info_layout.addWidget(key_widget, 1, 1)

            warning_label = QLabel("")
            warning_label.setStyleSheet("color: #ff6b6b;")
            info_layout.addWidget(warning_label, 2, 0, 1, 2)

            group_layout.addLayout(info_layout)

            active_label = QLabel("Active Model: None")
            active_label.setStyleSheet("font-weight: 600;")
            group_layout.addWidget(active_label)

            models_table = QTableWidget(0, 3)
            models_table.setHorizontalHeaderLabels(["Model Name", "Model ID", "Active"])
            header = models_table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.Stretch)
            header.setSectionResizeMode(1, QHeaderView.Stretch)
            header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
            models_table.verticalHeader().setVisible(False)
            models_table.setSelectionBehavior(QAbstractItemView.SelectRows)
            models_table.setSelectionMode(QAbstractItemView.SingleSelection)
            models_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            models_table.setAlternatingRowColors(True)
            group_layout.addWidget(models_table)

            button_row = QHBoxLayout()
            add_model_btn = QPushButton("Add Model")
            add_model_btn.setToolTip("Add a new OCR model definition")
            remove_model_btn = QPushButton("Remove Model")
            remove_model_btn.setToolTip("Remove the selected model")
            set_active_btn = QPushButton("Set Active")
            set_active_btn.setToolTip("Mark the selected model as active")
            button_row.addWidget(add_model_btn)
            button_row.addWidget(remove_model_btn)
            button_row.addWidget(set_active_btn)
            button_row.addStretch()
            group_layout.addLayout(button_row)

            add_model_btn.clicked.connect(partial(self._add_ocr_model, provider_key))
            remove_model_btn.clicked.connect(partial(self._remove_ocr_model, provider_key))
            set_active_btn.clicked.connect(partial(self._set_active_ocr_model, provider_key))

            self.ocr_provider_widgets[provider_key] = {
                'url_edit': url_edit,
                'api_key_edit': api_key_edit,
                'models_table': models_table,
                'warning_label': warning_label,
                'active_label': active_label,
                'model_radio_group': None
            }

            layout.addWidget(group)

        layout.addStretch()

    # ------------------------------------------------------------------
    # Loading Helpers
    # ------------------------------------------------------------------
    def _load_from_settings(self):
        self._load_translation_settings()
        self._load_ocr_settings()

    def _load_translation_settings(self):
        apis = self.temp_settings.get('apis', {})
        for provider, widgets in self.translation_provider_widgets.items():
            list_widget: QListWidget = widgets['list']
            list_widget.clear()
            keys = apis.get(provider, {}).get('keys', [])
            for key_info in keys:
                # Handle legacy string format or dict
                if isinstance(key_info, str):
                    key_info = {'name': key_info[:8], 'value': key_info, 'active': False}
                
                name = key_info.get('name') or (key_info.get('value') or '')[:8]
                active = bool(key_info.get('active'))
                display = f"[ON] {name}" if active else f"[OFF] {name}"
                item = QListWidgetItem(display)
                item.setData(Qt.UserRole, key_info)
                if active:
                    item.setForeground(QColor('#5de6c1'))
                list_widget.addItem(item)

    def _load_ocr_settings(self):
        ocr_settings = self.temp_settings.get('ocr', {})
        for provider, widgets in self.ocr_provider_widgets.items():
            cfg = ocr_settings.get(provider, {})
            widgets['url_edit'].setText(cfg.get('url', ''))
            widgets['api_key_edit'].setText(cfg.get('api_key', ''))
            self._populate_ocr_models_table(provider)

    # ------------------------------------------------------------------
    # Translation API handlers
    # ------------------------------------------------------------------
    def _on_add_translation_key(self, provider):
        name, ok = QInputDialog.getText(self, 'Add API Key', 'Key name (label):')
        if not ok or not name.strip():
            return
        value, ok2 = QInputDialog.getText(self, 'Add API Key', 'Key value:')
        if not ok2 or not value.strip():
            return
        keys = self.temp_settings.setdefault('apis', {}).setdefault(provider, {}).setdefault('keys', [])
        keys.append({'name': name.strip(), 'value': value.strip(), 'active': False})
        self._load_translation_settings()

    def _on_remove_translation_key(self, provider):
        widgets = self.translation_provider_widgets[provider]
        list_widget: QListWidget = widgets['list']
        item = list_widget.currentItem()
        if not item:
            return
        key_data = item.data(Qt.UserRole)
        keys = self.temp_settings.setdefault('apis', {}).setdefault(provider, {}).setdefault('keys', [])
        self.temp_settings['apis'][provider]['keys'] = [k for k in keys if k.get('value') != key_data.get('value')]
        self._load_translation_settings()

    def _on_set_translation_active(self, provider):
        widgets = self.translation_provider_widgets[provider]
        list_widget: QListWidget = widgets['list']
        item = list_widget.currentItem()
        if not item:
            return
        key_data = item.data(Qt.UserRole)
        keys = self.temp_settings.setdefault('apis', {}).setdefault(provider, {}).setdefault('keys', [])
        for entry in keys:
            entry['active'] = (entry.get('value') == key_data.get('value'))
        self._load_translation_settings()

    # ------------------------------------------------------------------
    # AI OCR handlers
    # ------------------------------------------------------------------
    def _populate_ocr_models_table(self, provider):
        widgets = self.ocr_provider_widgets[provider]
        table: QTableWidget = widgets['models_table']
        # Disconnect previous signal if any
        old_group = widgets.get('model_radio_group')
        if old_group is not None:
            try:
                old_group.buttonToggled.disconnect()
            except Exception:
                pass

        table.setRowCount(0)
        models = self.temp_settings.setdefault('ocr', {}).setdefault(provider, {}).setdefault('models', [])
        radio_group = QButtonGroup(table)
        radio_group.setExclusive(True)

        for row, model in enumerate(models):
            table.insertRow(row)
            name_item = QTableWidgetItem(model.get('name', ''))
            id_item = QTableWidgetItem(model.get('id', ''))
            name_item.setFlags(name_item.flags() ^ Qt.ItemIsEditable)
            id_item.setFlags(id_item.flags() ^ Qt.ItemIsEditable)
            table.setItem(row, 0, name_item)
            table.setItem(row, 1, id_item)

            radio = QRadioButton()
            radio.setProperty('row', row)
            radio.setChecked(bool(model.get('active')))
            radio_widget = QWidget()
            radio_layout = QHBoxLayout(radio_widget)
            radio_layout.setContentsMargins(0, 0, 0, 0)
            radio_layout.setAlignment(Qt.AlignCenter)
            radio_layout.addWidget(radio)
            table.setCellWidget(row, 2, radio_widget)
            radio_group.addButton(radio, row)

        def on_radio_toggled(button, checked, prov=provider):
            self._on_ocr_radio_toggled(prov, button, checked)

        radio_group.buttonToggled.connect(on_radio_toggled)
        widgets['model_radio_group'] = radio_group
        self._update_ocr_active_display(provider)

    def _on_ocr_radio_toggled(self, provider, button, checked):
        if not checked:
            return
        row = button.property('row')
        models = self.temp_settings.setdefault('ocr', {}).setdefault(provider, {}).setdefault('models', [])
        for idx, model in enumerate(models):
            model['active'] = (idx == row)
        self._update_ocr_active_display(provider)

    def _update_ocr_active_display(self, provider):
        widgets = self.ocr_provider_widgets[provider]
        table: QTableWidget = widgets['models_table']
        models = self.temp_settings.setdefault('ocr', {}).setdefault(provider, {}).setdefault('models', [])
        active_index = next((idx for idx, m in enumerate(models) if m.get('active')), -1)

        if active_index >= 0:
            active_name = models[active_index].get('name', '(Unnamed)')
            widgets['active_label'].setText(f"Active Model: {active_name}")
        else:
            widgets['active_label'].setText("Active Model: None")

        for row in range(table.rowCount()):
            is_active = (row == active_index)
            for col in range(2):
                item = table.item(row, col)
                if not item:
                    continue
                if is_active:
                    item.setBackground(QColor('#234162'))
                    item.setForeground(QColor('#f3f6fb'))
                else:
                    item.setBackground(QColor('#1a2634'))
                    item.setForeground(QColor('#e0e8f5'))
            widget = table.cellWidget(row, 2)
            if widget:
                pal = widget.palette()
                pal.setColor(widget.backgroundRole(), QColor('#234162') if is_active else QColor('#1a2634'))
                widget.setPalette(pal)
                widget.setAutoFillBackground(True)

    def _add_ocr_model(self, provider):
        name, ok = QInputDialog.getText(self, "Add OCR Model", "Model Name:")
        if not ok or not name.strip():
            return
        model_id, ok2 = QInputDialog.getText(self, "Add OCR Model", "Model ID:")
        if not ok2 or not model_id.strip():
            return

        models = self.temp_settings.setdefault('ocr', {}).setdefault(provider, {}).setdefault('models', [])
        is_first = len(models) == 0
        models.append({'name': name.strip(), 'id': model_id.strip(), 'active': is_first})
        if is_first:
            # ensure only one active
            for model in models[1:]:
                model['active'] = False
        self._populate_ocr_models_table(provider)

    def _remove_ocr_model(self, provider):
        widgets = self.ocr_provider_widgets[provider]
        table: QTableWidget = widgets['models_table']
        row = table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Remove Model", "Select a model to remove.")
            return
        confirm = QMessageBox.question(self, "Remove Model", "Remove selected model?",
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if confirm != QMessageBox.Yes:
            return
        models = self.temp_settings.setdefault('ocr', {}).setdefault(provider, {}).setdefault('models', [])
        if row >= len(models):
            return
        was_active = bool(models[row].get('active'))
        models.pop(row)
        if was_active and models:
            models[0]['active'] = True
            for model in models[1:]:
                model['active'] = False
        self._populate_ocr_models_table(provider)

    def _set_active_ocr_model(self, provider):
        widgets = self.ocr_provider_widgets[provider]
        table: QTableWidget = widgets['models_table']
        row = table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Set Active", "Select a model to activate.")
            return
        models = self.temp_settings.setdefault('ocr', {}).setdefault(provider, {}).setdefault('models', [])
        if row >= len(models):
            return
        for idx, model in enumerate(models):
            model['active'] = (idx == row)
        self._populate_ocr_models_table(provider)

    # ------------------------------------------------------------------
    # Save / Close helpers
    # ------------------------------------------------------------------
    def _browse_tesseract(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Select Tesseract executable')
        if path:
            self.tess_path_edit.setText(path)

    def _apply_latest_form_values(self):
        self._validation_messages = []
        ocr_settings = self.temp_settings.setdefault('ocr', {})
        for provider, widgets in self.ocr_provider_widgets.items():
            provider_cfg = ocr_settings.setdefault(provider, {})
            url = widgets['url_edit'].text().strip()
            api_key = widgets['api_key_edit'].text().strip()
            provider_cfg['url'] = url
            provider_cfg['api_key'] = api_key
            models = provider_cfg.setdefault('models', [])

            warnings = []
            if models:
                if not url:
                    warnings.append("API URL is required when models are configured.")
                if not api_key:
                    warnings.append("API Key is required when models are configured.")
            widgets['warning_label'].setText("\n".join(warnings))
            if warnings:
                display_name = self.OCR_PROVIDERS.get(provider, provider.title())
                for msg in warnings:
                    self._validation_messages.append(f"{display_name}: {msg}")

        tess_cfg = self.temp_settings.setdefault('tesseract', {})
        tess_cfg['path'] = self.tess_path_edit.text().strip()
        tess_cfg['auto_detected'] = False

        return not self._validation_messages

    def export_settings(self):
        if not self._apply_latest_form_values():
            return None
        return {
            'apis': copy.deepcopy(self.temp_settings.get('apis', {})),
            'ocr': copy.deepcopy(self.temp_settings.get('ocr', {})),
            'tesseract': copy.deepcopy(self.temp_settings.get('tesseract', {})),
        }

    def validation_messages(self):
        return list(getattr(self, '_validation_messages', []))

class OpenRouterSettingsPanel(QWidget):
    def __init__(self, initial_settings=None, parent=None):
        super().__init__(parent)

        translate_cfg = (initial_settings or SETTINGS).get('translate', {})
        self.data = copy.deepcopy(translate_cfg.get('openrouter', {}))
        self.data.setdefault('url', "https://openrouter.ai/api/v1/chat/completions")
        self.data.setdefault('api_key', "")
        self.data.setdefault('models', [])
        self.models = copy.deepcopy(self.data.get('models', []))

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget(self)
        layout.addWidget(self.tabs, 1)

        self._build_api_tab()
        self._build_models_tab()

    def _build_api_tab(self):
        api_widget = QWidget()
        form = QFormLayout(api_widget)
        self.url_edit = QLineEdit(self.data.get('url', ''))
        self.api_key_edit = QLineEdit(self.data.get('api_key', ''))
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        key_layout = QHBoxLayout()
        key_layout.setContentsMargins(0, 0, 0, 0)
        key_layout.addWidget(self.api_key_edit, 1)
        self.api_key_toggle = QToolButton()
        self.api_key_toggle.setText("Show")
        self.api_key_toggle.setCheckable(True)
        self.api_key_toggle.toggled.connect(self._toggle_api_key_visibility)
        key_layout.addWidget(self.api_key_toggle)
        key_widget = QWidget()
        key_widget.setLayout(key_layout)
        form.addRow("API URL", self.url_edit)
        form.addRow("API Key", key_widget)
        # Provider tuning
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(5, 600)
        self.timeout_spin.setValue(int(self.data.get('timeout', 60) or 60))
        form.addRow("Timeout (s)", self.timeout_spin)
        self.retries_spin = QSpinBox()
        self.retries_spin.setRange(0, 10)
        self.retries_spin.setValue(int(self.data.get('retries', 3) or 3))
        form.addRow("Retries", self.retries_spin)
        self.backoff_spin = QDoubleSpinBox()
        self.backoff_spin.setRange(0.1, 10.0)
        self.backoff_spin.setSingleStep(0.1)
        self.backoff_spin.setValue(float(self.data.get('backoff', 1.5) or 1.5))
        form.addRow("Backoff factor", self.backoff_spin)
        help_label = QLabel("Tip: Find your OpenRouter API key at https://openrouter.ai/account")
        help_label.setStyleSheet("color: #9cb4d0;")
        help_label.setWordWrap(True)
        form.addRow(help_label)
        self.tabs.addTab(api_widget, "API Configuration")

    def _build_models_tab(self):
        models_widget = QWidget()
        vbox = QVBoxLayout(models_widget)
        info = QLabel("Add translation models to call via OpenRouter. Multiple models can be active at the same time.")
        info.setWordWrap(True)
        info.setStyleSheet("color: #9cb4d0;")
        vbox.addWidget(info)

        self.models_table = QTableWidget(0, 4, self)
        self.models_table.setHorizontalHeaderLabels(["Model Name", "Model ID", "Description", "Active"])
        header = self.models_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.models_table.verticalHeader().setVisible(False)
        self.models_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.models_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.models_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.models_table.setAlternatingRowColors(True)
        vbox.addWidget(self.models_table, 1)

        button_row = QHBoxLayout()
        self.add_button = QPushButton("Add Model")
        self.add_button.clicked.connect(self._add_model)
        self.edit_button = QPushButton("Edit Model")
        self.edit_button.clicked.connect(self._edit_model)
        self.remove_button = QPushButton("Remove Model")
        self.remove_button.clicked.connect(self._remove_model)
        button_row.addWidget(self.add_button)
        button_row.addWidget(self.edit_button)
        button_row.addWidget(self.remove_button)
        button_row.addStretch()
        vbox.addLayout(button_row)

        self.tabs.addTab(models_widget, "Models")
        self._refresh_models_table()

    def _toggle_api_key_visibility(self, checked: bool):
        self.api_key_edit.setEchoMode(QLineEdit.Normal if checked else QLineEdit.Password)
        self.api_key_toggle.setText("Hide" if checked else "Show")

    def _refresh_models_table(self):
        self.models_table.setRowCount(0)
        for row, model in enumerate(self.models):
            name = model.get('name', '')
            model_id = model.get('id', '')
            desc = model.get('description', '')
            active = bool(model.get('active', True))
            self.models_table.insertRow(row)
            name_item = QTableWidgetItem(name)
            model_item = QTableWidgetItem(model_id)
            desc_item = QTableWidgetItem(desc)
            for item in (name_item, model_item, desc_item):
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
            self.models_table.setItem(row, 0, name_item)
            self.models_table.setItem(row, 1, model_item)
            self.models_table.setItem(row, 2, desc_item)
            checkbox = QCheckBox()
            checkbox.setChecked(active)
            checkbox.stateChanged.connect(lambda state, r=row: self._set_model_active(r, state))
            container = QWidget()
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setAlignment(Qt.AlignCenter)
            layout.addWidget(checkbox)
            self.models_table.setCellWidget(row, 3, container)

    def _set_model_active(self, row: int, state: int):
        if 0 <= row < len(self.models):
            self.models[row]['active'] = (state == Qt.Checked)

    def _add_model(self):
        dialog = ModelEditDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            values = dialog.get_values()
            if not values['name'] or not values['id']:
                QMessageBox.warning(self, "Invalid Model", "Model name and ID are required.")
                return
            self.models.append({
                'name': values['name'],
                'id': values['id'],
                'description': values['description'],
                'active': True
            })
            self._refresh_models_table()

    def _edit_model(self):
        row = self.models_table.currentRow()
        if not (0 <= row < len(self.models)):
            QMessageBox.information(self, "Edit Model", "Select a model to edit.")
            return
        model = self.models[row]
        dialog = ModelEditDialog(self, name=model.get('name', ''), model_id=model.get('id', ''), description=model.get('description', ''))
        if dialog.exec_() == QDialog.Accepted:
            values = dialog.get_values()
            if not values['name'] or not values['id']:
                QMessageBox.warning(self, "Invalid Model", "Model name and ID are required.")
                return
            model.update({
                'name': values['name'],
                'id': values['id'],
                'description': values['description']
            })
            self._refresh_models_table()

    def _remove_model(self):
        row = self.models_table.currentRow()
        if not (0 <= row < len(self.models)):
            QMessageBox.information(self, "Remove Model", "Select a model to remove.")
            return
        confirm = QMessageBox.question(self, "Remove Model", f"Remove model '{self.models[row].get('name', '')}'?",
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if confirm == QMessageBox.Yes:
            self.models.pop(row)
            self._refresh_models_table()

    def _on_lang_orientation_changed(self, lang_code, value):
        # Ensure mapping exists
        lang_map = SETTINGS.setdefault('lang_orientation', {})
        lang_map[lang_code] = value
        save_settings(SETTINGS)

    def export_settings(self):
        self.data['url'] = self.url_edit.text().strip() or "https://openrouter.ai/api/v1/chat/completions"
        self.data['api_key'] = self.api_key_edit.text().strip()
        try:
            self.data['timeout'] = int(self.timeout_spin.value())
            self.data['retries'] = int(self.retries_spin.value())
            self.data['backoff'] = float(self.backoff_spin.value())
        except Exception:
            pass
        self.data['models'] = copy.deepcopy(self.models)
        return copy.deepcopy(self.data)

    def get_settings(self):
        return self.export_settings()
