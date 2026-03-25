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
from src.utils.geometry import *
from src.ui.dialogs import *
from src.core.fonts import *
from src.core.models import EnhancedResult

class TypesetArea:
    def __init__(
        self,
        rect,
        text,
        font,
        color,
        polygon=None,
        orientation='horizontal',
        effect='none',
        effect_intensity=20.0,
        bezier_points=None,
        bubble_enabled=False,
        segments=None,
        bubble_fill='#ffffff',
        bubble_outline='#000000',
        bubble_outline_width=3.0,
        text_outline=False,
        text_outline_width=2.0,
        text_outline_color='#000000',
        text_outline_style='stroke',
        alignment='center',
        line_spacing=1.1,
        char_spacing=100.0,
        margins=None,
        history_id=None,
        original_text="",
        translation_style="",
        review_notes=None,
        overrides=None,
        rotation=0.0,
        cleanup_rect=None,
        cleanup_polygon=None,
        gradient_enabled=False,
        gradient_colors=None,
        gradient_angle=0.0,
    ):
        self.rect = rect
        # Cleanup geometry keeps track of the original mask/box so it doesn't move when text is repositioned
        try:
            self.cleanup_rect = QRect(cleanup_rect) if cleanup_rect is not None else QRect(rect)
        except Exception:
            self.cleanup_rect = QRect(rect)
        self.rotation = float(rotation) if rotation is not None else 0.0
        self.text = text or ""
        self.font_info = self.font_to_dict(font)
        self.color_info = color.name()
        self.polygon = polygon
        try:
            self.cleanup_polygon = QPolygon(cleanup_polygon) if cleanup_polygon is not None else (QPolygon(polygon) if polygon is not None else None)
        except Exception:
            self.cleanup_polygon = None
        self.orientation = orientation
        self.effect = effect
        self.effect_intensity = effect_intensity
        self.bezier_points = bezier_points or [
            {'x': 0.25, 'y': 0.2},
            {'x': 0.75, 'y': 0.2},
        ]
        self.bubble_enabled = bubble_enabled
        self.bubble_fill = bubble_fill
        self.bubble_outline = bubble_outline
        self.bubble_outline_width = bubble_outline_width
        self.text_outline = bool(text_outline)
        try:
            self.text_outline_width = max(0.0, float(text_outline_width))
        except Exception:
            self.text_outline_width = 2.0
        if isinstance(text_outline_color, QColor):
            self.text_outline_color = text_outline_color.name()
        else:
            self.text_outline_color = str(text_outline_color or '#000000')
        self.text_outline_style = (text_outline_style or 'stroke').lower()
        self.alignment = alignment
        self.line_spacing = line_spacing
        self.char_spacing = char_spacing
        self.margins = margins or {'top': 0, 'right': 0, 'bottom': 0, 'left': 0}
        self.history_id = history_id
        self.original_text = original_text or ""
        self.translation_style = translation_style or ""
        self.review_notes = review_notes or {}
        self.overrides = overrides if isinstance(overrides, dict) else {}
        self.gradient_enabled = bool(gradient_enabled)
        self.gradient_colors = gradient_colors or ["#FF0000", "#0000FF"]
        self.gradient_angle = float(gradient_angle) if gradient_angle is not None else 0.0
        self.text_segments = segments if segments is not None else self._build_segments_from_plain(self.text, font, color)
        self.ensure_defaults()

    def ensure_defaults(self):
        if not hasattr(self, 'overrides') or not isinstance(self.overrides, dict):
            self.overrides = {}
        if not hasattr(self, 'history_id'): self.history_id = None
        if not hasattr(self, 'original_text') or self.original_text is None: self.original_text = ''
        if not hasattr(self, 'translation_style') or self.translation_style is None: self.translation_style = ''
        if not hasattr(self, 'review_notes') or self.review_notes is None: self.review_notes = {}
        if not hasattr(self, 'orientation'): self.orientation = 'horizontal'
        if not hasattr(self, 'effect'): self.effect = 'none'
        if not hasattr(self, 'effect_intensity'): self.effect_intensity = 20.0
        if not hasattr(self, 'rotation') or self.rotation is None:
            try:
                self.rotation = float(self.rotation)
            except Exception:
                self.rotation = 0.0

        if not getattr(self, 'bezier_points', None):
            self.bezier_points = [{'x': 0.25, 'y': 0.2}, {'x': 0.75, 'y': 0.2}]
        if not hasattr(self, 'bubble_enabled'): self.bubble_enabled = False
        if not getattr(self, 'bubble_fill', None): self.bubble_fill = '#ffffff'
        if not getattr(self, 'bubble_outline', None): self.bubble_outline = '#000000'
        if not hasattr(self, 'bubble_outline_width'): self.bubble_outline_width = 3.0
        if not hasattr(self, 'text_outline'): self.text_outline = False
        if not hasattr(self, 'text_outline_width'):
            self.text_outline_width = 2.0
        else:
            try:
                self.text_outline_width = max(0.0, float(self.text_outline_width))
            except Exception:
                self.text_outline_width = 2.0
        if not getattr(self, 'text_outline_color', None):
            self.text_outline_color = '#000000'
        if not getattr(self, 'text_outline_style', None):
            self.text_outline_style = 'stroke'
        else:
            style = str(self.text_outline_style).lower()
            self.text_outline_style = style if style in ('stroke', 'glow') else 'stroke'
        if not hasattr(self, 'alignment'): self.alignment = 'center'
        if not hasattr(self, 'line_spacing') or self.line_spacing is None: self.line_spacing = 1.1
        if not hasattr(self, 'char_spacing') or self.char_spacing is None: self.char_spacing = 100.0
        if 'letterSpacing' not in self.font_info:
            self.font_info['letterSpacing'] = self.char_spacing
            self.font_info['letterSpacingType'] = QFont.PercentageSpacing
        if not getattr(self, 'margins', None): self.margins = {'top': 0, 'right': 0, 'bottom': 0, 'left': 0}
        if not getattr(self, 'cleanup_rect', None):
            try:
                self.cleanup_rect = QRect(self.rect)
            except Exception:
                self.cleanup_rect = QRect()
        if not hasattr(self, 'cleanup_polygon') or self.cleanup_polygon is None:
            try:
                self.cleanup_polygon = QPolygon(self.polygon) if getattr(self, 'polygon', None) is not None else None
            except Exception:
                self.cleanup_polygon = None
        if not hasattr(self, 'gradient_enabled'):
            self.gradient_enabled = False
        if not hasattr(self, 'gradient_colors') or not isinstance(self.gradient_colors, list):
            self.gradient_colors = ["#FF0000", "#0000FF"]
        if not hasattr(self, 'gradient_angle'):
            self.gradient_angle = 0.0
        if not getattr(self, 'text_segments', None):
            self.text_segments = self._build_segments_from_plain(self.text, self.get_font(), self.get_color())
        if not getattr(self, 'text', None):
            self.text = self._segments_to_plain_text(self.text_segments)

    def get_text_rect(self):
        try:
            return QRect(self.rect)
        except Exception:
            return QRect()

    def get_cleanup_rect(self):
        try:
            if not getattr(self, 'cleanup_rect', None):
                self.cleanup_rect = QRect(self.rect)
            return QRect(self.cleanup_rect)
        except Exception:
            return QRect(self.rect) if hasattr(self, 'rect') else QRect()

    def set_cleanup_rect(self, rect):
        try:
            self.cleanup_rect = QRect(rect)
        except Exception:
            self.cleanup_rect = QRect()

    def get_cleanup_polygon(self):
        poly = getattr(self, 'cleanup_polygon', None)
        if poly is None:
            return None
        try:
            return QPolygon(poly)
        except Exception:
            return None

    def set_cleanup_polygon(self, polygon):
        try:
            self.cleanup_polygon = QPolygon(polygon) if polygon is not None else None
        except Exception:
            self.cleanup_polygon = None

    def get_extra(self, key, default=None):
        return getattr(self, key, default)

    def get_overrides(self):
        return self.overrides

    def has_override(self, key):
        return isinstance(self.overrides, dict) and key in self.overrides

    def get_override(self, key, default=None):
        if self.has_override(key):
            return self.overrides[key]
        return default

    def set_override(self, key, value):
        if not isinstance(self.overrides, dict):
            self.overrides = {}
        self.overrides[key] = value

    def clear_override(self, key):
        if isinstance(self.overrides, dict):
            self.overrides.pop(key, None)

    def clear_overrides(self):
        if isinstance(self.overrides, dict):
            self.overrides.clear()

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.ensure_defaults()

    @staticmethod
    def font_to_dict(font):
        manager = get_font_manager()
        data = {
            'family': font.family(),
            'pointSize': font.pointSizeF() if hasattr(font, 'pointSizeF') else font.pointSize(),
            'weight': font.weight(),
            'italic': font.italic(),
            'underline': font.underline(),
            'letterSpacing': font.letterSpacing(),
            'letterSpacingType': font.letterSpacingType(),
        }
        if manager:
            display_name = manager.display_name_for_font(font, fallback=None)
            if display_name:
                data['displayName'] = display_name
        return data

    @staticmethod
    def font_from_dict(info):
        if isinstance(info, QFont):
            return QFont(info)

        font_manager = get_font_manager()
        base_font = None
        if isinstance(info, dict) and font_manager:
            display_name = info.get('displayName')
            if display_name:
                base_font = font_manager.create_qfont(display_name)

        font = QFont(base_font) if isinstance(base_font, QFont) else QFont()
        if not isinstance(info, dict):
            if not font.family():
                font.setFamily('Arial')
            if font.pointSize() <= 0:
                font.setPointSize(14)
            if font.weight() < 0:
                font.setWeight(QFont.Normal)
            return font

        family = info.get('family')
        if family:
            font.setFamily(str(family))

        point_size = coerce_float(info.get('pointSize', 14.0), default=14.0, minimum=1.0)
        if hasattr(font, 'setPointSizeF'):
            font.setPointSizeF(point_size)
        else:
            font.setPointSize(coerce_int(point_size, default=14, minimum=1))

        weight_value = info.get('weight', QFont.Normal)
        try:
            font.setWeight(coerce_int(weight_value, default=QFont.Normal))
        except Exception:
            font.setWeight(QFont.Normal)

        font.setItalic(bool(info.get('italic', False)))
        font.setUnderline(bool(info.get('underline', False)))

        spacing_type = coerce_int(info.get('letterSpacingType', QFont.PercentageSpacing), default=QFont.PercentageSpacing)
        spacing_value = coerce_float(info.get('letterSpacing', 100.0), default=100.0)
        font.setLetterSpacing(spacing_type, spacing_value)
        return font
    
    @classmethod
    def _sanitize_segments(cls, segments, fallback_font_dict, fallback_color):
        sanitized = []
        if not isinstance(segments, list):
            return sanitized
        fallback_font_dict = fallback_font_dict or cls.font_to_dict(cls.font_from_dict({}))
        fallback_color = fallback_color or '#000000'
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            seg_copy = copy.deepcopy(seg)
            font_info = seg_copy.get('font')
            if isinstance(font_info, QFont):
                seg_copy['font'] = cls.font_to_dict(font_info)
            elif isinstance(font_info, dict):
                seg_copy['font'] = cls.font_to_dict(cls.font_from_dict(font_info))
            else:
                seg_copy['font'] = fallback_font_dict
            seg_copy['color'] = seg_copy.get('color') or fallback_color
            seg_copy['text'] = seg_copy.get('text', '')
            seg_copy['underline'] = bool(seg_copy.get('underline', False))
            sanitized.append(seg_copy)
        return sanitized
    
    def to_payload(self):
        base_font = copy.deepcopy(self.font_info)
        segments = self._sanitize_segments(self.text_segments or [], base_font, self.color_info)
        margins = self.get_margins() if hasattr(self, 'get_margins') else getattr(self, 'margins', {})
        sanitized_margins = {key: coerce_int(margins.get(key, 0)) for key in ('top', 'right', 'bottom', 'left')}
        bezier_points = []
        for pt in self.bezier_points or []:
            if isinstance(pt, dict):
                bx = coerce_float(pt.get('x', 0.0))
                by = coerce_float(pt.get('y', 0.0))
                bezier_points.append({'x': bx, 'y': by})
            elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                bx = coerce_float(pt[0])
                by = coerce_float(pt[1])
                bezier_points.append({'x': bx, 'y': by})
        bezier_points = bezier_points or None
        return {
            'rect': rect_to_dict(self.rect),
            'cleanup_rect': rect_to_dict(self.get_cleanup_rect()),
            'text': self.text or '',
            'font': base_font,
            'color': self.color_info,
            'polygon': polygon_to_list(self.polygon),
            'cleanup_polygon': polygon_to_list(self.get_cleanup_polygon() or self.polygon),
            'orientation': self.get_orientation(),
            'effect': self.get_effect(),
            'effect_intensity': float(self.get_effect_intensity()),
            'bezier_points': bezier_points,
            'bubble_enabled': bool(self.bubble_enabled),
            'segments': segments,
            'bubble_fill': getattr(self, 'bubble_fill', '#ffffff') or '#ffffff',
            'bubble_outline': getattr(self, 'bubble_outline', '#000000') or '#000000',
            'bubble_outline_width': float(getattr(self, 'bubble_outline_width', 3.0) or 3.0),
            'text_outline': bool(self.has_text_outline()),
            'text_outline_width': float(self.get_text_outline_width()),
            'text_outline_color': self.get_text_outline_color().name(),
            'text_outline_style': self.get_text_outline_style(),
            'alignment': self.get_alignment(),
            'line_spacing': float(self.get_line_spacing()),
            'char_spacing': float(self.get_char_spacing()),
            'rotation': float(self.get_rotation()),
            'margins': sanitized_margins,
            'history_id': self.history_id,
            'original_text': self.original_text or '',
            'translation_style': self.translation_style or '',
            'review_notes': copy.deepcopy(self.review_notes if isinstance(self.review_notes, dict) else {}),
            'overrides': copy.deepcopy(self.get_overrides() if isinstance(self.get_overrides(), dict) else {}),
            'gradient_enabled': getattr(self, 'gradient_enabled', False),
            'gradient_colors': getattr(self, 'gradient_colors', None),
            'gradient_angle': getattr(self, 'gradient_angle', 0.0),
        }
    
    @classmethod
    def from_payload(cls, data, fallback_font=None, fallback_color=None):
        if fallback_font is None:
            fallback_font = QFont('Arial', 12)
        if fallback_color is None:
            fallback_color = QColor('#000000')
        if not isinstance(data, dict):
            return cls(QRect(), '', fallback_font, fallback_color)
        rect = dict_to_rect(data.get('rect'))
        font = cls.font_from_dict(data.get('font')) or fallback_font
        color_value = data.get('color', fallback_color.name()) or fallback_color.name()
        color = QColor(color_value)
        polygon = list_to_polygon(data.get('polygon'))
        cleanup_rect = dict_to_rect(data.get('cleanup_rect')) if data.get('cleanup_rect') else None
        cleanup_polygon = list_to_polygon(data.get('cleanup_polygon')) if data.get('cleanup_polygon') else None
        orientation = data.get('orientation', 'horizontal') or 'horizontal'
        effect = data.get('effect', 'none') or 'none'
        effect_intensity = coerce_float(data.get('effect_intensity'), default=20.0)
        bezier_raw = data.get('bezier_points')
        bezier_points = None
        if isinstance(bezier_raw, list):
            normalized = []
            for pt in bezier_raw:
                if isinstance(pt, dict):
                    normalized.append({'x': coerce_float(pt.get('x', 0.0)), 'y': coerce_float(pt.get('y', 0.0))})
                elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    normalized.append({'x': coerce_float(pt[0]), 'y': coerce_float(pt[1])})
            bezier_points = normalized if normalized else None
        bubble_enabled = bool(data.get('bubble_enabled', False))
        bubble_fill = data.get('bubble_fill', '#ffffff') or '#ffffff'
        bubble_outline = data.get('bubble_outline', '#000000') or '#000000'
        bubble_outline_width = coerce_float(data.get('bubble_outline_width'), default=3.0, minimum=0.0)
        text_outline = bool(data.get('text_outline', False))
        text_outline_width = coerce_float(data.get('text_outline_width'), default=2.0, minimum=0.0)
        text_outline_color = data.get('text_outline_color', '#000000') or '#000000'
        text_outline_style = (data.get('text_outline_style') or 'stroke')
        alignment = data.get('alignment', 'center') or 'center'
        line_spacing = coerce_float(data.get('line_spacing'), default=1.1)
        line_spacing = max(0.6, min(line_spacing, 5.0))
        char_spacing = coerce_float(data.get('char_spacing'), default=100.0)
        char_spacing = max(10.0, min(char_spacing, 500.0))
        rotation = coerce_float(data.get('rotation', 0.0))
        margins_data = data.get('margins')
        margins = {'top': 0, 'right': 0, 'bottom': 0, 'left': 0}
        if isinstance(margins_data, dict):
            for key in margins:
                margins[key] = coerce_int(margins_data.get(key, 0))
        segments = cls._sanitize_segments(data.get('segments'), cls.font_to_dict(font), color.name())
        history_id = data.get('history_id') or data.get('id')
        if history_id is not None:
            history_id = str(history_id)
        else:
            history_id = None
        original_text = data.get('original_text', '') or ''
        translation_style = data.get('translation_style', '') or ''
        review_notes = data.get('review_notes')
        if not isinstance(review_notes, dict):
            review_notes = {}
        overrides = data.get('overrides')
        if not isinstance(overrides, dict):
            overrides = {}
        legacy_override_keys = ('use_inpaint', 'use_background_box')
        for key in legacy_override_keys:
            if key in review_notes and key not in overrides:
                overrides[key] = review_notes.get(key)
                review_notes.pop(key, None)
    
        gradient_enabled = bool(data.get('gradient_enabled', False))
        gradient_colors = data.get('gradient_colors')
        gradient_angle = coerce_float(data.get('gradient_angle', 0.0))

        area = cls(
            rect,
            data.get('text', '') or '',
            font,
            color,
            polygon=polygon,
            orientation=orientation,
            effect=effect,
            effect_intensity=effect_intensity,
            bezier_points=bezier_points,
            bubble_enabled=bubble_enabled,
            segments=segments or None,
            bubble_fill=bubble_fill,
            bubble_outline=bubble_outline,
            bubble_outline_width=bubble_outline_width,
            text_outline=text_outline,
            text_outline_width=text_outline_width,
            text_outline_color=text_outline_color,
            text_outline_style=text_outline_style,
            alignment=alignment,
            line_spacing=line_spacing,
            char_spacing=char_spacing,
            margins=margins,
            history_id=history_id,
            original_text=original_text,
            translation_style=translation_style,
            review_notes=review_notes,
            overrides=overrides,
            rotation=rotation,
            cleanup_rect=cleanup_rect if cleanup_rect is not None else rect,
            cleanup_polygon=cleanup_polygon if cleanup_polygon is not None else polygon,
            gradient_enabled=gradient_enabled,
            gradient_colors=gradient_colors,
            gradient_angle=gradient_angle,
        )
        return area
    
    def get_font(self):
        return self.font_from_dict(self.font_info)

    def get_color(self):
        return QColor(self.color_info)

    def get_rotation(self):
        try:
            return float(self.rotation)
        except Exception:
            return 0.0

    def set_rotation(self, value):
        try:
            self.rotation = float(value)
        except Exception:
            self.rotation = 0.0

    def segment_to_qfont(self, segment):
        info = segment.get('font', self.font_info) or self.font_info
        font = self.font_from_dict(info)
        underline = segment.get('underline')
        if underline is None and isinstance(info, dict):
            underline = info.get('underline', False)
        font.setUnderline(bool(underline))
        return font

    def segment_to_color(self, segment):
        return QColor(segment.get('color', self.color_info))

    def get_segments(self):
        self.ensure_defaults()
        return self.text_segments

    def has_text_outline(self):
        return bool(getattr(self, 'text_outline', False)) and self.get_text_outline_width() > 0.0

    def get_text_outline_width(self):
        try:
            return max(0.0, float(getattr(self, 'text_outline_width', 2.0)))
        except Exception:
            return 2.0

    def get_text_outline_color(self):
        value = getattr(self, 'text_outline_color', '#000000')
        if isinstance(value, QColor):
            color = QColor(value)
        else:
            color = QColor(str(value))
        if not color.isValid():
            color = QColor('#000000')
        return color

    def get_text_outline_style(self):
        style = getattr(self, 'text_outline_style', 'stroke') or 'stroke'
        style = str(style).lower()
        return style if style in ('stroke', 'glow') else 'stroke'

    def set_segments(self, segments):
        self.text_segments = segments or []
        self.text = self._segments_to_plain_text(self.text_segments)

    def update_plain_text(self, text):
        font = self.get_font()
        color = self.get_color()
        self.text = text or ''
        self.text_segments = self._build_segments_from_plain(self.text, font, color)

    def get_orientation(self):
        return getattr(self, 'orientation', 'horizontal') or 'horizontal'

    def get_effect(self):
        return getattr(self, 'effect', 'none') or 'none'

    def get_effect_intensity(self):
        try:
            return float(getattr(self, 'effect_intensity', 20.0) or 20.0)
        except (TypeError, ValueError):
            return 20.0

    def get_bezier_points(self):
        self.ensure_defaults()
        return self.bezier_points

    def get_bubble_fill_color(self):
        return QColor(getattr(self, 'bubble_fill', '#ffffff'))

    def get_bubble_outline_color(self):
        return QColor(getattr(self, 'bubble_outline', '#000000'))

    def get_alignment(self):
        return getattr(self, 'alignment', 'center') or 'center'

    def get_line_spacing(self):
        try:
            val = float(getattr(self, 'line_spacing', 1.1) or 1.1)
            return max(0.6, min(val, 5.0))
        except (TypeError, ValueError):
            return 1.1

    def get_char_spacing(self):
        try:
            val = float(getattr(self, 'char_spacing', 100.0) or 100.0)
            return max(10.0, min(val, 500.0))
        except (TypeError, ValueError):
            return 100.0

    def get_margins(self):
        margins = getattr(self, 'margins', {'top': 0, 'right': 0, 'bottom': 0, 'left': 0})
        for key in ('top', 'right', 'bottom', 'left'):
            if key not in margins:
                margins[key] = 12
        return margins

    def _build_segments_from_plain(self, text, font, color):
        segment = {
            'text': text or '',
            'font': self.font_to_dict(font),
            'color': color.name(),
            'underline': font.underline(),
        }
        return [segment]

    def _segments_to_plain_text(self, segments):
        if not segments:
            return ''
        return ''.join(seg.get('text', '') for seg in segments)

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
        self.hovered_area = None
        self.trash_icon_rect = QRect()
        self.edit_icon_rect = QRect()
        self.hovering_edit_icon = False

        self.transform_mode = False
        self.transform_handles = {}
        self.transform_hover_handle = None
        self._set_active_transform(None)
        self.transform_handle_size = 14

        # --- Variabel baru untuk deteksi interaktif ---
        self.pending_bubble_polygon = None
        self.pending_bubble_rect = QRect()
        self.pending_trash_icon_rect = QRect()
        self.hovering_pending_trash = False

        self.detected_items = [] # List of dicts {'polygon': QPolygon, 'text': str}
        self.hovered_item_index = -1
        # Hovered handle index for pen tool vertex highlighting
        self.hovered_handle_index = -1
        # Debug flag: when True, draw large red markers for polygon points and print coords
        self._debug_draw_pen_points = False

        self._transform_update_timer = QTimer(self)
        self._transform_update_timer.setSingleShot(True)
        self._transform_update_timer.timeout.connect(self._emit_transform_redraw)

    def _set_active_transform(self, transform_data):
        self.active_transform = transform_data
        try:
            self.main_window.set_transform_preview_active(bool(transform_data))
        except Exception:
            pass

    def get_selection_mode(self):
        return self.main_window.selection_mode_combo.currentText()

    def get_polygon_points(self):
        return self.polygon_points
    
    def set_transform_mode(self, enabled):
        if self.transform_mode == enabled:
            return
        self.transform_mode = bool(enabled)
        self._set_active_transform(None)
        self.transform_handles.clear()
        self.transform_hover_handle = None
        if self.transform_mode:
            self._refresh_transform_handles()
        self.update()
    
    def _rotate_point(self, point, angle_deg):
        rad = math.radians(angle_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        return QPointF(point.x() * cos_a - point.y() * sin_a,
                       point.x() * sin_a + point.y() * cos_a)

    def _image_point_to_widget(self, point):
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull():
            return QPointF(point)
        scale = getattr(self.main_window, "zoom_factor", 1.0) or 1.0
        label_size = self.size()
        pix_size = pixmap.size()
        offset_x = max(0.0, (label_size.width() - pix_size.width()) / 2.0)
        offset_y = max(0.0, (label_size.height() - pix_size.height()) / 2.0)
        return QPointF(point.x() * scale + offset_x, point.y() * scale + offset_y)

    def _widget_point_to_image(self, pos):
        result = self.main_window.unzoom_coords(pos, as_point=True)
        if isinstance(result, QPoint):
            return QPointF(result)
        if isinstance(result, QPointF):
            return result
        return None

    def _area_polygon(self, area):
        rect = QRectF(area.rect)
        points = [QPointF(rect.topLeft()), QPointF(rect.topRight()), QPointF(rect.bottomRight()), QPointF(rect.bottomLeft())]
        rotation = area.get_rotation() if hasattr(area, 'get_rotation') else float(getattr(area, 'rotation', 0.0))
        if abs(rotation) > 0.01:
            center = rect.center()
            transform = QTransform()
            transform.translate(center.x(), center.y())
            transform.rotate(rotation)
            transform.translate(-center.x(), -center.y())
            points = [transform.map(pt) for pt in points]
        return QPolygonF(points)

    def _area_polygon_widget(self, area):
        return [self._image_point_to_widget(pt) for pt in self._area_polygon(area)]

    def _point_in_area(self, area, image_point):
        if image_point is None:
            return False
        polygon = self._area_polygon(area)
        return polygon.containsPoint(QPointF(image_point), Qt.OddEvenFill)

    def _area_at_widget_pos(self, pos):
        image_point = self._widget_point_to_image(pos)
        if image_point is None:
            return None
        for area in reversed(getattr(self.main_window, 'typeset_areas', [])):
            if self._point_in_area(area, image_point):
                return area
        return None

    def _image_dimensions(self):
        pixmap = getattr(self.main_window, 'original_pixmap', None)
        if pixmap is not None and not pixmap.isNull():
            return pixmap.width(), pixmap.height()
        pil_image = getattr(self.main_window, 'current_image_pil', None)
        if pil_image is not None:
            try:
                return pil_image.width, pil_image.height
            except AttributeError:
                try:
                    w, h = pil_image.size
                    return w, h
                except Exception:
                    pass
        return None, None

    def _clamp_rectf_to_image(self, rectf):
        w, h = self._image_dimensions()
        if rectf is None or w is None or h is None or w <= 0 or h <= 0:
            return QRectF(rectf)
        x = max(0.0, min(rectf.x(), float(w - 1)))
        y = max(0.0, min(rectf.y(), float(h - 1)))
        width = rectf.width()
        height = rectf.height()
        width = max(1.0, min(width, float(w) - x))
        height = max(1.0, min(height, float(h) - y))
        return QRectF(x, y, width, height)

    def _update_area_polygon_from_delta(self, area, orig_polygon_points, dx, dy):
        if not orig_polygon_points:
            return
        w, h = self._image_dimensions()
        updated_points = []
        for pt in orig_polygon_points:
            new_x = pt.x() + dx
            new_y = pt.y() + dy
            if w is not None and w > 0:
                new_x = max(0.0, min(new_x, w - 1))
            if h is not None and h > 0:
                new_y = max(0.0, min(new_y, h - 1))
            updated_points.append(QPoint(int(round(new_x)), int(round(new_y))))
        if updated_points:
            area.polygon = QPolygon(updated_points)

    def _update_area_polygon_for_scale(self, area, orig_polygon_points, anchor_point, orig_rect, new_rect):
        if not orig_polygon_points or anchor_point is None:
            return
        w, h = self._image_dimensions()
        orig_width = max(1.0, float(orig_rect.width()))
        orig_height = max(1.0, float(orig_rect.height()))
        new_width = max(1.0, float(new_rect.width()))
        new_height = max(1.0, float(new_rect.height()))
        scale_x = new_width / orig_width
        scale_y = new_height / orig_height
        anchor_x = anchor_point.x()
        anchor_y = anchor_point.y()
        updated_points = []
        for pt in orig_polygon_points:
            vec_x = pt.x() - anchor_x
            vec_y = pt.y() - anchor_y
            new_x = anchor_x + vec_x * scale_x
            new_y = anchor_y + vec_y * scale_y
            if w is not None and w > 0:
                new_x = max(0.0, min(new_x, w - 1))
            if h is not None and h > 0:
                new_y = max(0.0, min(new_y, h - 1))
            updated_points.append(QPoint(int(round(new_x)), int(round(new_y))))
        if updated_points:
            area.polygon = QPolygon(updated_points)

    def _build_local_transforms(self, area):
        rotation = area.get_rotation() if hasattr(area, 'get_rotation') else float(getattr(area, 'rotation', 0.0))
        center = QPointF(area.rect.center())
        to_local = QTransform()
        to_local.translate(-center.x(), -center.y())
        to_local.rotate(-rotation)
        from_local, invertible = to_local.inverted()
        if not invertible:
            from_local = QTransform()
            from_local.translate(center.x(), center.y())
        return to_local, from_local, center, rotation

    def _refresh_transform_handles(self):
        self.transform_handles.clear()
        if not self.transform_mode:
            return
        area = getattr(self.main_window, 'selected_typeset_area', None)
        if not area or area not in getattr(self.main_window, 'typeset_areas', []):
            return
        polygon = self._area_polygon_widget(area)
        if len(polygon) < 4:
            return
        handle_size = float(self.transform_handle_size)
        half = handle_size / 2.0
        for key, point in zip(('nw', 'ne', 'se', 'sw'), polygon):
            self.transform_handles[key] = QRectF(point.x() - half, point.y() - half, handle_size, handle_size)
        center_widget = self._image_point_to_widget(QPointF(area.rect.center()))
        top_center = QPointF((polygon[0].x() + polygon[1].x()) / 2.0, (polygon[0].y() + polygon[1].y()) / 2.0)
        direction = QPointF(top_center.x() - center_widget.x(), top_center.y() - center_widget.y())
        length = math.hypot(direction.x(), direction.y())
        if length < 1e-3:
            direction = QPointF(0.0, -1.0)
        else:
            direction = QPointF(direction.x() / length, direction.y() / length)
        rotation_offset = max(30.0, handle_size * 2.0)
        rotation_center = QPointF(top_center.x() + direction.x() * rotation_offset,
                                  top_center.y() + direction.y() * rotation_offset)
        self.transform_handles['rotate'] = QRectF(rotation_center.x() - half, rotation_center.y() - half, handle_size, handle_size)
        self.transform_handles['_points_widget'] = polygon
        self.transform_handles['_rotation_line'] = (top_center, rotation_center)
        self.transform_handles['_center_widget'] = center_widget

    def _emit_transform_redraw(self):
        if not self.main_window:
            return
        try:
            self.main_window.schedule_typeset_redraw(45)
        except Exception:
            traceback.print_exc()

    def _set_transform_hover_handle(self, handle):
        if self.transform_hover_handle == handle:
            return
        self.transform_hover_handle = handle
        if not self.transform_mode or self.active_transform:
            return
        if handle == 'rotate':
            self.setCursor(Qt.SizeAllCursor)
        elif handle in ('nw', 'se'):
            self.setCursor(Qt.SizeFDiagCursor)
        elif handle in ('ne', 'sw'):
            self.setCursor(Qt.SizeBDiagCursor)
        else:
            self.setCursor(Qt.OpenHandCursor)

    def _handle_transform_mouse_press(self, event):
        if not self.transform_mode or self.main_window.is_in_confirmation_mode:
            return False
        if event.button() not in (Qt.LeftButton, Qt.RightButton):
            return False
        self._refresh_transform_handles()
        selected_area = getattr(self.main_window, 'selected_typeset_area', None)
        area_at = self._area_at_widget_pos(event.pos())
        if area_at and area_at is not selected_area:
            self.main_window.set_selected_area(area_at)
            selected_area = area_at
            self._refresh_transform_handles()
        if event.button() == Qt.RightButton:
            return selected_area is not None
        if not selected_area:
            self._set_active_transform(None)
            self.update()
            return False
        image_point = self._widget_point_to_image(event.pos())
        if image_point is None:
            return False
        handle = None
        for key, rect in self.transform_handles.items():
            if key.startswith('_'):
                continue
            try:
                if rect and hasattr(rect, 'contains') and rect.contains(QPointF(event.pos())):
                    handle = key
                    break
            except Exception:
                # Defensive: ignore malformed transform handle entries
                continue
        if handle == 'rotate':
            center = QPointF(selected_area.rect.center())
            vector = QPointF(image_point.x() - center.x(), image_point.y() - center.y())
            if math.hypot(vector.x(), vector.y()) < 1e-3:
                return False
            base_rotation = selected_area.get_rotation() if hasattr(selected_area, 'get_rotation') else float(getattr(selected_area, 'rotation', 0.0))
            start_angle = math.degrees(math.atan2(vector.y(), vector.x()))
            transform = {
                'type': 'rotate',
                'area': selected_area,
                'center': center,
                'base_rotation': base_rotation,
                'start_angle': start_angle,
                'orig_polygon': [QPointF(pt) for pt in selected_area.polygon] if getattr(selected_area, 'polygon', None) else None,
            }
            self._set_active_transform(transform)
            self.transform_hover_handle = None
            self.setCursor(Qt.ClosedHandCursor)
            return True
        elif handle in ('nw', 'ne', 'se', 'sw'):
            anchor_map = {
                'nw': ('bottomRight', 'topLeft'),
                'ne': ('bottomLeft', 'topRight'),
                'se': ('topLeft', 'bottomRight'),
                'sw': ('topRight', 'bottomLeft'),
            }
            anchor_attr, _ = anchor_map[handle]
            orig_rect = QRectF(selected_area.rect)
            rotation = selected_area.get_rotation() if hasattr(selected_area, 'get_rotation') else float(getattr(selected_area, 'rotation', 0.0))
            center = QPointF(orig_rect.center())
            to_local = QTransform()
            to_local.translate(-center.x(), -center.y())
            to_local.rotate(-rotation)
            from_local, invertible = to_local.inverted()
            if not invertible:
                from_local = QTransform()
                from_local.rotate(rotation)
                from_local.translate(center.x(), center.y())
            anchor_point = QPointF(getattr(orig_rect, anchor_attr)())
            x_dir = -1 if handle in ('nw', 'sw') else 1
            y_dir = -1 if handle in ('nw', 'ne') else 1
            transform = {
                'type': 'scale',
                'area': selected_area,
                'handle': handle,
                'orig_rect': orig_rect,
                'anchor_point': anchor_point,
                'to_local': to_local,
                'from_local': from_local,
                'rotation': rotation,
                'x_dir': x_dir,
                'y_dir': y_dir,
                'min_size': 12.0,
                'orig_polygon': [QPointF(pt) for pt in selected_area.polygon] if getattr(selected_area, 'polygon', None) else None,
            }
            self._set_active_transform(transform)
            self.transform_hover_handle = None
            self.setCursor(Qt.SizeFDiagCursor if handle in ('nw', 'se') else Qt.SizeBDiagCursor)
            return True
        elif self._point_in_area(selected_area, image_point):
            transform = {
                'type': 'move',
                'area': selected_area,
                'start_mouse': QPointF(image_point),
                'orig_rect': QRectF(selected_area.rect),
                'orig_polygon': [QPointF(pt) for pt in selected_area.polygon] if getattr(selected_area, 'polygon', None) else None,
            }
            self._set_active_transform(transform)
            self.transform_hover_handle = None
            self.setCursor(Qt.ClosedHandCursor)
            return True
        else:
            self.main_window.clear_selected_area()
            self._set_active_transform(None)
            self.update()
            return False

    def _handle_transform_mouse_move(self, event):
        if not self.transform_mode:
            return False
        self.current_mouse_pos = event.pos()
        if self.active_transform:
            try:
                image_point = self._widget_point_to_image(event.pos())
                transform_type = self.active_transform.get('type') if isinstance(self.active_transform, dict) else None
                if transform_type == 'move':
                    self._update_transform_move(image_point)
                elif transform_type == 'rotate':
                    self._update_transform_rotate(image_point)
                elif transform_type == 'scale':
                    self._update_transform_scale(image_point)
            except Exception as exc:
                # Defensive: if something unexpected happens during transform, cancel active transform to avoid crash
                try:
                    print(f"Warning: transform move error: {exc}")
                except Exception:
                    pass
                self._set_active_transform(None)
                self._refresh_transform_handles()
                self.update()
                return False
            self._refresh_transform_handles()
            self.update()
            return True
        self._refresh_transform_handles()
        handle = None
        for key, rect in self.transform_handles.items():
            if key.startswith('_'):
                continue
            if rect.contains(QPointF(event.pos())):
                handle = key
                break
        self._set_transform_hover_handle(handle)
        self.hovered_area = self._area_at_widget_pos(event.pos())
        self.update()
        return True

    def _handle_transform_mouse_release(self, event):
        if not self.transform_mode or not self.active_transform:
            return False
        if event.button() != Qt.LeftButton:
            return False
        self._set_active_transform(None)
        self._refresh_transform_handles()
        self._set_transform_hover_handle(None)
        self.setCursor(Qt.OpenHandCursor)
        if hasattr(self.main_window, 'redraw_all_typeset_areas'):
            self.main_window.redraw_all_typeset_areas()
        self.update()
        return True

    def _update_transform_move(self, image_point):
        try:
            if image_point is None:
                return
            info = self.active_transform
            # Validate transform info
            if not info or not isinstance(info, dict):
                return
            if any(k not in info for k in ('area', 'orig_rect', 'start_mouse')):
                return

            area = info.get('area')
            start_rect = QRectF(info.get('orig_rect'))
            start_mouse = info.get('start_mouse')
            if start_mouse is None:
                return

            # Compute deltas and translate
            delta_x = image_point.x() - start_mouse.x()
            delta_y = image_point.y() - start_mouse.y()
            start_rect.translate(delta_x, delta_y)

            # Ensure area object is valid and has rect attribute
            if area is None:
                return
            clamped_rectf = self._clamp_rectf_to_image(start_rect)
            new_rect = QRect(
                int(round(clamped_rectf.x())),
                int(round(clamped_rectf.y())),
                max(1, int(round(clamped_rectf.width()))),
                max(1, int(round(clamped_rectf.height())))
            )

            orig_rect = QRectF(info.get('orig_rect'))
            dx = new_rect.x() - int(round(orig_rect.x()))
            dy = new_rect.y() - int(round(orig_rect.y()))

            try:
                area.rect = new_rect
            except Exception:
                return

            if info.get('orig_polygon'):
                try:
                    self._update_area_polygon_from_delta(area, info['orig_polygon'], dx, dy)
                except Exception:
                    area.polygon = QPolygon([new_rect.topLeft(), new_rect.topRight(), new_rect.bottomRight(), new_rect.bottomLeft()])
        except Exception as exc:
            try:
                print(f"Warning: _update_transform_move failed: {exc}")
            except Exception:
                pass
            return
        if hasattr(self.main_window, 'redo_stack'):
            try:
                self.main_window.redo_stack.clear()
            except Exception:
                pass
        if not self._transform_update_timer.isActive():
            self._transform_update_timer.start(16)

    def _update_transform_rotate(self, image_point):
        try:
            if image_point is None:
                return
            info = self.active_transform
            if not info or 'area' not in info or 'center' not in info or 'base_rotation' not in info or 'start_angle' not in info:
                return
            area = info.get('area')
            center = info.get('center')
            base_rotation = info.get('base_rotation')
            start_angle = info.get('start_angle')
            vector = QPointF(image_point.x() - center.x(), image_point.y() - center.y())
            if math.hypot(vector.x(), vector.y()) < 1e-3:
                return
            current_angle = math.degrees(math.atan2(vector.y(), vector.x()))
            new_rotation = (base_rotation + (current_angle - start_angle)) % 360.0
            if hasattr(area, 'set_rotation'):
                area.set_rotation(new_rotation)
            else:
                try:
                    area.rotation = float(new_rotation)
                except Exception:
                    pass
            if hasattr(self.main_window, 'redo_stack'):
                try:
                    self.main_window.redo_stack.clear()
                except Exception:
                    pass
            if not self._transform_update_timer.isActive():
                self._transform_update_timer.start(16)
        except Exception as exc:
            try:
                print(f"Warning: _update_transform_rotate failed: {exc}")
            except Exception:
                pass
            return

    def _update_transform_scale(self, image_point):
        try:
            if image_point is None:
                return
            info = self.active_transform
            if not info:
                return
            # Required fields
            required = ('area', 'to_local', 'from_local', 'anchor_point', 'x_dir', 'y_dir')
            if any(k not in info for k in required):
                return
            area = info['area']
            to_local = info['to_local']
            from_local = info['from_local']
            anchor_local = to_local.map(info['anchor_point'])
            current_local = to_local.map(QPointF(image_point))
            min_size = float(info.get('min_size', 12.0))
            x_dir = info['x_dir']
            y_dir = info['y_dir']
            raw_width = (anchor_local.x() - current_local.x()) if x_dir == -1 else (current_local.x() - anchor_local.x())
            raw_height = (anchor_local.y() - current_local.y()) if y_dir == -1 else (current_local.y() - anchor_local.y())
            width = max(min_size, raw_width)
            height = max(min_size, raw_height)
            new_center_local = QPointF(anchor_local.x() - x_dir * width / 2.0,
                                       anchor_local.y() - y_dir * height / 2.0)
            new_center_global = from_local.map(new_center_local)
            new_rectf = QRectF(new_center_global.x() - width / 2.0,
                               new_center_global.y() - height / 2.0,
                               width,
                               height)
            clamped_rectf = self._clamp_rectf_to_image(new_rectf)
            new_rect = QRect(
                int(round(clamped_rectf.x())),
                int(round(clamped_rectf.y())),
                max(1, int(round(clamped_rectf.width()))),
                max(1, int(round(clamped_rectf.height())))
            )
            try:
                area.rect = new_rect
            except Exception:
                return

            orig_rect = QRectF(info.get('orig_rect', QRectF(area.rect)))
            orig_polygon = info.get('orig_polygon')
            anchor_point = info.get('anchor_point')
            if orig_polygon and anchor_point is not None:
                try:
                    self._update_area_polygon_for_scale(
                        area,
                        orig_polygon,
                        anchor_point,
                        orig_rect,
                        clamped_rectf
                    )
                except Exception:
                    area.polygon = QPolygon([new_rect.topLeft(), new_rect.topRight(), new_rect.bottomRight(), new_rect.bottomLeft()])

            if hasattr(self.main_window, 'redo_stack'):
                try:
                    self.main_window.redo_stack.clear()
                except Exception:
                    pass
            if not self._transform_update_timer.isActive():
                self._transform_update_timer.start(16)
        except Exception as exc:
            try:
                print(f"Warning: _update_transform_scale failed: {exc}")
            except Exception:
                pass
            return

    def set_pending_item(self, polygon):
        """Menetapkan item yang terdeteksi untuk konfirmasi pengguna."""
        self.clear_selection()
        if polygon and not polygon.isEmpty():
            self.pending_bubble_polygon = polygon
            self.pending_bubble_rect = polygon.boundingRect()
        else:
            self.pending_bubble_polygon = None
            self.pending_bubble_rect = QRect()
        self.update()

    def cancel_pending_item(self):
        """Membatalkan item yang menunggu konfirmasi."""
        self.pending_bubble_polygon = None
        self.pending_bubble_rect = QRect()
        self.main_window.statusBar().showMessage("Deteksi dibatalkan.", 2000)
        self.update()

    def set_detected_items(self, items):
        self.detected_items = items
        self.hovered_item_index = -1
        self.update()

    def clear_detected_items(self):
        self.detected_items = []
        self.update()

    def mouseDoubleClickEvent(self, event):
        if not self.main_window.original_pixmap: return
        try:
            if getattr(self.main_window, 'dispatch_mouse_shortcut', None):
                if self.main_window.dispatch_mouse_shortcut('double', event.button()):
                    return
        except Exception:
            pass
        if self.main_window.is_in_confirmation_mode: return
        unzoomed_pos = self.main_window.unzoom_coords(event.pos(), as_point=True)
        if unzoomed_pos:
            posf = QPointF(unzoomed_pos) if isinstance(unzoomed_pos, QPoint) else unzoomed_pos
            for area in reversed(self.main_window.typeset_areas):
                if self._point_in_area(area, posf):
                    self.areaDoubleClicked.emit(area)
                    return

    def mousePressEvent(self, event):
        if not self.main_window.original_pixmap: return
        # Allow app-level mouse shortcuts to intercept the event
        try:
            if getattr(self.main_window, 'dispatch_mouse_shortcut', None):
                if self.main_window.dispatch_mouse_shortcut('press', event.button()):
                    return
        except Exception:
            pass
        
        # --- Logika baru untuk menangani item yang menunggu konfirmasi ---
        if self.pending_bubble_polygon:
            unzoomed_pos = self.main_window.unzoom_coords(event.pos(), as_point=True)
            if self.pending_trash_icon_rect.contains(event.pos()):
                self.cancel_pending_item()
                return

            if unzoomed_pos and self.pending_bubble_polygon.containsPoint(unzoomed_pos, Qt.OddEvenFill):
                if event.button() == Qt.RightButton:
                    self.main_window.confirm_pending_item(self.pending_bubble_polygon)
                    self.pending_bubble_polygon = None # Hapus setelah dikonfirmasi
                    self.update()
                    return
                if event.button() == Qt.MiddleButton:
                    self.cancel_pending_item()
                    return

        if self.main_window.is_in_confirmation_mode:
            if event.button() == Qt.LeftButton and self.hovered_item_index != -1:
                self.main_window.remove_detected_item(self.hovered_item_index)
                self.hovered_item_index = -1
                return

        # Determine selection mode early so pen/manual modes aren't blocked by hovered area logic
        mode = self.get_selection_mode()
        manual_rect = "Manual Text (Rect)" in mode
        manual_polygon = "Manual Text (Pen)" in mode
        pen_mode = (mode == "Pen Tool")
        transform_mode = (mode == "Transform (Hand)")

        if transform_mode and not self.main_window.is_in_confirmation_mode:
            if self._handle_transform_mouse_press(event):
                return

        # If hovering over an existing area, allow area-level actions unless we're in pen/manual-pen mode
        if self.hovered_area and not (pen_mode or manual_polygon or transform_mode):
            if self.trash_icon_rect.contains(event.pos()):
                if self.main_window.is_in_confirmation_mode: return
                self.main_window.delete_typeset_area(self.hovered_area)
                self.hovered_area = None
                self.update()
                return
            if self.edit_icon_rect.contains(event.pos()):
                if self.main_window.is_in_confirmation_mode: return
                self.hovering_edit_icon = False
                self.main_window.start_inline_edit(self.hovered_area)
                self.hovered_area = None
                self.update()
                return
            # Right-click on an area shows context menu for area-level actions
            mode = self.get_selection_mode()
            manual_polygon = "Manual Text (Pen)" in mode
            pen_mode = (mode == "Pen Tool")
            
            if event.button() == Qt.RightButton and not (pen_mode or manual_polygon):
                try:
                    from PyQt5.QtWidgets import QMenu, QAction
                    menu = QMenu(self)
                    
                    # Copy Action
                    copy_action = menu.addAction("Copy Typeset")
                    menu.addSeparator()
                    
                    revert_action = menu.addAction("Revert to Global Defaults")
                    
                    action = menu.exec_(self.mapToGlobal(event.pos()))
                    
                    if action == copy_action:
                        self.main_window.set_selected_area(self.hovered_area)
                        self.main_window.copy_selected_typeset_area()
                        return

                    if action == revert_action:
                        notes = self.hovered_area.review_notes if isinstance(getattr(self.hovered_area, 'review_notes', {}), dict) else {}
                        for legacy in ('manual_inpaint', 'manual'):
                            if legacy in notes:
                                notes.pop(legacy, None)
                        self.hovered_area.review_notes = notes
                        if hasattr(self.hovered_area, 'clear_override'):
                            self.hovered_area.clear_override('use_inpaint')
                            self.hovered_area.clear_override('use_background_box')
                        try:
                            self.main_window.redraw_all_typeset_areas()
                        except Exception:
                            pass
                        if self.main_window.selected_typeset_area is self.hovered_area:
                            self.main_window._sync_cleanup_controls_from_selection()
                        self.main_window.statusBar().showMessage("Area reverted to global defaults.", 2500)
                        self.update()
                        return
                except Exception:
                    pass

        # Right click on empty space -> Paste option
        mode = self.get_selection_mode()
        manual_polygon = "Manual Text (Pen)" in mode
        pen_mode = (mode == "Pen Tool")

        if event.button() == Qt.RightButton and not self.hovered_area and not self.main_window.is_in_confirmation_mode and not self.pending_bubble_polygon and not (pen_mode or manual_polygon):
            try:
                clipboard = QApplication.clipboard()
                text = clipboard.text()
                if text and ('"type": "manga_ocr_typeset"' in text):
                    from PyQt5.QtWidgets import QMenu
                    menu = QMenu(self)
                    paste_action = menu.addAction("Paste Typeset")
                    action = menu.exec_(self.mapToGlobal(event.pos()))
                    if action == paste_action:
                        self.main_window.paste_typeset_area()
                        return
            except Exception:
                pass

        if event.button() == Qt.LeftButton:
            # If in pen/manual-pen mode, do not let hovered_area clicks block pen selection
            if (
                self.hovered_area
                and not self.trash_icon_rect.contains(event.pos())
                and not self.edit_icon_rect.contains(event.pos())
                and not (pen_mode or manual_polygon)
            ):
                if not self.main_window.is_in_confirmation_mode:
                    self.main_window.set_selected_area(self.hovered_area)
                self.update()
                return
            if not self.main_window.is_in_confirmation_mode and not self.hovered_area:
                self.main_window.clear_selected_area()

        if event.button() == Qt.LeftButton:
            mode = self.get_selection_mode()
            manual_rect = "Manual Text (Rect)" in mode
            manual_polygon = "Manual Text (Pen)" in mode
            if ("Bubble Finder" in mode or "Direct OCR" in mode or manual_rect) and not manual_polygon:
                self.clear_selection()
                self.selection_start = event.pos()
                self.selection_end = event.pos()
                self.dragging = True
            elif mode == "Pen Tool" or manual_polygon:
                if not self.polygon_points:
                    self.clear_selection()
                self.polygon_points.append(event.pos())
                # If debug flag enabled, log the added point (widget coords)
                if getattr(self, '_debug_draw_pen_points', False):
                    try:
                        print(f"[DEBUG] Added polygon point: {event.pos().x()},{event.pos().y()}")
                    except Exception:
                        pass
                self.main_window.update_pen_tool_buttons_visibility(True)
            self.update()
    def mouseMoveEvent(self, event):
        self.current_mouse_pos = event.pos()
        mode = self.get_selection_mode()
        transform_mode = False

        # Cek hover di atas ikon tong sampah untuk item yang menunggu
        if self.pending_bubble_polygon:
            new_hover_state = self.pending_trash_icon_rect.contains(self.current_mouse_pos)
            if self.hovering_pending_trash != new_hover_state:
                self.hovering_pending_trash = new_hover_state
                self.update() # Perbarui untuk mengubah warna ikon

        if self.main_window.is_in_confirmation_mode:
            unzoomed_pos = self.main_window.unzoom_coords(self.current_mouse_pos, as_point=True)
            new_hover_index = -1
            if unzoomed_pos:
                # Iterasi terbalik agar item di atas terdeteksi dulu
                for i in range(len(self.detected_items) - 1, -1, -1):
                    item = self.detected_items[i]
                    if item['polygon'].containsPoint(unzoomed_pos, Qt.OddEvenFill):
                        new_hover_index = i
                        break
            if self.hovered_item_index != new_hover_index:
                self.hovered_item_index = new_hover_index
                self.update()
        else:
            transform_mode = (mode == "Transform (Hand)")
            if transform_mode and self._handle_transform_mouse_move(event):
                return
            manual_polygon = "Manual Text (Pen)" in mode
            pen_mode = (mode == "Pen Tool")

            if pen_mode or manual_polygon:
                # In pen drawing modes we avoid changing hovered_area to prevent interference
                new_hover_area = self.hovered_area
            else:
                unzoomed_pos = self.main_window.unzoom_coords(self.current_mouse_pos, as_point=True)
                new_hover_area = None
                if unzoomed_pos:
                    posf = QPointF(unzoomed_pos) if isinstance(unzoomed_pos, QPoint) else unzoomed_pos
                    for area in reversed(self.main_window.typeset_areas):
                        if self._point_in_area(area, posf):
                            new_hover_area = area
                            break

            if self.hovered_area != new_hover_area:
                self.hovered_area = new_hover_area
                self.update()

        edit_hover_state = (
            bool(self.hovered_area)
            and not self.main_window.is_in_confirmation_mode
            and not transform_mode
            and not self.edit_icon_rect.isNull()
            and self.edit_icon_rect.contains(event.pos())
        )
        if edit_hover_state != self.hovering_edit_icon:
            self.hovering_edit_icon = edit_hover_state
            self.update()

        manual_rect = "Manual Text (Rect)" in mode
        manual_polygon = "Manual Text (Pen)" in mode
        if ("Bubble Finder" in mode or "Direct OCR" in mode or manual_rect) and not manual_polygon:
            if self.dragging:
                self.selection_end = self.current_mouse_pos
                self.update()
        elif mode == "Pen Tool" or manual_polygon:
            if self.polygon_points:
                # detect nearby handles for hover feedback
                self.hovered_handle_index = -1
                if self.current_mouse_pos:
                    for i, p in enumerate(self.polygon_points):
                        if QPointF(p).toPoint().manhattanLength() is not None:
                            dist = (p - self.current_mouse_pos).manhattanLength()
                            if dist <= 10:
                                self.hovered_handle_index = i
                                break
                self.update()
    def mouseReleaseEvent(self, event):
        mode = self.get_selection_mode()
        # Allow app-level mouse shortcuts to intercept release events
        try:
            if getattr(self.main_window, 'dispatch_mouse_shortcut', None):
                if self.main_window.dispatch_mouse_shortcut('release', event.button()):
                    return
        except Exception:
            pass
        manual_polygon = "Manual Text (Pen)" in mode
        manual_rect = "Manual Text (Rect)" in mode

        if mode == "Transform (Hand)" and self._handle_transform_mouse_release(event):
            return

        if event.button() == Qt.RightButton and (mode == "Pen Tool" or manual_polygon):
            if len(self.polygon_points) >= 3:
                self.main_window.confirm_pen_selection()
            else:
                self.main_window.cancel_pen_selection()
            return

        if event.button() == Qt.LeftButton:
            if ("Bubble Finder" in mode or "Direct OCR" in mode or manual_rect) and not manual_polygon:
                if self.dragging:
                    self.dragging = False
                    self.selection_rect = QRect(self.selection_start, self.selection_end).normalized()
                    
                    if self.selection_rect.width() > 10 and self.selection_rect.height() > 10:
                        unzoomed_rect = self.main_window.unzoom_coords(self.selection_rect)
                        if unzoomed_rect:
                            if "Bubble Finder" in mode:
                                self.main_window.find_bubble_in_rect(unzoomed_rect)
                            elif mode == "Direct OCR (Rect)":
                                self.main_window.process_rect_area(self.selection_rect)
                            elif mode == "Direct OCR (Oval)":
                                path = QPainterPath()
                                path.addEllipse(QRectF(self.selection_rect))
                                polygon = path.toFillPolygon().toPolygon()
                                self.main_window.process_polygon_area(list(polygon))
                            elif manual_rect:
                                self.main_window.process_rect_area(self.selection_rect)
                        else:
                            self.clear_selection()
                    
                    if "Bubble Finder" not in mode:
                        self.clear_selection()

                    self.update()
    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.main_window.original_pixmap: return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self.main_window.is_in_confirmation_mode and self.detected_items:
            painter.save()
            scale = self.main_window.zoom_factor
            pixmap = self.pixmap()
            if pixmap and not pixmap.isNull():
                label_size = self.size()
                pixmap_size = pixmap.size()
                offset_x = max(0, (label_size.width() - pixmap_size.width()) // 2)
                offset_y = max(0, (label_size.height() - pixmap_size.height()) // 2)
                painter.translate(offset_x, offset_y)
                painter.scale(scale, scale)

            for i, item in enumerate(self.detected_items):
                path = QPainterPath()
                path.addPolygon(QPolygonF(item['polygon']))

                if i == self.hovered_item_index:
                    painter.fillPath(path, QColor(255, 80, 80, 150))
                    painter.setPen(QPen(QColor(255, 100, 100), 3 / scale))
                else:
                    # Biru untuk bubble, Hijau untuk teks
                    fill_color = QColor(0, 120, 215, 100) if item['text'] is None else QColor(0, 180, 100, 100)
                    pen_color = QColor(90, 180, 255) if item['text'] is None else QColor(90, 220, 150)
                    painter.fillPath(path, fill_color)
                    painter.setPen(QPen(pen_color, 2 / scale))

                painter.drawPath(path)
            painter.restore()

        mode = self.get_selection_mode()
        if "Bubble Finder" in mode or "Direct OCR" in mode:
            if self.dragging:
                rect = QRect(self.selection_start, self.selection_end).normalized()
                # Glow effect: draw a soft semi-transparent larger rect behind
                glow_pen = QPen(QColor(30, 150, 240, 40), 10)
                glow_pen.setStyle(Qt.SolidLine)
                painter.setPen(glow_pen)
                if "Oval" in mode:
                    painter.drawEllipse(rect.adjusted(-4, -4, 4, 4))
                else:
                    painter.drawRoundedRect(rect.adjusted(-4, -4, 4, 4), 6, 6)

                # Main outline dashed
                outline_pen = QPen(QColor(0, 140, 255), 2, Qt.DashLine)
                painter.setPen(outline_pen)
                painter.setBrush(QColor(0, 140, 255, 55))
                if "Oval" in mode:
                    painter.drawEllipse(rect)
                else:
                    painter.drawRoundedRect(rect, 6, 6)
        elif (mode == "Pen Tool" or "Manual Text (Pen)" in mode) and self.polygon_points:
            base_pen = QPen(QColor(90, 220, 130), 2, Qt.SolidLine)
            base_pen.setCapStyle(Qt.RoundCap)
            painter.setPen(base_pen)
            # draw filled translucent polygon if more than 2 points
            if len(self.polygon_points) > 2:
                poly = QPolygon(self.polygon_points)
                painter.setBrush(QColor(80, 200, 120, 50))
                painter.drawPolygon(poly)
            # draw polyline
            painter.setBrush(Qt.NoBrush)
            painter.drawPolyline(QPolygon(self.polygon_points))

            # draw handles (stronger with hover) and small index labels
            handle_pen = QPen(QColor(20, 20, 20), 1)
            painter.setPen(handle_pen)
            for idx, pt in enumerate(self.polygon_points):
                hover = (idx == getattr(self, 'hovered_handle_index', -1))
                size = 14 if hover else 12
                handle_rect = QRect(pt.x() - size//2, pt.y() - size//2, size, size)
                # draw outline for high contrast
                painter.setPen(QPen(QColor(10, 10, 10), 2))
                painter.setBrush(QColor(255, 255, 255) if not hover else QColor(255, 200, 120))
                painter.drawEllipse(handle_rect)
                painter.setPen(QPen(QColor(30, 30, 30), 1))
                painter.drawText(handle_rect, Qt.AlignCenter, str(idx + 1))
                painter.setPen(handle_pen)

            # rubber-band to current mouse
            if self.current_mouse_pos:
                rubber_pen = QPen(QColor(160, 255, 180), 1, Qt.DashLine)
                painter.setPen(rubber_pen)
                painter.drawLine(self.polygon_points[-1], self.current_mouse_pos)

            # Debug overlay: draw big red markers on each polygon point when enabled
            if getattr(self, '_debug_draw_pen_points', False):
                dbg_pen = QPen(QColor(200, 20, 20), 2)
                dbg_brush = QBrush(QColor(200, 20, 20, 180))
                painter.setPen(dbg_pen)
                painter.setBrush(dbg_brush)
                for pt in self.polygon_points:
                    r = QRect(pt.x() - 8, pt.y() - 8, 16, 16)
                    painter.drawEllipse(r)
        
        # --- Gambar item yang menunggu konfirmasi ---
        if self.pending_bubble_polygon:
            painter.save()
            scale = self.main_window.zoom_factor
            pixmap = self.pixmap()
            if pixmap and not pixmap.isNull():
                label_size = self.size()
                pixmap_size = pixmap.size()
                offset_x = max(0, (label_size.width() - pixmap_size.width()) // 2)
                offset_y = max(0, (label_size.height() - pixmap_size.height()) // 2)
                painter.translate(offset_x, offset_y)
                painter.scale(scale, scale)

            path = QPainterPath()
            path.addPolygon(QPolygonF(self.pending_bubble_polygon))
            
            # Gambar outline
            painter.setPen(QPen(QColor(255, 200, 0), 3 / scale, Qt.DashLine))
            painter.fillPath(path, QColor(255, 215, 120, 120))
            painter.drawPath(path)
            painter.restore()
            
            # Gambar ikon di atasnya
            zoomed_rect = self.main_window.zoom_coords(self.pending_bubble_rect)
            icon_size = 24; margin = 5
            self.pending_trash_icon_rect = QRect(zoomed_rect.topRight().x() - icon_size - margin, zoomed_rect.topRight().y() + margin, icon_size, icon_size)
            
            # Ikon Tong Sampah
            trash_color = QColor(255, 100, 100, 220) if self.hovering_pending_trash else QColor(255, 80, 80, 200)
            painter.setBrush(trash_color); painter.setPen(Qt.NoPen)
            painter.drawEllipse(self.pending_trash_icon_rect)
            pen = QPen(Qt.white, 2); painter.setPen(pen)
            painter.drawLine(self.pending_trash_icon_rect.topLeft() + QPoint(6,6), self.pending_trash_icon_rect.bottomRight() - QPoint(6,6))
            painter.drawLine(self.pending_trash_icon_rect.topRight() - QPoint(6,-6), self.pending_trash_icon_rect.bottomLeft() + QPoint(6,-6))

            # Draw small hint text near pending polygon
            hint = "Right-click to confirm • Middle-click to cancel"
            hint_rect = QRect(self.pending_trash_icon_rect.left() - 8 - 200, self.pending_trash_icon_rect.top() - 28, 200, 20)
            painter.setPen(QPen(QColor(240, 240, 240, 220)))
            painter.setBrush(QColor(0, 0, 0, 120))
            painter.drawRoundedRect(hint_rect, 6, 6)
            painter.drawText(hint_rect, Qt.AlignCenter, hint)


        # Draw override badges for areas that have per-area overrides
        try:
            for area in (self.main_window.typeset_areas or []):
                overrides = []
                default_inpaint = self.main_window._default_cleanup_value('use_inpaint')
                default_box = self.main_window._default_cleanup_value('use_background_box')
                override_inpaint = area.get_override('use_inpaint', None)
                override_box = area.get_override('use_background_box', None)
                if override_inpaint is not None and bool(override_inpaint) != bool(default_inpaint):
                    overrides.append('inpaint')
                if override_box is not None and bool(override_box) != bool(default_box):
                    overrides.append('background box')
                if overrides:
                    # draw a small badge at area's top-left (zoomed)
                    try:
                        zoomed = self.main_window.zoom_coords(area.rect)
                        badge_rect = QRect(zoomed.left() + 4, zoomed.top() + 4, 12, 12)
                        painter.setBrush(QColor(255, 200, 60, 230))
                        painter.setPen(Qt.NoPen)
                        painter.drawEllipse(badge_rect)
                    except Exception:
                        pass
        except Exception:
            pass

        selected_area = getattr(self.main_window, 'selected_typeset_area', None)
        if selected_area and selected_area in getattr(self.main_window, 'typeset_areas', []):
            try:
                polygon_points = self._area_polygon_widget(selected_area)
                if polygon_points:
                    path = QPainterPath()
                    path.moveTo(polygon_points[0])
                    for pt in polygon_points[1:]:
                        path.lineTo(pt)
                    path.closeSubpath()
                    painter.setPen(QPen(QColor(90, 180, 255, 200), 2, Qt.DashLine))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawPath(path)
                    if self.transform_mode:
                        self._refresh_transform_handles()
                        rotation_line = self.transform_handles.get('_rotation_line')
                        if rotation_line:
                            painter.setPen(QPen(QColor(160, 160, 160, 200), 1, Qt.SolidLine))
                            painter.drawLine(rotation_line[0], rotation_line[1])
                        for key, rect in self.transform_handles.items():
                            if key.startswith('_'):
                                continue
                            is_hover = (self.transform_hover_handle == key)
                            is_active = False
                            if self.active_transform:
                                t_type = self.active_transform.get('type')
                                if key == 'rotate' and t_type == 'rotate':
                                    is_active = True
                                elif key in ('nw', 'ne', 'se', 'sw') and t_type == 'scale' and self.active_transform.get('handle') == key:
                                    is_active = True
                            if key == 'rotate':
                                base_color = QColor(255, 200, 80)
                                if is_hover or is_active:
                                    base_color = QColor(255, 230, 120)
                                painter.setPen(QPen(QColor(120, 120, 120), 1))
                                painter.setBrush(base_color)
                            else:
                                base_color = QColor(245, 245, 245)
                                if is_hover or is_active:
                                    base_color = QColor(120, 200, 255)
                                painter.setPen(QPen(QColor(30, 30, 30), 1))
                                painter.setBrush(base_color)
                            painter.drawEllipse(rect)
            except Exception:
                pass

        if self.hovered_area and not self.main_window.is_in_confirmation_mode and mode != "Transform (Hand)":
            zoomed_rect = self.main_window.zoom_coords(self.hovered_area.rect)
            icon_size = 32
            margin = 6

            # Trash Icon
            self.trash_icon_rect = QRect(zoomed_rect.topRight().x() - icon_size - margin, zoomed_rect.topRight().y() + margin, icon_size, icon_size)
            painter.setBrush(QColor(255, 80, 80, 200)); painter.setPen(Qt.NoPen)
            painter.drawEllipse(self.trash_icon_rect)
            pen = QPen(Qt.white, 2); painter.setPen(pen)
            painter.drawLine(self.trash_icon_rect.topLeft() + QPoint(6,6), self.trash_icon_rect.bottomRight() - QPoint(6,6))
            painter.drawLine(self.trash_icon_rect.topRight() - QPoint(6,-6), self.trash_icon_rect.bottomLeft() + QPoint(6,-6))

            # Edit Icon
            self.edit_icon_rect = QRect(self.trash_icon_rect.left() - icon_size - margin, self.trash_icon_rect.top(), icon_size, icon_size)
            edit_color = QColor(80, 150, 255, 200)
            if self.hovering_edit_icon:
                edit_color = QColor(110, 180, 255, 230)
            painter.setBrush(edit_color); painter.setPen(Qt.NoPen)
            painter.drawEllipse(self.edit_icon_rect)
            painter.setPen(pen)
            # Draw a simple pencil
            poly = QPolygon([
                QPoint(10, 26), QPoint(10, 20), QPoint(20, 10),
                QPoint(24, 14), QPoint(14, 26), QPoint(10, 26)
            ])
            painter.drawPolyline(poly.translated(self.edit_icon_rect.topLeft()))
            painter.drawLine(QPoint(19,11)+self.edit_icon_rect.topLeft(), QPoint(22,14)+self.edit_icon_rect.topLeft())

            # If hovered area has overrides, show tooltip text near icons
            try:
                override_list = []
                default_inpaint = self.main_window._default_cleanup_value('use_inpaint')
                default_box = self.main_window._default_cleanup_value('use_background_box')
                override_inpaint = self.hovered_area.get_override('use_inpaint', None)
                override_box = self.hovered_area.get_override('use_background_box', None)
                if override_inpaint is not None and bool(override_inpaint) != bool(default_inpaint):
                    override_list.append(f"Inpaint: {override_inpaint}")
                if override_box is not None and bool(override_box) != bool(default_box):
                    override_list.append(f"Background box: {override_box}")
                if override_list:
                    hint = "Overrides: " + ", ".join(override_list)
                    hint_rect = QRect(zoomed_rect.left(), zoomed_rect.top() - 22, min(300, zoomed_rect.width()), 18)
                    painter.setPen(QPen(QColor(240, 240, 240, 220)))
                    painter.setBrush(QColor(0, 0, 0, 160))
                    painter.drawRoundedRect(hint_rect, 6, 6)
                    painter.drawText(hint_rect, Qt.AlignCenter, hint)
            except Exception:
                pass


    def clear_selection(self):
        self.selection_start = None
        self.selection_end = None
        self.selection_rect = QRect()
        self.dragging = False
        self.polygon_points = []
        self.current_mouse_pos = None
        # Jangan hapus item yang menunggu konfirmasi saat seleksi dibersihkan
        if self.main_window:
            self.main_window.update_pen_tool_buttons_visibility(False)
        self.update()
