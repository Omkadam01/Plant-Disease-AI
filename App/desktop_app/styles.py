"""Light green agriculture theme styles for PlantDiseaseAI."""

from __future__ import annotations

# Color palette
COLORS = {
    "primary": "#2E7D32",
    "primary_dark": "#1B5E20",
    "primary_light": "#4CAF50",
    "accent": "#66BB6A",
    "background": "#F1F8E9",
    "surface": "#FFFFFF",
    "surface_alt": "#E8F5E9",
    "text_primary": "#1B4332",
    "text_secondary": "#52796F",
    "text_muted": "#81A896",
    "border": "#C8E6C9",
    "success": "#43A047",
    "warning": "#F9A825",
    "danger": "#E53935",
    "confidence_high": "#2E7D32",
    "confidence_medium": "#F9A825",
    "confidence_low": "#E53935",
}

STYLESHEET = f"""
QMainWindow {{
    background-color: {COLORS['background']};
}}

QWidget {{
    font-family: "Segoe UI", "SF Pro Display", "Helvetica Neue", sans-serif;
    font-size: 13px;
    color: {COLORS['text_primary']};
}}

QLabel#appTitle {{
    font-size: 28px;
    font-weight: 700;
    color: {COLORS['primary_dark']};
}}

QLabel#appSubtitle {{
    font-size: 14px;
    color: {COLORS['text_secondary']};
}}

QLabel#sectionTitle {{
    font-size: 16px;
    font-weight: 600;
    color: {COLORS['primary']};
    padding-top: 8px;
}}

QPushButton {{
    background-color: {COLORS['primary']};
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 600;
    min-height: 20px;
}}

QPushButton:hover {{
    background-color: {COLORS['primary_dark']};
}}

QPushButton:pressed {{
    background-color: {COLORS['primary_dark']};
}}

QPushButton:disabled {{
    background-color: {COLORS['text_muted']};
}}

QPushButton#secondaryButton {{
    background-color: {COLORS['surface']};
    color: {COLORS['primary']};
    border: 2px solid {COLORS['primary']};
}}

QPushButton#secondaryButton:hover {{
    background-color: {COLORS['surface_alt']};
}}

QFrame#dropZone {{
    background-color: {COLORS['surface']};
    border: 2px dashed {COLORS['primary_light']};
    border-radius: 12px;
    min-height: 200px;
}}

QFrame#dropZone[dragActive="true"] {{
    background-color: {COLORS['surface_alt']};
    border-color: {COLORS['primary']};
}}

QFrame#predictionCard {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
    padding: 16px;
}}

QFrame#infoCard {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
    padding: 12px;
}}

QProgressBar {{
    border: none;
    border-radius: 6px;
    background-color: {COLORS['surface_alt']};
    text-align: center;
    color: white;
    font-weight: 600;
    min-height: 24px;
}}

QProgressBar::chunk {{
    border-radius: 6px;
    background-color: {COLORS['primary']};
}}

QScrollArea {{
    border: none;
    background-color: transparent;
}}

QTextEdit, QListWidget {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    padding: 8px;
}}

QStatusBar {{
    background-color: {COLORS['surface']};
    color: {COLORS['text_secondary']};
    border-top: 1px solid {COLORS['border']};
}}
"""
