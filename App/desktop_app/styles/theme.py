"""Modern white theme for PlantDiseaseAI desktop application."""

COLORS = {
    "background": "#F5F7FA",
    "surface": "#FFFFFF",
    "surface_alt": "#F0F2F5",
    "border": "#E4E7EB",
    "text_primary": "#1A1D21",
    "text_secondary": "#5F6B7A",
    "text_muted": "#9AA5B1",
    "accent": "#2563EB",
    "accent_hover": "#1D4ED8",
    "accent_light": "#EFF6FF",
    "success": "#059669",
    "success_bg": "#ECFDF5",
    "shadow": "rgba(15, 23, 42, 0.08)",
    "bar_fill": "#2563EB",
    "bar_bg": "#E8EDF3",
}

STYLESHEET = f"""
QMainWindow {{
    background-color: {COLORS['background']};
}}

QWidget {{
    font-family: "Segoe UI", "Inter", "SF Pro Display", sans-serif;
    font-size: 13px;
    color: {COLORS['text_primary']};
}}

QLabel#appTitle {{
    font-size: 22px;
    font-weight: 700;
    color: {COLORS['text_primary']};
}}

QLabel#appSubtitle {{
    font-size: 13px;
    color: {COLORS['text_secondary']};
}}

QLabel#sectionLabel {{
    font-size: 11px;
    font-weight: 600;
    color: {COLORS['text_muted']};
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

QFrame#card {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
}}

QPushButton {{
    background-color: {COLORS['accent']};
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 18px;
    font-weight: 600;
    min-height: 18px;
}}

QPushButton:hover {{
    background-color: {COLORS['accent_hover']};
}}

QPushButton:disabled {{
    background-color: {COLORS['text_muted']};
}}

QPushButton#secondaryBtn {{
    background-color: {COLORS['surface']};
    color: {COLORS['accent']};
    border: 1.5px solid {COLORS['accent']};
}}

QPushButton#secondaryBtn:hover {{
    background-color: {COLORS['accent_light']};
}}

QToolButton#predictionDropdown {{
    background: transparent;
    border: none;
    color: {COLORS['accent']};
    font-size: 16px;
    font-weight: 700;
    border-radius: 4px;
}}

QToolButton#predictionDropdown:hover {{
    background-color: {COLORS['accent_light']};
}}

QToolButton#predictionDropdown::menu-indicator {{
    image: none;
    width: 0;
}}

QComboBox {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    padding: 8px 12px;
    min-height: 20px;
}}

QComboBox::drop-down {{
    border: none;
    width: 24px;
}}

QComboBox QAbstractItemView {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    selection-background-color: {COLORS['accent_light']};
}}

QProgressBar {{
    border: none;
    border-radius: 4px;
    background-color: {COLORS['bar_bg']};
    min-height: 8px;
    max-height: 8px;
    text-align: center;
}}

QProgressBar::chunk {{
    background-color: {COLORS['bar_fill']};
    border-radius: 4px;
}}

QStatusBar {{
    background-color: {COLORS['surface']};
    color: {COLORS['text_secondary']};
    border-top: 1px solid {COLORS['border']};
    font-size: 12px;
}}
"""

CARD_STYLE = f"""
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
"""

IMAGE_PLACEHOLDER_STYLE = f"""
    background-color: {COLORS['surface_alt']};
    border: 1px dashed {COLORS['border']};
    border-radius: 10px;
    color: {COLORS['text_muted']};
    font-size: 12px;
"""
