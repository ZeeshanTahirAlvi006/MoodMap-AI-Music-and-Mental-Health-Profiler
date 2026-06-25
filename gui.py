import os
import sys
import re
from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QProgressBar, QComboBox, QSpinBox, QSlider, QFrame,
    QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea, QDialog,
    QMessageBox, QGridLayout, QSizePolicy, QToolTip, QTextBrowser
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPoint, QObject
from PyQt5.QtGui import QFont, QColor, QPalette, QBrush, QPainter, QPen
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Backend model imports
from domain import Respondent, GenreProfile, MentalHealthMetric
from engine import CorrelationEngine, predict_mental_health, AgeGroupAnalyzer
from recommender import RecommendationEngine
from ai_client import MistralClient, AIWorker
from pdf_export import PDFExporter
from dashboard import (
    create_genre_bar_chart, create_age_line_chart, create_anxiety_histogram,
    create_scatter_plot, create_correlation_heatmap, create_trend_chart,
    ReportGenerator
)

# Load environment variables
load_dotenv()

# Style constants for QSS
DARK_QSS = """
QMainWindow, #centralWidget, QStackedWidget, WelcomePage, QuestionnairePage, LoadingPage, ResultsPage, HistoryPage, ComparePage, QScrollArea, #scrollContent {
    background-color: #121212;
}
QWidget {
    color: #FFFFFF;
    font-family: 'Segoe UI', sans-serif;
}
QFrame.card {
    background-color: #181818;
    border-radius: 16px;
    border: 1px solid #282828;
}
QFrame.card_light {
    background-color: #282828;
    border-radius: 12px;
}
QLabel.title {
    font-size: 38px;
    font-weight: bold;
    color: #1DB954;
}
QLabel.heading {
    font-size: 24px;
    font-weight: bold;
    color: #FFFFFF;
}
QLabel.subheading {
    font-size: 18px;
    font-weight: 600;
    color: #FFFFFF;
}
QLabel.body {
    font-size: 14px;
    color: #FFFFFF;
}
QLabel.body_secondary {
    font-size: 14px;
    color: #B3B3B3;
}
QLabel.body_muted {
    font-size: 12px;
    color: #727272;
}
QPushButton.primary {
    background-color: #1DB954;
    color: #000000;
    font-weight: bold;
    border: none;
    border-radius: 20px;
    padding: 10px 24px;
    font-size: 14px;
}
QPushButton.primary:hover {
    background-color: #1ED760;
}
QPushButton.primary:disabled {
    background-color: #535353;
    color: #727272;
}
QPushButton.secondary {
    background-color: transparent;
    color: #FFFFFF;
    border: 1px solid #535353;
    border-radius: 20px;
    padding: 10px 24px;
    font-size: 14px;
}
QPushButton.secondary:hover {
    background-color: #282828;
    border-color: #B3B3B3;
}
QPushButton.danger {
    background-color: transparent;
    color: #E74C3C;
    border: 1px solid #E74C3C;
    border-radius: 20px;
    padding: 10px 24px;
    font-size: 14px;
}
QPushButton.danger:hover {
    background-color: rgba(231, 76, 60, 0.1);
}
QPushButton.toggle_active {
    background-color: #1DB954;
    color: #000000;
    font-weight: bold;
    border: none;
    border-radius: 8px;
    padding: 8px 16px;
}
QPushButton.toggle_inactive {
    background-color: #282828;
    color: #FFFFFF;
    border: 1px solid #535353;
    border-radius: 8px;
    padding: 8px 16px;
}
QPushButton.toggle_inactive:hover {
    background-color: #333333;
}
QProgressBar {
    border: none;
    background-color: #282828;
    height: 8px;
    border-radius: 4px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #1DB954;
    border-radius: 4px;
}
QComboBox {
    background-color: #282828;
    border: 1px solid #535353;
    border-radius: 8px;
    padding: 8px 12px;
    color: #FFFFFF;
    font-size: 14px;
}
QComboBox:focus {
    border-color: #1DB954;
}
QComboBox QAbstractItemView {
    background-color: #282828;
    color: #FFFFFF;
    selection-background-color: #1DB954;
    selection-color: #000000;
    border: 1px solid #535353;
}
QSpinBox {
    background-color: #282828;
    border: 1px solid #535353;
    border-radius: 8px;
    padding: 8px 12px;
    color: #FFFFFF;
    font-size: 14px;
}
QSpinBox:focus {
    border-color: #1DB954;
}
QSlider::groove:horizontal {
    border: none;
    height: 6px;
    background: #282828;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #1DB954;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #FFFFFF;
    border: 1px solid #535353;
    width: 16px;
    height: 16px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 8px;
}
QTableWidget {
    background-color: #181818;
    gridline-color: #282828;
    color: #FFFFFF;
    border: none;
    border-radius: 12px;
}
QHeaderView::section {
    background-color: #282828;
    color: #B3B3B3;
    padding: 8px;
    border: 1px solid #181818;
    font-weight: bold;
}
QScrollBar:vertical {
    background-color: #121212;
    width: 12px;
}
QScrollBar::handle:vertical {
    background-color: #282828;
    border-radius: 6px;
    min-height: 20px;
}
QScrollArea {
    border: none;
    background-color: transparent;
}
QTextBrowser {
    background-color: #181818;
    border: 1px solid #282828;
    border-radius: 12px;
    padding: 10px;
    color: #FFFFFF;
}
"""

LIGHT_QSS = """
QMainWindow, #centralWidget, QStackedWidget, WelcomePage, QuestionnairePage, LoadingPage, ResultsPage, HistoryPage, ComparePage, QScrollArea, #scrollContent {
    background-color: #F5F5F7;
}
QWidget {
    color: #1C1C1E;
    font-family: 'Segoe UI', sans-serif;
}
QFrame.card {
    background-color: #FFFFFF;
    border-radius: 16px;
    border: 1px solid #E5E5EA;
}
QFrame.card_light {
    background-color: #F2F2F7;
    border-radius: 12px;
}
QLabel.title {
    font-size: 38px;
    font-weight: bold;
    color: #1DB954;
}
QLabel.heading {
    font-size: 24px;
    font-weight: bold;
    color: #1C1C1E;
}
QLabel.subheading {
    font-size: 18px;
    font-weight: 600;
    color: #1C1C1E;
}
QLabel.body {
    font-size: 14px;
    color: #1C1C1E;
}
QLabel.body_secondary {
    font-size: 14px;
    color: #3A3A3C;
}
QLabel.body_muted {
    font-size: 12px;
    color: #8E8E93;
}
QPushButton.primary {
    background-color: #1DB954;
    color: #FFFFFF;
    font-weight: bold;
    border: none;
    border-radius: 20px;
    padding: 10px 24px;
    font-size: 14px;
}
QPushButton.primary:hover {
    background-color: #1ED760;
}
QPushButton.primary:disabled {
    background-color: #D1D1D6;
    color: #8E8E93;
}
QPushButton.secondary {
    background-color: transparent;
    color: #1C1C1E;
    border: 1px solid #C7C7CC;
    border-radius: 20px;
    padding: 10px 24px;
    font-size: 14px;
}
QPushButton.secondary:hover {
    background-color: #E5E5EA;
}
QPushButton.danger {
    background-color: transparent;
    color: #FF3B30;
    border: 1px solid #FF3B30;
    border-radius: 20px;
    padding: 10px 24px;
    font-size: 14px;
}
QPushButton.danger:hover {
    background-color: rgba(255, 59, 48, 0.1);
}
QPushButton.toggle_active {
    background-color: #1DB954;
    color: #FFFFFF;
    font-weight: bold;
    border: none;
    border-radius: 8px;
    padding: 8px 16px;
}
QPushButton.toggle_inactive {
    background-color: #FFFFFF;
    color: #1C1C1E;
    border: 1px solid #C7C7CC;
    border-radius: 8px;
    padding: 8px 16px;
}
QPushButton.toggle_inactive:hover {
    background-color: #E5E5EA;
}
QProgressBar {
    border: none;
    background-color: #E5E5EA;
    height: 8px;
    border-radius: 4px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #1DB954;
    border-radius: 4px;
}
QComboBox {
    background-color: #FFFFFF;
    border: 1px solid #C7C7CC;
    border-radius: 8px;
    padding: 8px 12px;
    color: #1C1C1E;
    font-size: 14px;
}
QComboBox:focus {
    border-color: #1DB954;
}
QComboBox QAbstractItemView {
    background-color: #FFFFFF;
    color: #1C1C1E;
    selection-background-color: #1DB954;
    selection-color: #FFFFFF;
    border: 1px solid #C7C7CC;
}
QSpinBox {
    background-color: #FFFFFF;
    border: 1px solid #C7C7CC;
    border-radius: 8px;
    padding: 8px 12px;
    color: #1C1C1E;
    font-size: 14px;
}
QSpinBox:focus {
    border-color: #1DB954;
}
QSlider::groove:horizontal {
    border: none;
    height: 6px;
    background: #E5E5EA;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #1DB954;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #FFFFFF;
    border: 1px solid #C7C7CC;
    width: 16px;
    height: 16px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 8px;
}
QTableWidget {
    background-color: #FFFFFF;
    gridline-color: #E5E5EA;
    color: #1C1C1E;
    border: 1px solid #E5E5EA;
    border-radius: 12px;
}
QHeaderView::section {
    background-color: #F2F2F7;
    color: #3A3A3C;
    padding: 8px;
    border: 1px solid #E5E5EA;
    font-weight: bold;
}
QScrollBar:vertical {
    background-color: #F2F2F7;
    width: 12px;
}
QScrollBar::handle:vertical {
    background-color: #C7C7CC;
    border-radius: 6px;
    min-height: 20px;
}
QScrollArea {
    border: none;
    background-color: transparent;
}
QTextBrowser {
    background-color: #FFFFFF;
    border: 1px solid #E5E5EA;
    border-radius: 12px;
    padding: 10px;
    color: #1C1C1E;
}
"""

DARK_CHART_STYLE = {
    'figure.facecolor': '#181818',
    'axes.facecolor': '#282828',
    'axes.edgecolor': '#535353',
    'axes.labelcolor': '#FFFFFF',
    'text.color': '#FFFFFF',
    'xtick.color': '#B3B3B3',
    'ytick.color': '#B3B3B3',
    'grid.color': '#535353',
    'grid.alpha': 0.3,
    'font.size': 10,
}

LIGHT_CHART_STYLE = {
    'figure.facecolor': '#FFFFFF',
    'axes.facecolor': '#F2F2F7',
    'axes.edgecolor': '#C7C7CC',
    'axes.labelcolor': '#1C1C1E',
    'text.color': '#1C1C1E',
    'xtick.color': '#3A3A3C',
    'ytick.color': '#3A3A3C',
    'grid.color': '#C7C7CC',
    'grid.alpha': 0.3,
    'font.size': 10,
}


def style_figure_for_theme(fig, theme_mode):
    style = DARK_CHART_STYLE if theme_mode == "dark" else LIGHT_CHART_STYLE
    
    # Set figure background
    fig.set_facecolor(style['figure.facecolor'])
    
    # Style each axis inside the figure
    for ax in fig.axes:
        ax.set_facecolor(style['axes.facecolor'])
        ax.tick_params(colors=style['xtick.color'], which='both')
        ax.xaxis.label.set_color(style['axes.labelcolor'])
        ax.yaxis.label.set_color(style['axes.labelcolor'])
        ax.title.set_color(style['text.color'])
        
        # Grid line colors
        ax.grid(color=style['grid.color'], alpha=style['grid.alpha'])
        
        # Spines (borders)
        for spine in ax.spines.values():
            spine.set_color(style['axes.edgecolor'])
            
        # Legend styling if present
        legend = ax.get_legend()
        if legend:
            frame = legend.get_frame()
            frame.set_facecolor(style['axes.facecolor'])
            frame.set_edgecolor(style['axes.edgecolor'])
            for text in legend.get_texts():
                text.set_color(style['text.color'])



def _parse_inline(text: str) -> str:
    import re
    # Escape HTML tags but preserve any tags we intentionally want to display.
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    # Bold: **bold** or __bold__
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__(.*?)__', r'<b>\1</b>', text)
    
    # Italic: *italic* or _italic_
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    text = re.sub(r'_(.*?)_', r'<i>\1</i>', text)
    
    # Code inline: `code`
    text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
    
    return text


def markdown_to_html(md_text: str, theme_mode: str = 'dark') -> str:
    if not md_text:
        return ""
    
    # Normalize line endings
    lines = md_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    html_lines = []
    
    # Styling variables based on active theme
    if theme_mode == 'dark':
        header_color = "#1DB954"   # Spotify Green
        border_color = "#333333"
        header_bg = "#222222"
        header_fg = "#FFFFFF"
        text_fg = "#E5E5EA"
        code_bg = "#2A2A2A"
        code_fg = "#1DB954"
    else:
        header_color = "#2E7D32"   # iOS Dark Green
        border_color = "#D1D1D6"
        header_bg = "#F2F2F7"
        header_fg = "#1C1C1E"
        text_fg = "#1C1C1E"
        code_bg = "#F2F2F7"
        code_fg = "#C0392B"

    # Inject theme-specific CSS stylesheet block
    style_block = f"""<style>
    h1, h2, h3, h4, h5, h6 {{
        color: {header_color};
        margin-top: 14px;
        margin-bottom: 6px;
        font-weight: bold;
    }}
    h1 {{ font-size: 18px; border-bottom: 1px solid {border_color}; padding-bottom: 4px; }}
    h2 {{ font-size: 16px; }}
    h3 {{ font-size: 15px; }}
    h4, h5, h6 {{ font-size: 14px; }}
    p {{
        margin-top: 5px;
        margin-bottom: 5px;
        line-height: 1.4;
        color: {text_fg};
    }}
    ul, ol {{
        margin-top: 5px;
        margin-bottom: 5px;
        padding-left: 22px;
    }}
    li {{
        margin-bottom: 3px;
        color: {text_fg};
    }}
    table {{
        border-collapse: collapse;
        width: 100%;
        margin-top: 12px;
        margin-bottom: 12px;
    }}
    th {{
        background-color: {header_bg};
        color: {header_fg};
        font-weight: bold;
        text-align: left;
        padding: 6px 10px;
    }}
    td {{
        padding: 6px 10px;
        color: {text_fg};
    }}
    code {{
        background-color: {code_bg};
        color: {code_fg};
        padding: 2px 4px;
        border-radius: 4px;
        font-family: monospace;
    }}
    hr {{
        border: 0;
        border-top: 1px solid {border_color};
        margin: 12px 0;
    }}
    </style>
    """
    html_lines.append(style_block)

    in_list = None  # can be 'ul', 'ol', or None
    in_table = False
    table_headers = None
    table_rows = []

    def is_table_row(line_str: str) -> bool:
        s = line_str.strip()
        return s.startswith('|') and s.endswith('|') and len(s) > 1

    def is_separator_row(line_str: str) -> bool:
        s = line_str.strip()
        if not (s.startswith('|') and s.endswith('|')):
            return False
        # Remove all separators/alignments
        chars = set(s)
        return chars.issubset({'|', '-', ':', ' ', '\t'})

    def close_list():
        nonlocal in_list
        if in_list == 'ul':
            html_lines.append("</ul>")
        elif in_list == 'ol':
            html_lines.append("</ol>")
        in_list = None

    def close_table():
        nonlocal in_table, table_headers, table_rows
        if in_table:
            table_html = []
            table_html.append(f'<table border="1" cellspacing="0" cellpadding="6" style="border-color: {border_color}; width: 100%;">')
            
            if table_headers:
                table_html.append("<thead><tr>")
                for h in table_headers:
                    table_html.append(f'<th style="background-color: {header_bg}; color: {header_fg}; border: 1px solid {border_color}; font-weight: bold; text-align: left;">{h}</th>')
                table_html.append("</tr></thead>")
            
            table_html.append("<tbody>")
            for row in table_rows:
                table_html.append("<tr>")
                for cell in row:
                    table_html.append(f'<td style="border: 1px solid {border_color}; color: {text_fg};">{cell}</td>')
                table_html.append("</tr>")
            table_html.append("</tbody></table>")
            
            html_lines.append("\n".join(table_html))
            in_table = False
            table_headers = None
            table_rows = []

    for line in lines:
        stripped = line.strip()
        
        # 1. Table Processing
        if is_table_row(line):
            close_list()
            if is_separator_row(line):
                continue
                
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            parsed_cells = [_parse_inline(c) for c in cells]
            
            if not in_table:
                in_table = True
                table_headers = parsed_cells
                table_rows = []
            else:
                table_rows.append(parsed_cells)
            continue
            
        # Non-table row: close table if active
        close_table()
        
        # 2. Horizontal Rules
        if stripped in ('---', '***', '___'):
            close_list()
            html_lines.append("<hr>")
            continue
            
        # 3. Headings (# to ######)
        if stripped.startswith("#"):
            close_list()
            num_hashes = 0
            for char in stripped:
                if char == '#':
                    num_hashes += 1
                else:
                    break
            
            heading_text = stripped[num_hashes:].strip()
            parsed_heading = _parse_inline(heading_text)
            
            tag_num = min(max(num_hashes, 1), 6)
            html_lines.append(f"<h{tag_num}>{parsed_heading}</h{tag_num}>")
            continue
            
        # 4. Bullet lists
        if stripped.startswith("- ") or stripped.startswith("* ") or stripped.startswith("+ "):
            if in_list != 'ul':
                close_list()
                html_lines.append("<ul>")
                in_list = 'ul'
            content = stripped[2:].strip()
            parsed_content = _parse_inline(content)
            html_lines.append(f"<li>{parsed_content}</li>")
            continue
            
        # 5. Ordered lists
        match_ol = re.match(r'^(\d+)\.\s+(.*)$', stripped)
        if match_ol:
            if in_list != 'ol':
                close_list()
                html_lines.append("<ol>")
                in_list = 'ol'
            content = match_ol.group(2).strip()
            parsed_content = _parse_inline(content)
            html_lines.append(f"<li>{parsed_content}</li>")
            continue
            
        # 6. Empty / Spacer lines
        if not stripped:
            close_list()
            continue
            
        # 7. Default paragraph line
        close_list()
        parsed_line = _parse_inline(line)
        html_lines.append(f"<p>{parsed_line}</p>")
        
    # Close any remaining structures
    close_list()
    close_table()
    
    return "\n".join(html_lines)


class SpinnerWidget(QWidget):
    """A premium custom spinning loader widget using QPainter drawing a rotating arc."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.angle = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.rotate)
        self.timer.start(40)  # 25 fps
        self.setMinimumSize(80, 80)
        self.setMaximumSize(80, 80)

    def rotate(self):
        self.angle = (self.angle + 12) % 360
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor("#1DB954"), 6)
        painter.setPen(pen)
        rect = self.rect().adjusted(6, 6, -6, -6)
        # Draw a beautiful 270 degree arc rotating around
        painter.drawArc(rect, -self.angle * 16, 270 * 16)


class WelcomePage(QWidget):
    """The onboarding welcome splash page."""
    start_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(20)

        # Centered Card
        card = QFrame(self)
        card.setObjectName("welcome_card")
        card.setProperty("class", "card")
        card_layout = QVBoxLayout(card)
        card_layout.setAlignment(Qt.AlignCenter)
        card_layout.setContentsMargins(60, 60, 60, 60)
        card_layout.setSpacing(25)

        # Title
        title_label = QLabel("MoodMap", self)
        title_label.setProperty("class", "title")
        title_label.setAlignment(Qt.AlignCenter)
        
        # Subtitle
        sub_label = QLabel("Music & Mental Health Profiler", self)
        sub_label.setProperty("class", "subheading")
        sub_label.setAlignment(Qt.AlignCenter)

        # Tagline
        tagline_label = QLabel("Discover how your music shapes your mind", self)
        tagline_label.setProperty("class", "body_secondary")
        tagline_label.setAlignment(Qt.AlignCenter)

        # Start button
        start_btn = QPushButton("Get Started →", self)
        start_btn.setProperty("class", "primary")
        start_btn.setCursor(Qt.PointingHandCursor)
        start_btn.clicked.connect(self.start_clicked.emit)

        # Add components to card
        card_layout.addWidget(title_label)
        card_layout.addWidget(sub_label)
        card_layout.addWidget(tagline_label)
        card_layout.addSpacing(15)
        card_layout.addWidget(start_btn)
        
        # Footer
        footer = QLabel("v1.0 · PFAI Project · UMT 2026", self)
        footer.setProperty("class", "body_muted")
        footer.setAlignment(Qt.AlignCenter)

        layout.addStretch()
        layout.addWidget(card)
        layout.addStretch()
        layout.addWidget(footer)


class QuestionnairePage(QWidget):
    """Questionnaire Page displaying one input question at a time."""
    submitted = pyqtSignal(object)
    back_to_welcome = pyqtSignal()

    def __init__(self, genres: list, services: list, parent=None):
        super().__init__(parent)
        self.genres = sorted(genres)
        self.services = sorted(services)
        self.current_index = 0
        self.answers = {}

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(20)

        # Header Row with Back to Splash Button
        header_layout = QHBoxLayout()
        self.splash_back_btn = QPushButton("← Welcome Screen", self)
        self.splash_back_btn.setProperty("class", "secondary")
        self.splash_back_btn.clicked.connect(self.back_to_welcome.emit)
        header_layout.addWidget(self.splash_back_btn)
        header_layout.addStretch()
        main_layout.addLayout(header_layout)

        # Progress Bar at the top
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        # Stack Widget holding the questionnaire sub-pages
        self.question_stack = QStackedWidget(self)
        self.question_stack.setProperty("class", "card")
        main_layout.addWidget(self.question_stack)

        # Create sub-pages
        self._create_question_pages()

        # Navigation row
        nav_layout = QHBoxLayout()
        self.back_btn = QPushButton("Back", self)
        self.back_btn.setProperty("class", "secondary")
        self.back_btn.clicked.connect(self.go_back)
        
        self.next_btn = QPushButton("Next →", self)
        self.next_btn.setProperty("class", "primary")
        self.next_btn.clicked.connect(self.go_next)

        nav_layout.addWidget(self.back_btn)
        nav_layout.addStretch()
        nav_layout.addWidget(self.next_btn)
        main_layout.addLayout(nav_layout)

        self.show_question(0)

    def _create_question_pages(self):
        # 1. Age (Page 0)
        self.age_widget = QWidget()
        l0 = QVBoxLayout(self.age_widget)
        l0.setContentsMargins(40, 40, 40, 40)
        l0.setSpacing(20)
        q0 = QLabel("Question 1: What is your age?", self.age_widget)
        q0.setProperty("class", "subheading")
        self.age_input = QSpinBox(self.age_widget)
        self.age_input.setRange(10, 89)
        self.age_input.setValue(22)
        l0.addWidget(q0)
        l0.addWidget(self.age_input)
        l0.addStretch()
        self.question_stack.addWidget(self.age_widget)

        # 2. Favorite Genre (Page 1)
        self.genre_widget = QWidget()
        l1 = QVBoxLayout(self.genre_widget)
        l1.setContentsMargins(40, 40, 40, 40)
        l1.setSpacing(20)
        q1 = QLabel("Question 2: What is your favorite music genre?", self.genre_widget)
        q1.setProperty("class", "subheading")
        self.genre_input = QComboBox(self.genre_widget)
        self.genre_input.addItems(self.genres)
        l1.addWidget(q1)
        l1.addWidget(self.genre_input)
        l1.addStretch()
        self.question_stack.addWidget(self.genre_widget)

        # 3. Hours per day (Page 2)
        self.hours_widget = QWidget()
        l2 = QVBoxLayout(self.hours_widget)
        l2.setContentsMargins(40, 40, 40, 40)
        l2.setSpacing(20)
        q2 = QLabel("Question 3: Average daily music listening time (hours)?", self.hours_widget)
        q2.setProperty("class", "subheading")
        
        self.hours_val_label = QLabel("3.0 hours", self.hours_widget)
        self.hours_val_label.setProperty("class", "body")
        
        self.hours_input = QSlider(Qt.Horizontal, self.hours_widget)
        self.hours_input.setRange(0, 48)  # 0 to 24 hours with 0.5 steps (0.5 * value)
        self.hours_input.setValue(6)      # default 3.0 hours
        self.hours_input.valueChanged.connect(self._on_hours_changed)
        
        l2.addWidget(q2)
        l2.addWidget(self.hours_val_label)
        l2.addWidget(self.hours_input)
        l2.addStretch()
        self.question_stack.addWidget(self.hours_widget)

        # 4. While working (Page 3)
        self.working_widget = QWidget()
        l3 = QVBoxLayout(self.working_widget)
        l3.setContentsMargins(40, 40, 40, 40)
        l3.setSpacing(20)
        q3 = QLabel("Question 4: Do you listen to music while working/studying?", self.working_widget)
        q3.setProperty("class", "subheading")
        
        btn_layout = QHBoxLayout()
        self.work_yes_btn = QPushButton("Yes", self.working_widget)
        self.work_yes_btn.setProperty("class", "toggle_active")
        self.work_no_btn = QPushButton("No", self.working_widget)
        self.work_no_btn.setProperty("class", "toggle_inactive")
        
        self.work_yes_btn.clicked.connect(lambda: self._set_toggle("while_working", "Yes"))
        self.work_no_btn.clicked.connect(lambda: self._set_toggle("while_working", "No"))
        self.answers["while_working"] = "Yes"  # default
        
        btn_layout.addWidget(self.work_yes_btn)
        btn_layout.addWidget(self.work_no_btn)
        l3.addWidget(q3)
        l3.addLayout(btn_layout)
        l3.addStretch()
        self.question_stack.addWidget(self.working_widget)

        # 5. Instrumentalist (Page 4)
        self.instrument_widget = QWidget()
        l4 = QVBoxLayout(self.instrument_widget)
        l4.setContentsMargins(40, 40, 40, 40)
        l4.setSpacing(20)
        q4 = QLabel("Question 5: Do you play a musical instrument regularly?", self.instrument_widget)
        q4.setProperty("class", "subheading")
        
        btn_layout4 = QHBoxLayout()
        self.inst_yes_btn = QPushButton("Yes", self.instrument_widget)
        self.inst_yes_btn.setProperty("class", "toggle_inactive")
        self.inst_no_btn = QPushButton("No", self.instrument_widget)
        self.inst_no_btn.setProperty("class", "toggle_active")
        
        self.inst_yes_btn.clicked.connect(lambda: self._set_toggle("instrumentalist", "Yes"))
        self.inst_no_btn.clicked.connect(lambda: self._set_toggle("instrumentalist", "No"))
        self.answers["instrumentalist"] = "No"  # default
        
        btn_layout4.addWidget(self.inst_yes_btn)
        btn_layout4.addWidget(self.inst_no_btn)
        l4.addWidget(q4)
        l4.addLayout(btn_layout4)
        l4.addStretch()
        self.question_stack.addWidget(self.instrument_widget)

        # 6. Streaming Service (Page 5)
        self.stream_widget = QWidget()
        l5 = QVBoxLayout(self.stream_widget)
        l5.setContentsMargins(40, 40, 40, 40)
        l5.setSpacing(20)
        q5 = QLabel("Question 6: Which primary streaming service do you use?", self.stream_widget)
        q5.setProperty("class", "subheading")
        self.stream_input = QComboBox(self.stream_widget)
        self.stream_input.addItems(self.services)
        l5.addWidget(q5)
        l5.addWidget(self.stream_input)
        l5.addStretch()
        self.question_stack.addWidget(self.stream_widget)

        # 7. Summary page (Page 6)
        self.summary_widget = QWidget()
        self.l6 = QVBoxLayout(self.summary_widget)
        self.l6.setContentsMargins(40, 40, 40, 40)
        self.l6.setSpacing(15)
        self.q6_title = QLabel("Confirm Your Responses", self.summary_widget)
        self.q6_title.setProperty("class", "subheading")
        
        self.summary_details = QLabel(self.summary_widget)
        self.summary_details.setProperty("class", "body")
        self.summary_details.setWordWrap(True)
        
        self.l6.addWidget(self.q6_title)
        self.l6.addWidget(self.summary_details)
        self.l6.addStretch()
        self.question_stack.addWidget(self.summary_widget)

    def _on_hours_changed(self, val):
        self.hours_val_label.setText(f"{val * 0.5:.1f} hours")

    def _set_toggle(self, field, value):
        self.answers[field] = value
        if field == "while_working":
            if value == "Yes":
                self.work_yes_btn.setProperty("class", "toggle_active")
                self.work_no_btn.setProperty("class", "toggle_inactive")
            else:
                self.work_yes_btn.setProperty("class", "toggle_inactive")
                self.work_no_btn.setProperty("class", "toggle_active")
            self.work_yes_btn.style().unpolish(self.work_yes_btn)
            self.work_yes_btn.style().polish(self.work_yes_btn)
            self.work_no_btn.style().unpolish(self.work_no_btn)
            self.work_no_btn.style().polish(self.work_no_btn)
        elif field == "instrumentalist":
            if value == "Yes":
                self.inst_yes_btn.setProperty("class", "toggle_active")
                self.inst_no_btn.setProperty("class", "toggle_inactive")
            else:
                self.inst_yes_btn.setProperty("class", "toggle_inactive")
                self.inst_no_btn.setProperty("class", "toggle_active")
            self.inst_yes_btn.style().unpolish(self.inst_yes_btn)
            self.inst_yes_btn.style().polish(self.inst_yes_btn)
            self.inst_no_btn.style().unpolish(self.inst_no_btn)
            self.inst_no_btn.style().polish(self.inst_no_btn)

    def show_question(self, index):
        self.current_index = index
        self.question_stack.setCurrentIndex(index)
        self.progress_bar.setValue(int((index / 6.0) * 100))

        # Back button configuration
        if index == 0:
            self.back_btn.setEnabled(False)
        else:
            self.back_btn.setEnabled(True)

        # Next button text configuration
        if index == 6:
            self._update_summary()
            self.next_btn.setText("Submit Analysis ✓")
        else:
            self.next_btn.setText("Next →")

    def _update_summary(self):
        hours = self.hours_input.value() * 0.5
        summary_text = (
            f"<b>Age:</b> {self.age_input.value()}<br>"
            f"<b>Favorite Genre:</b> {self.genre_input.currentText()}<br>"
            f"<b>Daily Listening Hours:</b> {hours:.1f} hours<br>"
            f"<b>Listen While Working:</b> {self.answers.get('while_working', 'Yes')}<br>"
            f"<b>Play Instrument:</b> {self.answers.get('instrumentalist', 'No')}<br>"
            f"<b>Streaming Service:</b> {self.stream_input.currentText()}"
        )
        self.summary_details.setText(summary_text)

    def go_next(self):
        if self.current_index < 6:
            self.show_question(self.current_index + 1)
        else:
            # Build Respondent object and submit
            hours = self.hours_input.value() * 0.5
            respondent = Respondent(
                age=self.age_input.value(),
                fav_genre=self.genre_input.currentText(),
                hours_per_day=hours,
                while_working=self.answers.get('while_working', 'Yes'),
                instrumentalist=self.answers.get('instrumentalist', 'No'),
                streaming_service=self.stream_input.currentText()
            )
            self.submitted.emit(respondent)

    def go_back(self):
        if self.current_index > 0:
            self.show_question(self.current_index - 1)


class LoadingPage(QWidget):
    """The Loading analysis pipeline page with a spinner and rotating messages."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(25)

        # Spinner container
        self.spinner = SpinnerWidget(self)
        layout.addWidget(self.spinner, 0, Qt.AlignCenter)

        # Message label
        self.msg_label = QLabel("Syncing your vibe...", self)
        self.msg_label.setProperty("class", "subheading")
        self.msg_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.msg_label)

        # Simulate details progress
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(10)
        self.progress_bar.setFixedWidth(300)
        layout.addWidget(self.progress_bar, 0, Qt.AlignCenter)

        self.messages = [
            "Loading your music profile...",
            "Analyzing your genre cohort...",
            "Computing correlations...",
            "Crunching the numbers...",
            "Building your mental health profile...",
            "Syncing with Clinical AI...",
            "Formatting insights...",
            "Almost there..."
        ]
        self.msg_index = 0
        self.progress_val = 10

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_progress)

    def start(self):
        self.msg_index = 0
        self.progress_val = 10
        self.msg_label.setText(self.messages[0])
        self.progress_bar.setValue(10)
        self.timer.start(1200)

    def stop(self):
        self.timer.stop()

    def _update_progress(self):
        self.msg_index = (self.msg_index + 1) % len(self.messages)
        self.msg_label.setText(self.messages[self.msg_index])
        
        self.progress_val = min(95, self.progress_val + 12)
        self.progress_bar.setValue(self.progress_val)


class ResultsPage(QWidget):
    """Results Dashboard Page displaying score cards, graphs, tables, and AI insights."""
    start_over_clicked = pyqtSignal()
    compare_clicked = pyqtSignal(object)  # Emits the current Respondent object
    history_clicked = pyqtSignal()
    export_pdf_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.report = None
        self.respondent_obj = None

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Scroll Area for the entire dashboard
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        scroll_content = QWidget()
        scroll_content.setObjectName("scrollContent")
        self.scroll_layout = QVBoxLayout(scroll_content)
        self.scroll_layout.setContentsMargins(10, 10, 10, 10)
        self.scroll_layout.setSpacing(25)

        # 1. Header Row
        header_layout = QHBoxLayout()
        title_label = QLabel("🎵 Your MoodMap Profile", scroll_content)
        title_label.setProperty("class", "heading")
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        # Theme/Alert Banner if AI Client fails
        self.ai_alert_banner = QFrame(scroll_content)
        self.ai_alert_banner.setProperty("class", "card")
        self.ai_alert_banner.setStyleSheet("background-color: rgba(243, 156, 18, 0.15); border: 1px solid #F39C12;")
        banner_layout = QHBoxLayout(self.ai_alert_banner)
        banner_layout.setContentsMargins(12, 6, 12, 6)
        self.ai_alert_label = QLabel("⚠️ AI Client Offline: Showing local analytical prediction matches.", self.ai_alert_banner)
        self.ai_alert_label.setStyleSheet("color: #F39C12; font-weight: bold; font-size: 13px;")
        banner_layout.addWidget(self.ai_alert_label)
        self.ai_alert_banner.setVisible(False)
        self.scroll_layout.addWidget(self.ai_alert_banner)
        
        self.scroll_layout.addLayout(header_layout)

        # 2. Score Cards (4 cards in a row)
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(15)
        
        self.cards = {}
        for metric in ['Anxiety', 'Depression', 'Insomnia', 'OCD']:
            card = QFrame(scroll_content)
            card.setProperty("class", "card")
            card.setMinimumHeight(150)
            
            c_lay = QVBoxLayout(card)
            c_lay.setContentsMargins(15, 15, 15, 15)
            c_lay.setSpacing(5)

            m_lbl = QLabel(metric, card)
            m_lbl.setProperty("class", "body_secondary")
            m_lbl.setAlignment(Qt.AlignCenter)
            
            score_lbl = QLabel("0.0", card)
            score_lbl.setProperty("class", "title")
            score_lbl.setAlignment(Qt.AlignCenter)
            score_lbl.setStyleSheet("font-size: 42px; font-weight: bold;")
            
            sev_lbl = QLabel("LOW", card)
            sev_lbl.setAlignment(Qt.AlignCenter)
            sev_lbl.setStyleSheet("font-weight: bold; font-size: 12px; padding: 4px 8px; border-radius: 8px;")
            
            pct_lbl = QLabel("Percentile: 0%", card)
            pct_lbl.setProperty("class", "body_muted")
            pct_lbl.setAlignment(Qt.AlignCenter)

            c_lay.addWidget(m_lbl)
            c_lay.addWidget(score_lbl)
            c_lay.addWidget(sev_lbl)
            c_lay.addWidget(pct_lbl)
            cards_layout.addWidget(card)
            
            self.cards[metric] = {
                'frame': card,
                'score': score_lbl,
                'sev': sev_lbl,
                'pct': pct_lbl
            }
        self.scroll_layout.addLayout(cards_layout)

        # 3. Two-Column Layout: Cohort vs Recommendations
        middle_layout = QHBoxLayout()
        middle_layout.setSpacing(20)

        # Left Column: Cohort Table
        cohort_card = QFrame(scroll_content)
        cohort_card.setProperty("class", "card")
        cohort_lay = QVBoxLayout(cohort_card)
        cohort_lay.setContentsMargins(20, 20, 20, 20)
        
        table_title = QLabel("📊 Cohort Comparison", cohort_card)
        table_title.setProperty("class", "subheading")
        self.confidence_label = QLabel("Based on cohort calculations", cohort_card)
        self.confidence_label.setProperty("class", "body_muted")
        
        self.table = QTableWidget(cohort_card)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Metric", "You", "Genre Avg", "Age Avg", "Delta"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setFixedHeight(170)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.NoSelection)

        cohort_lay.addWidget(table_title)
        cohort_lay.addWidget(self.confidence_label)
        cohort_lay.addWidget(self.table)
        middle_layout.addWidget(cohort_card, 3)

        # Right Column: Recommendations + BPM Insights
        recs_card = QFrame(scroll_content)
        recs_card.setProperty("class", "card")
        recs_lay = QVBoxLayout(recs_card)
        recs_lay.setContentsMargins(20, 20, 20, 20)
        recs_lay.setSpacing(10)

        recs_title = QLabel("💡 Actionable Recommendations", recs_card)
        recs_title.setProperty("class", "subheading")
        
        self.recs_text = QLabel(recs_card)
        self.recs_text.setProperty("class", "body")
        self.recs_text.setWordWrap(True)
        
        insights_title = QLabel("🔗 Statistical Insights", recs_card)
        insights_title.setProperty("class", "subheading")
        
        self.insights_text = QLabel(recs_card)
        self.insights_text.setProperty("class", "body_secondary")
        self.insights_text.setWordWrap(True)

        recs_lay.addWidget(recs_title)
        recs_lay.addWidget(self.recs_text)
        recs_lay.addSpacing(10)
        recs_lay.addWidget(insights_title)
        recs_lay.addWidget(self.insights_text)
        middle_layout.addWidget(recs_card, 2)

        self.scroll_layout.addLayout(middle_layout)

        # 4. AI insights Box
        ai_box = QFrame(scroll_content)
        ai_box.setProperty("class", "card")
        ai_lay = QVBoxLayout(ai_box)
        ai_lay.setContentsMargins(20, 20, 20, 20)
        ai_lay.setSpacing(10)
        
        ai_title = QLabel("🤖 Clinical AI Interpretation", ai_box)
        ai_title.setProperty("class", "subheading")
        
        self.ai_browser = QTextBrowser(ai_box)
        self.ai_browser.setMinimumHeight(220)
        self.ai_browser.setOpenExternalLinks(True)
        
        ai_lay.addWidget(ai_title)
        ai_lay.addWidget(self.ai_browser)
        self.scroll_layout.addWidget(ai_box)

        # 5. Charts Grid (Matplotlib Figures)
        charts_title = QLabel("📈 Data Visualizations", scroll_content)
        charts_title.setProperty("class", "subheading")
        self.scroll_layout.addWidget(charts_title)

        self.charts_grid = QGridLayout()
        self.charts_grid.setSpacing(15)
        self.scroll_layout.addLayout(self.charts_grid)

        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

        # 6. Action buttons footer
        actions_layout = QHBoxLayout()
        actions_layout.setContentsMargins(10, 10, 10, 10)
        
        self.export_png_btn = QPushButton("📥 Export Charts", self)
        self.export_png_btn.setProperty("class", "secondary")
        self.export_png_btn.clicked.connect(self.export_charts)
        
        self.export_pdf_btn = QPushButton("📄 Export PDF Report", self)
        self.export_pdf_btn.setProperty("class", "secondary")
        self.export_pdf_btn.clicked.connect(self.export_pdf_clicked.emit)

        self.compare_btn = QPushButton("🔀 Compare Session", self)
        self.compare_btn.setProperty("class", "primary")
        self.compare_btn.clicked.connect(self._on_compare_clicked)

        self.history_btn = QPushButton("📈 View History Logs", self)
        self.history_btn.setProperty("class", "primary")
        self.history_btn.clicked.connect(self.history_clicked.emit)

        self.reset_btn = QPushButton("🔄 Start Over", self)
        self.reset_btn.setProperty("class", "secondary")
        self.reset_btn.clicked.connect(self.start_over_clicked.emit)

        actions_layout.addWidget(self.export_png_btn)
        actions_layout.addWidget(self.export_pdf_btn)
        actions_layout.addStretch()
        actions_layout.addWidget(self.compare_btn)
        actions_layout.addWidget(self.history_btn)
        actions_layout.addWidget(self.reset_btn)
        
        main_layout.addLayout(actions_layout)

    def display_results(self, report: dict, figures: list, respondent_obj, ai_insights: dict = None, theme_mode: str = 'dark'):
        self.report = report
        self.respondent_obj = respondent_obj
        
        # Apply theme style to all figures
        for fig in figures:
            style_figure_for_theme(fig, theme_mode)
            
        # 1. Alert Banner
        if ai_insights is None:
            self.ai_alert_banner.setVisible(True)
            self.ai_browser.setHtml(
                "<h3>Insights Offline</h3>"
                "<p>The Clinical AI therapist is currently offline or the environment API key is missing. "
                "Local statistical calculations match successfully. Check the recommendation panel above for advice.</p>"
            )
        else:
            self.ai_alert_banner.setVisible(False)
            insights = ai_insights.get("insights", "No insights generated.")
            recs = ai_insights.get("recommendations", "")
            
            # Convert markdown to html
            html_insights = markdown_to_html(insights, theme_mode)
            html_recs = markdown_to_html(recs, theme_mode)
            
            full_html = f"<div>{html_insights}</div>"
            if recs and recs != 'No structured recommendations found':
                full_html += f"<hr style='border: 0; border-top: 1px solid #535353; margin-top: 15px; margin-bottom: 15px;'><div>{html_recs}</div>"
            self.ai_browser.setHtml(full_html)

        # 2. Score Cards
        metrics = report['metrics']
        for name, data in metrics.items():
            card = self.cards[name]
            score = data['score']
            sev = data['severity'].upper()
            pct = data['percentile']
            color = data['severity_color']

            # Adjust bright green/red on light mode for better readability
            adjusted_color = color
            if theme_mode == "light":
                if color == "#1DB954":  # Bright green
                    adjusted_color = "#2E7D32"
                elif color == "#E74C3C":  # Bright red
                    adjusted_color = "#C0392B"

            card['score'].setText(f"{score:.1f}")
            card['score'].setStyleSheet(f"font-size: 42px; font-weight: bold; color: {adjusted_color};")
            card['sev'].setText(sev)
            card['sev'].setStyleSheet(f"font-weight: bold; font-size: 11px; padding: 4px 8px; border-radius: 8px; background-color: {adjusted_color}; color: #FFFFFF;")
            card['pct'].setText(f"Percentile: {pct:.1f}%")

        # 3. Confidence label
        cohort_size = report.get('genre_cohort_size', 0)
        self.confidence_label.setText(f"Based on {cohort_size} cohort respondents with similar favorite genres.")

        # 4. Cohort Table
        self.table.setRowCount(0)
        comparison = report['comparison']
        age_stats = report.get('age_bracket_stats', {})
        for i, metric in enumerate(['Anxiety', 'Depression', 'Insomnia', 'OCD']):
            self.table.insertRow(i)
            row_data = comparison.get(metric, {})
            age_mean = age_stats.get(metric, 0.0)
            
            self.table.setItem(i, 0, QTableWidgetItem(metric))
            self.table.setItem(i, 1, QTableWidgetItem(f"{row_data.get('user_score', 0.0):.1f}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{row_data.get('cohort_mean', 0.0):.1f}"))
            self.table.setItem(i, 3, QTableWidgetItem(f"{age_mean:.1f}"))
            
            delta = row_data.get('delta', 0.0)
            sign = "+" if delta > 0 else ""
            delta_item = QTableWidgetItem(f"{sign}{delta:.1f}")
            
            # Color-code delta: green for negative delta (lower score is good), red for positive delta (higher score is bad)
            if delta < 0:
                delta_color = "#1DB954" if theme_mode == "dark" else "#2E7D32"
                delta_item.setForeground(QColor(delta_color))
            elif delta > 0:
                delta_color = "#E74C3C" if theme_mode == "dark" else "#D32F2F"
                delta_item.setForeground(QColor(delta_color))
            
            self.table.setItem(i, 4, delta_item)

        # 5. Recommendations Text
        recs = report.get('recommendations', [])
        if not recs:
            # Fallback to local recommend static values if not supplied
            recs = [
                "Consider adjusting average daily listening time down slightly to alleviate Anxiety.",
                "Swap high-BPM active music with relaxing ambient/classical tracks during focused work."
            ]
        self.recs_text.setText("<br>".join(f"• {r}" for r in recs))

        # 6. Insights Text
        strongest_hours = report.get('strongest_hours_correlation', {})
        strongest_bpm = report.get('strongest_bpm_correlation', {})
        
        hours_m = strongest_hours.get('metric', 'N/A')
        hours_r = strongest_hours.get('r', 0.0)
        bpm_m = strongest_bpm.get('metric', 'N/A')
        bpm_r = strongest_bpm.get('r', 0.0)

        insights_str = (
            f"Daily Hours show strongest correlation with <b>{hours_m}</b> (r = {hours_r:.3f}).<br>"
            f"Music BPM shows strongest correlation with <b>{bpm_m}</b> (r = {bpm_r:.3f})."
        )
        self.insights_text.setText(insights_str)

        # 7. Clear old charts and populate new matplotlib canvases
        self._clear_layout(self.charts_grid)
        
        if len(figures) >= 5:
            # Grid layout for charts
            c1 = FigureCanvasQTAgg(figures[0])
            c2 = FigureCanvasQTAgg(figures[1])
            c3 = FigureCanvasQTAgg(figures[2])
            c4 = FigureCanvasQTAgg(figures[3])
            c5 = FigureCanvasQTAgg(figures[4])

            # Set size policies
            for c in [c1, c2, c3, c4, c5]:
                c.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
                c.setMinimumHeight(280)

            # Helper to wrap canvas in card frame for premium rounded borders
            def wrap_in_card(canvas):
                frame = QFrame()
                frame.setProperty("class", "card")
                lay = QVBoxLayout(frame)
                lay.setContentsMargins(10, 10, 10, 10)
                lay.addWidget(canvas)
                return frame

            self.charts_grid.addWidget(wrap_in_card(c1), 0, 0)
            self.charts_grid.addWidget(wrap_in_card(c2), 0, 1)
            self.charts_grid.addWidget(wrap_in_card(c3), 1, 0)
            self.charts_grid.addWidget(wrap_in_card(c4), 1, 1)
            
            c5.setMinimumHeight(350)
            self.charts_grid.addWidget(wrap_in_card(c5), 2, 0, 1, 2)

    def _clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            elif child.layout():
                self._clear_layout(child.layout())

    def _on_compare_clicked(self):
        if self.respondent_obj:
            self.compare_clicked.emit(self.respondent_obj)

    def export_charts(self):
        try:
            # Simply trigger report generator to save figures in output directory
            os.makedirs("output", exist_ok=True)
            chart_names = ['genre_bar', 'age_line', 'anxiety_histogram', 'scatter_plot', 'correlation_heatmap']
            
            i = 0
            for j in range(self.charts_grid.count()):
                widget = self.charts_grid.itemAt(j).widget()
                if widget:
                    canvas = widget.findChild(FigureCanvasQTAgg) if isinstance(widget, QFrame) else widget
                    if isinstance(canvas, FigureCanvasQTAgg):
                        if i < len(chart_names):
                            path = os.path.join("output", f"{chart_names[i]}.png")
                            canvas.figure.savefig(path, dpi=150, bbox_inches='tight', facecolor=canvas.figure.get_facecolor(), edgecolor='none')
                            i += 1
            
            QMessageBox.information(
                self, "Export Success", 
                "All 5 visualizations exported successfully to the <b>output/</b> folder as PNGs."
            )
        except Exception as e:
            QMessageBox.warning(self, "Export Failed", f"Could not export charts: {str(e)}")


class HistoryPage(QWidget):
    """History Logs screen with a list of past sessions and trend plot."""
    back_clicked = pyqtSignal()

    def __init__(self, history_manager, parent=None):
        super().__init__(parent)
        self.hm = history_manager

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # 1. Header
        header_layout = QHBoxLayout()
        title_label = QLabel("📈 Session History Logs", self)
        title_label.setProperty("class", "heading")
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        self.clear_btn = QPushButton("Clear All History", self)
        self.clear_btn.setProperty("class", "danger")
        self.clear_btn.clicked.connect(self._clear_history)
        header_layout.addWidget(self.clear_btn)
        main_layout.addLayout(header_layout)

        # 2. Chart container (Trend lines)
        self.chart_card = QFrame(self)
        self.chart_card.setProperty("class", "card")
        self.chart_card_layout = QVBoxLayout(self.chart_card)
        self.chart_card_layout.setContentsMargins(15, 15, 15, 15)
        
        self.placeholder_label = QLabel("Run 2 or more sessions to view score trends across time.", self.chart_card)
        self.placeholder_label.setProperty("class", "body_secondary")
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.chart_card_layout.addWidget(self.placeholder_label)
        
        self.trend_canvas = None
        self.chart_card.setMinimumHeight(300)
        main_layout.addWidget(self.chart_card)

        # 3. Table
        table_card = QFrame(self)
        table_card.setProperty("class", "card")
        table_lay = QVBoxLayout(table_card)
        table_lay.setContentsMargins(10, 10, 10, 10)

        self.table = QTableWidget(table_card)
        self.table.setColumnCount(10)
        self.table.setHorizontalHeaderLabels([
            "ID", "Timestamp", "Age", "Genre", "Hours", 
            "Anxiety", "Depression", "Insomnia", "OCD", "Actions"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        table_lay.addWidget(self.table)
        main_layout.addWidget(table_card)

        # 4. Footer back button
        footer_layout = QHBoxLayout()
        back_btn = QPushButton("← Back to Results", self)
        back_btn.setProperty("class", "secondary")
        back_btn.clicked.connect(self.back_clicked.emit)
        footer_layout.addWidget(back_btn)
        footer_layout.addStretch()
        main_layout.addLayout(footer_layout)

    def load_history(self, theme_mode="dark"):
        sessions = self.hm.get_all_sessions()
        count = len(sessions)

        # 1. Update Trend Chart
        if self.trend_canvas:
            self.chart_card_layout.removeWidget(self.trend_canvas)
            self.trend_canvas.deleteLater()
            self.trend_canvas = None

        trend_data = self.hm.get_trend_data()
        fig = create_trend_chart(trend_data, count)

        if fig:
            style_figure_for_theme(fig, theme_mode)
            self.placeholder_label.setVisible(False)
            self.trend_canvas = FigureCanvasQTAgg(fig)
            self.trend_canvas.setMinimumHeight(260)
            self.chart_card_layout.addWidget(self.trend_canvas)
        else:
            self.placeholder_label.setVisible(True)

        # 2. Populate Table
        self.table.setRowCount(0)
        for i, row in enumerate(sessions):
            self.table.insertRow(i)
            
            # format date
            dt = row.get('timestamp', '')
            try:
                dt_obj = datetime.fromisoformat(dt)
                time_str = dt_obj.strftime("%Y-%m-%d %H:%M")
            except:
                time_str = dt[:16].replace('T', ' ')

            self.table.setItem(i, 0, QTableWidgetItem(row.get('id', '')))
            self.table.setItem(i, 1, QTableWidgetItem(time_str))
            self.table.setItem(i, 2, QTableWidgetItem(row.get('age', '')))
            self.table.setItem(i, 3, QTableWidgetItem(row.get('genre', '')))
            self.table.setItem(i, 4, QTableWidgetItem(f"{float(row.get('hours', 0.0)):.1f}"))
            self.table.setItem(i, 5, QTableWidgetItem(f"{float(row.get('anxiety', 0.0)):.1f}"))
            self.table.setItem(i, 6, QTableWidgetItem(f"{float(row.get('depression', 0.0)):.1f}"))
            self.table.setItem(i, 7, QTableWidgetItem(f"{float(row.get('insomnia', 0.0)):.1f}"))
            self.table.setItem(i, 8, QTableWidgetItem(f"{float(row.get('ocd', 0.0)):.1f}"))

            # Delete button
            del_btn = QPushButton("Delete", self)
            del_btn.setStyleSheet("color: #E74C3C; font-weight: bold; background: transparent; border: none; padding: 2px;")
            del_btn.setCursor(Qt.PointingHandCursor)
            
            # Use closure to capture session ID
            sid = int(row.get('id', 0))
            del_btn.clicked.connect(lambda checked, s=sid: self._delete_session(s))
            
            self.table.setCellWidget(i, 9, del_btn)

    def _delete_session(self, sid):
        reply = QMessageBox.question(
            self, "Delete Record",
            f"Are you sure you want to delete session record #{sid}?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.hm.delete_session(sid)
            self.load_history()

    def _clear_history(self):
        reply = QMessageBox.question(
            self, "Clear History",
            "Are you sure you want to permanently clear all session history?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.hm.clear_all()
            self.load_history()


class ComparePage(QWidget):
    """Side-by-side Comparative Mode Page."""
    back_clicked = pyqtSignal()

    def __init__(self, data_loader, parent=None):
        super().__init__(parent)
        self.dl = data_loader
        self.original_respondent = None
        self.original_predicted = None

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # Title
        title_label = QLabel("🔀 Comparative Simulation Mode", self)
        title_label.setProperty("class", "heading")
        main_layout.addWidget(title_label)

        # Split layout
        split_layout = QHBoxLayout()
        split_layout.setSpacing(20)

        # Left Pane: Modify Inputs
        input_card = QFrame(self)
        input_card.setProperty("class", "card")
        input_lay = QVBoxLayout(input_card)
        input_lay.setContentsMargins(20, 20, 20, 20)
        input_lay.setSpacing(15)

        input_title = QLabel("Modify Parameters", input_card)
        input_title.setProperty("class", "subheading")
        input_lay.addWidget(input_title)

        # Fields
        form_grid = QGridLayout()
        form_grid.setSpacing(10)

        form_grid.addWidget(QLabel("Age (10-89):", input_card), 0, 0)
        self.age_input = QSpinBox(input_card)
        self.age_input.setRange(10, 89)
        form_grid.addWidget(self.age_input, 0, 1)

        form_grid.addWidget(QLabel("Favorite Genre:", input_card), 1, 0)
        self.genre_input = QComboBox(input_card)
        self.genre_input.addItems(sorted(self.dl.get_unique_genres()))
        form_grid.addWidget(self.genre_input, 1, 1)

        form_grid.addWidget(QLabel("Hours per day:", input_card), 2, 0)
        self.hours_val_label = QLabel("0.0 hours", input_card)
        self.hours_input = QSlider(Qt.Horizontal, input_card)
        self.hours_input.setRange(0, 48)
        self.hours_input.valueChanged.connect(self._on_hours_changed)
        
        hours_lay = QHBoxLayout()
        hours_lay.addWidget(self.hours_input)
        hours_lay.addWidget(self.hours_val_label)
        form_grid.addLayout(hours_lay, 2, 1)

        form_grid.addWidget(QLabel("Listen while working:", input_card), 3, 0)
        self.work_input = QComboBox(input_card)
        self.work_input.addItems(["Yes", "No"])
        form_grid.addWidget(self.work_input, 3, 1)

        form_grid.addWidget(QLabel("Play instrument:", input_card), 4, 0)
        self.inst_input = QComboBox(input_card)
        self.inst_input.addItems(["Yes", "No"])
        form_grid.addWidget(self.inst_input, 4, 1)

        form_grid.addWidget(QLabel("Streaming service:", input_card), 5, 0)
        self.stream_input = QComboBox(input_card)
        self.stream_input.addItems(sorted(self.dl.get_unique_services()))
        form_grid.addWidget(self.stream_input, 5, 1)

        input_lay.addLayout(form_grid)

        # Run Compare button
        self.run_btn = QPushButton("Simulate Delta Difference 🔀", input_card)
        self.run_btn.setProperty("class", "primary")
        self.run_btn.clicked.connect(self._run_comparison)
        input_lay.addWidget(self.run_btn)
        input_lay.addStretch()

        split_layout.addWidget(input_card, 2)

        # Right Pane: Comparison Scores
        results_card = QFrame(self)
        results_card.setProperty("class", "card")
        results_lay = QVBoxLayout(results_card)
        results_lay.setContentsMargins(20, 20, 20, 20)
        results_lay.setSpacing(15)

        results_title = QLabel("Delta Metrics Analysis", results_card)
        results_title.setProperty("class", "subheading")
        results_lay.addWidget(results_title)

        # Grid comparing metrics
        self.compare_grid = QGridLayout()
        self.compare_grid.setSpacing(15)
        
        # Header row in grid
        self.compare_grid.addWidget(QLabel("<b>Metric</b>", results_card), 0, 0)
        self.compare_grid.addWidget(QLabel("<b>Original</b>", results_card), 0, 1)
        self.compare_grid.addWidget(QLabel("<b>Simulated</b>", results_card), 0, 2)
        self.compare_grid.addWidget(QLabel("<b>Delta</b>", results_card), 0, 3)

        self.rows = {}
        for idx, metric in enumerate(['Anxiety', 'Depression', 'Insomnia', 'OCD'], start=1):
            m_lbl = QLabel(metric, results_card)
            orig_lbl = QLabel("0.0", results_card)
            new_lbl = QLabel("0.0", results_card)
            delta_lbl = QLabel("=", results_card)
            
            orig_lbl.setAlignment(Qt.AlignCenter)
            new_lbl.setAlignment(Qt.AlignCenter)
            delta_lbl.setAlignment(Qt.AlignCenter)

            self.compare_grid.addWidget(m_lbl, idx, 0)
            self.compare_grid.addWidget(orig_lbl, idx, 1)
            self.compare_grid.addWidget(new_lbl, idx, 2)
            self.compare_grid.addWidget(delta_lbl, idx, 3)

            self.rows[metric] = {
                'orig': orig_lbl,
                'new': new_lbl,
                'delta': delta_lbl
            }

        results_lay.addLayout(self.compare_grid)
        results_lay.addStretch()

        split_layout.addWidget(results_card, 3)
        main_layout.addLayout(split_layout)

        # Footer
        footer_layout = QHBoxLayout()
        back_btn = QPushButton("← Back to Results", self)
        back_btn.setProperty("class", "secondary")
        back_btn.clicked.connect(self.back_clicked.emit)
        footer_layout.addWidget(back_btn)
        footer_layout.addStretch()
        main_layout.addLayout(footer_layout)

    def _on_hours_changed(self, val):
        self.hours_val_label.setText(f"{val * 0.5:.1f} hours")

    def prepopulate(self, respondent: Respondent, predicted_scores: dict):
        self.original_respondent = respondent
        self.original_predicted = predicted_scores

        self.age_input.setValue(respondent.age)
        self.genre_input.setCurrentText(respondent.fav_genre)
        self.hours_input.setValue(int(respondent.hours_per_day * 2))
        self.work_input.setCurrentText(respondent.while_working)
        self.inst_input.setCurrentText(respondent.instrumentalist)
        self.stream_input.setCurrentText(respondent.streaming_service)

        # Display original scores
        for metric in ['Anxiety', 'Depression', 'Insomnia', 'OCD']:
            orig_val = predicted_scores.get(metric, 0.0)
            self.rows[metric]['orig'].setText(f"{orig_val:.2f}")
            self.rows[metric]['new'].setText("-")
            self.rows[metric]['delta'].setText("-")
            self.rows[metric]['delta'].setStyleSheet("")

    def _run_comparison(self):
        if not self.original_respondent:
            return

        # Build simulated respondent
        sim_resp = Respondent(
            age=self.age_input.value(),
            fav_genre=self.genre_input.currentText(),
            hours_per_day=self.hours_input.value() * 0.5,
            while_working=self.work_input.currentText(),
            instrumentalist=self.inst_input.currentText(),
            streaming_service=self.stream_input.currentText()
        )

        df = self.dl.get_data()
        ce = CorrelationEngine(df)
        sim_predicted = predict_mental_health(sim_resp, df, ce)

        for metric in ['Anxiety', 'Depression', 'Insomnia', 'OCD']:
            orig_val = self.original_predicted.get(metric, 0.0)
            sim_val = sim_predicted.get(metric, 0.0)
            delta = sim_val - orig_val

            self.rows[metric]['new'].setText(f"{sim_val:.2f}")
            
            sign = "+" if delta > 0 else ""
            delta_str = f"{sign}{delta:.2f}"

            if delta < -0.01:
                # Improvement (lower score is better)
                self.rows[metric]['delta'].setText(f"↓ {delta_str}")
                self.rows[metric]['delta'].setStyleSheet("color: #1DB954; font-weight: bold;")
            elif delta > 0.01:
                # Degradation (higher score is worse)
                self.rows[metric]['delta'].setText(f"↑ {delta_str}")
                self.rows[metric]['delta'].setStyleSheet("color: #E74C3C; font-weight: bold;")
            else:
                self.rows[metric]['delta'].setText(f"= 0.00")
                self.rows[metric]['delta'].setStyleSheet("color: #727272;")


class OnboardingTooltipManager(QObject):
    """Controls showing tooltip overlays during the questionnaire steps on first launch."""
    def __init__(self, page: QuestionnairePage, settings_manager, parent=None):
        super().__init__(parent)
        self.page = page
        self.sm = settings_manager
        self.tooltips = {
            0: ("Age Indicator", "Used to group you with respondents in the same life stage."),
            1: ("Fav Genre Selection", "Your primary music genre. Affects cohort calculations most strongly."),
            2: ("Listening Hours", "Self-reported average daily listening time. Highly correlated with Insomnia."),
            3: ("Work Activity Mode", "Whether you play music during focused work or study."),
            4: ("Instrument Activity", "Whether you play a musical instrument regularly."),
            5: ("Streaming Platforms", "Your primary platform — used as a behavioural modifier.")
        }

    def trigger_tooltip(self, index):
        # Only show on first launch!
        shown = self.sm.get_setting("onboarding_shown")
        if shown == "1":
            return
            
        if index not in self.tooltips:
            return

        title, text = self.tooltips[index]
        
        # Determine target widget
        target = None
        if index == 0:
            target = self.page.age_input
        elif index == 1:
            target = self.page.genre_input
        elif index == 2:
            target = self.page.hours_input
        elif index == 3:
            target = self.page.work_yes_btn
        elif index == 4:
            target = self.page.inst_no_btn
        elif index == 5:
            target = self.page.stream_input

        if target and target.isVisible():
            # Show tooltip relative to target
            pos = target.mapToGlobal(QPoint(target.width() // 2, -10))
            QToolTip.showText(pos, f"<b>{title}</b><br>{text}", target)

    def mark_completed(self):
        self.sm.set_setting("onboarding_shown", "1")


class MoodMapApp(QMainWindow):
    """The main MainWindow Shell coordinating the screens, layouts, stylesheets, and pipelines."""
    def __init__(self, data_loader, history_manager, parent=None):
        super().__init__(parent)
        self.dl = data_loader
        self.hm = history_manager
        self.theme_mode = "dark"
        self.current_respondent = None
        self.active_report = None
        self.active_chart_paths = []
        self.ai_worker = None

        self.setWindowTitle("MoodMap — Music & Mental Health Profiler")
        self.resize(1200, 800)

        # Style sheet
        self.setStyleSheet(DARK_QSS)

        # Main Central Widget
        self.central_widget = QWidget(self)
        self.central_widget.setObjectName("centralWidget")
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Theme Selector Toolbar Row at the top-right
        self.toolbar_layout = QHBoxLayout()
        self.toolbar_layout.setContentsMargins(15, 10, 15, 10)
        self.theme_toggle = QPushButton("☀️ Light Mode", self)
        self.theme_toggle.setFixedWidth(130)
        self.theme_toggle.setStyleSheet("background-color: transparent; border: 1px solid #535353; border-radius: 15px; padding: 6px; font-weight: bold;")
        self.theme_toggle.clicked.connect(self.toggle_theme)
        
        self.toolbar_layout.addStretch()
        self.toolbar_layout.addWidget(self.theme_toggle)
        self.main_layout.addLayout(self.toolbar_layout)

        # QStackedWidget holding all screens
        self.stacked_widget = QStackedWidget(self)
        self.main_layout.addWidget(self.stacked_widget)

        # Load dataset if not loaded
        if not self.dl.is_loaded:
            self.dl.get_data()

        # Create unique items from dataset
        genres = self.dl.get_unique_genres()
        services = self.dl.get_unique_services()

        # Instantiate screens
        self.welcome_page = WelcomePage(self)
        self.questionnaire_page = QuestionnairePage(genres, services, self)
        self.loading_page = LoadingPage(self)
        self.results_page = ResultsPage(self)
        self.history_page = HistoryPage(self.hm, self)
        self.compare_page = ComparePage(self.dl, self)

        # Add to stack
        self.stacked_widget.addWidget(self.welcome_page)       # 0
        self.stacked_widget.addWidget(self.questionnaire_page) # 1
        self.stacked_widget.addWidget(self.loading_page)       # 2
        self.stacked_widget.addWidget(self.results_page)       # 3
        self.stacked_widget.addWidget(self.history_page)       # 4
        self.stacked_widget.addWidget(self.compare_page)       # 5

        # Connect signals
        self.welcome_page.start_clicked.connect(lambda: self.navigate_to(1))
        self.questionnaire_page.back_to_welcome.connect(lambda: self.navigate_to(0))
        self.questionnaire_page.submitted.connect(self.run_analysis)
        
        self.results_page.start_over_clicked.connect(self._reset_and_start_over)
        self.results_page.compare_clicked.connect(self._open_compare_mode)
        self.results_page.history_clicked.connect(self._open_history_mode)
        self.results_page.export_pdf_clicked.connect(self._export_pdf_report)

        self.history_page.back_clicked.connect(lambda: self.navigate_to(3))
        self.compare_page.back_clicked.connect(lambda: self.navigate_to(3))

        # Tooltip manager
        self.tooltip_manager = OnboardingTooltipManager(self.questionnaire_page, self.hm, self)
        self.questionnaire_page.question_stack.currentChanged.connect(self.tooltip_manager.trigger_tooltip)

        # Navigate to Splash landing first
        self.navigate_to(0)

    def navigate_to(self, index):
        self.stacked_widget.setCurrentIndex(index)
        
        # Tooltip trigger on page 1 load
        if index == 1:
            self.tooltip_manager.trigger_tooltip(self.questionnaire_page.current_index)
            # Show Welcome Screen button on questionnaire
            self.questionnaire_page.splash_back_btn.setVisible(True)
        else:
            QToolTip.hideText()

    def toggle_theme(self):
        if self.theme_mode == "dark":
            self.theme_mode = "light"
            self.setStyleSheet(LIGHT_QSS)
            self.theme_toggle.setText("🌙 Dark Mode")
            self.theme_toggle.setStyleSheet("background-color: transparent; border: 1px solid #C7C7CC; border-radius: 15px; padding: 6px; font-weight: bold; color: #1C1C1E;")
            plt.rcParams.update(LIGHT_CHART_STYLE)
        else:
            self.theme_mode = "dark"
            self.setStyleSheet(DARK_QSS)
            self.theme_toggle.setText("☀️ Light Mode")
            self.theme_toggle.setStyleSheet("background-color: transparent; border: 1px solid #535353; border-radius: 15px; padding: 6px; font-weight: bold; color: #FFFFFF;")
            plt.rcParams.update(DARK_CHART_STYLE)

        # Refresh matplotlib charts if they are visible
        if self.active_report and self.stacked_widget.currentIndex() == 3:
            self._regenerate_and_display_charts()

    def _regenerate_and_display_charts(self):
        # Fetch current values, generate new figures, and update result screen canvases
        df = self.dl.get_data()
        gp = GenreProfile(self.current_respondent, df)
        age_analyzer = AgeGroupAnalyzer(df)
        ce = CorrelationEngine(df)
        metrics_objs = [
            MentalHealthMetric(name, score, df[name].values)
            for name, score in self.active_report['predicted_scores'].items()
        ]
        rg = ReportGenerator(self.current_respondent, gp, metrics_objs, age_analyzer, ce, df)
        new_figs = rg.create_all_figures()
        
        # Update display
        self.results_page.display_results(
            self.active_report, new_figs, self.current_respondent, 
            ai_insights=self.active_ai_insights, theme_mode=self.theme_mode
        )

    def _reset_and_start_over(self):
        # Reset question state
        self.questionnaire_page.age_input.setValue(22)
        self.questionnaire_page.genre_input.setCurrentIndex(0)
        self.questionnaire_page.hours_input.setValue(6)
        self.questionnaire_page.stream_input.setCurrentIndex(0)
        self.questionnaire_page.answers["while_working"] = "Yes"
        self.questionnaire_page.answers["instrumentalist"] = "No"
        self.questionnaire_page._set_toggle("while_working", "Yes")
        self.questionnaire_page._set_toggle("instrumentalist", "No")
        self.questionnaire_page.show_question(0)
        self.navigate_to(1)

    def _open_compare_mode(self, respondent):
        self.compare_page.prepopulate(respondent, self.active_report['predicted_scores'])
        self.navigate_to(5)

    def _open_history_mode(self):
        self.history_page.load_history(self.theme_mode)
        self.navigate_to(4)

    def run_analysis(self, respondent: Respondent):
        self.current_respondent = respondent
        
        # Navigate to Loading screen and spin up progress cycle
        self.navigate_to(2)
        self.loading_page.start()

        # Build analysis data
        df = self.dl.get_data()
        gp = GenreProfile(respondent, df)
        age_analyzer = AgeGroupAnalyzer(df)
        ce = CorrelationEngine(df)
        predicted_scores = predict_mental_health(respondent, df, ce)
        
        metrics_objs = [
            MentalHealthMetric(name, score, df[name].values)
            for name, score in predicted_scores.items()
        ]
        
        # Assembly Report payload
        rg = ReportGenerator(respondent, gp, metrics_objs, age_analyzer, ce, df)
        self.active_report = rg.generate_report()
        self.active_chart_paths = rg.export_visualizations(output_dir="output")

        # Static Recommendations
        rec_engine = RecommendationEngine(respondent, predicted_scores, df)
        static_recs = rec_engine.get_behaviour_recommendations()
        genre_rec = rec_engine.get_genre_recommendation()
        
        # Add local recommendations list to active report
        self.active_report['recommendations'] = static_recs
        self.active_report['genre_recommendation'] = genre_rec

        # Fetch API key
        api_key = os.getenv('MISTRAL_API_KEY')
        if not api_key:
            # Fallback immediately
            QTimer.singleShot(1500, lambda: self._on_analysis_finished(None))
            return

        # Start AI Worker Thread
        client = MistralClient(api_key=api_key)
        self.ai_worker = AIWorker(
            client=client,
            respondent_data=self.active_report['respondent'],
            predicted_scores=self.active_report['predicted_scores'],
            cohort_stats=self.active_report['cohort_stats'],
            static_recommendations=static_recs
        )
        self.ai_worker.finished.connect(self._on_analysis_finished)
        self.ai_worker.error.connect(self._on_analysis_error)
        self.ai_worker.start()

    def _on_analysis_finished(self, ai_insights):
        self.loading_page.stop()
        self.active_ai_insights = ai_insights
        
        # Save to History database CSV
        percentiles = self.active_report['percentiles']
        self.hm.save_session(self.current_respondent, self.active_report['predicted_scores'], percentiles)

        # Mark onboarding shown since analysis ran successfully
        self.tooltip_manager.mark_completed()

        # Update and render displays
        self._regenerate_and_display_charts()
        self.navigate_to(3)

    def _on_analysis_error(self, err_msg):
        self.loading_page.stop()
        self.active_ai_insights = None
        
        # Save history even if AI fails
        percentiles = self.active_report['percentiles']
        self.hm.save_session(self.current_respondent, self.active_report['predicted_scores'], percentiles)
        
        self.tooltip_manager.mark_completed()

        # Update displays (will trigger banner on Results page)
        self._regenerate_and_display_charts()
        self.navigate_to(3)

    def _export_pdf_report(self):
        if not self.active_report:
            return

        # Open file dialog or build default path
        os.makedirs("output", exist_ok=True)
        pdf_path = os.path.join("output", "MoodMap_Clinical_Report.pdf")

        try:
            # Gather AI Insights
            ai_data = self.active_ai_insights if self.active_ai_insights else {
                "insights": "Offline: No AI interpretation generated.",
                "recommendations": "Offline: Check local statistics guidelines."
            }

            exporter = PDFExporter(
                report=self.active_report,
                chart_paths=self.active_chart_paths,
                recommendations=self.active_report['recommendations'],
                genre_rec=self.active_report.get('genre_recommendation', {}),
                ai_insights=ai_data,
                output_dir="output"
            )
            generated_path = exporter.generate()
            
            QMessageBox.information(
                self, "PDF Compiled",
                f"Clinical PDF Report successfully generated at:<br><b>{os.path.abspath(generated_path)}</b>"
            )
        except Exception as e:
            QMessageBox.warning(self, "Export Failed", f"Could not export PDF report: {str(e)}")
