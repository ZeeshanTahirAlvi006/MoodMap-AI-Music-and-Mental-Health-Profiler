import os
from datetime import datetime

from fpdf import FPDF


class PDFExporter:
    """Generates a multi-page PDF document using fpdf2."""

    def __init__(
        self,
        report: dict,
        chart_paths: list,
        recommendations: list,
        genre_rec: dict,
        ai_insights: dict,
        output_dir: str = "output",
    ) -> None:
        """Initializes class parameters."""
        self.report: dict = report
        self.chart_paths: list = chart_paths
        self.recommendations: list = recommendations
        self.genre_rec: dict = genre_rec
        self.ai_insights: dict = ai_insights
        self.output_dir: str = output_dir

    def generate(self) -> str:
        """Orchestrates the document creation process and returns the
        absolute path of the generated PDF."""
        os.makedirs(self.output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"moodmap_report_{timestamp}.pdf"
        out_path = os.path.join(self.output_dir, filename)

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=20)

        # ── Page 1: Cover ───────────────────────────────────────────
        self._build_cover_page(pdf)

        # ── Page 2: Predicted Scores & Cohorts ──────────────────────
        self._build_scores_page(pdf)

        # ── Page 3: Actionable Recommendations ──────────────────────
        self._build_recommendations_page(pdf)

        # ── Page 4: AI Clinical Insights & Music Playlist ───────────
        self._build_ai_insights_page(pdf)

        # ── Pages 5+: Visual Charts ────────────────────────────────
        self._build_chart_pages(pdf)

        pdf.output(out_path)
        return os.path.abspath(out_path)

    # ═══════════════════════════════════════════════════════════════
    #  Private page-builder methods
    # ═══════════════════════════════════════════════════════════════

    def _build_cover_page(self, pdf: FPDF) -> None:
        """Page 1: Cover - Large centered title, timestamp,
        and user profile configuration."""
        pdf.add_page()

        # Dark background header area
        pdf.set_fill_color(24, 24, 24)
        pdf.rect(0, 0, 210, 297, 'F')

        # Accent bar at top
        pdf.set_fill_color(29, 185, 84)  # #1DB954
        pdf.rect(0, 0, 210, 8, 'F')

        # Title
        pdf.set_y(60)
        pdf.set_font('Helvetica', 'B', 36)
        pdf.set_text_color(29, 185, 84)
        pdf.cell(0, 20, 'MoodMap', ln=True, align='C')

        pdf.set_font('Helvetica', '', 18)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 12, 'Wellness Report', ln=True, align='C')

        # Decorative divider
        pdf.ln(10)
        pdf.set_draw_color(29, 185, 84)
        pdf.set_line_width(0.5)
        pdf.line(60, pdf.get_y(), 150, pdf.get_y())
        pdf.ln(15)

        # Timestamp
        pdf.set_font('Helvetica', '', 11)
        pdf.set_text_color(179, 179, 179)
        report_time = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        pdf.cell(0, 8, f'Generated: {report_time}', ln=True, align='C')
        pdf.ln(20)

        # User profile configuration
        respondent = self.report.get('respondent', {})
        if respondent:
            pdf.set_font('Helvetica', 'B', 14)
            pdf.set_text_color(29, 185, 84)
            pdf.cell(0, 10, 'User Profile', ln=True, align='C')
            pdf.ln(5)

            pdf.set_font('Helvetica', '', 11)
            pdf.set_text_color(255, 255, 255)

            profile_fields = [
                ('Age', respondent.get('Age', 'N/A')),
                ('Favorite Genre', respondent.get('Fav genre', 'N/A')),
                ('Hours per Day', respondent.get('Hours per day', 'N/A')),
                ('While Working', respondent.get('While working', 'N/A')),
                ('Instrumentalist', respondent.get('Instrumentalist', 'N/A')),
                (
                    'Streaming Service',
                    respondent.get('Primary streaming service', 'N/A'),
                ),
            ]

            for label, value in profile_fields:
                pdf.set_x(50)
                pdf.set_font('Helvetica', 'B', 11)
                pdf.set_text_color(179, 179, 179)
                pdf.cell(50, 8, f'{label}:', align='R')
                pdf.set_font('Helvetica', '', 11)
                pdf.set_text_color(255, 255, 255)
                pdf.cell(60, 8, f'  {value}', ln=True, align='L')

    def _build_scores_page(self, pdf: FPDF) -> None:
        """Page 2: Predicted Scores & Cohorts - Score cards with
        percentiles, severity, and genre comparison tables."""
        pdf.add_page()

        # Dark background
        pdf.set_fill_color(24, 24, 24)
        pdf.rect(0, 0, 210, 297, 'F')

        # Section title
        pdf.set_font('Helvetica', 'B', 18)
        pdf.set_text_color(29, 185, 84)
        pdf.cell(0, 12, 'Predicted Mental Health Scores', ln=True, align='C')
        pdf.ln(5)

        metrics_data = self.report.get('metrics', {})
        cohort_stats = self.report.get('cohort_stats', {})

        # Severity color map
        color_map = {
            'Low': (29, 185, 84),       # #1DB954
            'Moderate': (243, 156, 18),  # #F39C12
            'High': (231, 76, 60),       # #E74C3C
        }

        # Score cards
        for metric_name in ['Anxiety', 'Depression', 'Insomnia', 'OCD']:
            m = metrics_data.get(metric_name, {})
            score = m.get('score', 0.0)
            percentile = m.get('percentile', 0.0)
            severity = m.get('severity', 'Low')
            sev_color = color_map.get(severity, (255, 255, 255))

            # Card background
            pdf.set_fill_color(40, 40, 40)
            card_y = pdf.get_y()
            pdf.rect(15, card_y, 180, 22, 'F')

            # Metric name
            pdf.set_x(20)
            pdf.set_font('Helvetica', 'B', 13)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(45, 10, metric_name, align='L')

            # Score
            pdf.set_font('Helvetica', 'B', 13)
            pdf.set_text_color(*sev_color)
            pdf.cell(25, 10, f'{score:.1f}', align='C')

            # Severity badge
            pdf.set_font('Helvetica', '', 10)
            pdf.cell(30, 10, severity, align='C')

            # Percentile
            pdf.set_text_color(179, 179, 179)
            pdf.set_font('Helvetica', '', 10)
            pdf.cell(40, 10, f'Percentile: {percentile:.1f}%', align='C')

            # Genre average comparison
            genre_avg = cohort_stats.get(metric_name, 0.0)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font('Helvetica', '', 10)
            pdf.cell(35, 10, f'Genre Avg: {genre_avg:.1f}', ln=True, align='C')

            pdf.ln(5)

        # Genre Cohort Comparison Table
        pdf.ln(10)
        pdf.set_font('Helvetica', 'B', 14)
        pdf.set_text_color(29, 185, 84)
        pdf.cell(0, 10, 'Genre Cohort Comparison', ln=True, align='C')
        pdf.ln(3)

        # Table header
        pdf.set_fill_color(40, 40, 40)
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(29, 185, 84)

        col_widths = [40, 35, 35, 35, 35]
        headers = ['Metric', 'Your Score', 'Genre Avg', 'Delta', 'Direction']
        pdf.set_x(15)
        for i, header in enumerate(headers):
            pdf.cell(col_widths[i], 8, header, border=1, align='C', fill=True)
        pdf.ln()

        # Table rows
        comparison = self.report.get('comparison', {})
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(255, 255, 255)

        for metric_name in ['Anxiety', 'Depression', 'Insomnia', 'OCD']:
            comp = comparison.get(metric_name, {})
            user_s = comp.get('user_score', 0.0)
            cohort_m = comp.get('cohort_mean', 0.0)
            delta = comp.get('delta', 0.0)
            direction = comp.get('direction', 'N/A')

            pdf.set_x(15)
            pdf.set_fill_color(30, 30, 30)
            pdf.cell(col_widths[0], 8, metric_name, border=1, align='C', fill=True)
            pdf.cell(col_widths[1], 8, f'{user_s:.1f}', border=1, align='C', fill=True)
            pdf.cell(col_widths[2], 8, f'{cohort_m:.1f}', border=1, align='C', fill=True)
            pdf.cell(col_widths[3], 8, f'{delta:+.2f}', border=1, align='C', fill=True)
            pdf.cell(col_widths[4], 8, direction, border=1, align='C', fill=True)
            pdf.ln()

    def _build_recommendations_page(self, pdf: FPDF) -> None:
        """Page 3: Actionable Recommendations - Lists static
        behavior modification changes."""
        pdf.add_page()

        # Dark background
        pdf.set_fill_color(24, 24, 24)
        pdf.rect(0, 0, 210, 297, 'F')

        # Section title
        pdf.set_font('Helvetica', 'B', 18)
        pdf.set_text_color(29, 185, 84)
        pdf.cell(0, 12, 'Actionable Recommendations', ln=True, align='C')
        pdf.ln(8)

        # Static recommendations list
        if self.recommendations:
            for i, rec in enumerate(self.recommendations, 1):
                pdf.set_fill_color(40, 40, 40)
                rec_y = pdf.get_y()
                # Estimate height needed
                lines = pdf.multi_cell(
                    160, 7, f'{i}. {rec}', dry_run=True, output='LINES'
                )
                box_height = max(len(lines) * 7 + 6, 16)
                pdf.rect(15, rec_y, 180, box_height, 'F')

                pdf.set_x(20)
                pdf.set_font('Helvetica', 'B', 11)
                pdf.set_text_color(29, 185, 84)
                pdf.cell(8, 7, f'{i}.', align='L')

                pdf.set_font('Helvetica', '', 11)
                pdf.set_text_color(255, 255, 255)
                pdf.multi_cell(155, 7, rec)
                pdf.ln(4)
        else:
            pdf.set_font('Helvetica', 'I', 11)
            pdf.set_text_color(179, 179, 179)
            pdf.cell(
                0, 10,
                'No static recommendations generated.',
                ln=True, align='C',
            )

        # Genre swap recommendation
        if self.genre_rec:
            pdf.ln(10)
            pdf.set_font('Helvetica', 'B', 14)
            pdf.set_text_color(29, 185, 84)
            pdf.cell(0, 10, 'Genre Exploration Suggestion', ln=True, align='C')
            pdf.ln(3)

            pdf.set_font('Helvetica', '', 11)
            pdf.set_text_color(255, 255, 255)

            suggested = self.genre_rec.get('suggested_genre', 'N/A')
            reason = self.genre_rec.get('reason', '')

            pdf.set_x(20)
            pdf.multi_cell(
                170, 7,
                f'Consider exploring: {suggested}\n{reason}',
            )

    def _build_ai_insights_page(self, pdf: FPDF) -> None:
        """Page 4: AI Clinical Insights & Music Playlist - Displays
        dynamic insights and recommendations from Mistral API,
        followed by a medical disclaimer."""
        pdf.add_page()

        # Dark background
        pdf.set_fill_color(24, 24, 24)
        pdf.rect(0, 0, 210, 297, 'F')

        # Section title
        pdf.set_font('Helvetica', 'B', 18)
        pdf.set_text_color(29, 185, 84)
        pdf.cell(0, 12, 'AI Clinical Insights', ln=True, align='C')
        pdf.ln(5)

        insights_text = ''
        recommendations_text = ''

        if self.ai_insights:
            insights_text = self.ai_insights.get('insights', '')
            recommendations_text = self.ai_insights.get('recommendations', '')

        # AI Insights
        if insights_text:
            pdf.set_font('Helvetica', 'B', 12)
            pdf.set_text_color(30, 144, 255)  # #1E90FF
            pdf.cell(0, 8, 'Clinical Analysis', ln=True)
            pdf.ln(2)

            pdf.set_font('Helvetica', '', 10)
            pdf.set_text_color(255, 255, 255)

            # Clean markdown formatting for PDF
            cleaned = self._clean_markdown(insights_text)
            pdf.multi_cell(0, 6, cleaned)
            pdf.ln(5)
        else:
            pdf.set_font('Helvetica', 'I', 11)
            pdf.set_text_color(179, 179, 179)
            pdf.cell(
                0, 10,
                'AI insights not available for this report.',
                ln=True, align='C',
            )
            pdf.ln(5)

        # AI Music Recommendations
        if recommendations_text:
            pdf.set_font('Helvetica', 'B', 12)
            pdf.set_text_color(155, 89, 182)  # #9B59B6
            pdf.cell(0, 8, 'Recommended Playlists / Artists', ln=True)
            pdf.ln(2)

            pdf.set_font('Helvetica', '', 10)
            pdf.set_text_color(255, 255, 255)

            cleaned = self._clean_markdown(recommendations_text)
            pdf.multi_cell(0, 6, cleaned)
            pdf.ln(5)

        # ── Disclaimer block ───────────────────────────────────────
        self._add_disclaimer(pdf)

    def _build_chart_pages(self, pdf: FPDF) -> None:
        """Pages 5+: Embeds exported PNG chart figures,
        fitting one per page."""
        for chart_path in self.chart_paths:
            if not os.path.exists(chart_path):
                continue

            pdf.add_page()

            # Dark background
            pdf.set_fill_color(24, 24, 24)
            pdf.rect(0, 0, 210, 297, 'F')

            # Chart title from filename
            chart_name = os.path.splitext(os.path.basename(chart_path))[0]
            chart_title = chart_name.replace('_', ' ').title()

            pdf.set_font('Helvetica', 'B', 14)
            pdf.set_text_color(29, 185, 84)
            pdf.cell(0, 12, chart_title, ln=True, align='C')
            pdf.ln(5)

            # Embed chart image, centered and fitted to page width
            try:
                pdf.image(
                    chart_path,
                    x=10,
                    y=pdf.get_y(),
                    w=190,
                )
            except Exception:
                pdf.set_font('Helvetica', 'I', 10)
                pdf.set_text_color(179, 179, 179)
                pdf.cell(
                    0, 10,
                    f'Failed to embed chart: {chart_name}',
                    ln=True, align='C',
                )

    # ═══════════════════════════════════════════════════════════════
    #  Utility methods
    # ═══════════════════════════════════════════════════════════════

    def _add_disclaimer(self, pdf: FPDF) -> None:
        """Adds the medical disclaimer block at the current position."""
        disclaimer_text = (
            "Disclaimer: The insights and recommendations provided by "
            "MoodMap and its AI assistant are for educational and "
            "informational purposes only and do not constitute clinical "
            "diagnosis or medical advice. Please consult a qualified "
            "mental health professional for medical concerns."
        )

        pdf.ln(10)

        # Disclaimer box
        pdf.set_fill_color(40, 30, 30)
        pdf.set_draw_color(231, 76, 60)  # #E74C3C
        box_y = pdf.get_y()
        pdf.rect(10, box_y, 190, 30, 'DF')

        pdf.set_xy(15, box_y + 3)
        pdf.set_font('Helvetica', 'BI', 8)
        pdf.set_text_color(231, 76, 60)
        pdf.multi_cell(180, 5, disclaimer_text)

    @staticmethod
    def _clean_markdown(text: str) -> str:
        """Strips common markdown formatting characters for
        clean PDF rendering."""
        # Remove markdown bold/italic markers
        cleaned = text.replace('**', '')
        cleaned = cleaned.replace('__', '')
        cleaned = cleaned.replace('*', '')
        cleaned = cleaned.replace('_', ' ')

        # Remove markdown headers
        lines = cleaned.split('\n')
        processed = []
        for line in lines:
            stripped = line.lstrip('#').strip()
            if stripped:
                processed.append(stripped)
            else:
                processed.append('')
        cleaned = '\n'.join(processed)

        # Remove excess blank lines
        while '\n\n\n' in cleaned:
            cleaned = cleaned.replace('\n\n\n', '\n\n')

        return cleaned.strip()
