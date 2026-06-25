import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from typing import Optional


CHART_STYLE = {
    'figure.facecolor': '#181818',
    'axes.facecolor': '#282828',
    'axes.edgecolor': '#535353',
    'axes.labelcolor': '#FFFFFF',
    'text.color': '#FFFFFF',
    'xtick.color': '#B3B3B3',
    'ytick.color': '#B3B3B3',
    'grid.color': '#535353',
    'grid.alpha': 0.3,
    'font.size': 11,
}  # defines the style for the charts

plt.rcParams.update(CHART_STYLE)

METRIC_COLORS = {
    'Anxiety': '#1DB954',
    'Depression': '#1E90FF',
    'Insomnia': '#9B59B6',
    'OCD': '#E74C3C',
}  # defines the colors for each metric
METRICS = ['Anxiety', 'Depression', 'Insomnia', 'OCD']

# Age brackets matching AgeGroupAnalyzer in engine.py
AGE_BRACKETS = [
    (10, 17, "10-17"),
    (18, 24, "18-24"),
    (25, 34, "25-34"),
    (35, 44, "35-44"),
    (45, 54, "45-54"),
    (55, 64, "55-64"),
    (65, 89, "65-89"),
]


def _assign_age_bracket(age):
    """Assigns an age bracket label to a given age value."""
    for low, high, label in AGE_BRACKETS:
        if low <= age <= high:
            return label
    return "18-24"


def create_genre_bar_chart(df: pd.DataFrame) -> Figure:
    """Draws a grouped bar chart showing average Anxiety, Depression,
    Insomnia, and OCD scores for each genre."""
    grouped = df.groupby('Fav genre')[METRICS].mean().reset_index()
    n = len(grouped)
    x = np.arange(n)
    width = 0.15
    fig = Figure(figsize=(12, 6), dpi=100)
    ax = fig.add_subplot(111)
    for i, metric in enumerate(METRICS):
        ax.bar(
            x + (i - 1.5) * width,
            grouped[metric],
            width,
            label=metric,
            color=METRIC_COLORS[metric],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(grouped['Fav genre'], rotation=45, ha='right')
    ax.set_ylabel("Average Score")
    ax.set_title("Average Score by Genre", fontsize=15, pad=20)
    ax.legend(fontsize=10)
    fig.tight_layout()
    return fig


def create_age_line_chart(df: pd.DataFrame) -> Figure:
    """Draws a dual-axis line chart: primary axis shows average Anxiety
    per age bracket; secondary axis shows average Daily Listening Hours."""
    # Assign age brackets to each row
    working_df = df.copy()
    working_df['age_bracket'] = working_df['Age'].apply(_assign_age_bracket)

    # Compute means per bracket, preserving bracket order
    bracket_labels = [label for _, _, label in AGE_BRACKETS]
    anxiety_mean = working_df.groupby('age_bracket')['Anxiety'].mean()
    hours_mean = working_df.groupby('age_bracket')['Hours per day'].mean()

    # Filter to only brackets that exist in data, preserving order
    ordered_labels = [b for b in bracket_labels if b in anxiety_mean.index]
    anxiety_vals = [anxiety_mean[b] for b in ordered_labels]
    hours_vals = [hours_mean[b] for b in ordered_labels]

    fig = Figure(figsize=(12, 6), dpi=100)
    ax1 = fig.add_subplot(111)
    line1 = ax1.plot(
        ordered_labels, anxiety_vals,
        color=METRIC_COLORS['Anxiety'], label='Anxiety',
    )
    ax1.set_ylabel('Anxiety Level', color=METRIC_COLORS['Anxiety'])
    ax1.tick_params(axis='y', colors=METRIC_COLORS['Anxiety'])
    ax1.set_xlabel('Age Group', fontsize=12)
    ax1.set_title("Age vs Anxiety and Listening Time", fontsize=12, pad=20)

    ax2 = ax1.twinx()
    line2 = ax2.plot(
        ordered_labels, hours_vals,
        marker="s", color='#1DB954', label="Avg Listening Hours",
    )
    ax2.set_ylabel("Average Listening Hours", color='#1DB954', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#1DB954')
    lines = line1 + line2
    ax1.legend(lines, [l.get_label() for l in lines], loc='best', fontsize=10)
    fig.tight_layout()
    return fig


def create_anxiety_histogram(df: pd.DataFrame, user_score: float) -> Figure:
    """Draws a histogram of anxiety distribution with a vertical dashed
    red line at user_score labeled with the calculated percentile."""
    fig = Figure(figsize=(12, 6), dpi=100)
    ax = fig.add_subplot(111)
    bins = np.arange(0, 11)
    if not df.empty and 'Anxiety' in df.columns:
        counts, _, patches = ax.hist(
            df['Anxiety'], bins=bins,
            color=METRIC_COLORS['Anxiety'], edgecolor='white', alpha=0.7,
        )
        idx = max(0, min(int(np.floor(user_score)), len(patches) - 1))
        if len(patches) > idx >= 0:
            patches[idx].set_facecolor('#E74C3C')
        percentile = (df['Anxiety'] <= user_score).mean() * 100
        max_val = max(counts) if len(counts) > 0 else 1
    else:
        percentile = 0.0
        max_val = 1
    ax.axvline(
        x=user_score, color='#E74C3C', linestyle='--', linewidth=2.5,
        label=f'Your Score: {user_score:.1f}',
    )
    ax.text(
        user_score + 0.15, max_val * 0.85,
        f"Your Score: {user_score:.1f}\nPercentile: {percentile:.1f}%",
        color='#E74C3C', fontweight='bold',
        bbox=dict(facecolor='#181818', alpha=0.8, edgecolor='#535353'),
    )
    ax.set_xlabel('Anxiety Score')
    ax.set_ylabel("Frequency")
    ax.set_title('Anxiety Score Distribution & Your Percentile')
    ax.set_xlim(-0.5, 10.5)
    ax.set_xticks(range(11))
    fig.tight_layout()
    return fig


def create_scatter_plot(
    df: pd.DataFrame, user_hours: float, user_depression: float,
) -> Figure:
    """Draws a scatter plot of daily hours (X) vs depression (Y) with
    a linear regression trend line and the user's point highlighted."""
    clean = df[['Hours per day', 'Depression']].dropna()
    hours = clean['Hours per day'].to_numpy()
    depression = clean['Depression'].to_numpy()
    fig = Figure(figsize=(7, 5), dpi=100)
    ax = fig.add_subplot(111)
    ax.scatter(
        hours, depression, alpha=0.4, s=20,
        color='#1DB954', label='Survey Respondents',
    )
    if len(hours) >= 2 and np.std(hours) > 0:
        m, b = np.polyfit(hours, depression, 1)
        x_line = np.linspace(hours.min(), hours.max(), 100)
        ax.plot(
            x_line, m * x_line + b,
            color='#D9534F', linewidth=2, label=f'Trend (slope={m:.2f})',
        )
    ax.scatter(
        [user_hours], [user_depression], marker='*', s=400,
        color='#F0AD4E', edgecolors='black', linewidths=1.2,
        zorder=5, label='Your data point',
    )
    ax.set_xlabel('Hours/Day')
    ax.set_ylabel('Depression Score')
    ax.set_title("Listening Hours v/s Depression")
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    return fig


def create_correlation_heatmap(df: pd.DataFrame) -> Figure:
    """6×6 heatmap showing Pearson correlations between Hours per day,
    BPM, Anxiety, Depression, Insomnia, and OCD."""
    cols = ['Hours per day', 'BPM', 'Anxiety', 'Depression', 'Insomnia', 'OCD']
    corr = df[cols].corr(method='pearson')
    fig = Figure(figsize=(12, 6), dpi=100)
    ax = fig.add_subplot(111)
    im = ax.imshow(corr.values, cmap='RdYlGn_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(cols, rotation=45, ha='right', fontsize=10)
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr.values[i, j]
            text_color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(
                j, i, f'{val:.2f}', ha='center', va='center',
                color=text_color, fontsize=10, fontweight='bold',
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Correlation')
    ax.set_title('Correlation Heatmap', fontsize=15, pad=20)
    fig.tight_layout()
    return fig


def create_trend_chart(
    trend_data: dict, session_count: int,
) -> Optional[Figure]:
    """Draws line charts for Anxiety, Depression, Insomnia, and OCD
    across all user sessions. Returns None if session_count < 2."""
    if session_count < 2:
        return None
    fig = Figure(figsize=(12, 6), dpi=100)
    ax = fig.add_subplot(111)
    session = list(range(1, session_count + 1))
    for metric in METRICS:
        values = trend_data.get(metric, [])
        if values:
            ax.plot(
                session[:len(values)], values, marker='o', linewidth=2,
                color=METRIC_COLORS[metric], label=metric, markersize=6,
            )
    ax.set_xlabel("Session", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title('Mental Health Trend Across Sessions', fontsize=15, pad=20)
    ax.set_ylim(0, 10.5)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


class ReportGenerator:
    """Prepares analysis report data and processes visualizations."""

    def __init__(
        self, respondent, genre_profile, metrics_objs: list,
        age_analyzer, corr_engine, df: pd.DataFrame,
    ) -> None:
        """Initializes references to all analysis components."""
        self.respondent = respondent
        self.genre_profile = genre_profile
        self.metrics_objs = metrics_objs
        self.age_analyzer = age_analyzer
        self.corr_engine = corr_engine
        self.df = df

    def generate_report(self) -> dict:
        """Assembles a details dictionary containing summary fields,
        metrics predictions, percentiles, cohort details, correlation
        scores, and BPM calculations."""
        # Build metrics payload
        metrics_data = {}
        predicted_scores = {}
        percentiles = {}
        for m in self.metrics_objs:
            metrics_data[m.name] = {
                'score': m.score,
                'percentile': m.percentile,
                'severity': m.severity,
                'severity_color': m.get_severity_color(),
                'dataset_mean': m.dataset_mean,
                'dataset_std': m.dataset_std,
            }
            predicted_scores[m.name] = m.score
            percentiles[m.name] = m.percentile

        # Cohort stats
        cohort_stats = self.genre_profile.get_cohort_stats()

        # Comparison
        comparison = self.genre_profile.compare_to_cohort(predicted_scores)

        # Age bracket info
        bracket_label = self.age_analyzer.assign_bracket(self.respondent.age)
        bracket_stats = self.age_analyzer.get_bracket_stats(self.respondent.age)

        # Correlation info
        strongest_hours = self.corr_engine.get_strongest()
        strongest_bpm = self.corr_engine.get_strongest_bpm()

        return {
            'respondent': self.respondent.to_dict(),
            'metrics': metrics_data,
            'predicted_scores': predicted_scores,
            'percentiles': percentiles,
            'cohort_stats': cohort_stats,
            'comparison': comparison,
            'age_bracket': bracket_label,
            'age_bracket_stats': bracket_stats,
            'correlations': self.corr_engine.correlations,
            'bpm_correlations': self.corr_engine.bpm_correlations,
            'strongest_hours_correlation': {
                'metric': strongest_hours[0],
                'r': strongest_hours[1],
            },
            'strongest_bpm_correlation': {
                'metric': strongest_bpm[0],
                'r': strongest_bpm[1],
            },
            'genre_cohort_size': self.genre_profile.cohort_size,
        }

    def create_all_figures(self) -> list:
        """Generates and returns a list containing all 5 Matplotlib
        figures."""
        predicted_scores = {m.name: m.score for m in self.metrics_objs}

        figs = [
            create_genre_bar_chart(self.df),
            create_age_line_chart(self.df),
            create_anxiety_histogram(
                self.df, predicted_scores.get('Anxiety', 0.0),
            ),
            create_scatter_plot(
                self.df,
                self.respondent.hours_per_day,
                predicted_scores.get('Depression', 0.0),
            ),
            create_correlation_heatmap(self.df),
        ]
        return figs

    def export_visualizations(self, output_dir: str = "output") -> list:
        """Creates directory if missing. Saves all 5 charts as PNG
        files using dpi=150 and tight margins. Returns list of
        absolute file paths."""
        os.makedirs(output_dir, exist_ok=True)

        chart_names = [
            'genre_bar', 'age_line', 'anxiety_histogram',
            'scatter_plot', 'correlation_heatmap',
        ]
        figures = self.create_all_figures()

        paths = []
        for name, fig in zip(chart_names, figures):
            path = os.path.join(output_dir, f'{name}.png')
            fig.savefig(path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor(), edgecolor='none')
            paths.append(os.path.abspath(path))

        return paths
