import numpy as np
import pandas as pd

from domain import Respondent


METRICS = ['Anxiety', 'Depression', 'Insomnia', 'OCD']


class AgeGroupAnalyzer:
    """Partitions dataset by age to extract age bracket means."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df: pd.DataFrame = df
        self.brackets: list = [
            (10, 17, "10-17"),
            (18, 24, "18-24"),
            (25, 34, "25-34"),
            (35, 44, "35-44"),
            (45, 54, "45-54"),
            (55, 64, "55-64"),
            (65, 89, "65-89"),
        ]
        self.bracket_stats: dict = self._precompute_bracket_stats()

    def _precompute_bracket_stats(self) -> dict:
        """Pre-computes mean metrics for every age bracket."""
        stats = {}
        for low, high, label in self.brackets:
            bracket_df = self.df[
                (self.df['Age'] >= low) & (self.df['Age'] <= high)
            ]
            if len(bracket_df) > 0:
                bracket_means = {
                    m: float(bracket_df[m].mean()) for m in METRICS
                }
                bracket_means['cohort_size'] = len(bracket_df)
            else:
                bracket_means = {m: 0.0 for m in METRICS}
                bracket_means['cohort_size'] = 0
            stats[label] = bracket_means
        return stats

    def assign_bracket(self, age: int) -> str:
        """Finds matching bracket string for the given age.
        Defaults to '18-24' if no bracket matches."""
        for low, high, label in self.brackets:
            if low <= age <= high:
                return label
        return "18-24"

    def get_bracket_boundaries(self, age: int) -> tuple:
        """Returns (min_age, max_age) boundary for the user's bracket."""
        for low, high, label in self.brackets:
            if low <= age <= high:
                return (low, high)
        return (18, 24)

    def get_bracket_stats(self, age: int) -> dict:
        """Returns mean metrics and cohort size for the user's bracket."""
        label = self.assign_bracket(age)
        return self.bracket_stats.get(label, {m: 0.0 for m in METRICS})


class CorrelationEngine:
    """Calculates Pearson product-moment correlation coefficients
    using NumPy."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df: pd.DataFrame = df
        self.correlations: dict = self._compute_correlations_for_column(
            'Hours per day'
        )
        self.bpm_correlations: dict = self._compute_correlations_for_column(
            'BPM'
        )

    def _compute_correlations_for_column(self, column: str) -> dict:
        """Computes Pearson r between the given column and each
        mental health metric.

        Formula:
            r_xy = sum((x_i - x_bar)(y_i - y_bar)) /
                   sqrt(sum((x_i - x_bar)^2) * sum((y_i - y_bar)^2))

        Implementation: np.corrcoef(col_values, metric_values)[0, 1]

        Zero-Variance Guard: If standard deviation of either column
        is zero, return correlation as 0.0.
        """
        results = {}
        # Drop rows where the target column is NaN
        valid_df = self.df.dropna(subset=[column])

        for metric in METRICS:
            # Also drop rows where the metric is NaN
            subset = valid_df.dropna(subset=[metric])

            if len(subset) < 2:
                results[metric] = 0.0
                continue

            col_values = subset[column].values.astype(float)
            metric_values = subset[metric].values.astype(float)

            # Zero-Variance Guard
            if np.std(col_values) == 0.0 or np.std(metric_values) == 0.0:
                results[metric] = 0.0
                continue

            r = np.corrcoef(col_values, metric_values)[0, 1]

            # Handle NaN from corrcoef (e.g. constant arrays)
            if np.isnan(r):
                r = 0.0

            results[metric] = float(r)

        return results

    def get_strongest(self) -> tuple:
        """Returns metric name and r value with maximum absolute
        correlation to daily hours."""
        if not self.correlations:
            return ('Anxiety', 0.0)
        strongest = max(
            self.correlations.items(), key=lambda x: abs(x[1])
        )
        return strongest

    def get_strongest_bpm(self) -> tuple:
        """Returns metric name and r value with maximum absolute
        correlation to music BPM."""
        if not self.bpm_correlations:
            return ('Anxiety', 0.0)
        strongest = max(
            self.bpm_correlations.items(), key=lambda x: abs(x[1])
        )
        return strongest


def predict_mental_health(
    respondent: Respondent,
    df: pd.DataFrame,
    correlation_engine: CorrelationEngine = None,
) -> dict:
    """Predicts mental health scores for a respondent using
    weighted cohort averages, hours deviation, and behavioral modifiers.

    Steps:
        1. Filter cohorts (intersection, genre, age group)
        2. Base score (weighted means)
        3. Daily hours deviation adjustment
        4. Behavioral modifier adjustments
        5. Clamping to [0.0, 10.0]
    """
    # Initialize correlation engine if not provided
    if correlation_engine is None:
        correlation_engine = CorrelationEngine(df)

    # Set up age group analyzer
    age_analyzer = AgeGroupAnalyzer(df)
    user_bracket = age_analyzer.assign_bracket(respondent.age)
    low, high = age_analyzer.get_bracket_boundaries(respondent.age)

    # ── Step 1: Filter cohorts ──────────────────────────────────────
    genre_cohort = df[df['Fav genre'] == respondent.fav_genre]
    age_cohort = df[(df['Age'] >= low) & (df['Age'] <= high)]
    intersection_cohort = genre_cohort[
        (genre_cohort['Age'] >= low) & (genre_cohort['Age'] <= high)
    ]

    predicted = {}

    for metric in METRICS:
        # ── Step 2: Base Score ──────────────────────────────────────
        global_mean = float(df[metric].mean())

        # Genre cohort mean (fall back to global if empty)
        if len(genre_cohort) > 0:
            genre_mean = float(genre_cohort[metric].mean())
        else:
            genre_mean = global_mean

        # Age cohort mean
        if len(age_cohort) > 0:
            age_mean = float(age_cohort[metric].mean())
        else:
            age_mean = global_mean

        # Intersection cohort mean
        if len(intersection_cohort) >= 10:
            intersection_mean = float(intersection_cohort[metric].mean())
            base = (
                0.60 * intersection_mean
                + 0.25 * genre_mean
                + 0.15 * age_mean
            )
        else:
            base = 0.50 * genre_mean + 0.50 * age_mean

        score = base

        # ── Step 3: Daily Hours Deviation ───────────────────────────
        if len(genre_cohort) > 0 and 'Hours per day' in genre_cohort.columns:
            genre_mean_hours = float(
                genre_cohort['Hours per day'].dropna().mean()
            )
        else:
            genre_mean_hours = float(df['Hours per day'].dropna().mean())

        hours_dev = respondent.hours_per_day - genre_mean_hours
        r_m = correlation_engine.correlations.get(metric, 0.0)
        score += hours_dev * r_m

        # ── Step 4: Behavioral Modifier Adjustments ─────────────────
        behavioral_columns = {
            'While working': respondent.while_working,
            'Instrumentalist': respondent.instrumentalist,
            'Primary streaming service': respondent.streaming_service,
        }

        for col_name, user_value in behavioral_columns.items():
            if col_name not in df.columns:
                continue

            modifier_cohort = df[df[col_name] == user_value]

            if len(modifier_cohort) >= 5:
                modifier_mean = float(modifier_cohort[metric].mean())
                delta = 0.15 * (modifier_mean - global_mean)
            else:
                delta = 0.0

            score += delta

        # ── Step 5: Clamping ────────────────────────────────────────
        score = float(np.clip(score, 0.0, 10.0))
        predicted[metric] = round(score, 2)

    return predicted
