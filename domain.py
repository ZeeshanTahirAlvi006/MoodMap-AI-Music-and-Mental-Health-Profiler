import numpy as np
import pandas as pd


class Respondent:
    """Stores questionnaire responses for the active session."""

    def __init__(
        self,
        age: int,
        fav_genre: str,
        hours_per_day: float,
        while_working: str,
        instrumentalist: str,
        streaming_service: str,
    ) -> None:
        """Initializes respondent parameters."""
        self.age: int = age
        self.fav_genre: str = fav_genre
        self.hours_per_day: float = hours_per_day
        self.while_working: str = while_working
        self.instrumentalist: str = instrumentalist
        self.streaming_service: str = streaming_service

    def to_dict(self) -> dict:
        """Returns attributes mapped to CSV dataset columns."""
        return {
            'Age': self.age,
            'Fav genre': self.fav_genre,
            'Hours per day': self.hours_per_day,
            'While working': self.while_working,
            'Instrumentalist': self.instrumentalist,
            'Primary streaming service': self.streaming_service,
        }

    def display_summary(self) -> str:
        """Returns a multi-line summary string of the respondent's responses."""
        return (
            f"Age: {self.age}\n"
            f"Favorite Genre: {self.fav_genre}\n"
            f"Hours per Day: {self.hours_per_day}\n"
            f"While Working: {self.while_working}\n"
            f"Instrumentalist: {self.instrumentalist}\n"
            f"Streaming Service: {self.streaming_service}"
        )


class GenreProfile(Respondent):
    """Analyzes the cohort of users who share the respondent's
    favorite genre."""

    def __init__(self, respondent: Respondent, df: pd.DataFrame) -> None:
        """Calls super().__init__() with properties from respondent
        and extracts the cohort."""
        super().__init__(
            age=respondent.age,
            fav_genre=respondent.fav_genre,
            hours_per_day=respondent.hours_per_day,
            while_working=respondent.while_working,
            instrumentalist=respondent.instrumentalist,
            streaming_service=respondent.streaming_service,
        )
        self.df: pd.DataFrame = df
        self.cohort: pd.DataFrame = df[df['Fav genre'] == self.fav_genre]
        self.cohort_size: int = len(self.cohort)

    def get_cohort_stats(self) -> dict:
        """Computes means for each mental health metric within the cohort.
        Returns dict with metric means and cohort size."""
        metrics = ['Anxiety', 'Depression', 'Insomnia', 'OCD']

        if self.cohort_size == 0:
            return {m: 0.0 for m in metrics} | {'cohort_size': 0}

        stats = {}
        for m in metrics:
            stats[m] = float(self.cohort[m].mean())
        stats['cohort_size'] = self.cohort_size
        return stats

    def compare_to_cohort(self, predicted_scores: dict) -> dict:
        """Calculates delta = user_score - cohort_mean for each metric.
        Returns structured comparison per metric."""
        cohort_stats = self.get_cohort_stats()
        comparison = {}

        for metric in ['Anxiety', 'Depression', 'Insomnia', 'OCD']:
            user_score = predicted_scores.get(metric, 0.0)
            cohort_mean = cohort_stats.get(metric, 0.0)
            delta = user_score - cohort_mean

            if delta > 0:
                direction = 'above'
            elif delta < 0:
                direction = 'below'
            else:
                direction = 'equal to'

            comparison[metric] = {
                'user_score': round(user_score, 2),
                'cohort_mean': round(cohort_mean, 2),
                'delta': round(delta, 2),
                'direction': direction,
            }

        return comparison


class MentalHealthMetric:
    """Encapsulates a single mental-health score outcome with
    percentile ranking and severity classification."""

    def __init__(
        self, name: str, score: float, dataset_values: np.ndarray
    ) -> None:
        """Initializes values, calculates standard deviation/mean,
        and calls percentile and severity methods."""
        self.name: str = name
        self.score: float = score
        self.dataset_values: np.ndarray = dataset_values
        self.dataset_mean: float = float(np.mean(dataset_values))
        self.dataset_std: float = float(np.std(dataset_values))
        self.percentile: float = self.compute_percentile()
        self.severity: str = self.get_severity_label()

    def compute_percentile(self) -> float:
        """Computes the percentile rank of the user's score within
        the dataset distribution.

        Formula:
            percentile = (count of values <= score) / N_total * 100
        """
        if len(self.dataset_values) == 0:
            return 0.0
        return float(
            np.sum(self.dataset_values <= self.score)
            / len(self.dataset_values)
            * 100
        )

    def get_severity_label(self) -> str:
        """Returns severity label based on score thresholds.
        score <= 3.0 -> 'Low', score <= 6.0 -> 'Moderate', else -> 'High'."""
        if self.score <= 3.0:
            return 'Low'
        elif self.score <= 6.0:
            return 'Moderate'
        else:
            return 'High'

    def get_severity_color(self) -> str:
        """Returns hex color code for the severity level.
        Low -> #1DB954, Moderate -> #F39C12, High -> #E74C3C."""
        if self.severity == 'Low':
            return '#1DB954'
        elif self.severity == 'Moderate':
            return '#F39C12'
        else:
            return '#E74C3C'
