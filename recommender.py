import pandas as pd

METRICS = ['Anxiety', 'Depression', 'Insomnia', 'OCD']


class RecommendationEngine:
    """Compiles static behavioral suggestions and suggests
    alternative musical genres."""

    def __init__(
        self, respondent, predicted_scores: dict, df: pd.DataFrame,
    ) -> None:
        """Stores references and extracts user favorite genre cohort."""
        self.respondent = respondent
        self.predicted_scores = predicted_scores
        self.df = df
        self.genre_cohort = df[df['Fav genre'] == respondent.fav_genre]

    def get_behaviour_recommendations(self) -> list[str]:
        """Generates behavioral recommendations based on listening hours,
        work habits, and instrument playing status."""
        recommendations = []

        # Hours Check
        if not self.genre_cohort.empty:
            cohort_mean_hours = self.genre_cohort['Hours per day'].mean()
            if self.respondent.hours_per_day > cohort_mean_hours + 1.0:
                hours_diff = self.respondent.hours_per_day - cohort_mean_hours
                for metric in METRICS:
                    metric_col = self.genre_cohort[metric]
                    if len(metric_col) > 0:
                        std = metric_col.std()
                        reduction = hours_diff * (
                            std / max(cohort_mean_hours, 1.0)
                        )
                        if reduction >= 0.3:
                            recommendations.append(
                                f"Consider reducing your daily listening "
                                f"time. You listen {hours_diff:.1f} hrs "
                                f"above your genre's average "
                                f"({cohort_mean_hours:.1f} hrs). Reducing "
                                f"could lower your {metric} score by "
                                f"~{reduction:.1f} points."
                            )
                            break

        # Work Check
        if self.respondent.while_working == 'Yes':
            highest_metric = max(
                METRICS, key=lambda m: self.predicted_scores.get(m, 0.0),
            )
            yes_cohort = self.df[self.df['While working'] == 'Yes']
            no_cohort = self.df[self.df['While working'] == 'No']
            if not yes_cohort.empty and not no_cohort.empty:
                yes_mean = yes_cohort[highest_metric].mean()
                no_mean = no_cohort[highest_metric].mean()
                diff = yes_mean - no_mean
                if diff >= 0.5:
                    recommendations.append(
                        f"Consider not listening to music while working. "
                        f"People who don't listen while working score "
                        f"{diff:.1f} points lower on {highest_metric} "
                        f"on average."
                    )

        # Instrument Check
        if self.respondent.instrumentalist == 'No':
            yes_inst = self.df[self.df['Instrumentalist'] == 'Yes']
            no_inst = self.df[self.df['Instrumentalist'] == 'No']
            if not yes_inst.empty and not no_inst.empty:
                improved_count = 0
                for metric in METRICS:
                    yes_mean = yes_inst[metric].mean()
                    no_mean = no_inst[metric].mean()
                    if no_mean - yes_mean >= 0.4:
                        improved_count += 1
                if improved_count >= 2:
                    recommendations.append(
                        "Consider learning a musical instrument. "
                        "Instrumentalists score notably lower on "
                        "multiple mental health metrics."
                    )

        return recommendations

    def get_genre_recommendation(self) -> dict:
        """Computes the average composite mental health score for all
        genres, identifies the best genre, and returns a comparison
        with the user's current genre.

        Returns dict with keys matching pdf_export.py expectations:
        'suggested_genre', 'reason', plus detailed scores."""
        genre_composites = {}
        for genre in self.df['Fav genre'].dropna().unique():
            genre_df = self.df[self.df['Fav genre'] == genre]
            composite = genre_df[METRICS].mean().mean()
            genre_composites[genre] = composite

        best_genre = min(genre_composites, key=genre_composites.get)
        best_score = genre_composites.get(best_genre)

        user_genre = self.respondent.fav_genre
        user_score = genre_composites.get(user_genre, 0.0)
        improvement = round(user_score - best_score, 2)

        return {
            'suggested_genre': best_genre,
            'reason': (
                f"Listeners of {best_genre} have a lower average composite "
                f"mental health score ({best_score:.2f}) compared to your "
                f"genre {user_genre} ({user_score:.2f}). "
                f"Potential improvement: {improvement:.2f} points."
            ),
            'recommended_score': best_score,
            'current_score': user_score,
            'current_genre': user_genre,
            'improvement': improvement,
        }
