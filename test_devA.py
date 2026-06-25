"""Quick smoke test for all Developer A modules."""
from data_loader import DataLoader
from domain import Respondent, GenreProfile, MentalHealthMetric
from engine import AgeGroupAnalyzer, CorrelationEngine, predict_mental_health
import numpy as np

# 1. DataLoader
print("=" * 60)
print("1. DataLoader")
print("=" * 60)
dl = DataLoader("archive/mxmh_survey_results.csv")
df = dl.get_data()
print(f"   Loaded: {len(df)} rows, {len(df.columns)} cols")
print(f"   Genres: {dl.get_unique_genres()}")
print(f"   Services: {dl.get_unique_services()}")
print(f"   BPM nulls after clean: {df['BPM'].isna().sum()}")
print()

# 2. Domain - Respondent
print("=" * 60)
print("2. Respondent")
print("=" * 60)
resp = Respondent(
    age=22, fav_genre="Rock", hours_per_day=4.0,
    while_working="Yes", instrumentalist="No",
    streaming_service="Spotify"
)
print(resp.display_summary())
print(f"   to_dict: {resp.to_dict()}")
print()

# 3. Domain - GenreProfile
print("=" * 60)
print("3. GenreProfile")
print("=" * 60)
gp = GenreProfile(resp, df)
print(f"   Cohort size: {gp.cohort_size}")
stats = gp.get_cohort_stats()
print(f"   Cohort stats: {stats}")
print()

# 4. Engine - AgeGroupAnalyzer
print("=" * 60)
print("4. AgeGroupAnalyzer")
print("=" * 60)
aga = AgeGroupAnalyzer(df)
print(f"   Bracket for age 22: {aga.assign_bracket(22)}")
print(f"   Boundaries: {aga.get_bracket_boundaries(22)}")
print(f"   Bracket stats: {aga.get_bracket_stats(22)}")
print()

# 5. Engine - CorrelationEngine
print("=" * 60)
print("5. CorrelationEngine")
print("=" * 60)
ce = CorrelationEngine(df)
print(f"   Hours correlations: {ce.correlations}")
print(f"   BPM correlations: {ce.bpm_correlations}")
print(f"   Strongest (hours): {ce.get_strongest()}")
print(f"   Strongest (BPM): {ce.get_strongest_bpm()}")
print()

# 6. Engine - predict_mental_health
print("=" * 60)
print("6. predict_mental_health")
print("=" * 60)
predicted = predict_mental_health(resp, df, ce)
print(f"   Predicted scores: {predicted}")
print()

# 7. Domain - MentalHealthMetric
print("=" * 60)
print("7. MentalHealthMetric")
print("=" * 60)
for metric_name, score in predicted.items():
    mhm = MentalHealthMetric(metric_name, score, df[metric_name].values)
    print(f"   {mhm.name}: score={mhm.score:.2f}, "
          f"percentile={mhm.percentile:.1f}%, "
          f"severity={mhm.severity} ({mhm.get_severity_color()}), "
          f"mean={mhm.dataset_mean:.2f}, std={mhm.dataset_std:.2f}")
print()

# 8. GenreProfile - compare_to_cohort
print("=" * 60)
print("8. compare_to_cohort")
print("=" * 60)
comparison = gp.compare_to_cohort(predicted)
for metric_name, comp in comparison.items():
    print(f"   {metric_name}: {comp}")
print()

# 9. PDF Export
print("=" * 60)
print("9. PDFExporter")
print("=" * 60)
from pdf_export import PDFExporter

metrics_payload = {}
for metric_name, score in predicted.items():
    mhm = MentalHealthMetric(metric_name, score, df[metric_name].values)
    metrics_payload[metric_name] = {
        'score': mhm.score,
        'percentile': mhm.percentile,
        'severity': mhm.severity,
    }

report = {
    'respondent': resp.to_dict(),
    'metrics': metrics_payload,
    'cohort_stats': stats,
    'comparison': comparison,
}

exporter = PDFExporter(
    report=report,
    chart_paths=[],
    recommendations=["Reduce listening hours before sleep", "Try ambient genres for focus"],
    genre_rec={"suggested_genre": "Classical", "reason": "Lower anxiety cohort average"},
    ai_insights={"insights": "Based on analysis...", "recommendations": "Try calm playlists"},
    output_dir="output",
)
path = exporter.generate()
print(f"   PDF generated at: {path}")
print()
print("ALL TESTS PASSED")
