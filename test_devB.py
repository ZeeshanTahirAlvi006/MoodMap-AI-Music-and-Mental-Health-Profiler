"""Integration test for all Developer B modules + data flow verification."""
from data_loader import DataLoader
from domain import Respondent, GenreProfile, MentalHealthMetric
from engine import AgeGroupAnalyzer, CorrelationEngine, predict_mental_health
from recommender import RecommendationEngine
from history import HistoryManager
from dashboard import (
    ReportGenerator,
    create_genre_bar_chart,
    create_age_line_chart,
    create_anxiety_histogram,
    create_scatter_plot,
    create_correlation_heatmap,
    create_trend_chart,
)
from pdf_export import PDFExporter
import os
import tempfile

# ── Load data ────────────────────────────────────────────────
print("=" * 60)
print("PHASE 1: Data Pipeline")
print("=" * 60)
dl = DataLoader("archive/mxmh_survey_results.csv")
df = dl.get_data()
print(f"  Loaded {len(df)} rows, {len(df.columns)} cols")
print()

# ── Respondent + Engine ──────────────────────────────────────
resp = Respondent(
    age=22, fav_genre="Rock", hours_per_day=4.0,
    while_working="Yes", instrumentalist="No",
    streaming_service="Spotify",
)
ce = CorrelationEngine(df)
predicted = predict_mental_health(resp, df, ce)
print(f"  Predicted scores: {predicted}")

# ── Domain objects ───────────────────────────────────────────
gp = GenreProfile(resp, df)
age_analyzer = AgeGroupAnalyzer(df)
metrics_objs = [
    MentalHealthMetric(name, score, df[name].values)
    for name, score in predicted.items()
]

# ══════════════════════════════════════════════════════════════
# DEVELOPER B TESTS
# ══════════════════════════════════════════════════════════════

# ── Recommender ──────────────────────────────────────────────
print("=" * 60)
print("PHASE 2: RecommendationEngine")
print("=" * 60)
rec_engine = RecommendationEngine(resp, predicted, df)

# Test plural method name (DEV-6 fix)
behaviour_recs = rec_engine.get_behaviour_recommendations()
print(f"  Behaviour recommendations ({len(behaviour_recs)}):")
for r in behaviour_recs:
    print(f"    - {r}")

# Test genre recommendation keys match pdf_export (DEV-7 fix)
genre_rec = rec_engine.get_genre_recommendation()
print(f"  Genre recommendation:")
assert 'suggested_genre' in genre_rec, "FAIL: missing 'suggested_genre' key"
assert 'reason' in genre_rec, "FAIL: missing 'reason' key"
print(f"    suggested_genre: {genre_rec['suggested_genre']}")
print(f"    reason: {genre_rec['reason']}")
print(f"    improvement: {genre_rec['improvement']}")
print()

# ── History ──────────────────────────────────────────────────
print("=" * 60)
print("PHASE 3: HistoryManager")
print("=" * 60)
# Use temp files for testing
tmp_hist = os.path.join("output", "_test_history.csv")
tmp_settings = os.path.join("output", "_test_settings.csv")
hm = HistoryManager(history_path=tmp_hist, settings_path=tmp_settings)

percentiles = {m.name: m.percentile for m in metrics_objs}
sid = hm.save_session(resp, predicted, percentiles)
print(f"  Saved session ID: {sid}")

# Test plural method name (DEV-8 fix)
sessions = hm.get_all_sessions()
print(f"  Total sessions: {len(sessions)}")
assert len(sessions) == 1, f"FAIL: expected 1 session, got {len(sessions)}"

trend_data = hm.get_trend_data()
print(f"  Trend data keys: {list(trend_data.keys())}")

hm.set_setting("api_key", "test_key_123")
assert hm.get_setting("api_key") == "test_key_123", "FAIL: setting not saved"
print(f"  Settings read/write: OK")

count = hm.get_session_count()
print(f"  Session count: {count}")

hm.delete_session(sid)
assert hm.get_session_count() == 0, "FAIL: delete_session didn't work"
print(f"  delete_session: OK")

# Cleanup temp files
os.remove(tmp_hist)
os.remove(tmp_settings)
print()

# ── Dashboard Charts ─────────────────────────────────────────
print("=" * 60)
print("PHASE 4: Dashboard Chart Functions")
print("=" * 60)

fig1 = create_genre_bar_chart(df)
print(f"  create_genre_bar_chart: OK ({type(fig1).__name__})")

fig2 = create_age_line_chart(df)
print(f"  create_age_line_chart: OK ({type(fig2).__name__})")

fig3 = create_anxiety_histogram(df, predicted['Anxiety'])
print(f"  create_anxiety_histogram: OK ({type(fig3).__name__})")

fig4 = create_scatter_plot(df, resp.hours_per_day, predicted['Depression'])
print(f"  create_scatter_plot: OK ({type(fig4).__name__})")

fig5 = create_correlation_heatmap(df)
print(f"  create_correlation_heatmap: OK ({type(fig5).__name__})")

fig6 = create_trend_chart(trend_data, 1)
assert fig6 is None, "FAIL: should return None for session_count < 2"
print(f"  create_trend_chart (count=1): None (correct)")
print()

# ── ReportGenerator ──────────────────────────────────────────
print("=" * 60)
print("PHASE 5: ReportGenerator")
print("=" * 60)
rg = ReportGenerator(resp, gp, metrics_objs, age_analyzer, ce, df)

report = rg.generate_report()
print(f"  generate_report keys: {list(report.keys())}")
assert 'respondent' in report, "FAIL: missing 'respondent'"
assert 'metrics' in report, "FAIL: missing 'metrics'"
assert 'predicted_scores' in report, "FAIL: missing 'predicted_scores'"
assert 'percentiles' in report, "FAIL: missing 'percentiles'"
assert 'cohort_stats' in report, "FAIL: missing 'cohort_stats'"
assert 'comparison' in report, "FAIL: missing 'comparison'"
print(f"  Report data validated: OK")

chart_paths = rg.export_visualizations(output_dir="output")
print(f"  export_visualizations: {len(chart_paths)} charts exported")
for p in chart_paths:
    assert os.path.exists(p), f"FAIL: chart not found: {p}"
    print(f"    {os.path.basename(p)}: OK")
print()

# ── Full Pipeline: PDF with real charts ──────────────────────
print("=" * 60)
print("PHASE 6: Full Pipeline — PDF with Charts")
print("=" * 60)
exporter = PDFExporter(
    report=report,
    chart_paths=chart_paths,
    recommendations=behaviour_recs,
    genre_rec=genre_rec,
    ai_insights={"insights": "AI test insights...", "recommendations": "Try calm playlists"},
    output_dir="output",
)
pdf_path = exporter.generate()
print(f"  PDF generated: {pdf_path}")
assert os.path.exists(pdf_path), "FAIL: PDF not created"
print()

# ── Data Flow Verification ───────────────────────────────────
print("=" * 60)
print("PHASE 7: Data Flow Verification")
print("=" * 60)

# Verify respondent.to_dict() keys match what ai_client expects
resp_dict = resp.to_dict()
expected_keys = ['Age', 'Fav genre', 'Hours per day', 'While working',
                 'Instrumentalist', 'Primary streaming service']
for k in expected_keys:
    assert k in resp_dict, f"FAIL: respondent.to_dict() missing key: {k}"
print(f"  respondent.to_dict() keys match ai_client prompt: OK")

# Verify cohort_stats keys match what ai_client expects
cohort = gp.get_cohort_stats()
for k in ['Anxiety', 'Depression', 'Insomnia', 'OCD', 'cohort_size']:
    assert k in cohort, f"FAIL: cohort_stats missing key: {k}"
print(f"  cohort_stats keys match ai_client prompt: OK")

# Verify genre_rec keys match what pdf_export expects
assert 'suggested_genre' in genre_rec, "FAIL: genre_rec missing 'suggested_genre'"
assert 'reason' in genre_rec, "FAIL: genre_rec missing 'reason'"
print(f"  genre_rec keys match pdf_export: OK")

# Verify metrics data flows from MentalHealthMetric → report → PDF
for metric in ['Anxiety', 'Depression', 'Insomnia', 'OCD']:
    assert metric in report['metrics'], f"FAIL: report missing metric {metric}"
    m = report['metrics'][metric]
    assert 'score' in m, f"FAIL: metric {metric} missing 'score'"
    assert 'percentile' in m, f"FAIL: metric {metric} missing 'percentile'"
    assert 'severity' in m, f"FAIL: metric {metric} missing 'severity'"
print(f"  metrics data flow (MentalHealthMetric -> report -> PDF): OK")

print()
print("=" * 60)
print("ALL TESTS PASSED — DATA FLOWS CORRECTLY")
print("=" * 60)
