"""
Unit tests for the performance benchmark suite.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from benchmark import BenchmarkRunner, load_benchmark_prompts, format_markdown_report


def test_load_benchmark_prompts():
    prompts_10 = load_benchmark_prompts(10)
    assert len(prompts_10) == 10
    assert all(isinstance(p, str) and len(p) > 5 for p in prompts_10)


def test_benchmark_runner_mock_execution():
    runner = BenchmarkRunner(mock_mode=True)
    report = runner.run(prompt_count=2, seed_start=100)

    assert report["total_prompts"] == 2
    assert report["success_count"] == 2
    assert report["success_rate_percent"] == 100.0
    assert report["oom_rate_percent"] == 0.0
    assert report["latency"]["median_seconds"] > 0
    assert len(report["detailed_results"]) == 2

    # Markdown formatter validation
    md = format_markdown_report(report)
    assert "Velaris Motion" in md
    assert "Summary Metrics" in md
    assert "100.0%" in md
