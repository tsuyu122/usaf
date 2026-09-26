"""Tests for the evaluation report helpers.

compare_reports and save_report are the only parts of the eval package with no
coverage, and both are reachable from the CLI: eval_cli writes its report
through save_report and --compare feeds the file back into compare_reports.
"""
import json

from usaf.eval.benchmark import BenchmarkConfig, BenchmarkResults
from usaf.eval.report import compare_reports, save_report


def _res(results, name="tiny-moe"):
    return BenchmarkResults(config=BenchmarkConfig(), model_name=name, results=results)


def test_compare_reports_computes_the_delta():
    before = _res({"d": {"perplexity": 122.52, "loss": 4.8089, "tokens": 65}})
    after = _res({"d": {"perplexity": 118.10, "loss": 4.7712, "tokens": 65}})
    c = compare_reports(before, after)
    assert c["model_before"] == "tiny-moe"
    assert c["model_after"] == "tiny-moe"
    assert c["datasets"]["d"]["ppl_change"] == -4.42
    assert abs(c["datasets"]["d"]["loss_change"] - (-0.0377)) < 1e-4


def test_compare_reports_unions_the_dataset_lists():
    """A dataset present in only one report must not be dropped silently."""
    before = _res({"a": {"perplexity": 10.0, "loss": 2.0}})
    after = _res({"b": {"perplexity": 20.0, "loss": 3.0}})
    c = compare_reports(before, after)
    assert sorted(c["datasets"]) == ["a", "b"]
    # No counterpart means no delta to report, not a zero.
    assert c["datasets"]["a"]["ppl_change"] is None
    assert c["datasets"]["b"]["ppl_change"] is None


def test_save_report_round_trips(tmp_path):
    r = _res({"d": {"perplexity": 118.10, "loss": 4.7712, "tokens": 65}})
    p = save_report(r, tmp_path / "nested" / "r.json")
    back = json.loads((tmp_path / "nested" / "r.json").read_text(encoding="utf-8"))
    assert back["model"] == "tiny-moe"
    assert back["results"]["d"]["perplexity"] == 118.10
    assert "config" in back and "timestamp" in back and "elapsed_s" in back
    assert p.endswith("r.json")


def test_save_report_writes_compact_json_when_asked(tmp_path):
    r = _res({"d": {"perplexity": 1.0, "loss": 0.0}})
    p = tmp_path / "c.json"
    save_report(r, p, pretty=False)
    text = p.read_text(encoding="utf-8")
    assert "\n" not in text
    assert json.loads(text)["model"] == "tiny-moe"
