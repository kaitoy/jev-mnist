"""Offline tests for the misclassification gallery."""

import numpy as np

from jev_mnist import errors
from jev_mnist.errors import Filters, Row, read_rows, render_html, select_rows, summary_html

ROWS = [
    Row(0, 7, 7, 0.99, ""),
    Row(1, 4, 9, 0.80, ""),
    Row(2, 4, 9, 0.95, ""),
    Row(3, 3, 5, 0.40, ""),
    Row(4, 2, None, None, "TypeSafeAPIError: boom"),
]


def indices(rows):
    return [r.index for r in rows]


def test_select_only_wrong_sorted_by_confidence():
    assert indices(select_rows(ROWS, Filters())) == [2, 1, 3]


def test_select_label_and_prediction():
    assert indices(select_rows(ROWS, Filters(label=4, prediction=9))) == [2, 1]
    assert indices(select_rows(ROWS, Filters(prediction=5))) == [3]


def test_select_confidence_range():
    assert indices(select_rows(ROWS, Filters(min_confidence=0.5, max_confidence=0.9))) == [1]


def test_select_api_errors_and_sort_and_limit():
    assert indices(select_rows(ROWS, Filters(include_errors=True, sort="index"))) == [1, 2, 3, 4]
    assert indices(select_rows(ROWS, Filters(include_errors=True))) == [2, 1, 3, 4]
    assert indices(select_rows(ROWS, Filters(limit=1))) == [2]


def test_render_html():
    page = render_html([(ROWS[1], b"png"), (ROWS[4], b"png")], "T", "all misclassified")
    assert page.count("<img") == 2
    assert '4 → <span class="pred">9</span>' in page
    assert "API error" in page


def test_summary_html():
    page = summary_html(ROWS)
    assert "5 (answered: 4, errors: 1)" in page
    assert "25.00% (1/4)" in page and "20.00% (1/5)" in page
    assert "<tr><th>4</th><td>0</td><td>0</td><td>0</td><td>0</td><td class=\"diag\">0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>2</td></tr>" in page


def test_main_end_to_end(tmp_path, monkeypatch):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "index,label,prediction,confidence,correct,error\n"
        "0,7,7,0.9900,True,\n"
        "1,4,9,0.8000,False,\n"
        "2,2,,,False,TypeSafeAPIError: boom\n"
    )
    assert indices(read_rows(csv_path)) == [0, 1, 2]
    monkeypatch.setattr(errors, "load_test_set", lambda data_dir: (np.zeros((3, 28, 28), dtype=np.uint8), np.array([7, 4, 2])))
    out = tmp_path / "errors.html"
    assert errors.main([str(csv_path), "--output", str(out)]) == 0
    page = out.read_text(encoding="utf-8")
    assert page.count("<img") == 1 and "#1" in page
