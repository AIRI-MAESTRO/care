"""Cyrillic-safe ROUGE helpers (mirrors Platform validate.py)."""

from __future__ import annotations

from statistics import mean


def _lcs_length(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    for i in range(m):
        cur = [0] * (n + 1)
        for j in range(n):
            if a[i] == b[j]:
                cur[j + 1] = prev[j] + 1
            else:
                cur[j + 1] = max(cur[j + 1], prev[j + 1], cur[j])
        prev = cur
    return prev[n]


def _f1_from_overlap(overlap: int, pred_len: int, ref_len: int) -> float:
    if overlap <= 0 or pred_len <= 0 or ref_len <= 0:
        return 0.0
    precision = overlap / pred_len
    recall = overlap / ref_len
    return (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0


def word_rouge_pair(pred: str, ref: str, metric: str = "ROUGE-L") -> float:
    pred_t = str(pred).split()
    ref_t = str(ref).split()
    if metric == "ROUGE-L":
        lcs = _lcs_length(pred_t, ref_t)
        return _f1_from_overlap(lcs, len(pred_t), len(ref_t))
    if metric == "ROUGE-2":
        pred_bi = list(zip(pred_t, pred_t[1:]))
        ref_bi = list(zip(ref_t, ref_t[1:]))
        if not pred_bi or not ref_bi:
            return 0.0
        overlap = sum(1 for bg in pred_bi if bg in set(ref_bi))
        return _f1_from_overlap(overlap, len(pred_bi), len(ref_bi))
    if not pred_t or not ref_t:
        return 0.0
    ref_set = set(ref_t)
    overlap = sum(1 for tok in pred_t if tok in ref_set)
    return _f1_from_overlap(overlap, len(pred_t), len(ref_t))


def word_rouge_corpus(
    predictions: list[str], references: list[str], metric: str = "ROUGE-L",
) -> float:
    scores = [word_rouge_pair(p, r, metric) for p, r in zip(predictions, references)]
    return float(mean(scores)) if scores else 0.0


class TestWordRouge:
    def test_cyrillic_exact_match_is_one(self) -> None:
        ref = "Таяние ледников из-за изменения климата повышает уровень моря."
        score = word_rouge_pair(ref, ref, "ROUGE-L")
        assert score >= 0.99

    def test_cyrillic_paraphrase_is_nonzero(self) -> None:
        ref = (
            "Таяние ледников из-за изменения климата повышает уровень моря "
            "и угрожает прибрежным городам."
        )
        pred = (
            "Климатические изменения ускоряют таяние ледников, повышая "
            "уровень моря и угрожая прибрежным городам."
        )
        score = word_rouge_pair(pred, ref, "ROUGE-L")
        assert score >= 0.45

    def test_latin_english_mismatch_to_russian_is_zero(self) -> None:
        ref = "Таяние ледников из-за изменения климата."
        pred = "Accelerated glacier melting due to climate change."
        score = word_rouge_pair(pred, ref, "ROUGE-L")
        assert score == 0.0
