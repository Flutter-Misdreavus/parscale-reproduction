"""Regression checks for observed worker bugs; no GPU or network access."""
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate
import cache_diagnostic_v2 as diagnostic


class ToyCausal(torch.nn.Module):
    def __init__(self, nonfinite=False):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.nonfinite = nonfinite

    def forward(self, input_ids, attention_mask=None, past_key_values=None,
                use_cache=True, labels=None):
        logits = torch.nn.functional.one_hot(input_ids, 8).float() * 2
        if self.nonfinite:
            logits.fill_(float("nan"))
        loss = None if labels is None else torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, 8), labels[:, 1:].reshape(-1))
        return SimpleNamespace(logits=logits, past_key_values=object(), loss=loss)


class Regression(unittest.TestCase):
    def test_utilization_telemetry_preserves_memory_gate(self):
        for free, expected in [(19055, True), (18431, False)]:
            with patch.object(evaluate.subprocess, "check_output", return_value=f"GPU-test, {free}, 95\n"):
                result = evaluate.gpu_admission("GPU-test", 18432, util_max=None)
                self.assertEqual(result["passed"], expected)
                self.assertEqual(result["samples"][0]["utilization_percent"], 95)
                self.assertTrue(result["requirement"]["utilization_record_only"])

    def test_single_failed_admission_does_not_poll(self):
        with patch.object(evaluate.subprocess, "check_output",
                          side_effect=["GPU-test, 19055, 12\n", "GPU-test, 19055, 0\n"]) as command:
            self.assertFalse(evaluate.gpu_admission("GPU-test", 18432)["passed"])
            self.assertEqual(command.call_count, 1)

    def test_resource_boundaries(self):
        for free, util, expected in [(18432, 10, True), (18431, 0, False), (19055, 11, False)]:
            with patch.object(evaluate.subprocess, "check_output", return_value=f"GPU-test, {free}, {util}\n"):
                self.assertEqual(evaluate.gpu_admission("GPU-test",18432)["passed"],expected)

    def test_prefill_one_and_many_concatenate(self):
        model = ToyCausal()
        ids = np.arange(64) % 8
        reference = diagnostic.full_logits(model, ids)
        for prefill in (1, 32):
            actual = diagnostic.cached_logits(model, ids, prefill)
            self.assertEqual(actual.shape, (64, 8))
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_scored_block_returns_record_and_ids(self):
        rec, ids = evaluate.scored_block(ToyCausal(), np.arange(5), "cpu")
        self.assertEqual(rec["token_count_scored"],4)
        self.assertEqual(tuple(ids.shape),(1,5))
        self.assertTrue(math.isfinite(rec["nll_sum_nats"]))
        self.assertLess(abs(rec["manual_minus_label"]),1e-6)

    def test_nan_stops_scoring(self):
        with self.assertRaisesRegex(RuntimeError,"non-finite"):
            evaluate.scored_block(ToyCausal(nonfinite=True), np.arange(5), "cpu")

    def test_paired_bootstrap_all_pairs_and_unequal_lengths(self):
        data = {"P1": [(30,10),(100,100)], "P2": [(29,10),(90,100)],
                "P4": [(28,10),(80,100)]}
        result = diagnostic.paired_bootstrap(data)
        self.assertEqual(set(result),{"P2-P1","P4-P1","P4-P2"})
        for pair in result.values():
            self.assertAlmostEqual(pair["ci95_high"],-.1 if pair is not result["P4-P1"] else -.2)
        broken = dict(data, P2=[(29,11),(90,100)])
        with self.assertRaises(ValueError):
            diagnostic.paired_bootstrap(broken)


if __name__ == "__main__":
    unittest.main(verbosity=2)
