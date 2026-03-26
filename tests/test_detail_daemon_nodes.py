"""Tests for modules/detail_daemon_nodes.py — Detailer Daemon node and schedule helpers."""
import unittest

# numpy may be mocked in the test harness when not installed
try:
    import numpy as np
    _HAS_NUMPY = not hasattr(np, '_mock_name')  # real numpy won't have this
except (ImportError, AttributeError):
    _HAS_NUMPY = False

from modules.detail_daemon_nodes import (
    UmeAiRT_Detailer_Daemon,
    make_detail_daemon_schedule,
)


@unittest.skipUnless(_HAS_NUMPY, "numpy not available (mocked in test harness)")
class TestMakeDetailDaemonSchedule(unittest.TestCase):
    """Unit tests for the pure schedule function (no ComfyUI mocks needed)."""

    def test_returns_array_of_correct_length(self):
        result = make_detail_daemon_schedule(
            steps=20, start=0.2, end=0.8, bias=0.5, amount=1.0,
            exponent=1.0, start_offset=0.0, end_offset=0.0, fade=0.0, smooth=True
        )
        self.assertEqual(len(result), 20)

    def test_zero_amount_gives_zeros(self):
        result = make_detail_daemon_schedule(
            steps=10, start=0.0, end=1.0, bias=0.5, amount=0.0,
            exponent=1.0, start_offset=0.0, end_offset=0.0, fade=0.0, smooth=False
        )
        np.testing.assert_array_almost_equal(result, np.zeros(10))

    def test_full_fade_gives_zeros(self):
        result = make_detail_daemon_schedule(
            steps=15, start=0.0, end=1.0, bias=0.5, amount=1.0,
            exponent=1.0, start_offset=0.0, end_offset=0.0, fade=1.0, smooth=False
        )
        np.testing.assert_array_almost_equal(result, np.zeros(15))

    def test_schedule_values_within_range(self):
        result = make_detail_daemon_schedule(
            steps=30, start=0.1, end=0.9, bias=0.5, amount=2.0,
            exponent=1.0, start_offset=0.0, end_offset=0.0, fade=0.0, smooth=True
        )
        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result <= 2.0 + 1e-9))

    def test_single_step(self):
        """Edge case: 1 step should not crash."""
        result = make_detail_daemon_schedule(
            steps=1, start=0.0, end=1.0, bias=0.5, amount=1.0,
            exponent=1.0, start_offset=0.0, end_offset=0.0, fade=0.0, smooth=False
        )
        self.assertEqual(len(result), 1)

    def test_smooth_vs_linear_different(self):
        smooth = make_detail_daemon_schedule(
            steps=20, start=0.2, end=0.8, bias=0.5, amount=1.0,
            exponent=1.0, start_offset=0.0, end_offset=0.0, fade=0.0, smooth=True
        )
        linear = make_detail_daemon_schedule(
            steps=20, start=0.2, end=0.8, bias=0.5, amount=1.0,
            exponent=1.0, start_offset=0.0, end_offset=0.0, fade=0.0, smooth=False
        )
        # They should differ (smooth applies cosine interpolation)
        self.assertFalse(np.array_equal(smooth, linear))


class TestDetailerDaemon(unittest.TestCase):
    def test_input_types_required(self):
        inputs = UmeAiRT_Detailer_Daemon.INPUT_TYPES()
        req = inputs["required"]
        self.assertIn("gen_pipe", req)
        self.assertIn("enabled", req)
        self.assertIn("detail_amount", req)

    def test_input_types_has_schedule_params(self):
        inputs = UmeAiRT_Detailer_Daemon.INPUT_TYPES()
        req = inputs["required"]
        for key in ["start", "end", "bias", "exponent", "smooth", "fade"]:
            self.assertIn(key, req, f"Missing '{key}' in required inputs")

    def test_input_types_has_optional_overrides(self):
        inputs = UmeAiRT_Detailer_Daemon.INPUT_TYPES()
        opt = inputs.get("optional", {})
        self.assertIn("steps", opt)
        self.assertIn("cfg", opt)
        self.assertIn("seed", opt)

    def test_return_types(self):
        self.assertEqual(UmeAiRT_Detailer_Daemon.RETURN_TYPES, ("UME_PIPELINE",))

    def test_function_name(self):
        self.assertEqual(UmeAiRT_Detailer_Daemon.FUNCTION, "process")

    def test_category(self):
        self.assertEqual(UmeAiRT_Detailer_Daemon.CATEGORY, "UmeAiRT/Pipeline/Post-Processing")


if __name__ == "__main__":
    unittest.main()
