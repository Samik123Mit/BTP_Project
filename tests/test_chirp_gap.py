"""Scientific guardrails for the reconstruction experiment."""
import unittest
import numpy as np

from src.chirp_gap import (
    METHODS, Profile, TrainingPrior, corrupt, forward, metrics, reconstruct,
)
from src.chirp_raster import damage_raster, extract_trace, PIXELS_PER_SAMPLE


class GapReconstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(51)
        t = np.arange(256)
        basis = np.stack([np.sin(.13*t), np.cos(.13*t), np.sin(.29*t)])
        cls.training = rng.normal(size=(90, 3)) @ basis
        cls.reference = np.array([.7, -.2, .4]) @ basis
        cls.prior = TrainingPrior(cls.training, rank=32)
        cls.profile = Profile("test", ((80, 160),))
        cls.observed, cls.mask = corrupt(cls.reference, cls.profile, 10)

    def test_erased_values_cannot_influence_predictions(self):
        poison = self.observed.copy()
        poison[~self.mask] = 1234567.
        for method in METHODS:
            if method == "zero_fill":
                continue
            with self.subTest(method=method):
                first = reconstruct(method, self.observed, self.mask, self.profile, self.prior)
                second = reconstruct(method, poison, self.mask, self.profile, self.prior)
                np.testing.assert_allclose(first, second, rtol=1e-9, atol=1e-9)

    def test_gap_only_preserves_measured_samples(self):
        for method in METHODS:
            with self.subTest(method=method):
                prediction = reconstruct(method, self.observed, self.mask, self.profile, self.prior)
                np.testing.assert_array_equal(prediction[self.mask], self.observed[self.mask])

    def test_low_rank_method_recovers_heldout_synthetic_waveform(self):
        prediction = reconstruct("pca_r32_l01", self.observed, self.mask, self.profile, self.prior)
        baseline = reconstruct("linear", self.observed, self.mask, self.profile, self.prior)
        self.assertLess(
            metrics(self.reference, prediction, self.mask)["gap_rmse"],
            .05 * metrics(self.reference, baseline, self.mask)["gap_rmse"],
        )

    def test_noise_is_repeatable_and_holes_are_zero(self):
        profile = Profile("mixed", ((80, 160),), .04, .8, 2, .8)
        first, mask = corrupt(self.reference, profile, 45)
        second, _ = corrupt(self.reference, profile, 45)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[~mask], 0)

    def test_known_corruption_is_linear_before_noise_and_mask(self):
        profile = Profile("mixed", ((80, 160),), .04, .8, 4, .8)
        a, b = self.training[:2]
        np.testing.assert_allclose(forward(2*a-b, profile), 2*forward(a, profile)-forward(b, profile), atol=1e-12)

    def test_missing_region_metrics_ignore_visible_region_improvements(self):
        prediction = self.reference.copy()
        prediction[~self.mask] += 2
        score = metrics(self.reference, prediction, self.mask)
        self.assertAlmostEqual(score["gap_rmse"], 2.)
        self.assertAlmostEqual(score["visible_rmse"], 0.)

    def test_no_observations_is_explicitly_rejected(self):
        with self.assertRaisesRegex(ValueError, "No observed"):
            reconstruct("pca_r32_l01", self.observed, np.zeros_like(self.mask), self.profile, self.prior)

    def test_raster_digitizer_reads_visible_pixels(self):
        image, mask = damage_raster(self.reference, self.profile, 5., 7)
        observed, valid = extract_trace(image, mask, 5.)
        np.testing.assert_array_equal(valid, mask)
        self.assertLess(np.sqrt(np.mean((observed[mask]-self.reference[mask])**2)), .05)
        # Changing all pixels in erased columns must have no effect on inference data.
        image[:, np.repeat(~mask, PIXELS_PER_SAMPLE)] = 0
        changed, changed_valid = extract_trace(image, mask, 5.)
        np.testing.assert_array_equal(observed, changed)
        np.testing.assert_array_equal(valid, changed_valid)


if __name__ == "__main__":
    unittest.main()
