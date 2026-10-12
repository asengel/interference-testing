"""Regression coverage for the four-panel CSV output (no interactive display required)."""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from interference.models import (ObservationWell, Reservoir, SourceWell,
                                 rate_levels_to_events, simulate_line_source)

RUNNER = Path(__file__).resolve().parents[1] / "examples" / "run.py"
spec = importlib.util.spec_from_file_location("four_panel_case_runner", RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class FourPanelPlotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.case = Path(self.temp.name) / "independent_pulse"
        self.case.mkdir()
        (self.case / "wells.csv").write_text(
            "well,role,x_ft,y_ft,rw_ft\nA,active,0,0,0.3\nB,observation,1000,0,0.3\n")
        (self.case / "rates.csv").write_text(
            "well,time_hr,rate_stbd\nA,0,500\nA,12,0\n")
        self.k, self.pct = 100., 1e-6
        self.times = np.array([1., 2., 5., 12., 14., 20., 35., 48.])
        source = SourceWell("A", 0, 0, rate_levels_to_events([0, 12], [500, 0]), .3)
        obs = ObservationWell("B", 1000, 0, .3)
        self.measured = simulate_line_source(
            self.times, source, obs, self.k,
            Reservoir(50, 1, 1, phi_ct_psi_inv=self.pct))
        with (self.case / "pressure.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(("well", "time_hr", "dp_psi"))
            writer.writerows(("B", t, dp) for t, dp in zip(self.times, self.measured))
        self.settings = {
            "case_name": "Plot test", "model": "line_source", "run_mode": "manual",
            "thickness_ft": "50", "viscosity_cp": "1", "formation_volume_factor_rb_stb": "1",
            "permeability_md": "100", "storativity_psi_inv": "1e-6",
            "simulation_end_hr": "50", "boundary_type": "none",
        }
        self.save_inputs()

    def save_inputs(self):
        with (self.case / "inputs.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(("parameter", "value"))
            w.writerows(self.settings.items())

    def plot(self):
        inp, m, t, y, tr, qr, src, obs, res, bounds = runner._load_case(self.case)
        fit, pred, k, pct, summary = runner._manual_case(
            m, inp, t, y, src, obs, res, bounds)
        tdense = runner._simulation_times(inp, t, tr)
        fig = runner._make_plot("Plot test", t, y, tr, qr, src, obs,
                                res, k, pct, pred, fit,
                                self.case / "four_panel.png", mode="manual", tdense=tdense)
        self.addCleanup(plt.close, fig)
        return fig

    def test_four_panels_for_pulse_with_dimensionless_observations(self):
        fig = self.plot()
        self.assertTrue((self.case / "four_panel.png").exists())
        self.assertEqual(len(fig.axes), 4)
        pressure, type_curve, residual, rates = fig.axes
        self.assertEqual(pressure.get_xscale(), "log")
        self.assertEqual(type_curve.get_xscale(), "log")
        self.assertEqual(type_curve.get_yscale(), "log")
        self.assertIn("$t_D/r_D^2$", type_curve.get_xlabel())
        self.assertTrue(any("Theis" in x.get_label() for x in type_curve.lines))
        self.assertTrue(any("Pulse / variable rate" in x.get_text() for x in type_curve.texts))
        # The plotted observed dimensionless coordinates must be based on
        # 500 STB/day (initial rate), NOT the zero rate after shut-in.
        scaled_points = type_curve.collections[0].get_offsets()
        expected_u = .0002637 * self.k * self.times / (self.pct * 1000**2)
        expected_pD = self.measured * self.k * 50 / (141.2 * 500)
        np.testing.assert_allclose(scaled_points[:, 0], expected_u, rtol=1e-8)
        np.testing.assert_allclose(scaled_points[:, 1], expected_pD, rtol=1e-8)
        np.testing.assert_allclose(residual.collections[0].get_offsets()[:, 1],
                                   np.zeros(self.times.size), atol=1e-10)
        step = rates.lines[0]
        np.testing.assert_allclose(step.get_ydata(), [500, 0, 0])

    def test_forward_mode_without_observations_still_shows_type_and_rate(self):
        (self.case / "pressure.csv").unlink()
        fig = self.plot()
        self.assertEqual(len(fig.axes), 4)
        self.assertEqual(fig.axes[1].get_xscale(), "log")
        self.assertEqual(fig.axes[1].get_yscale(), "log")
        self.assertTrue(any("No measured" in label.get_text() for label in fig.axes[2].texts))

    def test_negative_injection_rates_normalize_to_positive_type_curve(self):
        (self.case / "rates.csv").write_text(
            "well,time_hr,rate_stbd\nA,0,-500\nA,12,0\n")
        source = SourceWell("A", 0, 0, rate_levels_to_events([0, 12], [-500, 0]), .3)
        obs = ObservationWell("B", 1000, 0, .3)
        values = simulate_line_source(self.times, source, obs, self.k,
            Reservoir(50, 1, 1, phi_ct_psi_inv=self.pct))
        with (self.case / "pressure.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(("well", "time_hr", "dp_psi"))
            writer.writerows(("B", t, dp) for t, dp in zip(self.times, values))
        fig = self.plot()
        points = fig.axes[1].collections[0].get_offsets()
        self.assertTrue(np.all(points[:, 1] > 0))


if __name__ == "__main__":
    unittest.main()
