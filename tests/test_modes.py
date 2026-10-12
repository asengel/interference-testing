"""Regression tests for CSV manual/auto modes and independent cases.

Run from repository root:
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from interference.models import (ObservationWell, Reservoir, SourceWell,
                                 rate_levels_to_events, simulate_line_source)
from interference.schema import validate_input_schema

RUNNER = Path(__file__).resolve().parents[1] / "examples" / "run.py"
spec = importlib.util.spec_from_file_location("interference_csv_runner", RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ManualModeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "standalone"
        self.path.mkdir()
        (self.path / "wells.csv").write_text(
            "well,role,x_ft,y_ft,rw_ft\nA,active,0,0,0.3\nB,observation,1000,0,0.3\n")
        (self.path / "rates.csv").write_text(
            "well,time_hr,rate_stbd\nA,0,300\nA,12,0\n")
        self.rows = {
            "case_name": "Synthetic validation",
            "model": "line_source", "run_mode": "manual",
            "thickness_ft": "50", "viscosity_cp": "1.0",
            "formation_volume_factor_rb_stb": "1.0",
            "permeability_md": "100", "storativity_psi_inv": "1e-6",
            "simulation_end_hr": "48", "simulation_points": "80",
            "boundary_type": "none",
        }
        self.save_inputs()

    def save_inputs(self):
        with (self.path / "inputs.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(("parameter", "value"))
            writer.writerows(self.rows.items())

    def run_case(self):
        return runner.run_directory(self.path, show=False)

    def test_manual_no_pressures_generates_forward_prediction(self):
        png, txt, info = self.run_case()
        self.assertTrue(png.exists())
        self.assertTrue(txt.exists())
        self.assertIn("run_mode = manual", info)
        self.assertIn("RMSE = not available", info)
        with (self.path / "results" / "standalone_predicted.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertGreaterEqual(len(rows), 80)
        self.assertEqual(rows[0]["time_hr"], "0.0")
        self.assertTrue(np.isfinite([float(r["dp_psi"]) for r in rows]).all())

    def test_manual_observations_calculate_correct_rmse(self):
        src = SourceWell("A", 0, 0, rate_levels_to_events([0, 12], [300, 0]), 0.3)
        obs = ObservationWell("B", 1000, 0, 0.3)
        res = Reservoir(50, 1, 1, phi_ct_psi_inv=1e-6)
        times = np.array([2.0, 6.0, 12.0, 22.0, 35.0])
        dp = simulate_line_source(times, src, obs, 100, res)
        with (self.path / "pressure.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(("well", "time_hr", "dp_psi"))
            w.writerows(("B", t, p) for t, p in zip(times, dp))
        png, txt, info = self.run_case()
        self.assertIn("RMSE = 0.000000 psi", info)
        self.assertTrue(png.exists())

    def test_constant_pressure_and_no_flow_both_supported(self):
        predictions = {}
        for kind in ("none", "no_flow", "constant_pressure"):
            self.rows["boundary_type"] = kind
            if kind != "none":
                self.rows.update(boundary_distance_ft="2000", boundary_azimuth_deg="0")
            self.save_inputs()
            self.run_case()
            with (self.path / "results" / "standalone_predicted.csv").open() as f:
                rows = list(csv.DictReader(f))
            predictions[kind] = np.array([float(r["dp_psi"]) for r in rows])
        self.assertGreater(predictions["no_flow"].max(), predictions["none"].max())
        self.assertLess(predictions["constant_pressure"].max(), predictions["none"].max())

    def test_auto_recovers_single_boundary_distance(self):
        """Inverse fitting estimates an empty distance for both image signs."""
        src = SourceWell("A", 0, 0, rate_levels_to_events([0, 12], [300, 0]), 0.3)
        obs = ObservationWell("B", 1000, 0, 0.3)
        res = Reservoir(50, 1, 1, phi_ct_psi_inv=1e-6)
        times = np.r_[np.arange(1, 15), np.arange(16, 81, 3)].astype(float)
        for kind in ("no_flow", "constant_pressure"):
            with self.subTest(kind=kind):
                self.rows.update(run_mode="auto", boundary_type=kind,
                                 boundary_distance_ft="", boundary_azimuth_deg="0",
                                 simulation_end_hr="100")
                self.save_inputs()
                from interference.models import LinearBoundary
                dp = simulate_line_source(times, src, obs, 100, res,
                                          (LinearBoundary(1800, 0, 0, kind),))
                with (self.path / "pressure.csv").open("w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(("well", "time_hr", "dp_psi"))
                    writer.writerows(("B", t, v) for t, v in zip(times, dp))
                png, txt, summary = self.run_case()
                self.assertTrue(png.is_file())
                import re
                m = re.search(r"fitted boundary distance = ([0-9.]+) ft", summary)
                self.assertIsNotNone(m)
                self.assertAlmostEqual(float(m.group(1)), 1800, delta=2.0)
                self.assertIn("RMSE = 0.000000 psi", summary)


    def test_auto_requires_pressure_data(self):
        self.rows["run_mode"] = "auto"
        self.save_inputs()
        with self.assertRaisesRegex(ValueError, "requires pressure.csv"):
            self.run_case()

    def test_manual_requires_fixed_parameters(self):
        self.rows.pop("permeability_md")
        self.save_inputs()
        with self.assertRaisesRegex(ValueError, "permeability_md"):
            self.run_case()

    def test_manual_without_pressures_requires_end_time(self):
        self.rows.pop("simulation_end_hr")
        self.save_inputs()
        with self.assertRaisesRegex(ValueError, "simulation_end_hr"):
            self.run_case()


    def test_manual_dispatch_supports_all_advanced_models(self):
        # Isolate runner parameter/geometry wiring from expensive analytical kernels.
        cases = [
            ("closed_rectangle", {"rectangle_width_height_ratio": "2",
                "active_well_x_percent": "50", "active_well_y_percent": "50",
                "rectangle_height_ft": "3000"}, "simulate_closed_rectangle"),
            ("leaky", {"leakage_factor_ft": "4000"}, "simulate_leaky_line_source"),
            ("active_storage_skin", {"active_well_storage_cd": "100",
                "active_well_skin": "1"}, "simulate_active_storage_skin"),
            ("double_porosity", {"fracture_storage_fraction": "0.1",
                "interporosity_flow_parameter": "1e-4",
                "interporosity_model": "pss"}, "simulate_double_porosity"),
        ]
        for model, extra, function in cases:
            with self.subTest(model=model):
                self.rows["model"] = model
                self.rows.update(extra)
                self.save_inputs()
                inputs, mod, t, y, tr, qr, src, obs, res, b = runner._load_case(self.path)
                def fake_prediction(times, *args, **kwargs):
                    return np.full_like(np.asarray(times, float), 1.23)
                with patch.object(runner, function, side_effect=fake_prediction) as fake:
                    result, predict, k, pct, info = runner._manual_case(
                        mod, inputs, t, y, src, obs, res, b)
                    answer = predict(np.array([1.0, 2.0]))
                    self.assertTrue(np.all(answer == 1.23))
                    self.assertEqual(k, 100)
                    fake.assert_called_once()
                # Clear model-specific settings to avoid ambiguity in subsequent cases.
                for key in extra:
                    self.rows.pop(key)

    def test_modes_and_parameter_bounds(self):
        for invalid in ("forward", "fit", "AUTO_NOW"):
            self.rows["run_mode"] = invalid
            with self.assertRaisesRegex(ValueError, "manual.*auto"):
                validate_input_schema(self.rows)
        self.rows["run_mode"] = "manual"
        self.rows["permeability_md"] = "-5"
        with self.assertRaisesRegex(ValueError, "permeability_md"):
            validate_input_schema(self.rows)


if __name__ == "__main__":
    unittest.main()
