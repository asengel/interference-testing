"""Focused regression tests for CSV input and physically constrained fits.

Run after copying the update into the cloned repository:
    python -m unittest discover -s tests -p 'test_validation.py' -v
"""
import tempfile
import unittest
from pathlib import Path
import numpy as np

from interference.schema import canonicalize_inputs, validate_input_schema
from interference.validation import validate_history, validate_reservoir


class InputValidationTests(unittest.TestCase):
    def test_pressure_rejects_nan(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_history([1, 2, 3], [1, np.nan, 2], label="pressure")

    def test_unsorted_times(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            validate_history([1, 3, 2], [1, 2, 3], label="pressure")

    def test_duplicate_times(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            validate_history([1, 2, 2], [1, 2, 3], label="rates")

    def test_bad_reservoir(self):
        with self.assertRaisesRegex(ValueError, "positive|> 0"):
            validate_reservoir(0, 0.54, 1)

    def test_simple_inputs_no_rectangle_rows(self):
        values = {"model": "line_source", "thickness_ft": "607", "viscosity_cp": "0.54",
                  "formation_volume_factor_rb_stb": "1", "fit_loss": "linear"}
        validate_input_schema(values)
        self.assertEqual(canonicalize_inputs(values)["h_ft"], "607")

    def test_rectangle_percentages(self):
        values = {"model": "closed_rectangle", "thickness_ft": "23", "viscosity_cp": "0.8",
                  "formation_volume_factor_rb_stb": "1.12", "rectangle_width_height_ratio": "2",
                  "active_well_x_percent": "50", "active_well_y_percent": "75"}
        validate_input_schema(values)
        v = canonicalize_inputs(values)
        self.assertEqual(float(v["source_xD"]), 0.5)
        self.assertEqual(float(v["source_yD"]), 0.75)

    def test_auto_boundary_accepts_blank_distance(self):
        values = {"model": "line_source", "run_mode": "auto", "thickness_ft": "607", "viscosity_cp": "0.54",
                  "formation_volume_factor_rb_stb": "1", "boundary_type": "no_flow",
                  "boundary_distance_ft": "", "boundary_azimuth_deg": "0"}
        validate_input_schema(values)
        values["boundary_type"] = "constant_pressure"
        validate_input_schema(values)

    def test_manual_boundary_requires_distance(self):
        values = {"model": "line_source", "run_mode": "manual", "thickness_ft": "607", "viscosity_cp": "0.54",
                  "formation_volume_factor_rb_stb": "1", "boundary_type": "no_flow",
                  "permeability_md": "100", "storativity_psi_inv": "1e-6", "boundary_azimuth_deg": "0"}
        with self.assertRaisesRegex(ValueError, "boundary_distance_ft"):
            validate_input_schema(values)

    def test_auto_boundary_requires_azimuth(self):
        values = {"model": "line_source", "run_mode": "auto", "thickness_ft": "607", "viscosity_cp": "0.54",
                  "formation_volume_factor_rb_stb": "1", "boundary_type": "no_flow",
                  "boundary_distance_ft": ""}
        with self.assertRaisesRegex(ValueError, "boundary_azimuth_deg"):
            validate_input_schema(values)

    def test_boundary_rejected_for_rectangle(self):
        values = {"model": "closed_rectangle", "thickness_ft": "23", "viscosity_cp": "0.8",
                  "formation_volume_factor_rb_stb": "1.12", "rectangle_width_height_ratio": "2",
                  "active_well_x_percent": "50", "active_well_y_percent": "75",
                  "boundary_type": "no_flow", "boundary_distance_ft": "1000", "boundary_azimuth_deg": "0"}
        with self.assertRaisesRegex(ValueError, "only with model=line_source"):
            validate_input_schema(values)


if __name__ == "__main__":
    unittest.main()
