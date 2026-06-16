"""Input-validation guards fail fast with clear errors instead of NaNs/crashes."""

import pytest

import pdeforge as pf
from pdeforge.verify import grid_convergence_index


def test_gci_rejects_non_monotone_grids():
    # hs must be strictly increasing fine -> coarse.
    with pytest.raises(ValueError, match="strictly increasing"):
        grid_convergence_index([0.04, 0.02, 0.01], [5.0, 5.1, 5.2])


def test_gci_rejects_zero_successive_difference():
    with pytest.raises(ValueError, match="must differ"):
        grid_convergence_index([0.01, 0.02, 0.04], [5.0, 5.0, 5.1])


def test_study_requires_at_least_three_grids():
    with pytest.raises(ValueError, match="at least 3"):
        pf.run_convergence_study(pf.heat_1d(), resolutions=(17, 33))


def test_study_rejects_non_constant_refinement_ratio():
    # Nested but ratios differ (2x then 4x) -> per-cell order would be wrong.
    with pytest.raises(ValueError, match="constant refinement ratio"):
        pf.run_convergence_study(pf.heat_1d(), resolutions=(17, 33, 129))


def test_study_rejects_non_nested_grids():
    # 40 is not a nested refinement of 17 -> cannot co-locate fields.
    with pytest.raises(ValueError):
        pf.run_convergence_study(pf.heat_1d(), resolutions=(17, 40, 65))
