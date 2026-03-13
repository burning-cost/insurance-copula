"""
Smoke tests for package imports and public API surface.
"""

from __future__ import annotations


class TestPublicImports:

    def test_import_package(self):
        import insurance_copula.vine
        assert hasattr(insurance_copula.vine, "__version__")

    def test_version_string(self):
        from insurance_copula.vine import __version__
        assert isinstance(__version__, str)
        parts = __version__.split(".")
        assert len(parts) == 3

    def test_import_panel_dataset(self):
        from insurance_copula.vine import PanelDataset
        assert PanelDataset is not None

    def test_import_occurrence_marginal(self):
        from insurance_copula.vine import OccurrenceMarginal
        assert OccurrenceMarginal is not None

    def test_import_severity_marginal(self):
        from insurance_copula.vine import SeverityMarginal
        assert SeverityMarginal is not None

    def test_import_two_part_dvine(self):
        from insurance_copula.vine import TwoPartDVine
        assert TwoPartDVine is not None

    def test_import_predict_functions(self):
        from insurance_copula.vine import (
            predict_claim_prob,
            predict_severity_quantile,
            predict_premium,
        )
        assert callable(predict_claim_prob)
        assert callable(predict_severity_quantile)
        assert callable(predict_premium)

    def test_import_relativity_functions(self):
        from insurance_copula.vine import (
            extract_relativity_curve,
            compare_to_ncd,
        )
        assert callable(extract_relativity_curve)
        assert callable(compare_to_ncd)

    def test_all_exports(self):
        import insurance_copula.vine as ivl
        expected = [
            "PanelDataset",
            "OccurrenceMarginal",
            "SeverityMarginal",
            "TwoPartDVine",
            "predict_claim_prob",
            "predict_severity_quantile",
            "predict_premium",
            "extract_relativity_curve",
            "compare_to_ncd",
        ]
        for name in expected:
            assert hasattr(ivl, name), f"Missing export: {name}"

    def test_dvine_fit_result_importable(self):
        from insurance_copula.vine._dvine import DVineFitResult
        assert DVineFitResult is not None

    def test_stationary_dvine_importable(self):
        from insurance_copula.vine._dvine import StationaryDVine
        assert StationaryDVine is not None

    def test_plot_module_importable(self):
        from insurance_copula.vine import _plot
        assert hasattr(_plot, "plot_tau_by_lag")
        assert hasattr(_plot, "plot_experience_surface")
        assert hasattr(_plot, "plot_pit_diagnostics")
        assert hasattr(_plot, "plot_bic_by_truncation")
