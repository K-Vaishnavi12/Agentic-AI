"""ML subpackage — forecasting, anomaly detection, sentiment, legal classification, evaluation.

Public API: import what you need from the submodules. Heavy imports (torch,
transformers, statsmodels) are kept inside functions / submodules so importing
``src.ml`` itself stays cheap.
"""

__all__ = ["config", "registry"]
