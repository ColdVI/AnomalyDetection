"""Independent ADS-B physics-consistency anomaly detection pipeline."""

from src.adsb_behavioral.physics_residuals import MODEL_FEATURES, add_physics_residuals

__all__ = ["MODEL_FEATURES", "add_physics_residuals"]
