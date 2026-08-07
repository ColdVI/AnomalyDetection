"""Frozen configuration for the RflyMAD direct deep-learning experiment."""

from __future__ import annotations

FEATURE_COLUMNS = (
    "gps_speed_calc_mps",
    "gps_accel_mps2",
    "vertical_rate_calc",
    "local_alt_m",
    "local_vertical_rate_mps",
    "gps_speed_residual",
    "vertical_rate_residual",
    "roll_deg",
    "roll_rate",
    "pitch_deg",
    "pitch_rate",
    "yaw_rate",
    "roll_setpoint_error",
    "roll_rate_error",
    "pitch_setpoint_error",
    "pitch_rate_error",
    "yaw_rate_error",
    "attitude_error_mag",
    "actuator_roll_cmd",
    "actuator_pitch_cmd",
    "actuator_yaw_cmd",
    "actuator_thrust_cmd",
    "actuator_effort",
    "control_strain",
    "vel_test_ratio",
    "pos_test_ratio",
    "hgt_test_ratio",
    "mag_test_ratio",
    "actuator_output_std",
    "actuator_output_range",
    "pos_horiz_accuracy",
    "pos_vert_accuracy",
    "vibe_0",
    "vibe_1",
    "vibe_2",
)

MODEL_NAMES = ("lstm_ae", "dense_ae", "usad")
BUDGETS = {"critical": 2.0, "advisory": 12.0}
MIN_RECALL = {"critical": 0.30, "advisory": 0.50}

WINDOW = 50
WINDOW_STRIDE = 5
MAX_GAP_S = 2.0
DECISION_STRIDE_S = 1.0
SCALE_CLIP = 10.0

MAX_EPOCHS = 40
PATIENCE = 5
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
GRADIENT_CLIP = 1.0

CUSUM_BLOCK_SECONDS = 60.0
CUSUM_BOOTSTRAP_HOURS = 50.0
CUSUM_K = 0.5
CUSUM_REFRACTORY_SECONDS = 30.0

RHO_THRESHOLD = 0.8
