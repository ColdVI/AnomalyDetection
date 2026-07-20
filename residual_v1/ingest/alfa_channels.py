"""Pre-registered ALFA channel inventory.

Names are the canonical columns written by :mod:`residual_v1.ingest.alfa`;
topics mirror the processed ALFA CSV suffixes.
"""

from residual_v1.schema import ChannelSpec


CHANNELS: tuple[ChannelSpec, ...] = (
    ChannelSpec("roll_cmd", "mavros-nav_info-roll", "rad", -3.141593, 3.141593, 25.0, True, "command"),
    ChannelSpec("roll", "mavros-nav_info-roll", "rad", -3.141593, 3.141593, 25.0, True, "response"),
    ChannelSpec("pitch_cmd", "mavros-nav_info-pitch", "rad", -1.570797, 1.570797, 25.0, True, "command"),
    ChannelSpec("pitch", "mavros-nav_info-pitch", "rad", -1.570797, 1.570797, 25.0, True, "response"),
    ChannelSpec("yaw_cmd", "mavros-nav_info-yaw", "rad", -3.141593, 3.141593, 25.0, True, "command"),
    ChannelSpec("yaw", "mavros-nav_info-yaw", "rad", -3.141593, 3.141593, 25.0, True, "response"),
    ChannelSpec("airspeed_cmd", "mavros-nav_info-airspeed", "m_s", 0.0, 60.0, 25.0, False, "command"),
    ChannelSpec("airspeed", "mavros-nav_info-airspeed", "m_s", 0.0, 60.0, 25.0, False, "response"),
    ChannelSpec("roll_rate", "mavros-imu-data", "rad_s", -20.0, 20.0, 50.0, False, "response"),
    ChannelSpec("pitch_rate", "mavros-imu-data", "rad_s", -20.0, 20.0, 50.0, False, "response"),
    ChannelSpec("yaw_rate", "mavros-imu-data", "rad_s", -20.0, 20.0, 50.0, False, "response"),
    ChannelSpec("accel_x", "mavros-imu-data", "m_s2", -100.0, 100.0, 50.0, False, "context"),
    ChannelSpec("accel_y", "mavros-imu-data", "m_s2", -100.0, 100.0, 50.0, False, "context"),
    ChannelSpec("accel_z", "mavros-imu-data", "m_s2", -100.0, 100.0, 50.0, False, "context"),
    ChannelSpec("quat_x", "mavros-imu-data", "unitless", -1.000001, 1.000001, 50.0, False, "context"),
    ChannelSpec("quat_y", "mavros-imu-data", "unitless", -1.000001, 1.000001, 50.0, False, "context"),
    ChannelSpec("quat_z", "mavros-imu-data", "unitless", -1.000001, 1.000001, 50.0, False, "context"),
    ChannelSpec("quat_w", "mavros-imu-data", "unitless", -1.000001, 1.000001, 50.0, False, "context"),
    ChannelSpec("aileron_cmd", "mavros-rc-out", "pwm_delta", -1000.0, 1000.0, 20.0, False, "command"),
    ChannelSpec("elevator_cmd", "mavros-rc-out", "pwm_delta", -1000.0, 1000.0, 20.0, False, "command"),
    ChannelSpec("throttle_pwm", "mavros-rc-out", "pwm", 800.0, 2200.0, 20.0, False, "command"),
    ChannelSpec("rudder_cmd", "mavros-rc-out", "pwm_delta", -1000.0, 1000.0, 20.0, False, "command"),
    ChannelSpec("throttle_cmd", "mavros-vfr_hud", "ratio", 0.0, 1.0, 5.0, False, "command"),
    ChannelSpec("ground_speed", "mavros-vfr_hud", "m_s", 0.0, 80.0, 5.0, False, "context"),
    ChannelSpec("climb_rate", "mavros-vfr_hud", "m_s", -30.0, 30.0, 5.0, False, "response"),
    ChannelSpec("altitude", "mavros-global_position-global", "m", -500.0, 10000.0, 5.0, False, "context"),
    ChannelSpec("latitude", "mavros-global_position-global", "deg", -90.0, 90.0, 5.0, False, "context"),
    ChannelSpec("longitude", "mavros-global_position-global", "deg", -180.0, 180.0, 5.0, True, "context"),
    ChannelSpec("local_vx", "mavros-local_position-velocity", "m_s", -100.0, 100.0, 10.0, False, "response"),
    ChannelSpec("local_vy", "mavros-local_position-velocity", "m_s", -100.0, 100.0, 10.0, False, "response"),
    ChannelSpec("local_vz", "mavros-local_position-velocity", "m_s", -100.0, 100.0, 10.0, False, "response"),
    ChannelSpec("xtrack_error", "mavros-nav_info-errors", "m", -5000.0, 5000.0, 10.0, False, "response"),
    ChannelSpec("waypoint_distance", "mavros-nav_info-errors", "m", 0.0, 5000.0, 10.0, False, "context"),
    ChannelSpec("path_dev_x", "mavctrl-path_dev", "m", -5000.0, 5000.0, 10.0, False, "response"),
    ChannelSpec("path_dev_y", "mavctrl-path_dev", "m", -5000.0, 5000.0, 10.0, False, "response"),
    ChannelSpec("path_dev_z", "mavctrl-path_dev", "m", -5000.0, 5000.0, 10.0, False, "response"),
)
