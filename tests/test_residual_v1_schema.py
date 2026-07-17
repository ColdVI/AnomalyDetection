import pytest

from residual_v1.ingest.alfa_channels import CHANNELS as ALFA_CHANNELS
from residual_v1.ingest.rfly_channels import CHANNELS as RFLY_CHANNELS
from residual_v1.schema import ChannelSpec


def test_channel_spec_validates_bounds_and_frequency():
    with pytest.raises(ValueError, match="valid_min"):
        ChannelSpec("bad", "topic", "u", 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="nominal_hz"):
        ChannelSpec("bad", "topic", "u", 0.0, 1.0, 0.0)


@pytest.mark.parametrize("channels", [ALFA_CHANNELS, RFLY_CHANNELS])
def test_channel_inventory_names_are_unique_and_valid(channels):
    names = [channel.name for channel in channels]
    assert len(names) == len(set(names))
    assert all(channel.valid_min < channel.valid_max for channel in channels)
    assert all(channel.nominal_hz > 0 for channel in channels)
    assert all(channel.role in {"response", "command", "context"} for channel in channels)


def test_battery_is_context_only():
    battery = next(channel for channel in RFLY_CHANNELS if channel.name == "battery_voltage")
    assert battery.role == "context"
