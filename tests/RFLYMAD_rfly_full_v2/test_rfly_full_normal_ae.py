from __future__ import annotations

import pandas as pd
import torch

from gecmis_calismalar.rfly_full.normal_ae import TemporalConvAutoencoder, _normal_split


def test_temporal_autoencoder_preserves_window_and_value_shape():
    model = TemporalConvAutoencoder(channels_in=20, channels_out=10)
    assert model(torch.zeros(3, 100, 20)).shape == (3, 100, 10)


def test_normal_split_rotates_whole_groups_without_faults():
    rows = []
    for domain in ("Real", "HIL", "SIL"):
        for group in ("a", "b", "c"):
            for index in range(2):
                rows.append({
                    "canonical_case_id": f"{domain}_{group}_{index}",
                    "domain": domain,
                    "split": "development",
                    "evaluation_role": "normal_reference",
                    "split_group_id": f"{domain}:{group}",
                })
        rows.append({
            "canonical_case_id": f"{domain}_fault",
            "domain": domain,
            "split": "development",
            "evaluation_role": "fault_detection",
            "split_group_id": f"{domain}:fault",
        })
    frame = pd.DataFrame(rows)
    train0, validation0, groups0 = _normal_split(frame, validation_rotation=0)
    train1, validation1, groups1 = _normal_split(frame, validation_rotation=1)
    assert groups0 != groups1
    for train, validation, groups in (
        (train0, validation0, groups0), (train1, validation1, groups1)
    ):
        assert set(train.canonical_case_id).isdisjoint(validation.canonical_case_id)
        assert train.evaluation_role.eq("normal_reference").all()
        assert validation.evaluation_role.eq("normal_reference").all()
        for domain, group in groups.items():
            assert validation.loc[validation.domain.eq(domain), "split_group_id"].eq(group).all()
            assert not train.split_group_id.eq(group).any()
