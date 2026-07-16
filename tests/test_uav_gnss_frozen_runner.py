from uav_gnss.frozen_runner import require_prior_role


def _report(current_pass: bool):
    methods = {
        method: {
            "passes_gate": current_pass,
            "events": {"recall": 1.0 if method == "lstm" else 0.0},
        }
        for method in ("px4_native", "cusum", "lstm")
    }
    return {
        "contracts": {
            "critical": {"methods": {key: dict(value) for key, value in methods.items()}},
            "advisory": {"methods": {key: dict(value) for key, value in methods.items()}},
        }
    }


def test_rehearsal_cannot_resurrect_failed_development():
    result = require_prior_role(_report(True), _report(False))
    assert result["selected_critical_method"] is None
    assert result["preliminary_status"] == "no_go_on_current_role"
    assert not any(
        method["passes_gate"]
        for method in result["contracts"]["critical"]["methods"].values()
    )

