# Mentor Packet: ML-14 + RFLYMAD Status

## One-minute summary

We are building UAV anomaly detection, but the model must be judged by two
numbers together: how many true anomaly events it catches, and how many false
operator notifications it creates per hour.

ML-14 improved the false-alarm drift problem strongly, but did not yet restore
enough recall to pass the operating gate.

## Key numbers

- SEAD labels: 1246
- SEAD parsed flights: 1244
- SEAD normal flights parsed: 898
- Development normal sessions: 115
- Holdout flights kept closed: 200
- D1 rebuild gate: passed
- D2 FA drift gate: passed
- Median FA shift ratio: 5.21 -> 1.90
- Relative FA drift drop: 63.6%
- D3 operating gate: failed
- Best critical under FA<=2: ml14_fusion/cusum: recall=0.043, FA/h=1.60
- Best advisory under FA<=12: ml14_fusion/cusum: recall=0.126, FA/h=9.95

## RFLYMAD status

- Parsed RFLYMAD flights: 490
- Real-Motor: 242
- Real-Sensors: 197
- Real-No_Fault normal: 51
- Normal needed for registered split: 61
- Missing normal flights: 10
- RFLY status: parsed_but_gate_not_ready

## How recall is computed

1. Each anomaly interval is converted into a boolean timeline: `y_true=True`
   inside the annotated anomaly interval.
2. The model produces an anomaly score for each time bucket.
3. A decision policy turns scores into alarms. We count only new alarm starts,
   not every row where the alarm remains active.
4. One anomaly event is counted as detected only if a new alarm starts inside
   that event interval.
5. `event_onset_recall = detected_events / n_events`.
6. `false_alarms_per_hour = false_alarm_events / normal_hours`, where false
   alarms are new alarm starts outside anomaly intervals.

So recall 0.126 means: among all true anomaly events in the evaluated splits,
about 12.6% produced a timely new operator alarm.

## Mentor-ready sentence

The refresh fixed a major measurement problem: false alarms no longer explode
from validation to test as badly. Median FA drift dropped from
5.21x budget to
1.90x budget. But the model became
too conservative, so recall is still far below the operating target. RFLYMAD is
the right next data source for motor/sensor faults, but it still needs
10 more normal real flights before a
registered split/evaluation is honest.

## Files

- `gate_status_summary.csv`
- `ml14_best_operational_rows.csv`
- `ml14_drift_shift_cells.csv`
- `ml13_vs_ml14_category_recall.csv`
- `rflymad_label_counts.csv`
- `figures/ml14_fa_drift_old_vs_new.png`
- `figures/ml14_recall_vs_false_alarm.png`
- `figures/rflymad_parsed_pool.png`
