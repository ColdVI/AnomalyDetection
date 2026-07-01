Purpose & context
Anıl is an intern (stajyer) on an 8-week UAV data engineering project, working alongside two teammates. Anıl's individual focus (weeks 5–8) is anomaly detection in UAV flight telemetry data using Isolation Forest and LSTM-Autoencoder methods, targeting GPS spoofing, sensor failures, actuator faults, and unauthorized interventions. A teammate owns the Kafka/MinIO big data infrastructure (Bronze/Silver/Gold medallion pipeline), while Anıl is responsible for the ML and model training components. The project also requires a formal literature review chapter suitable for an internship/project report.
Three primary data sources serve distinct roles:

ALFA dataset — labeled fixed-wing UAV actuator/motor fault telemetry; trains the actuator fault model
UAV Attack IEEE DataPort dataset — ULOG-converted CSVs with benign/malicious GPS spoofing, jamming, DoS scenarios on PX4 multirotor; trains the GPS attack model
ADS-B live feed (adsb.lol) — unlabeled live/historical crewed aircraft tracking; feeds inference and normal-flight baseline

Two Kaggle CSVs (SurveilDrone-Net23 and a disaster operations dataset) were evaluated and found to be synthetic (GPS teleportation artifacts, integer-only altitudes, velocity-position inconsistencies) — deprioritized for model training, retained for pipeline/dashboard demonstration only.
Whelan et al. (2020) serves as the primary baseline paper (Autoencoder achieving ~95% average F1 on UAV sensor attack detection); the ALFA dataset paper anchors actuator fault benchmarking.

Current state

Architecture: Bronze/Silver/Gold medallion pipeline established. Bronze lands raw data with provenance metadata only; Silver performs per-source cleaning and intra-source merge_asof time-alignment with interpolation; Gold produces a unified common_uav_events table via UNION with source_type discriminator (never timestamp-joined across sources).
Modeling approach confirmed: Unsupervised/semi-supervised novelty detection — train exclusively on normal (benign-only) flight data, detect deviations as anomalies. ALFA labels are used only for threshold calibration and evaluation (precision/recall/F1, sequence-level metrics, Maximum/Average Detection Time), never for supervised training.
Separate models per domain — fixed-wing (ALFA), multirotor (UAV Attack), and ADS-B are architecturally incompatible for weight sharing; same methodology applied independently to each.
Key technical clarifications locked in:

ALFA's labeled data lives in processed/ topic-CSVs (pandas-readable); pymavlink applies only to raw DataFlash binary logs (.bin/.tlog), not CSVs
ALFA topics arrive at different frequencies (5–50 Hz); merge_asof time-alignment is a prerequisite before any residual feature computation
UAV Attack has significant class imbalance (large benign vs. small malicious); attack timing recorded separately since ULOG timestamps are stripped on conversion
ADS-B (adsb.lol) eliminates OAuth2/token management vs. prior OpenSky approach
IEEE DataPort UAV Attack column schema should be inspected from actual downloaded files before finalizing Silver schema


Codex integration: Three persistent repo-root files produced — phased implementation plan (bronze_implementasyon_plani.md), AGENTS.md (behavioral rules Codex reads automatically), and MEMORY.md (consolidated knowledge base). Recommended pattern: feed Codex one phase at a time, with real sample files in _input/ directories before running loader phases.
MLflow + MinIO identified as the artifact store and model registry; every trained model artifact tagged with dataset version, features, hyperparameters, and metrics.


On the horizon

Staged training plan (Faz 0–3):

Faz 0: Feature inventory across sources
Faz 1: Basic kinematic features + Isolation Forest across all sources (pipeline validation)
Faz 2: Advanced features + both methods on ALFA and UAV Attack
Faz 3: LSTM-AE on windowed advanced features for collective/contextual anomaly detection


Periodic retraining automation is most meaningful for the live ADS-B stream (one-class/novelty detection on accumulated normal flight data); ALFA and IEEE are static datasets where automation value is reproducibility and versioning
Silver schema for UAV Attack pending actual DataPort file inspection
Team review of Bronze layer before moving to Silver


Key learnings & principles

Domain separation is non-negotiable: Fixed-wing UAV, multirotor UAV, and crewed aircraft ADS-B have incompatible feature spaces — no cross-domain model weight transfer; same methodology, separate models
Novelty detection rationale: Live ADS-B has no labels and contains unknown anomaly types, so supervised classification on ALFA's fault types would not generalize; the semi-supervised framing is the correct choice for this project
Most discriminative features:

ALFA: commanded-vs-measured residuals from mavros/nav_info/ topics (roll, pitch, yaw, velocity, airspeed) and path deviation (mavctrl/path_dev) — these are computed columns, not raw sensor columns
UAV Attack: GPS-IMU cross-sensor inconsistency, GPS signal quality, magnetometer data, MAVLink message rate/ID sequence patterns (for DoS)


Synthetic data detection: GPS-velocity channel inconsistency (large spatial displacement vs. reported low velocity across consecutive timestamps) is a reliable structural artifact of naively generated tabular synthetic data
Bronze principle: Raw data with lineage only — all unit conversions, coordinate scaling, and harmonization deferred to Silver
Architecture deviation acknowledged: The teammate's data source substitutions (adsb.lol for OpenSky, ALFA + UAV Attack for generic MAVLink) serve Anıl's anomaly detection project better than the original course sources; this should be formally acknowledged with the team and mentor


Approach & patterns

Works in Turkish; prefers explanations humanized first (intuitive analogies, then formulas) before technical depth
Collaborative and iterative — seeks both conceptual grounding and concrete actionable next steps in the same conversation
Report/literature review requirements are highly specific: build every technique from prerequisite concepts upward, define all terms, present all formulas in LaTeX with verbal reading and intuitive explanation, technical content in English
Validates assumptions against real files rather than relying on assumed conventions (e.g., UAV Attack filename schema)


Tools & resources

Pipeline: Kafka, MinIO/S3, medallion architecture (Bronze/Silver/Gold)
ML tooling: MLflow (artifact store + model registry pointed at MinIO), Isolation Forest, LSTM-Autoencoder
Data libraries: pandas, merge_asof, pyulog, adsb.lol API
Codex CLI for implementation automation; AGENTS.md for persistent behavioral rules
Key references: Whelan et al. (2020) UAV sensor attack paper, ALFA dataset paper, Chandola et al. anomaly taxonomy, UAVAttack and ALFA research papers (on hand)