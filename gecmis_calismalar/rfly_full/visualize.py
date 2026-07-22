"""Development-only visual diagnostics for the RflyMAD v2 workstream.

The plots in this module are exploratory diagnostics.  Feature values from the
locked test allocation are deliberately never loaded.  Split composition may
be reported from manifest metadata, but no locked-test distribution is used to
select features, thresholds or models.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import RobustScaler
from sklearn.manifold import TSNE

from gecmis_calismalar.rfly_full.pipeline import FEATURES, ROOT


V2_ROOT = ROOT / "artifacts/rfly_full/v2"
MANIFEST_PATH = V2_ROOT / "dataset_manifest.csv"
DEFAULT_OUTPUT = V2_ROOT / "visuals"
FAMILY_ORDER = ["NoFault", "Motor", "Propeller", "Sensor", "Environment"]
DOMAIN_ORDER = ["Real", "HIL", "SIL"]
COLORS = {
    "NoFault": "#2a9d8f",
    "Motor": "#e76f51",
    "Propeller": "#f4a261",
    "Sensor": "#9b5de5",
    "Environment": "#457b9d",
    "Real": "#d62828",
    "HIL": "#003049",
    "SIL": "#669bbc",
}


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    rows: list[str],
    columns: list[str],
    title: str,
    *,
    vmin: float,
    vmax: float,
    cmap: str,
    annotate: bool = False,
    fmt: str = ".0f",
) -> None:
    image = ax.imshow(values, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks(np.arange(len(columns)), columns, rotation=55, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(rows)), rows, fontsize=8)
    if annotate:
        midpoint = (vmin + vmax) / 2
        for i in range(len(rows)):
            for j in range(len(columns)):
                value = values[i, j]
                text_color = "white" if value > midpoint else "black"
                label = format(value, fmt)
                if fmt == ".0f" and vmax <= 1.0:
                    label = f"{value:.0%}"
                ax.text(j, i, label, ha="center", va="center", fontsize=8, color=text_color)
    plt.colorbar(image, ax=ax, fraction=0.028, pad=0.02)


def load_development_rows(manifest: pd.DataFrame) -> pd.DataFrame:
    """Load only development feature rows and attach frozen manifest metadata."""
    development = manifest.loc[manifest["split"].eq("development")].copy()
    valid_ids = set(development["case_id"].astype(str))
    columns = ["case_id", "t_rel_s", "fault_active", *FEATURES]
    frames: list[pd.DataFrame] = []
    for raw_path in sorted(development["parsed_path"].dropna().astype(str).unique()):
        path = Path(raw_path)
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=columns)
        frame = frame.loc[frame["case_id"].astype(str).isin(valid_ids)]
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError("No development parquet rows were found")
    rows = pd.concat(frames, ignore_index=True)
    metadata = development.set_index("case_id")
    for name in ("domain", "fault_family", "system_fault", "cv_fold", "split_group_id"):
        rows[name] = rows["case_id"].map(metadata[name])
    rows["fault_active"] = rows["fault_active"].fillna(False).astype(bool)
    rows["system_fault"] = rows["system_fault"].fillna(False).astype(bool)
    rows["analysis_phase"] = "nonfault_exposure"
    rows.loc[rows["fault_family"].eq("NoFault"), "analysis_phase"] = "normal"
    rows.loc[rows["fault_family"].eq("Environment"), "analysis_phase"] = "environment_inactive"
    rows.loc[
        rows["fault_family"].eq("Environment") & rows["fault_active"], "analysis_phase"
    ] = "environment_active"
    rows.loc[rows["system_fault"] & rows["fault_active"], "analysis_phase"] = "fault_active"
    return rows


def plot_composition(manifest: pd.DataFrame, output: Path) -> None:
    counts = (
        manifest.groupby(["domain", "fault_family", "split"], observed=True)
        .size()
        .rename("flights")
        .reset_index()
    )
    counts.to_csv(output / "data_composition.csv", index=False)
    labels = [f"{domain}\n{family}" for domain in DOMAIN_ORDER for family in FAMILY_ORDER]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(15, 6))
    bottom = np.zeros(len(labels))
    for split, color in (("development", "#277da1"), ("locked_test", "#adb5bd")):
        values = []
        for domain in DOMAIN_ORDER:
            for family in FAMILY_ORDER:
                match = counts.loc[
                    counts["domain"].eq(domain)
                    & counts["fault_family"].eq(family)
                    & counts["split"].eq(split),
                    "flights",
                ]
                values.append(int(match.iloc[0]) if len(match) else 0)
        ax.bar(x, values, bottom=bottom, label=split.replace("_", " "), color=color)
        bottom += np.asarray(values)
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("Uçuş sayısı")
    ax.set_title("RflyMAD veri bileşimi — kilitli test yalnız manifest sayımıdır", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    _save(fig, output / "01_data_composition.png")


def plot_missingness(rows: pd.DataFrame, output: Path) -> None:
    table = rows.groupby(["domain", "fault_family"], observed=True)[list(FEATURES)].apply(
        lambda frame: frame.notna().mean()
    )
    table = table.reindex(
        pd.MultiIndex.from_tuples(
            [(d, f) for d in DOMAIN_ORDER for f in FAMILY_ORDER if (d, f) in table.index],
            names=["domain", "fault_family"],
        )
    )
    table.to_csv(output / "feature_completeness_by_domain_family.csv")
    fig, ax = plt.subplots(figsize=(17, 7))
    _heatmap(
        ax,
        table.to_numpy(float),
        [f"{d} / {f}" for d, f in table.index],
        list(table.columns),
        "Development verisi özellik doluluğu",
        vmin=0,
        vmax=1,
        cmap="YlGnBu",
    )
    _save(fig, output / "02_feature_completeness_heatmap.png")


def _top_pairs(correlation: pd.DataFrame, cohort: str) -> pd.DataFrame:
    records = []
    for i, left in enumerate(correlation.columns):
        for right in correlation.columns[i + 1 :]:
            value = correlation.loc[left, right]
            if pd.notna(value):
                records.append({"cohort": cohort, "feature_a": left, "feature_b": right, "spearman_rho": value, "abs_rho": abs(value)})
    return pd.DataFrame(records).sort_values("abs_rho", ascending=False)


def plot_correlations(rows: pd.DataFrame, output: Path) -> None:
    normal = rows.loc[rows["analysis_phase"].eq("normal"), list(FEATURES)]
    active = rows.loc[rows["analysis_phase"].eq("fault_active"), list(FEATURES)]
    usable = [
        feature
        for feature in FEATURES
        if normal[feature].notna().mean() >= 0.25
        and active[feature].notna().mean() >= 0.25
        and normal[feature].nunique(dropna=True) > 1
        and active[feature].nunique(dropna=True) > 1
    ]
    corr_normal = normal[usable].corr(method="spearman")
    corr_active = active[usable].corr(method="spearman")
    delta = corr_active - corr_normal
    corr_normal.to_csv(output / "correlation_normal.csv")
    corr_active.to_csv(output / "correlation_fault_active.csv")
    delta.to_csv(output / "correlation_delta_fault_minus_normal.csv")
    pairs = pd.concat([_top_pairs(corr_normal, "normal"), _top_pairs(corr_active, "fault_active")])
    pairs.to_csv(output / "top_correlated_feature_pairs.csv", index=False)
    delta_pairs = _top_pairs(delta, "fault_minus_normal").rename(columns={"spearman_rho": "rho_delta", "abs_rho": "abs_delta"})
    delta_pairs.to_csv(output / "top_correlation_changes.csv", index=False)
    for number, matrix, title, cmap, vmin, vmax, filename in (
        (3, corr_normal, "Normal uçuşlarda Spearman korelasyonu", "RdBu_r", -1, 1, "03_correlation_normal.png"),
        (4, corr_active, "Aktif sistem arızasında Spearman korelasyonu", "RdBu_r", -1, 1, "04_correlation_fault_active.png"),
        (5, delta, "Korelasyon değişimi: aktif arıza − normal", "PiYG", -1, 1, "05_correlation_delta.png"),
    ):
        fig, ax = plt.subplots(figsize=(14, 12))
        _heatmap(ax, matrix.to_numpy(float), usable, usable, title, vmin=vmin, vmax=vmax, cmap=cmap)
        fig.text(0.5, 0.01, "Yalnız development satırları; korelasyon nedensellik veya model başarısı değildir.", ha="center", fontsize=9)
        _save(fig, output / filename)


def build_flight_features(rows: pd.DataFrame, manifest: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    eligible = rows.loc[
        rows["fault_family"].eq("NoFault")
        | rows["analysis_phase"].isin(["fault_active", "environment_active"])
    ].copy()
    coverage = eligible[list(FEATURES)].notna().mean()
    usable = [feature for feature in FEATURES if coverage[feature] >= 0.55]
    grouped = eligible.groupby("case_id", observed=True)[usable]
    medians = grouped.median().add_suffix("__median")
    iqrs = (grouped.quantile(0.75) - grouped.quantile(0.25)).add_suffix("__iqr")
    flight = medians.join(iqrs).reset_index()
    meta = manifest.loc[manifest["split"].eq("development")].set_index("case_id")
    for name in ("domain", "fault_family", "system_fault", "cv_fold", "split_group_id"):
        flight[name] = flight["case_id"].map(meta[name])
    feature_columns = [column for column in flight.columns if "__" in column]
    return flight, feature_columns


def prepare_embedding_matrix(flight: pd.DataFrame, feature_columns: list[str]) -> tuple[np.ndarray, list[str]]:
    normal = flight["fault_family"].eq("NoFault")
    normal_coverage = flight.loc[normal, feature_columns].notna().mean()
    selected = [column for column in feature_columns if normal_coverage[column] >= 0.60]
    normal_variance = flight.loc[normal, selected].var(skipna=True)
    selected = [column for column in selected if np.isfinite(normal_variance[column]) and normal_variance[column] > 1e-12]
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler(quantile_range=(25, 75))
    imputer.fit(flight.loc[normal, selected])
    scaler.fit(imputer.transform(flight.loc[normal, selected]))
    matrix = scaler.transform(imputer.transform(flight[selected]))
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=10.0, neginf=-10.0)
    return np.clip(matrix, -10.0, 10.0), selected


def plot_embeddings_and_knn(
    flight: pd.DataFrame,
    matrix: np.ndarray,
    selected: list[str],
    output: Path,
    random_state: int,
) -> dict:
    components = min(12, matrix.shape[1], matrix.shape[0] - 1)
    pca_model = PCA(n_components=components, random_state=random_state)
    reduced = pca_model.fit_transform(matrix)
    perplexity = min(30.0, max(5.0, (len(flight) - 1) / 3))
    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=750,
        random_state=random_state,
        n_jobs=1,
    ).fit_transform(reduced)
    exported = flight[["case_id", "domain", "fault_family", "cv_fold", "split_group_id"]].copy()
    exported["pca_1"] = reduced[:, 0]
    exported["pca_2"] = reduced[:, 1]
    exported["tsne_1"] = embedding[:, 0]
    exported["tsne_2"] = embedding[:, 1]
    exported.to_csv(output / "flight_embedding_development.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for family in FAMILY_ORDER:
        mask = flight["fault_family"].eq(family).to_numpy()
        if mask.any():
            axes[0].scatter(reduced[mask, 0], reduced[mask, 1], s=8, alpha=0.48, label=family, color=COLORS[family])
            axes[1].scatter(embedding[mask, 0], embedding[mask, 1], s=8, alpha=0.48, label=family, color=COLORS[family])
    for domain in DOMAIN_ORDER:
        mask = flight["domain"].eq(domain).to_numpy()
        if mask.any():
            axes[2].scatter(embedding[mask, 0], embedding[mask, 1], s=8, alpha=0.48, label=domain, color=COLORS[domain])
    axes[0].set_title(f"PCA-2 (açıklanan varyans %{100 * pca_model.explained_variance_ratio_[:2].sum():.1f})", fontweight="bold")
    axes[1].set_title("t-SNE — arıza ailesi rengi", fontweight="bold")
    axes[2].set_title("Aynı t-SNE — domain rengi", fontweight="bold")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(markerscale=2, frameon=False, fontsize=8)
    fig.suptitle("Uçuş-düzeyi development temsili (arıza uçuşunda yalnız aktif aralık)", fontweight="bold")
    fig.text(0.5, 0.01, "t-SNE adaları başarı kanıtı değildir; domain/oturum etkisini teşhis etmek için kullanılır.", ha="center", fontsize=9)
    _save(fig, output / "06_pca_tsne_family_domain.png")

    neighbors = NearestNeighbors(n_neighbors=min(6, len(flight))).fit(reduced)
    neighbor_indices = neighbors.kneighbors(return_distance=False)[:, 1:]
    family_values = flight["fault_family"].astype(str).to_numpy()
    domain_values = flight["domain"].astype(str).to_numpy()
    family_agreement = (family_values[neighbor_indices] == family_values[:, None]).mean(axis=1)
    domain_agreement = (domain_values[neighbor_indices] == domain_values[:, None]).mean(axis=1)
    neighbor_table = flight[["case_id", "domain", "fault_family"]].copy()
    neighbor_table["knn_family_agreement"] = family_agreement
    neighbor_table["knn_domain_agreement"] = domain_agreement
    neighbor_table.to_csv(output / "knn_neighbor_agreement.csv", index=False)
    summary = neighbor_table.groupby("fault_family", observed=True)[["knn_family_agreement", "knn_domain_agreement"]].mean().reindex(FAMILY_ORDER)
    summary.to_csv(output / "knn_neighbor_agreement_summary.csv")
    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.18, summary["knn_family_agreement"], width=0.36, label="aynı arıza ailesi")
    ax.bar(x + 0.18, summary["knn_domain_agreement"], width=0.36, label="aynı domain")
    ax.set_xticks(x, summary.index)
    ax.set_ylim(0, 1)
    ax.set_ylabel("5 komşu içindeki oran")
    ax.set_title("k-NN komşuluğu neyi izliyor: arıza mı, domain mi?", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    _save(fig, output / "07_knn_neighbor_agreement.png")

    train = flight["cv_fold"].astype(int).ne(0).to_numpy()
    validation = ~train
    classifier = KNeighborsClassifier(n_neighbors=5, weights="distance")
    classifier.fit(matrix[train], family_values[train])
    prediction = classifier.predict(matrix[validation])
    labels = [label for label in FAMILY_ORDER if label in set(family_values)]
    counts = confusion_matrix(family_values[validation], prediction, labels=labels)
    normalized = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    pd.DataFrame(counts, index=labels, columns=labels).to_csv(output / "knn_fold0_confusion_counts.csv")
    report = classification_report(family_values[validation], prediction, labels=labels, output_dict=True, zero_division=0)
    (output / "knn_fold0_classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _heatmap(axes[0], counts, labels, labels, "k-NN fold-0 confusion — sayı", vmin=0, vmax=max(1, counts.max()), cmap="Blues", annotate=True)
    _heatmap(axes[1], normalized, labels, labels, "k-NN fold-0 confusion — satır oranı", vmin=0, vmax=1, cmap="Blues", annotate=True, fmt=".0f")
    for ax in axes:
        ax.set_xlabel("Tahmin")
        ax.set_ylabel("Gerçek")
    fig.text(0.5, 0.01, "Development içi teşhis; uçuş-düzeyi, grup-temelli fold. Operasyonel model sonucu değildir.", ha="center", fontsize=9)
    _save(fig, output / "08_knn_fold0_confusion_matrix.png")
    return {
        "flights": int(len(flight)),
        "features_before_pca": len(selected),
        "pca_components": components,
        "pca_2_explained_variance": float(pca_model.explained_variance_ratio_[:2].sum()),
        "tsne_perplexity": perplexity,
        "knn_fold0_flights": int(validation.sum()),
        "knn_fold0_accuracy": float((prediction == family_values[validation]).mean()),
        "mean_knn_family_agreement": float(family_agreement.mean()),
        "mean_knn_domain_agreement": float(domain_agreement.mean()),
    }


def plot_feature_shifts(rows: pd.DataFrame, output: Path) -> None:
    records = []
    for domain in DOMAIN_ORDER:
        normal = rows.loc[rows["domain"].eq(domain) & rows["analysis_phase"].eq("normal"), list(FEATURES)]
        active = rows.loc[rows["domain"].eq(domain) & rows["analysis_phase"].eq("fault_active"), list(FEATURES)]
        for feature in FEATURES:
            n = normal[feature].dropna()
            a = active[feature].dropna()
            if len(n) < 20 or len(a) < 20:
                continue
            iqr = n.quantile(0.75) - n.quantile(0.25)
            if not np.isfinite(iqr) or abs(iqr) < 1e-12:
                continue
            records.append({
                "domain": domain,
                "feature": feature,
                "normal_median": n.median(),
                "fault_active_median": a.median(),
                "normal_iqr": iqr,
                "robust_median_shift": (a.median() - n.median()) / iqr,
                "normal_rows": len(n),
                "fault_active_rows": len(a),
            })
    shifts = pd.DataFrame(records)
    shifts.to_csv(output / "feature_robust_shifts_by_domain.csv", index=False)
    ranking = shifts.groupby("feature")["robust_median_shift"].apply(lambda values: values.abs().median()).sort_values(ascending=False)
    top = list(ranking.head(15).index)
    table = shifts.pivot(index="feature", columns="domain", values="robust_median_shift").reindex(top).reindex(columns=DOMAIN_ORDER)
    clipped = table.clip(-5, 5)
    fig, ax = plt.subplots(figsize=(8, 8))
    _heatmap(ax, clipped.to_numpy(float), list(clipped.index), list(clipped.columns), "Aktif arıza − normal: sağlam medyan kayması", vmin=-5, vmax=5, cmap="RdBu_r", annotate=True, fmt=".1f")
    fig.text(0.5, 0.01, "Birim = aynı domaindeki normal IQR; ±5'te kırpılmıştır. Arıza ailesi karışımı nedeniyle nedensel değildir.", ha="center", fontsize=9)
    _save(fig, output / "09_feature_shift_by_domain.png")


def _binary_matrix(summary: dict) -> np.ndarray:
    return np.asarray([[summary["flight_tn"], summary["flight_fp"]], [summary["flight_fn"], summary["flight_tp"]]], dtype=float)


def plot_model_confusions(output: Path) -> None:
    ae_path = V2_ROOT / "dense_ae_diagnostics/summary.json"
    run_dirs = sorted((V2_ROOT / "supervised_tcn").glob("run_*"))
    if not ae_path.exists() or not run_dirs:
        return
    ae = json.loads(ae_path.read_text(encoding="utf-8"))["frozen_policy"]
    tcn = pd.read_csv(run_dirs[-1] / "operational_metrics.csv").set_index("policy")
    panels = [
        ("Dense AE — donmuş eşik", _binary_matrix(ae)),
        ("TCN smoke — kritik", _binary_matrix(tcn.loc["critical"].to_dict())),
        ("TCN smoke — danışma", _binary_matrix(tcn.loc["advisory"].to_dict())),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (title, matrix) in zip(axes, panels):
        _heatmap(ax, matrix, ["Gerçek normal", "Gerçek arıza"], ["Normal tahmini", "Alarm"], title, vmin=0, vmax=max(1, matrix.max()), cmap="Blues", annotate=True)
        ax.set_xlabel("Tahmin")
    fig.text(0.5, 0.01, "AE matrisi NoFault uçuşlarını ve sistem arızalarını; TCN matrisleri küçük 1-epoch smoke testini gösterir. Doğrudan kıyaslanmaz.", ha="center", fontsize=9)
    _save(fig, output / "10_model_confusion_matrices.png")


def plot_ae_phase_scores(output: Path) -> None:
    path = V2_ROOT / "dense_ae_diagnostics/score_distributions_by_phase.csv"
    summary_path = V2_ROOT / "dense_ae_diagnostics/summary.json"
    if not path.exists() or not summary_path.exists():
        return
    data = pd.read_csv(path)
    threshold = json.loads(summary_path.read_text(encoding="utf-8"))["frozen_policy"]["threshold"]
    phases = ["normal", "pre_fault", "fault_active", "post_fault", "environment_active"]
    data = data.loc[data["phase"].isin(phases)].copy()
    x_lookup = {phase: index for index, phase in enumerate(phases)}
    fig, ax = plt.subplots(figsize=(12, 6))
    offsets = {"Real": -0.18, "HIL": 0.0, "SIL": 0.18}
    for domain in DOMAIN_ORDER:
        subset = data.loc[data["domain"].eq(domain)]
        x = np.asarray([x_lookup[value] + offsets[domain] for value in subset["phase"]])
        sizes = 20 + 25 * np.log10(np.maximum(subset["count"].to_numpy(), 1))
        ax.scatter(x, subset["median"], s=sizes, alpha=0.75, color=COLORS[domain], label=domain, edgecolor="white", linewidth=0.4)
        for xi, (_, row) in zip(x, subset.iterrows()):
            ax.vlines(xi, row["median"], row["q95"], color=COLORS[domain], alpha=0.28, linewidth=1)
    ax.axhline(threshold, color="#d00000", linestyle="--", linewidth=1.5, label=f"donmuş eşik {threshold:.2f}")
    ax.set_xticks(range(len(phases)), [phase.replace("_", " ") for phase in phases])
    ax.set_ylabel("AE yeniden-yapılandırma skoru (medyan → q95)")
    ax.set_title("Dense AE skorları: faz ve domain ayrımı", fontweight="bold")
    ax.legend(frameon=False, ncol=4)
    ax.grid(axis="y", alpha=0.2)
    _save(fig, output / "11_ae_score_by_phase_domain.png")


def write_readme(output: Path, diagnostics: dict, selected: list[str]) -> None:
    text = f"""# RflyMAD v2 görsel teşhis paketi

Bu klasör yalnız **development** telemetri değerlerinden üretilmiştir. Kilitli
testin yalnız manifestteki uçuş sayıları `01` grafiğinde gösterilir; test
özellikleri korelasyon, PCA, t-SNE veya k-NN hesabına sokulmamıştır.

- Development telemetri satırı: {diagnostics['development_rows']:,}
- Uçuş-düzeyi temsil: {diagnostics['flights']:,}
- PCA öncesi medyan/IQR özelliği: {diagnostics['features_before_pca']}
- PCA ilk iki bileşen açıklanan varyans: %{100 * diagnostics['pca_2_explained_variance']:.2f}
- k-NN fold-0 keşif doğruluğu: %{100 * diagnostics['knn_fold0_accuracy']:.2f}
- Ortalama 5-NN aynı arıza ailesi oranı: %{100 * diagnostics['mean_knn_family_agreement']:.2f}
- Ortalama 5-NN aynı domain oranı: %{100 * diagnostics['mean_knn_domain_agreement']:.2f}
- Normal fold-0 doğrulama dağılımı: {diagnostics['normal_fold0_by_domain']}
- Normal fold 1–4 eğitim dağılımı: {diagnostics['normal_train_by_domain']}

## Doğru yorum

Korelasyon ilişkiyi, t-SNE yerel komşuluğu, k-NN ise mevcut temsilin yakınlık
yapısını gösterir. Bunlar operasyonel anomali başarısı değildir. Özellikle
t-SNE adalarının domain/oturum izleyip izlemediği, arıza ailesi görünümünden
birlikte değerlendirilmelidir. `08` k-NN matrisi development içi ve
grup-temelli fold-0 teşhisidir. `10` içindeki Dense AE ve TCN matrislerinin
veri kapsamları farklıdır; yan yana çizilmeleri doğrudan performans kıyası
anlamına gelmez.

Normal-only yaklaşımda yalnız development NoFault uçuşları öğrenme/eşik için
kullanılmalı; arıza etiketleri strict novelty değerlendirmesine kadar saklı
tutulmalıdır. Bilinen arızalar için etiketli TCN hattı ayrıca yürütülür.
Mevcut fold-0 normal doğrulaması yalnız HIL içerdiği için domain-bazlı eşik
kalibrasyonu yapılmadan bu split nihai eğitim sözleşmesi olarak kullanılmamalıdır.

Seçilen temsil kolonları: `{', '.join(selected)}`
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def render(output: Path = DEFAULT_OUTPUT, random_state: int = 20260721) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    manifest = pd.read_csv(MANIFEST_PATH)
    plot_composition(manifest, output)
    rows = load_development_rows(manifest)
    plot_missingness(rows, output)
    plot_correlations(rows, output)
    flight, feature_columns = build_flight_features(rows, manifest)
    matrix, selected = prepare_embedding_matrix(flight, feature_columns)
    diagnostics = plot_embeddings_and_knn(flight, matrix, selected, output, random_state)
    diagnostics["development_rows"] = int(len(rows))
    plot_feature_shifts(rows, output)
    plot_model_confusions(output)
    plot_ae_phase_scores(output)
    diagnostics["normal_flights_total"] = int(manifest["fault_family"].eq("NoFault").sum())
    diagnostics["normal_flights_development"] = int((manifest["fault_family"].eq("NoFault") & manifest["split"].eq("development")).sum())
    diagnostics["normal_flights_real_total"] = int((manifest["fault_family"].eq("NoFault") & manifest["domain"].eq("Real")).sum())
    normal_development = manifest.loc[
        manifest["fault_family"].eq("NoFault") & manifest["split"].eq("development")
    ].copy()
    diagnostics["normal_fold0_by_domain"] = {
        domain: int(
            (
                normal_development["domain"].eq(domain)
                & normal_development["cv_fold"].astype(int).eq(0)
            ).sum()
        )
        for domain in DOMAIN_ORDER
    }
    diagnostics["normal_train_by_domain"] = {
        domain: int(
            (
                normal_development["domain"].eq(domain)
                & normal_development["cv_fold"].astype(int).ne(0)
            ).sum()
        )
        for domain in DOMAIN_ORDER
    }
    diagnostics["development_only_features"] = True
    (output / "visual_diagnostics_summary.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    write_readme(output, diagnostics, selected)
    return diagnostics
