from __future__ import annotations

import numpy as np
import pandas as pd


def get_all_labels(df: pd.DataFrame) -> list[str]:
    return sorted({label for labels in df["LabelList"] for label in labels})


def build_multi_hot(df: pd.DataFrame, labels: list[str] | None = None) -> pd.DataFrame:
    all_labels = labels or get_all_labels(df)
    multi_hot = pd.DataFrame(0, index=df.index, columns=all_labels, dtype=np.uint8)
    label_sets = df["LabelList"].map(set)
    for label in all_labels:
        multi_hot[label] = label_sets.map(lambda labels_set: int(label in labels_set)).astype(np.uint8)
    return multi_hot


def compute_label_summary(multi_hot: pd.DataFrame) -> pd.DataFrame:
    counts = multi_hot.sum().sort_values(ascending=False)
    return pd.DataFrame(
        {
            "count": counts.astype(int),
            "prevalence_pct": counts / len(multi_hot) * 100.0,
        }
    )


def compute_labelset_summary(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    return df["Finding Labels"].value_counts().head(top_n).to_frame("count")


def compute_multilabel_summary(df: pd.DataFrame, num_classes: int) -> pd.DataFrame:
    rows = [
        ("label_cardinality", float(df["NumLabels"].mean())),
        ("label_density", float(df["NumLabels"].mean() / num_classes)),
        ("unique_labelsets", int(df["Finding Labels"].nunique())),
        ("multi_label_ratio_pct", float((df["NumLabels"] > 1).mean() * 100.0)),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def compute_cooccurrence(multi_hot: pd.DataFrame, exclude_label: str = "No Finding") -> pd.DataFrame:
    disease_labels = [label for label in multi_hot.columns if label != exclude_label]
    disease_hot = multi_hot[disease_labels]
    return disease_hot.T.dot(disease_hot)


def compute_pairwise_associations(
    multi_hot: pd.DataFrame,
    exclude_label: str = "No Finding",
) -> pd.DataFrame:
    disease_labels = [label for label in multi_hot.columns if label != exclude_label]
    disease_hot = multi_hot[disease_labels]
    label_support = disease_hot.sum(axis=0)
    n_rows = len(disease_hot)
    pair_rows: list[tuple[str, str, int, float, float, int, int]] = []

    for i, label_a in enumerate(disease_labels):
        a_mask = disease_hot[label_a].to_numpy(dtype=bool)
        support_a = int(label_support[label_a])
        p_a = support_a / n_rows
        for label_b in disease_labels[i + 1 :]:
            b_mask = disease_hot[label_b].to_numpy(dtype=bool)
            support_b = int(label_support[label_b])
            p_b = support_b / n_rows
            inter = int(np.logical_and(a_mask, b_mask).sum())
            if inter == 0:
                continue
            union = int(np.logical_or(a_mask, b_mask).sum())
            jaccard = inter / union if union else np.nan
            p_ab = inter / n_rows
            lift = p_ab / (p_a * p_b) if p_a * p_b else np.nan
            pair_rows.append((label_a, label_b, inter, jaccard, lift, support_a, support_b))

    return pd.DataFrame(
        pair_rows,
        columns=["label_a", "label_b", "co_count", "jaccard", "lift", "support_a", "support_b"],
    )


def compute_binary_group_risk(
    df: pd.DataFrame,
    multi_hot: pd.DataFrame,
    group_col: str,
    numerator_group: str,
    denominator_group: str,
) -> pd.DataFrame:
    """Compute per-label prevalence and risk ratio between two groups."""
    group_counts = df[group_col].value_counts()
    rows = []
    for label in multi_hot.columns:
        label_mask = multi_hot[label] == 1
        numerator_cases = int(((df[group_col] == numerator_group) & label_mask).sum())
        denominator_cases = int(((df[group_col] == denominator_group) & label_mask).sum())
        numerator_prev = numerator_cases / group_counts.get(numerator_group, np.nan)
        denominator_prev = denominator_cases / group_counts.get(denominator_group, np.nan)
        risk_ratio = numerator_prev / denominator_prev if denominator_prev and denominator_prev > 0 else np.nan
        log2_rr = np.log2(risk_ratio) if risk_ratio and risk_ratio > 0 else np.nan
        rows.append(
            (
                label,
                numerator_cases,
                denominator_cases,
                numerator_prev,
                denominator_prev,
                risk_ratio,
                log2_rr,
            )
        )
    return pd.DataFrame(
        rows,
        columns=[
            "label",
            f"{numerator_group}_cases",
            f"{denominator_group}_cases",
            f"{numerator_group}_prev",
            f"{denominator_group}_prev",
            f"risk_ratio_{numerator_group}_over_{denominator_group}",
            "log2_rr",
        ],
    ).sort_values("log2_rr", ascending=False)


def compute_age_bin_prevalence(
    df: pd.DataFrame,
    labels: list[str],
    age_bin_col: str = "AgeBin",
) -> pd.DataFrame:
    rows = []
    for age_bin, group in df.groupby(age_bin_col, observed=False):
        n_rows = len(group)
        if n_rows == 0:
            continue
        label_sets = group["LabelList"].map(set)
        for label in labels:
            count = int(label_sets.map(lambda labels_set: label in labels_set).sum())
            rows.append((str(age_bin), label, count, count / n_rows))
    return pd.DataFrame(rows, columns=["AgeBin", "Label", "count", "prevalence"])
