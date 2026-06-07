from __future__ import annotations

import numpy as np
import pandas as pd


def compute_patient_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    patient_visit_counts = df["Patient ID"].value_counts()
    df_sorted = df.sort_values(["Patient ID", "FollowUpNum"])

    stable_patients = 0
    varied_patients = 0
    nf_to_finding_patients = 0
    finding_to_nf_patients = 0
    both_direction_patients = 0
    consecutive_pairs = 0
    same_labelset_pairs = 0
    new_label_pairs = 0

    for _, group in df_sorted.groupby("Patient ID"):
        if len(group) < 2:
            continue
        labels_seq = group["Finding Labels"].tolist()
        if len(set(labels_seq)) == 1:
            stable_patients += 1
        else:
            varied_patients += 1

        has_up = False
        has_down = False
        for labels_a, labels_b in zip(labels_seq[:-1], labels_seq[1:]):
            consecutive_pairs += 1
            same_labelset_pairs += int(labels_a == labels_b)

            set_a = set(labels_a.split("|"))
            set_b = set(labels_b.split("|"))
            new_label_pairs += int(len(set_b - set_a) > 0)

            has_up = has_up or (labels_a == "No Finding" and labels_b != "No Finding")
            has_down = has_down or (labels_a != "No Finding" and labels_b == "No Finding")

        nf_to_finding_patients += int(has_up)
        finding_to_nf_patients += int(has_down)
        both_direction_patients += int(has_up and has_down)

    rows = [
        ("unique_patients", int(df["Patient ID"].nunique())),
        ("patients_with_multivisit", int((patient_visit_counts > 1).sum())),
        ("stable_patients_same_labelset", stable_patients),
        ("varied_patients", varied_patients),
        ("nf_to_finding_patients", nf_to_finding_patients),
        ("finding_to_nf_patients", finding_to_nf_patients),
        ("both_direction_patients", both_direction_patients),
        ("consecutive_visit_pairs", consecutive_pairs),
        ("same_labelset_pair_ratio", same_labelset_pairs / consecutive_pairs if consecutive_pairs else np.nan),
        ("new_label_appears_pair_ratio", new_label_pairs / consecutive_pairs if consecutive_pairs else np.nan),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def simulate_split_leakage(
    df: pd.DataFrame,
    multi_hot: pd.DataFrame,
    repeats: int = 80,
    seed: int = 123,
    train_frac: float = 0.8,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[tuple[str, int, float, float]] = []
    patients = df["Patient ID"].unique()

    for _ in range(repeats):
        idx = np.arange(len(df))
        rng.shuffle(idx)
        cut = int(train_frac * len(df))
        train_idx, val_idx = idx[:cut], idx[cut:]
        train_patients = set(df.iloc[train_idx]["Patient ID"])
        val_patients = set(df.iloc[val_idx]["Patient ID"])
        overlap = len(train_patients & val_patients)
        denominator = min(len(train_patients), len(val_patients))
        overlap_ratio = overlap / denominator if denominator else np.nan
        drift = (multi_hot.iloc[train_idx].mean(axis=0) - multi_hot.iloc[val_idx].mean(axis=0)).abs().mean()
        rows.append(("image_random", overlap, overlap_ratio, float(drift)))

        shuffled_patients = patients.copy()
        rng.shuffle(shuffled_patients)
        cut_patients = int(train_frac * len(shuffled_patients))
        train_patient_set = set(shuffled_patients[:cut_patients])
        val_patient_set = set(shuffled_patients[cut_patients:])
        train_mask = df["Patient ID"].isin(train_patient_set)
        val_mask = df["Patient ID"].isin(val_patient_set)
        drift = (multi_hot.loc[train_mask].mean(axis=0) - multi_hot.loc[val_mask].mean(axis=0)).abs().mean()
        rows.append(("patient_wise", 0, 0.0, float(drift)))

    return pd.DataFrame(
        rows,
        columns=["split_type", "patient_overlap", "overlap_ratio", "mean_abs_label_prev_drift"],
    )

