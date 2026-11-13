
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = PROJECT_ROOT / "datasets" / "faceforensics++" / "metadata"
SPLITS_DIR = PROJECT_ROOT / "datasets" / "faceforensics++" / "splits"

INPUT_CSV = METADATA_DIR / "ffpp_manifest.csv"

RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


def create_splits():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Manifest not found: {INPUT_CSV}\n"
            "Build the manifest after adding labelled videos."
        )

    df = pd.read_csv(INPUT_CSV)

    required = {
        "video_path", "label", "manipulation", "source_video_id"
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Manifest contains no videos.")

    if df["source_video_id"].isna().any():
        raise ValueError("Some videos have no source_video_id.")

    if df["source_video_id"].duplicated().sum() == 0:
        raise ValueError(
            "No repeated source IDs found. Check that original and "
            "manipulated videos share the same source_video_id."
        )

    if not abs(TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO - 1.0) < 1e-9:
        raise ValueError("Split ratios must add up to 1.")

    # First split: training vs. remaining data.
    first_split = GroupShuffleSplit(
        n_splits=1,
        train_size=TRAIN_RATIO,
        random_state=RANDOM_STATE,
    )

    train_idx, remaining_idx = next(
        first_split.split(df, groups=df["source_video_id"])
    )

    train_df = df.iloc[train_idx].copy()
    remaining_df = df.iloc[remaining_idx].copy()

    # Second split: validation vs. testing.
    relative_test_ratio = TEST_RATIO / (
        VALIDATION_RATIO + TEST_RATIO
    )

    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=relative_test_ratio,
        random_state=RANDOM_STATE,
    )

    val_idx, test_idx = next(
        second_split.split(
            remaining_df,
            groups=remaining_df["source_video_id"],
        )
    )

    val_df = remaining_df.iloc[val_idx].copy()
    test_df = remaining_df.iloc[test_idx].copy()

    # Verify that no source appears in multiple splits.
    train_groups = set(train_df["source_video_id"])
    val_groups = set(val_df["source_video_id"])
    test_groups = set(test_df["source_video_id"])

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df.to_csv(SPLITS_DIR / "validation.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "test.csv", index=False)

    print("Dataset splits created successfully.")
    print(f"Training:   {len(train_df)} videos")
    print(f"Validation: {len(val_df)} videos")
    print(f"Testing:    {len(test_df)} videos")
    print("Source-video leakage check: PASSED")


if __name__ == "__main__":
    create_splits()
