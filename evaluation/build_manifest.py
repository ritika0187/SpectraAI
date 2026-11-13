
from pathlib import Path
import csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "faceforensics++" / "raw_download"
METADATA_DIR = PROJECT_ROOT / "datasets" / "faceforensics++" / "metadata"

OUTPUT_CSV = METADATA_DIR / "ffpp_manifest.csv"

ORIGINAL_DIR = (
    DATASET_ROOT
    / "original_sequences"
    / "youtube"
    / "c40"
    / "videos"
)

FAKE_DIR = (
    DATASET_ROOT
    / "manipulated_sequences"
    / "Deepfakes"
    / "c40"
    / "videos"
)


def build_manifest():
    original_files = sorted(ORIGINAL_DIR.glob("*.mp4"))
    fake_files = sorted(FAKE_DIR.glob("*.mp4"))

    if not original_files or not fake_files:
        raise FileNotFoundError(
            "Original or manipulated videos were not found. "
            "Check the dataset directories."
        )

    original_ids = {file.stem for file in original_files}

    # Build connected source groups to prevent data leakage.
    parent = {video_id: video_id for video_id in original_ids}

    def find(video_id):
        while parent[video_id] != video_id:
            parent[video_id] = parent[parent[video_id]]
            video_id = parent[video_id]
        return video_id

    def union(first, second):
        root_first = find(first)
        root_second = find(second)

        if root_first != root_second:
            parent[root_second] = root_first

    fake_pairs = []

    for fake_file in fake_files:
        parts = fake_file.stem.split("_")

        if len(parts) != 2:
            raise ValueError(
                f"Unexpected manipulated filename: {fake_file.name}"
            )

        first, second = parts

        if first not in original_ids or second not in original_ids:
            raise ValueError(
                f"Missing corresponding original video for: "
                f"{fake_file.name}"
            )

        union(first, second)
        fake_pairs.append((fake_file, first, second))

    # Give each connected component a stable group ID.
    components = {}

    for video_id in original_ids:
        root = find(video_id)
        components.setdefault(root, []).append(video_id)

    group_ids = {}

    for members in components.values():
        group_id = "_".join(sorted(members, key=int))

        for video_id in members:
            group_ids[video_id] = group_id

    rows = []

    for original_file in original_files:
        video_id = original_file.stem

        rows.append({
            "video_path": str(original_file.resolve()),
            "label": "real",
            "manipulation": "original",
            "source_video_id": group_ids[video_id],
        })

    for fake_file, first, second in fake_pairs:
        rows.append({
            "video_path": str(fake_file.resolve()),
            "label": "fake",
            "manipulation": "Deepfakes",
            "source_video_id": group_ids[first],
        })

    rows.sort(key=lambda row: row["video_path"])

    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "video_path",
                "label",
                "manipulation",
                "source_video_id",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Manifest created successfully.")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Total videos: {len(rows)}")
    print(f"Real videos: {sum(r['label'] == 'real' for r in rows)}")
    print(f"Fake videos: {sum(r['label'] == 'fake' for r in rows)}")
    print(f"Source groups: {len(components)}")


if __name__ == "__main__":
    build_manifest()
