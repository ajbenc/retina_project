import pandas as pd

from src.retina_embeddings_dataset import load_retina_embeddings_dataset


def test_load_mbrset_merges_and_derives_tasks(tmp_path):
    # Embeddings
    emb_path = tmp_path / "emb.csv"
    pd.DataFrame(
        {
            "ImageName": ["1.1.jpg", "1.3.jpg", "2.1.jpg"],
            "0": [0.1, 0.2, 0.3],
            "1": [1.1, 1.2, 1.3],
        }
    ).to_csv(emb_path, index=False)

    # Labels
    labels_path = tmp_path / "labels.csv"
    pd.DataFrame(
        {
            "patient": [1, 1, 2],
            "age": [50, 50, 60],
            "sex": [1, 1, 0],
            "smoking": ["no", "yes", "no"],
            "file": ["1.1.jpg", "1.3.jpg", "2.1.jpg"],
            "final_icdr": [0, 2, 4],
            "final_edema": ["no", "yes", "no"],
        }
    ).to_csv(labels_path, index=False)

    ds = load_retina_embeddings_dataset(
        dataset="mbrset", embeddings_csv_path=emb_path, labels_csv_path=labels_path, view="macula"
    )

    # macula view should keep .1 and .3, so all three rows match the regex (1.1, 1.3, 2.1)
    assert len(ds.df) == 3

    # Tasks
    # 0 -> any_dr=0, referable=0, 3class=0
    row0 = ds.df.loc[ds.df["image_name"] == "1.1.jpg"].iloc[0]
    assert int(row0["task_any_dr"]) == 0
    assert int(row0["task_referable"]) == 0
    assert int(row0["task_3class"]) == 0

    # 2 + edema yes -> any_dr=1, referable=1, 3class=1
    row1 = ds.df.loc[ds.df["image_name"] == "1.3.jpg"].iloc[0]
    assert int(row1["task_any_dr"]) == 1
    assert int(row1["task_referable"]) == 1
    assert int(row1["task_3class"]) == 1

    # 4 -> any_dr=1, referable=1, 3class=2
    row2 = ds.df.loc[ds.df["image_name"] == "2.1.jpg"].iloc[0]
    assert int(row2["task_any_dr"]) == 1
    assert int(row2["task_referable"]) == 1
    assert int(row2["task_3class"]) == 2

    # Extra binary targets should be normalized to 0/1
    assert int(ds.df.loc[ds.df["image_name"] == "1.3.jpg"].iloc[0]["smoking"]) == 1


def test_load_brset_strips_extension_for_join(tmp_path):
    emb_path = tmp_path / "emb.csv"
    pd.DataFrame(
        {
            "ImageName": ["img00001.jpg", "img00002.jpg"],
            "0": [0.1, 0.2],
        }
    ).to_csv(emb_path, index=False)

    labels_path = tmp_path / "labels.csv"
    pd.DataFrame(
        {
            "image_id": ["img00001", "img00002"],
            "patient_id": [1, 1],
            "patient_age": [48, 48],
            "patient_sex": [1, 1],
            "DR_ICDR": [0, 3],
            "macular_edema": [0, 1],
            "diabetes": [1, 0],
        }
    ).to_csv(labels_path, index=False)

    ds = load_retina_embeddings_dataset(dataset="brset", embeddings_csv_path=emb_path, labels_csv_path=labels_path)
    assert len(ds.df) == 2

    # Second row has grade 3 and edema 1 => any_dr=1, referable=1, 3class=1
    row = ds.df.loc[ds.df["image_name"] == "img00002.jpg"].iloc[0]
    assert int(row["task_any_dr"]) == 1
    assert int(row["task_referable"]) == 1
    assert int(row["task_3class"]) == 1

    # Extra binary targets should be normalized to 0/1
    assert int(row["diabetes"]) == 0
