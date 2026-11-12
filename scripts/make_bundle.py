"""Package the trained model + processed data into one zip that others can run without retraining.

    python scripts/make_bundle.py                 # -> data/hm_ready_to_run.zip

Contents (unzip into the repo's data/ folder -> data/hm_project/...):
  week2_model/            trained tower weights, item embeddings, FAISS index, model config
  *.parquet / *.json      the processed tables the feature store and the API need
  images/                 thumbnails of the most popular articles (the rest show a colour swatch)

Not included: the raw Kaggle CSVs and the 30 GB of original photos.
"""
import argparse
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402

TABLES = ["item_table.parquet", "user_table.parquet", "user_history.parquet", "article_details.parquet",
          "interactions_train.parquet", "interactions_test.parquet", "vocabularies.json", "fallback_top12.json"]


def thumbnails(image_zip, article_ids, size=(360, 480)):
    from PIL import Image
    with zipfile.ZipFile(image_zip) as z:
        names = set(z.namelist())
        for article_id in article_ids:
            name = f"images/{article_id[:3]}/{article_id}.jpg"
            if name not in names:
                continue
            img = Image.open(io.BytesIO(z.read(name))).convert("RGB")
            img.thumbnail(size)
            buffer = io.BytesIO()
            img.save(buffer, "JPEG", quality=80, optimize=True)
            yield f"{article_id[:3]}/{article_id}.jpg", buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(ROOT / "data" / "hm_ready_to_run.zip"))
    parser.add_argument("--images", type=int, default=6000, help="thumbnails of the N most popular articles")
    args = parser.parse_args()

    data, model = Path(config.DATA_PATH), Path(config.MODEL_DIR)
    info = json.loads((data / "build_info.json").read_text())
    started = time.time()
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(model.iterdir()):
            z.write(f, f"hm_project/week2_model/{f.name}")
        for name in TABLES:
            z.write(data / name, f"hm_project/{name}")
        # build info without local paths of this computer
        public = {k: v for k, v in info.items() if k not in ("raw_source", "images_source")}
        public.update({"images_source": None, "bundle": "scripts/make_bundle.py",
                       "bundled_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        z.writestr("hm_project/build_info.json", json.dumps(public, indent=2))

        n_images = 0
        if args.images and info.get("images_source") and Path(info["images_source"]).is_file():
            items = pd.read_parquet(data / "item_table.parquet", columns=["article_id", "purchases_last_28d"])
            top = items.sort_values("purchases_last_28d", ascending=False)["article_id"].head(args.images).tolist()
            top = list(dict.fromkeys(json.loads((data / "fallback_top12.json").read_text()) + top))
            for name, jpeg in thumbnails(info["images_source"], top):
                z.writestr(f"hm_project/images/{name}", jpeg, compress_type=zipfile.ZIP_STORED)
                n_images += 1
                if n_images % 1000 == 0:
                    print(f"  thumbnails: {n_images:,}", flush=True)
        else:
            print("no Kaggle zip with photos found - bundle has no thumbnails")

    size = os.path.getsize(args.out) / 1e6
    print(f"bundle written: {args.out} ({size:.0f} MB, {n_images:,} thumbnails, {time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
