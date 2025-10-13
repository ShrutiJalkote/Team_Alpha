import io
import os
import sys
import numpy as np
import pandas as pd

from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LinearRegression

import torch
from transformers import ViTImageProcessor, ViTModel


def resolve_col(df: pd.DataFrame, name: str):
    lower_map = {c.lower(): c for c in df.columns}
    return lower_map.get(name.lower())


def get_vit():
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTModel.from_pretrained("google/vit-base-patch16-224")
    model.eval()
    return processor, model


def embed_image(pil_image: Image.Image, processor, model) -> np.ndarray:
    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    vec = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
    return vec


def fetch_image(url_in: str, retries: int = 2, timeout: int = 8) -> Image.Image:
    import requests
    last_err = None
    for _ in range(retries):
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            r = requests.get(str(url_in), headers=headers, timeout=timeout)
            if r.status_code == 200:
                return Image.open(io.BytesIO(r.content)).convert("RGB")
            last_err = f"HTTP {r.status_code}"
        except Exception as e:
            last_err = str(e)
    raise RuntimeError(last_err or "Failed to download image")


def main(argv: list[str]) -> int:
    # Args: [--text-only]
    text_only = "--text-only" in argv
    train_path = os.path.join("dataset", "train.csv")
    test_path = os.path.join("dataset", "test.csv")
    out_path = os.path.join("dataset", "test_out.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    col_sid_tr = resolve_col(train_df, "sample_id") or resolve_col(train_df, "id")
    col_img_tr = resolve_col(train_df, "image_link")
    col_txt_tr = resolve_col(train_df, "catalog_content")
    col_price_tr = resolve_col(train_df, "price")

    col_sid_te = resolve_col(test_df, "sample_id") or resolve_col(test_df, "id")
    col_img_te = resolve_col(test_df, "image_link")
    col_txt_te = resolve_col(test_df, "catalog_content")

    if not (col_sid_tr and col_price_tr and col_sid_te):
        raise RuntimeError("Required columns missing in train/test.")

    # Text model
    text_vectorizer = None
    text_model = None
    if col_txt_tr is not None and col_txt_te is not None:
        train_text_series = train_df[col_txt_tr].astype(str).fillna("")
        if train_text_series.str.len().sum() > 0:
            text_vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
            X_text = text_vectorizer.fit_transform(train_text_series)
            y = train_df[col_price_tr].astype(float).values
            text_model = Ridge(alpha=1.0)
            text_model.fit(X_text, y)

    # Image model (optional)
    img_model = None
    processor = None
    vit = None
    mean_price = float(train_df[col_price_tr].astype(float).mean()) if len(train_df) else 0.0

    if not text_only and col_img_tr is not None and col_img_te is not None:
        try:
            processor, vit = get_vit()
            X_img = []
            y_img = []
            max_rows = 2000
            use_tr = train_df.dropna(subset=[col_img_tr, col_price_tr]).head(max_rows)
            for _, row in use_tr.iterrows():
                try:
                    im = fetch_image(row[col_img_tr])
                    vec = embed_image(im, processor, vit)
                    price_val = float(row[col_price_tr])
                    if np.isfinite(price_val):
                        X_img.append(vec)
                        y_img.append(price_val)
                except Exception:
                    pass
            if len(X_img) >= 10:
                img_model = LinearRegression()
                img_model.fit(np.vstack(X_img), np.array(y_img))
        except Exception:
            img_model = None

    # Inference
    preds = []
    for _, row in test_df.iterrows():
        sid_val = row[col_sid_te]
        components = []
        # text
        if text_model is not None and text_vectorizer is not None and col_txt_te is not None:
            try:
                txt = str(row[col_txt_te]) if pd.notna(row[col_txt_te]) else ""
                Xte = text_vectorizer.transform([txt])
                p_txt = float(text_model.predict(Xte)[0])
                if np.isfinite(p_txt):
                    components.append(p_txt)
            except Exception:
                pass
        # image
        if img_model is not None and processor is not None and vit is not None and col_img_te is not None and not text_only:
            try:
                im = fetch_image(row[col_img_te])
                vec = embed_image(im, processor, vit)
                p_img = float(img_model.predict(vec.reshape(1, -1))[0])
                if np.isfinite(p_img):
                    components.append(p_img)
            except Exception:
                pass
        # combine
        pred = float(np.mean(components)) if components else mean_price
        preds.append((sid_val, pred))

    out_df = pd.DataFrame(preds, columns=["sample_id", "price"])
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(out_df)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


