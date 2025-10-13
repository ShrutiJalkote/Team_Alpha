# Advanced ML & Image Processing App

**Team Members:**
- Sushant Nichat 
- Vedant Tangadkar
- Krushna Borase
- Shruti Jalkote

## Overview

Interactive Streamlit app for tabular ML and image processing. Includes:
- Tabular ML (regression/classification) with feature selection and evaluation
- Image processing (ViT classification, denoising, optional barcode, OCR)
- Image-to-price training and prediction
- Batch generator to produce `dataset/test_out.csv`

## Key Features

### Tabular ML
- File upload (CSV/XLSX/TSV) or built-in datasets (titanic/tips/iris)
- Missing-value report, describe, shape, memory usage
- Correlation-based feature selection for large feature sets
- Iterative imputation; label encoding for categoricals
- Auto detect regression vs classification

### Models
- Regression: Linear, Decision Tree, Random Forest, SVR
- Classification: Logistic, Decision Tree, Random Forest, SVC

### Evaluation
- Regression: MSE, RMSE, MAE, R²
- Classification: Accuracy, Precision, Recall, F1, AUROC
- Confusion matrix; automatic best model highlighting

### Image Processing
- Upload/Camera/CSV image loader
- Vision Transformer (ViT) classification (top‑k)
- Denoising and preprocessing; annotated image download
- Optional barcode via pyzbar; OCR via easyocr

### Image→Price & Batch
- UI: Train image-price regressor (ViT embeddings + Linear Regression) from CSV (`image_link, price`) and predict current image price
- CLI: Generate `dataset/test_out.csv` matching `sample_test_out.csv` using only provided data

### 🎯 **Advanced Capabilities**
- **Model Download**: Export trained models in Pickle format
- **Prediction Interface**: Interactive sliders for single predictions
- **Data Visualization**: Multiple plot types (line, scatter, bar, histogram, box, violin, count, pair plots)
- **Error Handling**: Comprehensive error management with helpful suggestions
- **Memory Management**: Automatic optimization for large datasets

## Installation & Setup

Prerequisites: Python 3.8+, Windows/Linux/macOS

Quick Start
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
pip install -r requirements.txt
streamlit run Alpha.py
```
Open `http://localhost:8501`.

### Command-line batch (no UI)
```bash
# Text-only (TF‑IDF + Ridge on catalog_content)
python generate_test_out.py --text-only
# Image + text ensemble (ViT + Linear Regression, averaged with text)
python generate_test_out.py
```
Output: `dataset/test_out.csv` with header `sample_id,price`.

## Project Structure

```
ML/
├── Alpha.py                    # 🎯 Streamlit app (tabular + image)
├── generate_test_out.py        # 🔧 Batch generator for dataset/test_out.csv
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── venv/                      # Virtual environment
```

## Requirements

Core: streamlit, pandas, numpy, scikit-learn, seaborn, matplotlib, opencv-python, Pillow

Optional: transformers, torch, easyocr, pyzbar (barcode; needs VC++ on Windows)

## Submission Guide (Challenge)
1) `test_out.csv` (exact format as `sample_test_out.csv`)
- UI: Image tab → "Batch: Generate test_out.csv" → Run → Download
- CLI: `python generate_test_out.py` (or `--text-only`)

2) 1‑page methodology (see `Documentation_template.md`)
- Methodology: preprocessing, validation assumptions, runtime caps
- Models: TF‑IDF + Ridge (text); ViT embeddings + Linear Regression (image)
- Features: n‑grams(1–2), max_features=20k; ViT 224×224, mean‑pool tokens
- Ensemble: average of available text/image predictions
- Constraints: strictly no external price lookup/scraping/APIs/databases
- Fallback: train mean price if no signal

## Troubleshooting
Memory errors: reduce sample/features; prefer linear/tree models

Import errors: install optional deps as needed (`transformers`, `torch`, `easyocr`, `pyzbar`). For `pyzbar` on Windows, install Visual C++ Redistributable. Barcode is optional.

Large datasets: use feature selection and sampling for speed/memory

## License
MIT License


