import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, VotingRegressor, VotingClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
import pickle
import cv2
from PIL import Image

# Safe pyzbar import
PYZBAR_AVAILABLE = False
decode = None

def safe_import_pyzbar():
    global PYZBAR_AVAILABLE, decode
    try:
        from pyzbar.pyzbar import decode
        PYZBAR_AVAILABLE = True
        return True
    except Exception as e:
        print(f"pyzbar import error: {e}")
        print("Please install pyzbar dependencies. On Windows, you may need to install Visual C++ Redistributable.")
        PYZBAR_AVAILABLE = False
        decode = None
        return False

# Try to import pyzbar
safe_import_pyzbar()
import io
import torch
from transformers import ViTForImageClassification, ViTImageProcessor, ViTModel

st.title("Machine Learning & Image Processing Web App")

tab_selection = st.sidebar.radio("Select Mode", ["Tabular ML", "Image Processing"])

#---------------- ML TAB -----------------
if tab_selection == "Tabular ML":
    data_input = st.sidebar.radio("Data Source:", ["Upload your data", "Use example data"])
    df = None
    if data_input == "Upload your data":
        uploaded_file = st.sidebar.file_uploader("Upload CSV/XLSX/TSV", type=["csv", "xlsx", "tsv"])
        if uploaded_file:
            if uploaded_file.name.endswith('csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('xlsx'):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('tsv'):
                df = pd.read_csv(uploaded_file, sep='\t')
    else:
        dataset_name = st.selectbox("Select a default dataset", ["titanic", "tips", "iris"])
        df = sns.load_dataset(dataset_name)

    @st.cache_data
    def load_data(uploaded_file, dataset_name):
        if uploaded_file is not None:
            if uploaded_file.name.endswith('csv'):
                return pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('xlsx'):
                return pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('tsv'):
                return pd.read_csv(uploaded_file, sep='\t')
        elif dataset_name:
            return sns.load_dataset(dataset_name)
        return None

    @st.cache_resource
    def train_model(model, X_train, y_train):
        model.fit(X_train, y_train)
        return model

    if df is not None:
        st.success("Data loaded successfully!")
        st.write(df.head())
    else:
        st.warning("Please upload a dataset or select a default dataset.")
        st.stop()

    if df is not None:
        st.subheader("Missing/Null Values per Column")
        st.write(df.isna().sum())
        st.subheader("Data Shape")
        st.write(df.shape)
        st.subheader("Description")
        st.write(df.describe(include='all'))
        st.subheader("Column Names")
        st.write(df.columns.tolist())
        # Data size validation
        st.subheader("Data Size Information")
        st.write(f"Dataset shape: {df.shape}")
        st.write(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Warn about large datasets
        if df.shape[0] > 10000 or df.shape[1] > 1000:
            st.warning("⚠️ Large dataset detected! This may cause memory issues. Consider:")
            st.write("- Reducing the number of features")
            st.write("- Using a smaller sample of your data")
            st.write("- Choosing memory-efficient models (Decision Tree, Random Forest)")
        
        # Smart feature selection for large datasets
        if df.shape[1] > 100:
            st.subheader("Smart Feature Selection")
            target = st.selectbox("Select target column", df.columns.tolist())
            
            feature_selection_method = st.radio(
                "Choose feature selection method:",
                ["Manual selection", "Top 50 features", "Top 100 features", "Top 200 features"]
            )
            
            if feature_selection_method == "Manual selection":
                available_features = [col for col in df.columns if col != target]
                features = st.multiselect("Select features", available_features, default=available_features[:50])
            else:
                # Use correlation for feature selection
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if target in numeric_cols:
                    numeric_cols.remove(target)
                
                if len(numeric_cols) > 0:
                    correlations = df[numeric_cols].corrwith(df[target]).abs().sort_values(ascending=False)
                    n_features = int(feature_selection_method.split()[1])
                    top_features = correlations.head(n_features).index.tolist()
                    features = st.multiselect("Selected features (based on correlation)", top_features, default=top_features)
                else:
                    available_features = [col for col in df.columns if col != target]
                    features = st.multiselect("Select features", available_features, default=available_features[:50])
        else:
            features = st.multiselect("Select features", df.columns.tolist(), default=df.columns.tolist()[:-1])
            target = st.selectbox("Select target column", [col for col in df.columns if col not in features])
        
        # Feature count validation
        if len(features) > 1000:
            st.error(f"❌ Too many features selected ({len(features)}). Please select fewer than 1000 features to avoid memory issues.")
            st.stop()
        elif len(features) > 100:
            st.warning(f"⚠️ Many features selected ({len(features)}). This may cause memory issues with some models.")

        if pd.api.types.is_numeric_dtype(df[target]):
            st.success("Regression problem")
            problem_type = "regression"
        else:
            st.info("Classification problem")
            problem_type = "classification"

        encoders = {}
        for col in features:
            if df[col].dtype == object or df[col].dtype.name == 'category':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le

        target_encoder = None
        if problem_type == "classification":
            target_encoder = LabelEncoder()
            df[target] = target_encoder.fit_transform(df[target].astype(str))

        df = df.dropna(subset=[target])
        
        # Data sampling for large datasets
        if df.shape[0] > 10000:
            sample_size = st.slider("Sample size for training (to avoid memory issues)", 1000, min(10000, df.shape[0]), min(5000, df.shape[0]))
            df_sample = df.sample(n=sample_size, random_state=42)
            st.info(f"Using a sample of {sample_size} rows for training to avoid memory issues.")
        else:
            df_sample = df
        
        # Memory-efficient imputation
        if df_sample[features].isna().sum().sum() > 0:
            st.info("Imputing missing values...")
            imp = IterativeImputer(max_iter=10, random_state=42)  # Limit iterations for memory efficiency
            df_sample[features] = imp.fit_transform(df_sample[features])

        split_size = st.slider("Test size (%)", 10, 50, 20, step=5)
        X = df_sample[features]
        y = df_sample[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size/100, random_state=42)

        if problem_type == "regression":
            model_name = st.sidebar.selectbox("Regression model", ["All Models", "Linear Regression", "Decision Tree", "Random Forest", "Support Vector Machine"])
            # Memory-efficient model configurations
            if len(features) > 100 or X_train.shape[0] > 5000:
                st.info("Using memory-efficient model configurations for large datasets")
                model_dict = {
                    "Linear Regression": LinearRegression(),
                    "Decision Tree": DecisionTreeRegressor(max_depth=10),
                    "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=1),
                    "Support Vector Machine": SVR(kernel='linear')  # Linear kernel is more memory efficient
                }
            else:
                model_dict = {
                    "Linear Regression": LinearRegression(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "Random Forest": RandomForestRegressor(),
                    "Support Vector Machine": SVR()
                }
        else:
            model_name = st.sidebar.selectbox("Classification model", ["All Models", "Logistic Regression", "Decision Tree", "Random Forest", "Support Vector Machine"])
            # Memory-efficient model configurations
            if len(features) > 100 or X_train.shape[0] > 5000:
                st.info("Using memory-efficient model configurations for large datasets")
                model_dict = {
                    "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear'),
                    "Decision Tree": DecisionTreeClassifier(max_depth=10),
                    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=1),
                    "Support Vector Machine": SVC(kernel='linear', probability=True)  # Linear kernel is more memory efficient
                }
            else:
                model_dict = {
                    "Logistic Regression": LogisticRegression(),
                    "Decision Tree": DecisionTreeClassifier(),
                    "Random Forest": RandomForestClassifier(),
                    "Support Vector Machine": SVC(probability=True)
                }
        
        # If user chose to train all models, loop and compare
        trained_models = {}
        results_rows = []
        best_name = None
        model = None

        if model_name == "All Models":
            with st.spinner("Training all models..."):
                for name, mdl in model_dict.items():
                    try:
                        mdl.fit(X_train, y_train)
                        y_pred_local = mdl.predict(X_test)
                        if problem_type == "regression":
                            mse = mean_squared_error(y_test, y_pred_local)
                            rmse = mse ** 0.5
                            mae = mean_absolute_error(y_test, y_pred_local)
                            r2 = r2_score(y_test, y_pred_local)
                            results_rows.append({"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2})
                        else:
                            accuracy = accuracy_score(y_test, y_pred_local)
                            precision = precision_score(y_test, y_pred_local, average='weighted', zero_division=0)
                            recall = recall_score(y_test, y_pred_local, average='weighted', zero_division=0)
                            f1 = f1_score(y_test, y_pred_local, average='weighted', zero_division=0)
                            row = {"Model": name, "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1}
                            # Optional AUROC for binary
                            if len(set(y_test)) == 2 and hasattr(mdl, "predict_proba"):
                                try:
                                    auc = roc_auc_score(y_test, mdl.predict_proba(X_test)[:, 1])
                                    row["AUROC"] = auc
                                except Exception:
                                    pass
                            results_rows.append(row)
                        trained_models[name] = mdl
                    except Exception as e:
                        st.warning(f"Skipping {name} due to error: {e}")

            st.subheader("Model Comparison")
            if len(results_rows) > 0:
                results_df = pd.DataFrame(results_rows)
                st.write(results_df)
                if problem_type == "regression":
                    best_name = results_df.sort_values(by=["RMSE", "MAE"]).iloc[0]["Model"]
                else:
                    # Prefer F1, then Accuracy
                    sort_cols = ["F1"] + (["Accuracy"] if "Accuracy" in results_df.columns else [])
                    best_name = results_df.sort_values(by=sort_cols, ascending=False).iloc[0]["Model"]
                st.success(f"Best model selected: {best_name}")
                model = trained_models[best_name]
            else:
                st.error("No models could be trained successfully.")
                st.stop()
        else:
            model = model_dict[model_name]
            # Training with progress indicator and error handling
            try:
                with st.spinner(f"Training {model_name}..."):
                    model.fit(X_train, y_train)
                st.success(f"{model_name} trained successfully!")
            except MemoryError:
                st.error("❌ Memory error during training. Please:")
                st.write("- Reduce the number of features")
                st.write("- Use a smaller sample size")
                st.write("- Try Decision Tree or Random Forest models")
                st.stop()
            except Exception as e:
                st.error(f"❌ Error during training: {str(e)}")
                st.write("Try reducing the dataset size or number of features.")
                st.stop()

        # Evaluate selected/best model
        y_pred = model.predict(X_test)
        metrics_dict = {}
        if problem_type == "regression":
            mse = mean_squared_error(y_test, y_pred)
            rmse = mse ** 0.5
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write(f"MSE: {mse:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}, R2: {r2:.3f}")
            metrics_dict = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}
        else:
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            st.write(f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1:.3f}")
            cm = confusion_matrix(y_test, y_pred)
            st.write("Confusion Matrix:")
            st.write(cm)
            metrics_dict = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1}
        if problem_type == "classification":
            if len(set(y_test)) == 2 and hasattr(model, "predict_proba"):
                try:
                    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                    st.write(f"AUROC: {auc:.3f}")
                except Exception:
                    pass

        st.subheader("Evaluation Metrics")
        st.write(metrics_dict)
        st.success(f"Selected model: {best_name if model_name == 'All Models' else model_name}")

        # Optional ensemble using all trained models of the chosen type
        ensemble_model = None
        if model_name == "All Models" and len(trained_models) >= 2:
            if problem_type == "regression":
                estimators = [(n, m) for n, m in trained_models.items() if isinstance(m, (LinearRegression, DecisionTreeRegressor, RandomForestRegressor, SVR))]
                if len(estimators) >= 2:
                    ensemble_model = VotingRegressor(estimators=estimators)
            else:
                estimators = [(n, m) for n, m in trained_models.items() if isinstance(m, (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, SVC))]
                if len(estimators) >= 2:
                    # Use soft voting when possible
                    ensemble_model = VotingClassifier(estimators=estimators, voting='soft', n_jobs=None)

        if ensemble_model is not None:
            if st.checkbox("Enable ensemble prediction (average/soft voting)"):
                try:
                    with st.spinner("Fitting ensemble model..."):
                        ensemble_model.fit(X_train, y_train)
                    y_pred_ens = ensemble_model.predict(X_test)
                    if problem_type == "regression":
                        mse_e = mean_squared_error(y_test, y_pred_ens)
                        rmse_e = mse_e ** 0.5
                        mae_e = mean_absolute_error(y_test, y_pred_ens)
                        r2_e = r2_score(y_test, y_pred_ens)
                        st.info(f"Ensemble -> RMSE: {rmse_e:.3f}, MAE: {mae_e:.3f}, R2: {r2_e:.3f}")
                    else:
                        acc_e = accuracy_score(y_test, y_pred_ens)
                        f1_e = f1_score(y_test, y_pred_ens, average='weighted', zero_division=0)
                        st.info(f"Ensemble -> Accuracy: {acc_e:.3f}, F1: {f1_e:.3f}")
                except Exception as e:
                    st.warning(f"Ensemble training failed: {e}")

        if st.button("Download Model"):
            model_pickle = pickle.dumps(model)
            st.download_button("Download .pkl", model_pickle, file_name="model.pkl")
        if ensemble_model is not None and st.button("Download Ensemble Model"):
            try:
                ens_pickle = pickle.dumps(ensemble_model)
                st.download_button("Download ensemble.pkl", ens_pickle, file_name="ensemble_model.pkl")
            except Exception as e:
                st.warning(f"Unable to serialize ensemble: {e}")

        if st.checkbox("Want to make prediction?"):
            input_dict = {}
            st.subheader("Provide input data:")
            for feature in features:
                input_dict[feature] = st.slider(f"Input for {feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))
            input_df = pd.DataFrame([input_dict])
            for col, le in encoders.items():
                input_df[col] = le.transform(input_df[col].astype(str))
            pred = model.predict(input_df)
            if problem_type == "classification" and target_encoder:
                pred = target_encoder.inverse_transform([int(pred[0])])
            st.subheader("Prediction Result")
            st.write(pred[0])

        if data_input == "Upload your data" and df is not None:
            st.subheader("Select X and Y axis for Visualization")
            X_axis = st.selectbox('Select X-axis', df.columns)
            Y_axis = st.selectbox('Select Y-axis', df.columns)
            plot_type = st.selectbox('Select Plot Type', ('lineplot', 'scatterplot', 'barplot', 'histplot', 'boxplot', 'violinplot', 'countplot',"pairplot"))
            if st.button("Generate Plot"):
                import matplotlib.pyplot as plt
                plt.figure()
                if plot_type == 'lineplot':
                    fig = sns.lineplot(data=df, x=X_axis, y=Y_axis)
                elif plot_type == 'scatterplot':
                    fig = sns.scatterplot(data=df, x=X_axis, y=Y_axis)
                elif plot_type == 'barplot':
                    fig = sns.barplot(data=df, x=X_axis, y=Y_axis)
                elif plot_type == 'histplot':
                    fig = sns.histplot(data=df, x=X_axis)
                elif plot_type == 'boxplot':
                    fig = sns.boxplot(data=df, x=X_axis, y=Y_axis)
                elif plot_type == 'violinplot':
                    fig = sns.violinplot(data=df, x=X_axis, y=Y_axis)
                elif plot_type == 'countplot':
                    fig = sns.countplot(data=df, x=X_axis)
                elif plot_type == 'pairplot':
                    fig = sns.pairplot(df)
                st.pyplot(plt.gcf())

#---------------- IMAGE TAB -----------------
elif tab_selection == "Image Processing":
    st.header("Image Processing & Inference")
    img_source = st.radio("Image Source", ["Upload", "Camera", "CSV (image_link)"])
    img = None
    selected_text = None
    selected_title = None
    if img_source == "Upload":
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            img = Image.open(uploaded_image).convert("RGB")
    elif img_source == "Camera":
        img_file = st.camera_input("Take a Picture")
        if img_file:
            img = Image.open(img_file).convert("RGB")
    elif img_source == "CSV (image_link)":
        st.info("Load image from a CSV column named 'image_link'. Optionally use 'sample_id'.")
        csv_choice = st.radio("CSV Source", ["Upload CSV", "Use dataset/test.csv"]) 
        csv_df = None
        if csv_choice == "Upload CSV":
            csv_file = st.file_uploader("Upload CSV with image_link column", type=["csv"]) 
            if csv_file:
                try:
                    csv_df = pd.read_csv(csv_file)
                except Exception as e:
                    st.error(f"Failed to read CSV: {e}")
        else:
            try:
                csv_df = pd.read_csv("dataset/test.csv")
            except Exception as e:
                st.error(f"Unable to read dataset/test.csv: {e}")
        if csv_df is not None:
            if "image_link" not in csv_df.columns:
                st.error("CSV must contain an 'image_link' column")
            else:
                selector_mode = "Row Index"
                if "sample_id" in csv_df.columns:
                    selector_mode = st.radio("Select by", ["Row Index", "sample_id"]) 
                if selector_mode == "Row Index":
                    row_idx = st.number_input("Row index", min_value=0, max_value=max(0, len(csv_df)-1), value=0, step=1)
                    if len(csv_df) > 0:
                        row = csv_df.iloc[int(row_idx)]
                        url = str(row["image_link"]) if "image_link" in row else None
                        selected_text = str(row["catalog_content"]) if "catalog_content" in csv_df.columns else None
                        selected_title = str(row["sample_id"]) if "sample_id" in csv_df.columns else None
                    else:
                        url = None
                else:
                    sample_ids = csv_df["sample_id"].tolist()
                    sid = st.selectbox("sample_id", sample_ids)
                    sub = csv_df.loc[csv_df["sample_id"] == sid]
                    url = str(sub["image_link"].iloc[0]) if len(sub) > 0 else None
                    selected_text = str(sub["catalog_content"].iloc[0]) if (len(sub) > 0 and "catalog_content" in sub.columns) else None
                    selected_title = str(sid)

                def fetch_image(url_in, retries=3, timeout=10):
                    import requests
                    last_err = None
                    for _ in range(retries):
                        try:
                            headers = {"User-Agent": "Mozilla/5.0"}
                            r = requests.get(url_in, headers=headers, timeout=timeout)
                            if r.status_code == 200:
                                return Image.open(io.BytesIO(r.content)).convert("RGB")
                            last_err = f"HTTP {r.status_code}"
                        except Exception as e:
                            last_err = str(e)
                    raise RuntimeError(last_err or "Failed to download image")

                if url and st.button("Load Image from CSV"):
                    try:
                        img = fetch_image(url)
                    except Exception as e:
                        st.error(f"Failed to load image: {e}")

    if img:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Input Image", use_column_width=True)
        with col2:
            if selected_title:
                st.subheader(f"Sample: {selected_title}")
            if selected_text:
                st.subheader("Catalog Content")
                st.text_area("", selected_text, height=250)
        img_np = np.array(img)
        img_proc = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_proc = cv2.resize(img_proc, (224, 224))
        img_proc = cv2.fastNlMeansDenoisingColored(img_proc, None, 10, 10, 7, 21)
        img_norm = img_proc / 255.0

        if PYZBAR_AVAILABLE:
            barcodes = decode(Image.fromarray(img_np))
            for b in barcodes:
                pts = np.array([b.polygon], np.int32)
                cv2.polylines(img_proc, [pts], True, (0,255,0), 2)
                x, y = b.rect.left, b.rect.top
                cv2.putText(img_proc, b.data.decode('utf-8'), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

            st.subheader("Barcodes Detected:")
            if len(barcodes) == 0:
                st.write("No barcodes found.")
            for b in barcodes:
                st.write(f"Type: {b.type}, Data: {b.data.decode('utf-8')}")
        else:
            st.warning("Barcode detection is not available. Please install pyzbar dependencies.")

        st.subheader("Transformer-based Classification")
        run_inf = st.button("Run Classification/Detection")
        # OCR extraction
        st.subheader("OCR: Extract Text From Image")
        if st.button("Extract Title/Brand/Bullets/Description"):
            try:
                import easyocr
                reader = easyocr.Reader(['en'], gpu=False)
                result = reader.readtext(img_np)
                lines = [t[1] for t in result if isinstance(t, (list, tuple)) and len(t) > 1]
                full_text = "\n".join(lines)
                # Simple heuristics for fields
                title = lines[0] if len(lines) > 0 else ""
                brand = ""
                bullets = []
                description = ""
                # detect brand keyword
                for ln in lines[:10]:
                    if ln.lower().startswith("brand") or "brand" in ln.lower():
                        brand = ln
                        break
                # bullets indicated by hyphen/dot
                for ln in lines:
                    if ln.strip().startswith(("-", "•", "*")):
                        bullets.append(ln)
                # description heuristic: last paragraph-sized block
                if len(lines) > 5:
                    description = " ".join(lines[-10:])
                col_t, col_v = st.columns(2)
                with col_t:
                    st.markdown("**Title**")
                    st.text_area("", title, height=80)
                    st.markdown("**Brand**")
                    st.text_input("", brand)
                with col_v:
                    st.markdown("**Bullet Points**")
                    st.text_area("", "\n".join(bullets) if bullets else "", height=150)
                    st.markdown("**Product Description**")
                    st.text_area("", description, height=150)
            except Exception as e:
                st.warning(f"OCR failed: {e}")
        if run_inf:
            processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
            model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
            inputs = processor(images=img, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            top = torch.topk(probs, k=3)
            for idx, score in zip(top.indices[0], top.values[0]):
                label = model.config.id2label[idx.item()]
                st.write(f"{label}: {score.item():.2f}")

        st.image(cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB), caption="Annotated Image", use_column_width=True)
        # Download annotated image
        pil_annot = Image.fromarray(cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB))
        download_buf = io.BytesIO()
        pil_annot.save(download_buf, format="PNG")
        download_buf.seek(0)
        st.download_button("Download Annotated Image", download_buf, file_name="annotated_image.png", mime="image/png")

        # -------- Price Prediction from Image --------
        st.subheader("Price Prediction from Image")

        @st.cache_resource
        def get_vit_backbone():
            processor_local = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
            backbone = ViTModel.from_pretrained("google/vit-base-patch16-224")
            backbone.eval()
            return processor_local, backbone

        @st.cache_data
        def embed_image_to_vector(pil_image):
            processor_local, backbone = get_vit_backbone()
            inputs_local = processor_local(images=pil_image, return_tensors="pt")
            with torch.no_grad():
                outputs_local = backbone(**inputs_local)
            # Mean-pool token embeddings
            vec = outputs_local.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
            return vec

        # Training small regressor from CSV (image_link, price)
        with st.expander("Train/Load Price Model (CSV with image_link, price)"):
            src_choice = st.radio("Training data source", ["Upload CSV", "Use dataset/train.csv"], horizontal=True)
            csv_for_price = None
            if src_choice == "Upload CSV":
                price_csv = st.file_uploader("Upload CSV containing columns: image_link, price", type=["csv"], key="price_csv_up")
                if price_csv:
                    try:
                        csv_for_price = pd.read_csv(price_csv)
                    except Exception as e:
                        st.error(f"Failed to read CSV: {e}")
            else:
                try:
                    csv_for_price = pd.read_csv("dataset/train.csv")
                except Exception as e:
                    st.error(f"Unable to read dataset/train.csv: {e}")

            if csv_for_price is not None:
                required_cols = {"image_link", "price"}
                if not required_cols.issubset(set([c.lower() for c in csv_for_price.columns])):
                    st.warning("CSV must include 'image_link' and 'price' columns (case-insensitive).")
                else:
                    # Normalize column names
                    cols_lower = {c.lower(): c for c in csv_for_price.columns}
                    col_image = cols_lower["image_link"]
                    col_price = cols_lower["price"]
                    max_rows = st.slider("Max training rows", 50, 1000, 200, step=50)
                    if st.button("Train price regressor"):
                        sample_df = csv_for_price.dropna(subset=[col_image, col_price]).head(max_rows)
                        X_embeds = []
                        y_prices = []

                        def _fetch_image(url_in, retries=2, timeout=8):
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

                        with st.spinner("Embedding images and training regressor..."):
                            for _, row in sample_df.iterrows():
                                try:
                                    im = _fetch_image(row[col_image])
                                    vec = embed_image_to_vector(im)
                                    price_val = float(row[col_price])
                                    if np.isfinite(price_val):
                                        X_embeds.append(vec)
                                        y_prices.append(price_val)
                                except Exception:
                                    # Skip bad rows silently to keep UI clean
                                    pass

                            if len(X_embeds) >= 10:
                                try:
                                    # Use simple linear regression to map embeddings -> price
                                    reg = LinearRegression()
                                    reg.fit(np.vstack(X_embeds), np.array(y_prices))
                                    st.session_state["price_regressor"] = reg
                                    st.success(f"Trained on {len(y_prices)} samples")
                                except Exception as e:
                                    st.error(f"Training failed: {e}")
                            else:
                                st.warning("Not enough valid samples to train (need at least 10).")

        if st.session_state.get("price_regressor") is not None:
            if st.button("Predict Price from Current Image"):
                try:
                    vec_cur = embed_image_to_vector(img)
                    pred_price = float(st.session_state["price_regressor"].predict(vec_cur.reshape(1, -1))[0])
                    st.success(f"Estimated Price: {pred_price:.2f}")
                except Exception as e:
                    st.error(f"Price prediction failed: {e}")

        # -------- Batch generation of test_out.csv (no external price lookup) --------
        with st.expander("Batch: Generate test_out.csv (uses only provided train/test data)"):
            st.markdown("This will train simple regressors on provided train data and predict prices for test samples.")
            if st.button("Run batch training + inference"):
                try:
                    train_df = pd.read_csv("dataset/train.csv")
                    test_df = pd.read_csv("dataset/test.csv")
                except Exception as e:
                    st.error(f"Failed to read train/test CSVs: {e}")
                    train_df = None
                    test_df = None

                if train_df is not None and test_df is not None:
                    # Column resolution (case-insensitive)
                    def resolve_col(df, name):
                        lower_map = {c.lower(): c for c in df.columns}
                        return lower_map.get(name.lower())

                    col_sid_tr = resolve_col(train_df, "sample_id") or resolve_col(train_df, "id")
                    col_img_tr = resolve_col(train_df, "image_link")
                    col_txt_tr = resolve_col(train_df, "catalog_content")
                    col_price_tr = resolve_col(train_df, "price")

                    col_sid_te = resolve_col(test_df, "sample_id") or resolve_col(test_df, "id")
                    col_img_te = resolve_col(test_df, "image_link")
                    col_txt_te = resolve_col(test_df, "catalog_content")

                    if not (col_sid_tr and col_img_tr and col_price_tr and col_sid_te and col_img_te):
                        st.error("Required columns missing. Need sample_id/image_link/price in train and sample_id/image_link in test.")
                    else:
                        # Prepare text model
                        text_vectorizer = None
                        text_model = None
                        train_text = []
                        y_text = []
                        if col_txt_tr and col_txt_te:
                            train_text_series = train_df[col_txt_tr].astype(str).fillna("")
                            if train_text_series.str.len().sum() > 0:
                                text_vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
                                X_text = text_vectorizer.fit_transform(train_text_series)
                                try:
                                    y_text = train_df[col_price_tr].astype(float).values
                                    text_model = Ridge(alpha=1.0)
                                    text_model.fit(X_text, y_text)
                                except Exception:
                                    text_vectorizer = None
                                    text_model = None

                        # Prepare image model
                        img_model = None
                        X_img = []
                        y_img = []
                        mean_price = float(train_df[col_price_tr].astype(float).mean()) if len(train_df) else 0.0

                        def _fetch_image_batch(url_in, retries=2, timeout=8):
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

                        # Cap rows for runtime
                        max_train_rows = 2000
                        use_tr = train_df.dropna(subset=[col_img_tr, col_price_tr]).head(max_train_rows)
                        with st.spinner("Training image and/or text regressors..."):
                            # Image embeddings for a subset
                            for _, row in use_tr.iterrows():
                                try:
                                    im = _fetch_image_batch(row[col_img_tr])
                                    vec = embed_image_to_vector(im)
                                    price_val = float(row[col_price_tr])
                                    if np.isfinite(price_val):
                                        X_img.append(vec)
                                        y_img.append(price_val)
                                except Exception:
                                    pass
                            if len(X_img) >= 10:
                                try:
                                    img_model = LinearRegression()
                                    img_model.fit(np.vstack(X_img), np.array(y_img))
                                except Exception:
                                    img_model = None

                        # Inference on test
                        preds = []
                        with st.spinner("Running inference on test set..."):
                            for _, row in test_df.iterrows():
                                sid_val = row[col_sid_te]
                                pred_components = []
                                # Image-based prediction
                                if img_model is not None:
                                    try:
                                        im = _fetch_image_batch(row[col_img_te])
                                        vec = embed_image_to_vector(im)
                                        p_img = float(img_model.predict(vec.reshape(1, -1))[0])
                                        if np.isfinite(p_img):
                                            pred_components.append(p_img)
                                    except Exception:
                                        pass
                                # Text-based prediction
                                if text_model is not None and text_vectorizer is not None and col_txt_te:
                                    try:
                                        txt = str(row[col_txt_te]) if pd.notna(row[col_txt_te]) else ""
                                        Xte = text_vectorizer.transform([txt])
                                        p_txt = float(text_model.predict(Xte)[0])
                                        if np.isfinite(p_txt):
                                            pred_components.append(p_txt)
                                    except Exception:
                                        pass
                                # Combine or fallback
                                if len(pred_components) == 0:
                                    pred_price_row = mean_price
                                else:
                                    pred_price_row = float(np.mean(pred_components))
                                preds.append((sid_val, pred_price_row))

                        # Write CSV to disk and provide download
                        out_df = pd.DataFrame(preds, columns=["sample_id", "price"])
                        out_path = "dataset/test_out.csv"
                        try:
                            out_df.to_csv(out_path, index=False)
                            st.success(f"Saved predictions to {out_path}")
                        except Exception as e:
                            st.warning(f"Failed to save to disk: {e}")

                        # Download button
                        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                        st.download_button("Download test_out.csv", csv_bytes, file_name="test_out.csv", mime="text/csv")