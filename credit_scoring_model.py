import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from joblib import dump, load

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTE

from data_preparation import clean_lendingclub_data

# ----------------- CONFIG -----------------
RANDOM_SEED = 42
TEST_SIZE = 0.2

# ----------------- SESSION STATE -----------------
if "model" not in st.session_state:
    st.session_state.model = None
if "features" not in st.session_state:
    st.session_state.features = None

# ----------------- Helper Functions -----------------
def balance_data(X_train, y_train):
    smote = SMOTE(random_state=RANDOM_SEED)
    return smote.fit_resample(X_train, y_train)

def train_model(X_train, y_train):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    return report, roc_auc, cm, y_proba

def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    return fig

def plot_precision_recall(y_test, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(thresholds, precision[:-1], label='Precision')
    ax.plot(thresholds, recall[:-1], label='Recall')
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision-Recall Tradeoff")
    ax.legend()
    plt.tight_layout()
    return fig

def plot_feature_importance(model, feature_names, top_n=15):
    importances = model.named_steps["classifier"].feature_importances_
    fig, ax = plt.subplots(figsize=(6, 4))
    pd.Series(importances, index=feature_names).nlargest(top_n).plot(kind='barh', ax=ax)
    ax.set_title("Top Feature Importances")
    plt.tight_layout()
    return fig

def download_model(model, features):
    buffer = BytesIO()
    dump((model, features), buffer)
    buffer.seek(0)
    st.download_button("📥 Download Trained Model", buffer, file_name="credit_scoring_model.joblib")

# ----------------- UI CONFIG -----------------
st.set_page_config(page_title="Credit Scoring Model Trainer", layout="wide")
st.title("📈 LendingClub Credit Scoring Model Trainer")

# ----------------- TRAINING -----------------
uploaded_file = st.file_uploader("Upload Cleaned or Raw LendingClub CSV File", type=["csv"], key="train_upload")

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(raw_df.head())

    if st.checkbox("Clean and Prepare Data"):
        df = clean_lendingclub_data(raw_df, is_train=True)
        if df is not None:
            st.success("✅ Data cleaned successfully!")
            st.dataframe(df.head())
        else:
            st.stop()
    else:
        df = raw_df

    if st.button("🚀 Train Model"):
        st.info("Splitting and balancing data...")
        X = df.drop(columns=["loan_status"])
        y = df["loan_status"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
        )
        X_train_bal, y_train_bal = balance_data(X_train, y_train)

        st.info("Training model...")
        model = train_model(X_train_bal, y_train_bal)

        st.session_state.model = model
        st.session_state.features = X.columns.tolist()

        st.success("✅ Model trained successfully!")

        st.info("Evaluating model...")
        report, roc_auc, cm, y_proba = evaluate_model(model, X_test, y_test)

        st.markdown("### 📊 Classification Report")
        with st.expander("View Classification Report"):
            st.json(report)

        st.markdown("### 🎯 ROC-AUC Score")
        st.metric(label="ROC-AUC", value=f"{roc_auc:.4f}")

        st.markdown("### 🔷 Confusion Matrix")
        left, center, right = st.columns([1, 2, 1])
        with center:
            st.pyplot(plot_confusion_matrix(cm), use_container_width=False)

        st.markdown("### 📈 Precision-Recall Curve")
        with center:
            st.pyplot(plot_precision_recall(y_test, y_proba), use_container_width=False)

        st.markdown("### 🌟 Feature Importance")
        with center:
            st.pyplot(plot_feature_importance(model, X.columns), use_container_width=False)

        st.markdown("### 💾 Save Trained Model")
        download_model(model, st.session_state.features)

# ------------------ PREDICT FROM RAW FILE -------------------
st.markdown("---")
st.header("🔍 Predict Loan Status from Raw Data")

predict_file = st.file_uploader("Upload CSV File (with some loan_status=None)", type=["csv"], key="predict_upload")
model_file = st.file_uploader("Upload a Trained Model (.joblib)", type=["joblib"], key="model_upload")

if model_file is not None:
    st.session_state.model, st.session_state.features = load(model_file)
    st.success("✅ Model loaded successfully.")

if predict_file:
    df_raw = pd.read_csv(predict_file)
    st.subheader("Uploaded Prediction File")
    st.dataframe(df_raw.head())

    if st.session_state.model is None:
        st.warning("⚠️ No model loaded. Please upload a trained model.")
    else:
        try:
            # Clean data WITHOUT scaling (pipeline handles it)
            df_cleaned = clean_lendingclub_data(df_raw, is_train=False)

            # Use only rows where loan_status is missing
            df_unlabeled = df_raw[df_raw["loan_status"].isna()].copy()

            # Align features (assumes same order as training)
            X_unlabeled = df_cleaned.loc[df_unlabeled.index, st.session_state.features]

            # Model already includes scaler in pipeline
            y_pred = st.session_state.model.predict(X_unlabeled)
            y_proba = st.session_state.model.predict_proba(X_unlabeled)[:, 1]

            df_unlabeled["Predicted loan_status"] = y_pred
            df_unlabeled["Prediction Probability"] = y_proba

            # Format for display
            st.markdown("### 🧠 Predictions")
            st.write(f"Predicted {len(df_unlabeled)} loans")
            st.dataframe(df_unlabeled)

            # Export predictions
            csv_pred = df_unlabeled.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Predictions CSV", data=csv_pred, file_name="loan_predictions.csv", mime="text/csv")

        except KeyError as e:
            st.error(f"❌ Feature mismatch! Missing feature in data: {e}")
        except Exception as ex:
            st.error(f"❌ Unexpected error: {ex}")
