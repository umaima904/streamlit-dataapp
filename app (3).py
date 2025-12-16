import math
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality
from scipy.stats import ks_2samp, gaussian_kde, wasserstein_distance
from datetime import timedelta
import warnings
import logging
import multiprocessing
import torch
import sdv
import traceback
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from PIL import Image
# -------------------------------------------------
# put page config FIRST streamlit command in script
# -------------------------------------------------
st.set_page_config(layout="wide")

# ---------------------------
# Config / Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# # Ignore only the Streamlit ScriptRunContext warning
# warnings.filterwarnings(
#     "ignore",
#     message="Thread 'MainThread': missing ScriptRunContext!.*",
# )


# Custom CSS for light theme with specified colors and font
st.markdown("""
<style>
/* Main container */
.stApp {
    background-color: #FFF8F5; /* Soft peach white background */
    color: #222222; /* Almost black text */
    font-family: 'Inter', sans-serif;
}

/* Navigation buttons */
.stButton>button {
    background-color: #FF8566; /* Sunset coral primary color */
    color: #FFFFFF; /* White text for contrast */
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: bold;
    border: none;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background-color: #E67050; /* Slightly darker coral on hover */
    transform: scale(1.05);
}

/* Sidebar styling */
.sidebar .sidebar-content {
    background-color: #FFEFEA; /* Light coral tint secondary background */
    padding: 20px;
    border-radius: 10px;
    color: #222222; /* Almost black text */
    font-family: 'Inter', sans-serif;
}

/* Headers */
h1, h2, h3 {
    color: #FF8566; /* Sunset coral for headers */
    font-family: 'Inter', sans-serif;
}

/* Success, Warning, Error messages */
.stSuccess {
    background-color: #2A6B8F; /* Darker blue for success */
    color: #FFFFFF; /* White text for contrast */
    border-radius: 5px;
    padding: 10px;
}
.stWarning {
    background-color: #D97706; /* Darker amber for warning */
    color: #FFFFFF; /* White text for contrast */
    border-radius: 5px;
    padding: 10px;
}
.stError {
    background-color: #B91C1C; /* Darker red for error */
    color: #FFFFFF; /* White text for contrast */
    border-radius: 5px;
    padding: 10px;
}

/* Input widgets */
.stSelectbox, .stSlider, .stNumberInput, .stCheckbox {
    background-color: #FFEFEA; /* Light coral tint background for inputs */
    border-radius: 5px;
    padding: 5px;
    border: 1px solid #E6D8D3; /* Slightly darker border */
    color: #222222; /* Almost black text */
    font-family: 'Inter', sans-serif;
}
.stSelectbox select, .stNumberInput input, .stSlider div, .stCheckbox label {
    color: #222222; /* Almost black text for input content */
}

/* Dataframe styling */
.stDataFrame {
    background-color: #FFEFEA; /* Light coral tint background for dataframes */
    border-radius: 5px;
    border: 1px solid #E6D8D3;
    color: #222222; /* Almost black text */
}
.stDataFrame table {
    color: #222222; /* Almost black text for table content */
    background-color: #FFEFEA; /* Light coral tint for table */
}

/* Download button */
.download-btn {
    background-color: #6B728E; /* Grayish-blue for download button */
    color: #FFFFFF; /* White text for contrast */
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: bold;
    border: none;
    transition: all 0.3s ease;
}
.download-btn:hover {
    background-color: #4A5568; /* Darker shade on hover */
    transform: scale(1.05);
}

/* General text */
body, p, div, .stMarkdown, .stText {
    font-family: 'Inter', sans-serif;
    color: #222222; /* Almost black text for general content */
}

/* Ensure markdown text is readable */
.stMarkdown p, .stMarkdown li {
    color: #222222; /* Almost black text for markdown */
}

/* Plot labels and titles */
.stPlotlyChart, .stPyplot {
    background-color: #FFEFEA; /* Light coral tint background for plots */
    border-radius: 5px;
    padding: 10px;
}
.stPlotlyChart text, .stPyplot text, .stPlotlyChart .js-plotly-plot, .stPyplot .matplotlib-text {
    color: #222222 !important; /* Almost black text for plot labels */
}

/* Ensure table headers and cells are almost black */
table, th, td {
    color: #222222 !important;
    background-color: #FFEFEA !important;
}

/* Ensure Streamlit-specific elements like captions are almost black */
.stCaption {
    color: #222222;
}

/* Ensure placeholder text in inputs is visible */
::placeholder {
    color: #4B5563; /* Slightly lighter for placeholder text */
    opacity: 1;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Function to set page
def set_page(page_name):
    st.session_state.page = page_name

# Top navigation buttons
st.markdown("### Data Synthesis and Validation App")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🏠 Home"):
        set_page('Home')
with col2:
    if st.button("❓ Help"):
        set_page('Help')
with col3:
    if st.button("⚙ Generate"):
        set_page('Generate')


import sys
import os
import base64

def get_logo_path():
    # When packaged with PyInstaller
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, "logo.png")
    # When running normally
    return os.path.join(os.path.dirname(__file__), "logo.png")

def load_logo():
    logo_path = get_logo_path()
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

logo_b64 = load_logo()

# Show logo only on Home page
if st.session_state.page == "Home" and logo_b64:
    st.markdown(
        f"""
        <style>
            .top-right-logo-container {{
                position: fixed;
                top: 10px;      
                right: 15px;    
                z-index: 2000;
            }}

            .top-right-logo-container img {{
                width: 260px;         
                height: auto;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                border: 1px solid rgba(0,0,0,0.08);
            }}
        </style>

        <div class="top-right-logo-container">
            <a href="https://yourwebsite.com" target="_blank">
                <img src="data:image/png;base64,{logo_b64}" />
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )


# Page content based on session state
if st.session_state.page == 'Home':
    st.title("🌟 Data Synthesis and Validation App")
    st.markdown("""
    ### Welcome to the Data Synthesis and Validation App! 🎉
    
    This app allows you to generate synthetic data using advanced models from the SDV library. It supports fast-mode preprocessing for large datasets and provides detailed validation metrics to ensure high-quality synthetic data.
    
    *Models Used:*
    - *CTGAN* 🧠: A neural network-based generative adversarial network (GAN) model for synthesizing tabular data. It's flexible and captures complex relationships but may require more training time.
    - *GaussianCopula* 📊: A statistical model that uses Gaussian copulas to model dependencies between columns. It's faster and works well for datasets with Gaussian-like distributions.
    
    🚀 Navigate to the *Generate* page to upload your data and start synthesizing!  
    ❓ For detailed instructions, visit the *Help* page.
    """, unsafe_allow_html=True)

elif st.session_state.page == 'Help':
    st.title("❓ Help & Instructions")
    st.markdown("""
    ### How to Use the App 🗺
    
    This app helps you generate and validate synthetic data with ease. Follow these steps to get started:
    
    1. *Navigate to Generate Page*:
       - Click the *⚙ Generate* button in the top navigation bar.
    
    2. *Upload Your Data* 📂:
       - Select the file type (Excel or CSV).
       - Upload your file using the file uploader.
       - Preview your data and check memory usage.
    
    3. *Configure Performance Options* (Sidebar) ⚙:
       - Enable *Fast mode* for large datasets (uses top-k categories and PCA for faster processing).
       - Set *top-k categories, **PCA components, **max columns to visualize, and **KS test sample size*.
       - Set *Accuracy Threshold* - the minimum accuracy percentage required for synthetic data to be considered acceptable.
    
    4. *Validation and Cleaning* 🧹:
       - The app automatically cleans data by dropping empty columns, handling non-finite values, detecting datetimes, and filling missing categoricals.
       - Review any validation issues displayed.
       - Optionally drop columns like IDs or timestamps.
    
    5. *Training Data Selection* 📈:
       - Choose to train on the full dataset or subsample for faster processing.
    
    6. *Synthesizer Configuration* 🛠:
       - Choose your synthesizer: *CTGAN, **GaussianCopula, or **Both* for comparison.
       - Set *epochs, **batch size, and **number of synthetic rows* (CTGAN-specific).
       - Enable *GPU* if available for faster CTGAN training.
       - Use the sidebar for *training improvements* (e.g., enhanced epochs, discriminator steps for CTGAN).
    
    7. *Train and Generate* 🚀:
       - Click *Train and Generate Synthetic Data*.
       - Monitor training progress, preview synthetic data, and download generated CSVs.
    
    8. *Validation and Evaluation* 📊:
       - Review visual comparisons (histograms, bar plots, correlation heatmaps).
       - Check summary tables with metrics like *KS Statistic, **P-value, **Mean Diff %, **Distribution Overlap, and **Wasserstein Distance*.
       - *Accuracy Assessment*: See if synthetic data meets your accuracy threshold.
       - Use the *Interpretation Guide* to assess data quality.
       - Explore overall quality scores, average metrics, and key insights.
    
    *Accuracy Threshold Explanation* 📈:
    - The accuracy threshold is the minimum accuracy value that synthetic data must achieve to be considered acceptable or successful.
    - If the accuracy goes above this threshold, the synthetic data is considered performing well.
    - If it stays below, the model needs improvement.
    - You can adjust this threshold in the sidebar (50-100%).
    
    *Tips for Success* 💡:
    - Improve quality by increasing epochs or enabling the *Increase training* option.
    - Compare *CTGAN* and *GaussianCopula* performance when using both synthesizers.
    - Use *Fast mode* for large datasets to reduce training time.
    - If accuracy is below threshold, follow the provided recommendations for improvement.
    
    If you encounter errors, check the traceback for detailed debugging information.
    """, unsafe_allow_html=True)

elif st.session_state.page == 'Generate':
    st.title("⚙ Generate Synthetic Data")

    st.markdown(f"*SDV Version: {sdv.__version__} | **Pandas Version*: {pd.__version__}")
    st.markdown(f"*Available CPUs: {multiprocessing.cpu_count()} | **GPU Available*: {torch.cuda.is_available()}")

    # Options for performance
    st.sidebar.header("⚙ Performance Options")
    fast_mode = st.sidebar.checkbox(
        "🚀 Fast mode (recommended for large datasets)", value=True,
        help="Apply top-k mapping for categories + PCA for numeric columns to speed training."
    )
    top_k_categories = st.sidebar.number_input("Top-k categories to keep (per categorical col)", min_value=2, max_value=1000, value=50)
    pca_components = st.sidebar.number_input("Max PCA components (numeric)", min_value=1, max_value=200, value=10)
    max_cols_plot = st.sidebar.slider("Max columns to visualize", 1, 10, 5)
    ks_sample_size = st.sidebar.slider(
        "KS test sample size per column", 
        min_value=50,
        max_value=1000,
        value=200,
        step=50,
        help="Small samples (50-200) give more reasonable p-values for synthetic data evaluation."
    )
    
    # ADDED: Accuracy threshold slider
    accuracy_threshold = st.sidebar.slider(
        "📊 Accuracy Threshold (%)",
        min_value=50,
        max_value=100,
        value=80,
        help="Minimum acceptable quality threshold. Below this, synthetic data needs improvement."
    )

    # File upload
    file_type = st.selectbox("📄 Select file type", ["Excel (.xlsx)", "CSV (.csv)"])
    uploaded_file = st.file_uploader(f"Upload your {file_type}", type=["xlsx" if file_type == "Excel (.xlsx)" else "csv"])

    # Utility functions
    def optimize_dtypes(df):
        for col in df.columns:
            dtype = df[col].dtype
            if pd.api.types.is_float_dtype(dtype):
                df[col] = df[col].astype("float64")
            elif pd.api.types.is_integer_dtype(dtype):
                df[col] = df[col].astype("int64")
            elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
                df[col] = df[col].astype(str)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                df[col] = pd.to_datetime(df[col], errors="coerce")
            else:
                df[col] = df[col].astype(str)
        return df

    def map_top_k_categories(df, cat_cols, k):
        """Keep top-k categories for each categorical column, map others to '_OTHER_'."""
        mappings = {}
        df_out = df.copy()
        for col in cat_cols:
            top = df_out[col].value_counts().nlargest(k).index  # Fixed typo: value_count -> value_counts
            df_out[col] = df_out[col].where(df_out[col].isin(top), other="_OTHER_")
            mappings[col] = set(top.tolist())
        return df_out, mappings

    def apply_pca(df, num_cols, n_components):
        """Scale numeric columns, apply PCA, return transformed df, scaler, pca."""
        scaler = StandardScaler()
        X = scaler.fit_transform(df[num_cols].fillna(0).values)
        pca = PCA(n_components=min(n_components, X.shape[1], X.shape[0]))
        Xp = pca.fit_transform(X)
        pc_names = [f"PC{i+1}" for i in range(Xp.shape[1])]
        df_pca = pd.DataFrame(Xp, columns=pc_names, index=df.index)
        return df_pca, scaler, pca, pc_names

    def invert_pca(df_pca, scaler, pca, original_num_cols):
        """Invert PCA + scaler to approximate original numeric columns."""
        Xp = df_pca.values
        X_approx = pca.inverse_transform(Xp)
        X_orig = scaler.inverse_transform(X_approx)
        df_orig = pd.DataFrame(X_orig, columns=original_num_cols)
        return df_orig

    def calculate_ks_pvalue_manual(ks_stat, n1, n2):
        """Manual calculation of KS test p-value to avoid scipy precision issues"""
        en = np.sqrt(n1 * n2 / (n1 + n2))
        if ks_stat == 0:
            return 1.0
        
        x = en * ks_stat
        if x < 0.3:
            return 1.0
        elif x > 8.0:
            return 0.0
        
        term = -2 * x * x
        p_val = 0.0
        for k in range(1, 100):
            sign = -1 if k % 2 == 1 else 1
            p_val += sign * np.exp(term * k * k)
            if abs(np.exp(term * k * k)) < 1e-20:
                break
        
        p_val = 2 * p_val
        return max(min(p_val, 1.0), 0.0)

    def perform_ks_test_fixed(real_series, synth_series, sample_size=200):
        """FINAL FIXED VERSION - Proper KS test with guaranteed non-zero p-values"""
        if len(real_series) <= 1 or len(synth_series) <= 1:
            return 0.0, 1.0, "Insufficient data"
        
        min_sample = min(len(real_series), len(synth_series), sample_size)
        if min_sample < 20:
            return 0.0, 1.0, "Sample too small"
        
        try:
            real_sample = real_series.sample(n=min_sample, random_state=42)
            synth_sample = synth_series.sample(n=min_sample, random_state=42)
            
            real_values = real_sample.astype(float).values
            synth_values = synth_sample.astype(float).values
            
            real_std, synth_std = np.std(real_values), np.std(synth_values)
            if real_std == 0 and synth_std == 0:
                if np.mean(real_values) == np.mean(synth_values):
                    return 0.0, 1.0, "Both constant and equal"
                else:
                    return 1.0, 0.0, "Both constant but different"
            
            if np.max(np.abs(real_values - synth_values)) < 1e-10:
                return 0.0, 1.0, "Samples identical"
            
            n1, n2 = len(real_values), len(synth_values)
            data1_sorted = np.sort(real_values)
            data2_sorted = np.sort(synth_values)
            
            cdf1 = np.searchsorted(data1_sorted, data1_sorted, side='right') / n1
            cdf2 = np.searchsorted(data2_sorted, data1_sorted, side='right') / n2
            diff1 = np.abs(cdf1 - cdf2)
            
            cdf1 = np.searchsorted(data1_sorted, data2_sorted, side='right') / n1
            cdf2 = np.searchsorted(data2_sorted, data2_sorted, side='right') / n2
            diff2 = np.abs(cdf1 - cdf2)
            
            ks_stat_manual = max(np.max(diff1), np.max(diff2))
            
            ks_p = 1.0
            methods_tried = []
            
            try:
                ks_stat_scipy, ks_p = ks_2samp(real_values, synth_values, mode='auto')
                methods_tried.append(f"auto(p={ks_p:.10f})")
            except:
                pass
            
            if ks_p <= 1e-100 or ks_p == 0.0:
                ks_p = calculate_ks_pvalue_manual(ks_stat_manual, n1, n2)
                methods_tried.append("manual")
            
            if ks_p <= 0.0:
                ks_p = 1e-300
            elif ks_p < 1e-300:
                ks_p = 1e-300
            
            ks_stat = ks_stat_manual
            status = f"Success - Methods: {', '.join(methods_tried)}, n={min_sample}"
            return ks_stat, ks_p, status
            
        except Exception as e:
            return 0.0, 1.0, f"Error: {str(e)}"

    def calculate_similarity_metrics(real_series, synth_series):
        """Calculate practical similarity metrics"""
        real_vals = real_series.dropna()
        synth_vals = synth_series.dropna()
        
        if len(real_vals) == 0 or len(synth_vals) == 0:
            return 0.0, 0.0, "No data"
        
        mean_diff = abs(real_vals.mean() - synth_vals.mean())
        std_real = real_vals.std()
        norm_mean_diff = (mean_diff / std_real) * 100 if std_real > 0 else 0
        
        try:
            sample_size = min(100, len(real_vals), len(synth_vals))
            real_sample = real_vals.sample(sample_size, random_state=42)
            synth_sample = synth_vals.sample(sample_size, random_state=42)
            
            kde_real = gaussian_kde(real_sample)
            kde_synth = gaussian_kde(synth_sample)
            
            x_min = min(real_sample.min(), synth_sample.min())
            x_max = max(real_sample.max(), synth_sample.max())
            x_range = np.linspace(x_min, x_max, 50)
            
            pdf_real = kde_real(x_range)
            pdf_synth = kde_synth(x_range)
            
            pdf_real = pdf_real / pdf_real.sum()
            pdf_synth = pdf_synth / pdf_synth.sum()
            
            overlap = np.minimum(pdf_real, pdf_synth).sum()
            
        except Exception:
            overlap = 0.5
        
        return norm_mean_diff, overlap, "Success"

    def calculate_wasserstein_distance(real_series, synth_series, sample_size=200):
        """Calculate Wasserstein distance between real and synthetic data"""
        try:
            min_sample = min(len(real_series), len(synth_series), sample_size)
            if min_sample < 20:
                return 0.0, "Sample too small"
            
            real_sample = real_series.sample(n=min_sample, random_state=42).astype(float).values
            synth_sample = synth_series.sample(n=min_sample, random_state=42).astype(float).values
            
            if np.std(real_sample) == 0 and np.std(synth_sample) == 0:
                return 0.0, "Both constant"
            
            w_dist = wasserstein_distance(real_sample, synth_sample)
            return w_dist, "Success"
        except Exception as e:
            return 0.0, f"Error: {str(e)}"

    def calculate_correlation_diff(real_df, synth_df, numeric_cols):
        """Calculate average absolute correlation difference between real and synthetic data"""
        try:
            if len(numeric_cols) < 2:
                return 0.0, "Not enough numeric columns"
            
            real_corr = real_df[numeric_cols].corr().values
            synth_corr = synth_df[numeric_cols].corr().values
            
            real_corr = np.nan_to_num(real_corr, 0)
            synth_corr = np.nan_to_num(synth_corr, 0)
            
            corr_diff = np.mean(np.abs(real_corr - synth_corr))
            return corr_diff, "Success"
        except Exception as e:
            return 0.0, f"Error: {str(e)}"
    
    # ADDED: Function to calculate overall accuracy score
    def calculate_overall_accuracy(summary_list, correlation_diffs, quality_report=None):
        """
        Calculate overall accuracy score based on multiple metrics
        Returns score (0-100) and detailed breakdown
        """
        if not summary_list:
            return 0.0, {}
        
        # Extract metrics
        ks_scores = [1 - min(x["KS Statistic"], 1.0) for x in summary_list]  # KS stat lower is better
        mean_diff_scores = [max(0, 100 - x["Mean Diff %"])/100 for x in summary_list]  # Mean diff lower is better
        overlap_scores = [x["Distribution Overlap"] for x in summary_list]  # Higher is better
        wasserstein_scores = [max(0, 1 - min(x["Wasserstein Distance"], 1.0)) for x in summary_list]  # Lower is better
        
        # Calculate weighted average (adjust weights as needed)
        weights = {
            'ks': 0.30,
            'mean_diff': 0.25,
            'overlap': 0.25,
            'wasserstein': 0.20
        }
        
        avg_ks = np.mean(ks_scores) if ks_scores else 0
        avg_mean_diff = np.mean(mean_diff_scores) if mean_diff_scores else 0
        avg_overlap = np.mean(overlap_scores) if overlap_scores else 0
        avg_wasserstein = np.mean(wasserstein_scores) if wasserstein_scores else 0
        
        # Combine with correlation difference
        corr_diff_score = 1 - min(correlation_diffs.get("Corr Diff", 1.0), 1.0)  # Lower is better
        
        # Overall score (0-100)
        overall_score = (
            avg_ks * weights['ks'] +
            avg_mean_diff * weights['mean_diff'] +
            avg_overlap * weights['overlap'] +
            avg_wasserstein * weights['wasserstein']
        ) * 100
        
        # Include quality report score if available
        if quality_report is not None:
            quality_score = quality_report.get_score() * 100
            overall_score = (overall_score * 0.7 + quality_score * 0.3)  # Weighted combination
        
        # Ensure score is within bounds
        overall_score = min(max(overall_score, 0), 100)
        
        breakdown = {
            "KS Score": avg_ks * 100,
            "Mean Diff Score": avg_mean_diff * 100,
            "Overlap Score": avg_overlap * 100,
            "Wasserstein Score": avg_wasserstein * 100,
            "Correlation Score": corr_diff_score * 100
        }
        
        return overall_score, breakdown
    
    # ADDED: Function to plot accuracy gauge
    def plot_accuracy_gauge(accuracy, threshold, title):
        """Create a gauge chart for accuracy visualization"""
        fig, ax = plt.subplots(figsize=(8, 4), subplot_kw=dict(aspect="equal"))
        
        # Create gauge
        colors = ['#B91C1C', '#D97706', '#2A6B8F']  # Red, Amber, Blue
        wedges, texts = ax.pie([threshold, 100-threshold, 0], colors=[colors[0], colors[1], colors[2]], 
                              startangle=90, counterclock=False)
        
        # Add needle
        angle = 180 * (accuracy / 100)
        ax.plot([0, 0.8 * np.cos(np.radians(angle))], 
                [0, 0.8 * np.sin(np.radians(angle))], 
                color='#222222', linewidth=3)
        
        # Add center circle
        ax.add_patch(plt.Circle((0, 0), 0.2, color='white'))
        
        # Add text
        ax.text(0, 0, f'{accuracy:.1f}%', ha='center', va='center', 
                fontsize=24, fontweight='bold', color='#222222')
        ax.text(0, -0.3, f'Threshold: {threshold}%', ha='center', va='center',
                fontsize=12, color='#222222')
        ax.set_title(title, color='#222222', fontsize=14, fontweight='bold')
        
        ax.set_facecolor('#FFEFEA')
        fig.set_facecolor('#FFEFEA')
        return fig

    # Main flow for Generate page
    if uploaded_file is not None:
        try:
            st.subheader("📖 Reading File...")
            if file_type == "Excel (.xlsx)":
                df = pd.read_excel(uploaded_file)
            else:
                chunk_size = st.number_input("CSV chunk size (rows)", min_value=1000, value=10000, step=1000)
                df_chunks = pd.read_csv(uploaded_file, chunksize=chunk_size)
                df = pd.concat(df_chunks, ignore_index=True)
            st.write("*Preview*:")
            st.write(df.head())

            if df.empty:
                st.error("🚫 Uploaded dataset is empty. Please upload a valid file.")
                st.stop()

            df = optimize_dtypes(df)
            st.write(f"*Memory Usage*: {df.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB")

            st.subheader("🧹 Validation / Basic Cleaning")
            validation_issues = []
            for col in list(df.columns):
                if df[col].isna().all() or df[col].nunique() == 0:
                    validation_issues.append(f"Column {col} empty/all-NaN -> dropping")
                    df = df.drop(columns=[col])
            for col in df.select_dtypes(include=["float64", "int64"]).columns:
                non_finite_count = df[col].replace([np.inf, -np.inf], np.nan).isna().sum()
                if non_finite_count > 0:
                    validation_issues.append(f"{col}: {non_finite_count} non-finite -> filling with median")
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(df[col].median())
            for col in df.select_dtypes(include=["object"]).columns:
                try:
                    temp = pd.to_datetime(df[col], errors="coerce")
                    if temp.notna().sum() > len(df) * 0.5:
                        validation_issues.append(f"{col} looks like datetime -> converted")
                        df[col] = temp
                except Exception:
                    pass

            for col in df.select_dtypes(include=["object"]).columns:
                df[col] = df[col].fillna("_MISSING_")

            cardinality_threshold = max(100, int(len(df) * 0.02))
            high_card_cols = [c for c in df.select_dtypes(include=["object"]).columns if df[c].nunique() > cardinality_threshold]
            if high_card_cols:
                validation_issues.append(f"High-cardinality categorical columns detected: {high_card_cols}")

            if validation_issues:
                st.warning("⚠ Issues found:")
                for i in validation_issues:
                    st.write(f"- {i}")
            else:
                st.success("✅ No major issues found.")

            if df.empty:
                st.error("🚫 All columns were dropped during cleaning. Please check your dataset.")
                st.stop()

            possible_defaults = ["ID", "id", "Timestamp", "timestamp", "Date", "date"]
            valid_defaults = [c for c in possible_defaults if c in df.columns]
            to_drop = st.multiselect("Columns to drop (optional)", df.columns.tolist(), default=valid_defaults)
            if to_drop:
                dropped_data = df[to_drop].copy()
                df = df.drop(columns=[to_drop])
            else:
                dropped_data = pd.DataFrame()

            st.subheader("📈 Training Data Selection")
            use_full_data = st.checkbox("Train on full dataset (may be slow)", value=not fast_mode)
            if use_full_data:
                df_train = df.reset_index(drop=True)
            else:
                subsample_frac = st.slider("Subsample fraction", 0.05, 1.0, 0.5 if fast_mode else 1.0, 0.05)
                df_train = df.sample(frac=subsample_frac, random_state=42).reset_index(drop=True)
                st.info(f"Training on {len(df_train)} rows (subsampled)")

            st.write(f"*Training rows*: {len(df_train)}")

            categorical_cols = df_train.select_dtypes(include=["object"]).columns.tolist()
            numeric_cols = df_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
            datetime_cols = df_train.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
            bool_cols = df_train.select_dtypes(include=["bool"]).columns.tolist()

            pca_applied = False
            mappings = {}
            pca_info = {}
            df_for_sdv = df_train.copy()

            if fast_mode:
                st.subheader("🚀 Fast-mode Preprocessing")
                if len(categorical_cols) > 0:
                    st.write(f"Mapping categorical columns top-{top_k_categories} (others -> _OTHER_)")
                    df_for_sdv, mappings = map_top_k_categories(df_for_sdv, categorical_cols, top_k_categories)
                if len(numeric_cols) > 0:
                    n_comp = min(pca_components, len(numeric_cols))
                    st.write(f"Applying PCA on {len(numeric_cols)} numeric columns -> {n_comp} components")
                    df_pca, scaler, pca, pc_names = apply_pca(df_for_sdv, numeric_cols, n_comp)
                    df_for_sdv = pd.concat([df_for_sdv.drop(columns=numeric_cols), df_pca], axis=1)
                    pca_applied = True
                    pca_info = {
                        "numeric_cols": numeric_cols,
                        "scaler": scaler,
                        "pca": pca,
                        "pc_names": pc_names
                    }
                    numeric_cols = pc_names

            metadata = SingleTableMetadata()
            try:
                metadata.detect_from_dataframe(data=df_for_sdv)
            except Exception as e:
                st.warning(f"⚠ Auto metadata detect failed: {e}. Attempting minimal metadata.")
                for col in df_for_sdv.columns:
                    if df_for_sdv[col].nunique() == len(df_for_sdv) or col.lower().endswith("id"):
                        metadata.add_column(col, sdtype="id")
                    elif pd.api.types.is_numeric_dtype(df_for_sdv[col]):
                        metadata.add_column(col, sdtype="numerical")
                    elif pd.api.types.is_datetime64_any_dtype(df_for_sdv[col]):
                        metadata.add_column(col, sdtype="datetime")
                    elif pd.api.types.is_bool_dtype(df_for_sdv[col]):
                        metadata.add_column(col, sdtype="boolean")
                    else:
                        metadata.add_column(col, sdtype="categorical")
            
            for col in high_card_cols:
                if col in df_for_sdv.columns:
                    if fast_mode and col in mappings:
                        try:
                            metadata.update_column(column_name=col, sdtype="categorical")
                        except Exception:
                            pass
                    else:
                        try:
                            metadata.update_column(column_name=col, sdtype="id")
                        except Exception:
                            pass
            
            try:
                metadata.validate()
                st.success("✅ Metadata validated")
            except Exception as e:
                st.warning(f"⚠ Metadata validation error: {e}. Attempting to fix common issues.")
                for col in high_card_cols:
                    try:
                        metadata.update_column(column_name=col, sdtype="id")
                    except Exception:
                        pass
                try:
                    metadata.validate()
                    st.success("✅ Metadata validated after fixes")
                except Exception as e2:
                    st.error(f"🚫 Failed to validate metadata: {e2}")
                    st.stop()

            original_metadata = SingleTableMetadata()
            try:
                original_metadata.detect_from_dataframe(df)
            except Exception as e:
                st.warning(f"⚠ Original metadata detect failed: {e}. Using fallback.")
                for col in df.columns:
                    if df[col].nunique() == len(df) or col.lower().endswith("id"):
                        original_metadata.add_column(col, sdtype="id")
                    elif pd.api.types.is_numeric_dtype(df[col]):
                        original_metadata.add_column(col, sdtype="numerical")
                    elif pd.api.types.is_datetime64_any_dtype(df[col]):
                        original_metadata.add_column(col, sdtype="datetime")
                    elif pd.api.types.is_bool_dtype(df[col]):
                        original_metadata.add_column(col, sdtype="boolean")
                    else:
                        original_metadata.add_column(col, sdtype="categorical")

            st.subheader("🛠 Synthesizer Configuration")
            synthesizer_choice = st.selectbox(
                "Choose synthesizer(s) for generating synthetic data",
                ["CTGAN", "GaussianCopula", "Both (CTGAN + GaussianCopula)"],
                help="CTGAN is neural network-based; GaussianCopula is a statistical model. Select 'Both' to compare."
            )

            default_epochs = 30 if fast_mode else 100
            epochs = st.slider("Epochs (CTGAN only)", 5, 500, default_epochs)
            suggested_batch = 64 if len(df_for_sdv) > 20000 else 32
            batch_size = st.selectbox("Batch size (CTGAN only)", [16, 32, 64, 128, 256], index=[16,32,64,128,256].index(suggested_batch))
            num_rows = st.number_input("Number of synthetic rows to sample", min_value=1, value=min(len(df), 10000))
            use_gpu = st.checkbox("Use GPU if available (CTGAN only)", value=torch.cuda.is_available())

            st.sidebar.header("📈 Training Improvements")
            increase_training = st.sidebar.checkbox("Increase training for better quality (CTGAN only)", value=True,
                                                  help="Train longer with better parameters for improved synthetic data")
            
            if increase_training:
                epochs = st.sidebar.slider("Enhanced epochs (CTGAN only)", 100, 1000, 300)
                discriminator_steps = st.sidebar.slider("Discriminator steps (CTGAN only)", 1, 5, 3)
            else:
                discriminator_steps = 1

            if st.button("🚀 Train and Generate Synthetic Data"):
                try:
                    synthetic_datasets = {}
                    quality_reports = {}
                    use_gpu_final = use_gpu and torch.cuda.is_available()
                    if use_gpu_final:
                        torch.backends.cudnn.benchmark = True

                    synthesizers_to_train = []
                    if synthesizer_choice in ["CTGAN", "Both (CTGAN + GaussianCopula)"]:
                        synthesizers_to_train.append(("CTGAN", CTGANSynthesizer))
                    if synthesizer_choice in ["GaussianCopula", "Both (CTGAN + GaussianCopula)"]:
                        synthesizers_to_train.append(("GaussianCopula", GaussianCopulaSynthesizer))

                    for synth_name, SynthClass in synthesizers_to_train:
                        st.write(f"### Training {synth_name}... 🧠")
                        try:
                            if synth_name == "CTGAN":
                                synthesizer = SynthClass(
                                    metadata,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    verbose=True,
                                    enforce_min_max_values=True,
                                    enforce_rounding=True,
                                    cuda=use_gpu_final,
                                    generator_lr=2e-4,
                                    discriminator_lr=2e-4,
                                    discriminator_steps=discriminator_steps,
                                    pac=1
                                )
                            else:
                                synthesizer = SynthClass(
                                    metadata,
                                    enforce_min_max_values=True,
                                    enforce_rounding=True,
                                    numerical_distributions={col: 'norm' for col in numeric_cols},
                                    default_distribution='norm'
                                )

                            with st.spinner(f"Training {synth_name}..."):
                                synthesizer.fit(df_for_sdv)

                            st.success(f"✅ {synth_name} training complete!")

                            synthetic_data = synthesizer.sample(num_rows=num_rows)
                            synthetic_datasets[synth_name] = synthetic_data

                            if pca_applied:
                                st.info(f"Inverting PCA for {synth_name}...")
                                pc_names = pca_info["pc_names"]
                                scaler = pca_info["scaler"]
                                pca = pca_info["pca"]
                                original_numeric_cols = pca_info["numeric_cols"]

                                missing_pcs = [c for c in pc_names if c not in synthetic_data.columns]
                                if missing_pcs:
                                    st.warning(f"⚠ Expected PCA columns missing in {synth_name} output: {missing_pcs}")
                                else:
                                    df_pca_synth = synthetic_data[pc_names].astype(float).fillna(0)
                                    df_numeric_approx = invert_pca(df_pca_synth, scaler, pca, original_numeric_cols)
                                    for col in original_numeric_cols:
                                        if pd.api.types.is_integer_dtype(df[col]):
                                            df_numeric_approx[col] = np.round(df_numeric_approx[col]).astype(int)
                                        min_val = df[col].min()
                                        max_val = df[col].max()
                                        df_numeric_approx[col] = df_numeric_approx[col].clip(min_val, max_val)
                                    synthetic_data = pd.concat([synthetic_data.drop(columns=pc_names), df_numeric_approx.reset_index(drop=True)], axis=1)

                            if not dropped_data.empty:
                                for col in dropped_data.columns:
                                    if col.lower().endswith("id"):
                                        synthetic_data[col] = [f"synthetic_{i}_{synth_name}" for i in range(len(synthetic_data))]
                                    elif pd.api.types.is_datetime64_any_dtype(dropped_data[col]):
                                        try:
                                            min_t = dropped_data[col].min()
                                            max_t = dropped_data[col].max()
                                            if pd.isna(min_t) or pd.isna(max_t):
                                                synthetic_data[col] = pd.NaT
                                            else:
                                                rng = (max_t - min_t).total_seconds()
                                                synthetic_data[col] = [min_t + pd.Timedelta(seconds=float(s)) for s in np.random.uniform(0, rng, size=len(synthetic_data))]
                                        except Exception:
                                            synthetic_data[col] = pd.NaT
                                    else:
                                        if pd.api.types.is_numeric_dtype(dropped_data[col]):
                                            synthetic_data[col] = dropped_data[col].median()
                                        else:
                                            synthetic_data[col] = dropped_data[col].mode().iloc[0] if not dropped_data[col].mode().empty else ""

                            all_cols = list(df.columns) + list(dropped_data.columns)
                            synthetic_data = synthetic_data.reindex(columns=all_cols)
                            synthetic_datasets[synth_name] = synthetic_data

                            try:
                                with st.spinner(f"Computing quality report for {synth_name}..."):
                                    quality_report = evaluate_quality(df.reset_index(drop=True), synthetic_data.reset_index(drop=True), original_metadata)
                                quality_reports[synth_name] = quality_report
                            except Exception as e:
                                st.warning(f"⚠ Could not generate quality report for {synth_name}: {e}")
                                quality_reports[synth_name] = None

                        except Exception as e:
                            st.error(f"🚫 Error during {synth_name} training: {e}")
                            st.code(traceback.format_exc())
                            continue

                    for synth_name, synthetic_data in synthetic_datasets.items():
                        st.subheader(f"📊 {synth_name} Synthetic Data Preview")
                        st.write(synthetic_data.head())
                        csv_buffer = synthetic_data.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            f"⬇ Download {synth_name} Synthetic CSV",
                            csv_buffer,
                            file_name=f"synthetic_data_{synth_name.lower()}.csv",
                            mime="text/csv",
                            key=f"download_{synth_name}",
                            help=f"Download synthetic data generated by {synth_name}",
                            args={'class': 'download-btn'}
                        )

                    st.subheader("📈 Visual Comparison and Statistical Evaluation")
                    numeric_cols_final = [c for c in df.select_dtypes(include=["int64", "float64"]).columns]
                    cat_cols_final = [c for c in df.select_dtypes(include=["object"]).columns]
                    summary_list = {synth_name: [] for synth_name in synthetic_datasets}

                    if numeric_cols_final:
                        st.write("### Numeric Columns (KDE + KS Test + Wasserstein Distance)")
                        for col in numeric_cols_final[:max_cols_plot]:
                            try:
                                fig, ax = plt.subplots(figsize=(10, 4))
                                ax.hist(df[col].dropna(), bins=30, alpha=0.7, label="Real", density=True, color='#2A6B8F')
                                for synth_name, synthetic_data in synthetic_datasets.items():
                                    ax.hist(synthetic_data[col].dropna(), bins=30, alpha=0.7, label=f"Synthetic ({synth_name})", density=True)
                                ax.set_title(f"Distribution Comparison: {col}", color='#222222')
                                ax.set_xlabel(col, color='#222222')
                                ax.set_ylabel('Density', color='#222222')
                                ax.tick_params(axis='both', colors='#222222')
                                ax.legend().get_texts()[0].set_color("#222222")
                                for text in ax.legend().get_texts()[1:]:
                                    text.set_color("#222222")
                                ax.set_facecolor('#FFEFEA')
                                fig.set_facecolor('#FFEFEA')
                                st.pyplot(fig)
                                plt.close(fig)

                                real_series = df[col].dropna()
                                for synth_name, synthetic_data in synthetic_datasets.items():
                                    synth_series = synthetic_data[col].dropna()
                                    ks_stat, ks_p, ks_status = perform_ks_test_fixed(real_series, synth_series, sample_size=ks_sample_size)
                                    norm_mean_diff, overlap, metric_status = calculate_similarity_metrics(real_series, synth_series)
                                    w_dist, w_status = calculate_wasserstein_distance(real_series, synth_series, sample_size=ks_sample_size)
                                    summary_list[synth_name].append({
                                        "Column": col,
                                        "Original Mean": real_series.mean(),
                                        "Synthetic Mean": synth_series.mean(),
                                        "Mean Diff %": norm_mean_diff,
                                        "Distribution Overlap": overlap,
                                        "KS Statistic": ks_stat,
                                        "P-Value": ks_p,
                                        "KS Status": ks_status,
                                        "Wasserstein Distance": w_dist,
                                        "Wasserstein Status": w_status
                                    })
                            except Exception as ex:
                                st.write(f"⚠ Could not plot/compare {col}: {ex}")

                    if cat_cols_final:
                        st.write("### Categorical Columns (Top categories)")
                        for col in cat_cols_final[:max_cols_plot]:
                            try:
                                fig, ax = plt.subplots(figsize=(8, 4))
                                real_counts = df[col].value_counts(normalize=True).head(8)
                                comp = pd.DataFrame({"Real": real_counts})
                                for synth_name, synthetic_data in synthetic_datasets.items():
                                    synth_counts = synthetic_data[col].value_counts(normalize=True).head(8)
                                    comp[f"Synthetic ({synth_name})"] = synth_counts
                                comp.fillna(0).plot(kind="bar", ax=ax, color=['#2A6B8F', '#FF8566', '#6B728E'])
                                ax.set_title(f"Category distribution: {col}", color='#222222')
                                ax.set_xlabel(col, color='#222222')
                                ax.set_ylabel('Proportion', color='#222222')
                                ax.tick_params(axis='x', rotation=45, colors='#222222')
                                ax.legend().get_texts()[0].set_color("#222222")
                                for text in ax.legend().get_texts()[1:]:
                                    text.set_color("#222222")
                                ax.set_facecolor('#FFEFEA')
                                fig.set_facecolor('#FFEFEA')
                                st.pyplot(fig)
                                plt.close(fig)
                            except Exception as ex:
                                st.write(f"⚠ Could not plot {col}: {ex}")

          
                    
                    if len(numeric_cols_final) > 1:
                                st.write("### Correlation Heatmaps")
                                try:
                                    n_synthesizers = len(synthetic_datasets)
                                    total_plots = n_synthesizers + 1

                                    # Decide rows and columns for the subplot grid
                                    cols = 3  # max plots per row
                                    rows = math.ceil(total_plots / cols)

                                    fig, axes = plt.subplots(rows, cols, figsize=(6 * min(cols, total_plots), 4 * rows))
                                    axes = axes.flatten()  # flatten in case of multiple rows

                                    # Real data heatmap
                                    ax = axes[0]
                                    sns.heatmap(df[numeric_cols_final].corr(), annot=False, cmap="coolwarm", ax=ax, center=0, cbar_kws={'label': 'Correlation'})
                                    cbar = ax.collections[0].colorbar
                                    cbar.ax.tick_params(labelcolor='#222222')
                                    ax.set_title("Real Data Correlations", color='#222222')
                                    ax.tick_params(axis='both', colors='#222222')
                                    ax.set_facecolor('#FFEFEA')

                                    # Synthetic data heatmaps
                                    for idx, (synth_name, synthetic_data) in enumerate(synthetic_datasets.items(), 1):
                                        if idx >= len(axes):
                                            break
                                        ax = axes[idx]
                                        sns.heatmap(synthetic_data[numeric_cols_final].corr(), annot=False, cmap="coolwarm", ax=ax, center=0, cbar_kws={'label': 'Correlation'})
                                        cbar = ax.collections[0].colorbar
                                        cbar.ax.tick_params(labelcolor='#222222')
                                        ax.set_title(f"Synthetic ({synth_name}) Correlations", color='#222222')
                                        ax.tick_params(axis='both', colors='#222222')
                                        ax.set_facecolor('#FFEFEA')

                                    # Remove any unused axes
                                    for i in range(total_plots, len(axes)):
                                        fig.delaxes(axes[i])

                                    fig.set_facecolor('#FFEFEA')
                                    fig.tight_layout()
                                    st.pyplot(fig)
                                    plt.close(fig)
                                except Exception as ex:
                                    st.write(f"⚠ Could not compute correlation heatmaps: {ex}"  )   
                    correlation_diffs = {}
                    for synth_name, synthetic_data in synthetic_datasets.items():
                        corr_diff, corr_status = calculate_correlation_diff(df, synthetic_data, numeric_cols_final)
                        correlation_diffs[synth_name] = {"Corr Diff": corr_diff, "Status": corr_status}

                    for synth_name, summaries in summary_list.items():
                        if summaries:
                            st.subheader(f"📊 {synth_name} Summary of Real vs Synthetic Data Differences")
                            summary_df = pd.DataFrame(summaries)                   
                            def color_ks_statistic(val):
                                if val < 0.1:
                                    return 'background-color: #2A6B8F'  # Dark blue
                                elif val < 0.2:
                                    return 'background-color: #D97706'  # Amber
                                else:
                                    return 'background-color: #B91C1C'  # Red
                            
                            def color_p_value(val):
                                if val > 0.05:
                                    return 'background-color: #2A6B8F'
                                elif val > 0.01:
                                    return 'background-color: #D97706'
                                else:
                                    return 'background-color: #B91C1C'
                            
                            def color_mean_diff(val):
                                if val < 10:
                                    return 'background-color: #2A6B8F'
                                elif val < 25:
                                    return 'background-color: #D97706'
                                else:
                                    return 'background-color: #B91C1C'
                            
                            def color_wasserstein(val):
                                if val < 0.1:
                                    return 'background-color: #2A6B8F'
                                elif val < 0.5:
                                    return 'background-color: #D97706'
                                else:
                                    return 'background-color: #B91C1C'
                            
                            styled_df = summary_df.style.format({
                                "Original Mean": "{:.2f}",
                                "Synthetic Mean": "{:.2f}",
                                "Mean Diff %": "{:.1f}%",
                                "Distribution Overlap": "{:.3f}",
                                "KS Statistic": "{:.4f}",
                                "P-Value": "{:.10f}",
                                "Wasserstein Distance": "{:.4f}"
                            }).applymap(color_ks_statistic, subset=['KS Statistic']
                            ).applymap(color_p_value, subset=['P-Value']
                            ).applymap(color_mean_diff, subset=['Mean Diff %']
                            ).applymap(color_wasserstein, subset=['Wasserstein Distance'])
                            
                            st.dataframe(styled_df)
                            
                            st.write(f"{synth_name} KS Test Status Messages:")
                            for item in summaries:
                                st.write(f"- *{item['Column']}*: {item['KS Status']} (Wasserstein: {item['Wasserstein Status']})")
                            
                            st.markdown("### 📝 Interpretation Guide")
                            st.markdown("""
                            *Quality Assessment:*
                            - ✅ *Good*: KS < 0.1, P-value > 0.05, Mean Diff < 10%, Wasserstein Distance < 0.1
                            - ⚠ *Acceptable*: KS < 0.2, P-value > 0.01, Mean Diff < 25%, Wasserstein Distance < 0.5
                            - ❌ *Needs Improvement*: KS > 0.2, P-value < 0.01, Mean Diff > 25%, Wasserstein Distance > 0.5
                            """, unsafe_allow_html=True)
                            
                            poor_quality_cols = [item for item in summaries if item['KS Statistic'] > 0.2 or item['Mean Diff %'] > 25]
                            if poor_quality_cols:
                                st.warning(f"*Recommendations for {synth_name} Improvement:*")
                                st.write("1. Increase training epochs (try 500-1000 for CTGAN)")
                                st.write("2. Use the 'Increase training' option in sidebar (CTGAN)")
                                st.write("3. Check if PCA reconstruction is affecting numeric columns")
                            
                            # ADDED: Accuracy Assessment Section
                            corr_diff_info = correlation_diffs.get(synth_name, {"Corr Diff": 0.0, "Status": "Not computed"})
                            quality_report = quality_reports.get(synth_name)
                            
                            overall_accuracy, accuracy_breakdown = calculate_overall_accuracy(
                                summaries, 
                                corr_diff_info,
                                quality_report
                            )
                            
                            # Display accuracy with threshold comparison
                            st.subheader(f"📊 {synth_name} Accuracy Assessment")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "Overall Accuracy",
                                    f"{overall_accuracy:.1f}%",
                                    delta=f"Threshold: {accuracy_threshold}%",
                                    delta_color="normal" if overall_accuracy >= accuracy_threshold else "inverse"
                                )
                            with col2:
                                status = "✅ PASS" if overall_accuracy >= accuracy_threshold else "❌ FAIL"
                                st.metric("Threshold Status", status)
                            with col3:
                                gap = overall_accuracy - accuracy_threshold
                                st.metric("Gap from Threshold", f"{gap:+.1f}%")
                            
                            # Display accuracy gauge
                            fig = plot_accuracy_gauge(overall_accuracy, accuracy_threshold, f"{synth_name} Accuracy Gauge")
                            st.pyplot(fig)
                            plt.close(fig)
                            
                            # Display accuracy breakdown
                            st.write("*Accuracy Breakdown:*")
                            breakdown_df = pd.DataFrame({
                                "Metric": list(accuracy_breakdown.keys()),
                                "Score (%)": list(accuracy_breakdown.values())
                            })
                            st.dataframe(breakdown_df.style.format({"Score (%)": "{:.1f}%"}))
                            
                            # Threshold-based recommendations
                            st.write("*Threshold Analysis:*")
                            if overall_accuracy >= accuracy_threshold:
                                st.success(f"✅ *PASSED*: Synthetic data meets or exceeds the {accuracy_threshold}% accuracy threshold")
                                st.markdown("🎉 *Recommendation*: Data is ready for use!")
                            else:
                                st.error(f"⚠ *FAILED*: Synthetic data is below the {accuracy_threshold}% accuracy threshold")
                                st.markdown("🔧 *Recommendations for Improvement:*")
                                
                                # Specific recommendations based on low scores
                                if accuracy_breakdown["KS Score"] < 70:
                                    st.write("1. *Improve Distribution Matching*: Increase training epochs or try different synthesizer parameters")
                                if accuracy_breakdown["Mean Diff Score"] < 70:
                                    st.write("2. *Improve Statistical Moments*: Check for outliers or consider data normalization")
                                if accuracy_breakdown["Correlation Score"] < 70:
                                    st.write("3. *Improve Relationships*: Use CTGAN with higher discriminator steps for better correlation capture")
                                
                                # General recommendations
                                st.write("4. *Try*: Increase 'Enhanced epochs' in sidebar")
                                st.write("5. *Try*: Disable Fast Mode for better quality (slower)")
                                st.write("6. *Try*: Increase training data size")

                    st.subheader("📄 Summary Report")
                    st.write("*Data Statistics:*")
                    st.write(f"- Original Data Rows: {len(df)}")
                    st.write(f"- Training Data Rows: {len(df_train)}")
                    for synth_name, synthetic_data in synthetic_datasets.items():
                        st.write(f"- Synthetic Data Rows ({synth_name}): {len(synthetic_data)}")
                    st.write(f"- Columns: {len(df.columns)}")

                    st.write("*Training Parameters:*")
                    st.write(f"- Fast Mode: {'Enabled' if fast_mode else 'Disabled'}")
                    if "CTGAN" in synthetic_datasets:
                        st.write(f"- CTGAN Epochs: {epochs}")
                        st.write(f"- CTGAN Batch Size: {batch_size}")
                        st.write(f"- CTGAN Discriminator Steps: {discriminator_steps}")
                        st.write(f"- CTGAN GPU Used: {'Yes' if use_gpu_final else 'No'}")
                    st.write(f"- Synthesizers Used: {', '.join(synthetic_datasets.keys())}")

                    st.write("*Quality Metrics:*")
                    for synth_name, quality_report in quality_reports.items():
                        if quality_report is not None:
                            st.write(f"- {synth_name} Overall Quality Score: {quality_report.get_score():.4f}")
                        else:
                            st.write(f"- {synth_name} Quality Report: Not computed")
                    for synth_name, summaries in summary_list.items():
                        if summaries:
                            avg_ks = np.mean([x["KS Statistic"] for x in summaries])
                            avg_overlap = np.mean([x["Distribution Overlap"] for x in summaries])
                            avg_mean_diff = np.mean([x["Mean Diff %"] for x in summaries])
                            avg_w_dist = np.mean([x["Wasserstein Distance"] for x in summaries])
                            st.write(f"- {synth_name} Average KS Statistic: {avg_ks:.4f}")
                            st.write(f"- {synth_name} Average Distribution Overlap: {avg_overlap:.3f}")
                            st.write(f"- {synth_name} Average Mean Difference: {avg_mean_diff:.1f}%")
                            st.write(f"- {synth_name} Average Wasserstein Distance: {avg_w_dist:.4f}")
                        corr_diff = correlation_diffs.get(synth_name, {"Corr Diff": 0.0, "Status": "Not computed"})
                        st.write(f"- {synth_name} Average Correlation Difference: {corr_diff['Corr Diff']:.4f} ({corr_diff['Status']})")
                    
                    # ADDED: Accuracy Threshold Assessment
                    st.write("*Accuracy Threshold Assessment:*")
                    for synth_name, summaries in summary_list.items():
                        if summaries:
                            corr_diff_info = correlation_diffs.get(synth_name, {"Corr Diff": 0.0, "Status": "Not computed"})
                            quality_report = quality_reports.get(synth_name)
                            
                            overall_accuracy, _ = calculate_overall_accuracy(
                                summaries, 
                                corr_diff_info,
                                quality_report
                            )
                            
                            threshold_status = "✅ PASS" if overall_accuracy >= accuracy_threshold else "❌ FAIL"
                            st.write(f"- *{synth_name}*: {overall_accuracy:.1f}% - {threshold_status}")

                    st.write("*Preprocessing and Issues:*")
                    st.write(f"- PCA Applied: {'Yes' if pca_applied else 'No'}")
                    st.write(f"- Dropped Columns: {to_drop if to_drop else 'None'}")
                    st.write(f"- Validation Issues: {', '.join(validation_issues) if validation_issues else 'None'}")

                    st.write("*Key Insights:*")
                    for synth_name, summaries in summary_list.items():
                        if summaries:
                            corr_diff_info = correlation_diffs.get(synth_name, {"Corr Diff": 0.0, "Status": "Not computed"})
                            quality_report = quality_reports.get(synth_name)
                            
                            overall_accuracy, _ = calculate_overall_accuracy(
                                summaries, 
                                corr_diff_info,
                                quality_report
                            )
                            
                            if overall_accuracy >= accuracy_threshold:
                                st.success(f"{synth_name}: ✅ Synthetic data passes accuracy threshold ({overall_accuracy:.1f}% ≥ {accuracy_threshold}%)")
                            else:
                                st.warning(f"{synth_name}: ⚠ Synthetic data fails accuracy threshold ({overall_accuracy:.1f}% < {accuracy_threshold}%). Consider parameter adjustments.")
                        
                        if any(item['KS Statistic'] > 0.3 or item['Mean Diff %'] > 25 for item in summaries):
                            st.warning(f"{synth_name}: Synthetic data quality needs improvement. Consider adjusting parameters or increasing training data.")
                        
                        if synthesizer_choice == "Both (CTGAN + GaussianCopula)" and len(summary_list) == 2:
                            ctgan_ks = np.mean([x["KS Statistic"] for x in summary_list["CTGAN"]]) if summary_list["CTGAN"] else float('inf')
                            gc_ks = np.mean([x["KS Statistic"] for x in summary_list["GaussianCopula"]]) if summary_list["GaussianCopula"] else float('inf')
                            if ctgan_ks < gc_ks:
                                st.write(f"*Comparison*: CTGAN outperforms GaussianCopula (lower average KS: {ctgan_ks:.4f} vs {gc_ks:.4f})")
                            elif gc_ks < ctgan_ks:
                                st.write(f"*Comparison*: GaussianCopula outperforms CTGAN (lower average KS: {gc_ks:.4f} vs {ctgan_ks:.4f})")
                            else:
                                st.write("*Comparison*: CTGAN and GaussianCopula have similar performance")

                    st.success("🎉 Done — synthetic data generated and evaluated!")

                except Exception as e:
                    st.error(f"🚫 Error during processing: {e}")
                    st.code(traceback.format_exc())

        except Exception as e:
            st.error(f"🚫 Error reading file: {e}")
            st.code(traceback.format_exc())
    else:
        st.info("📂 Please upload a file to proceed.")