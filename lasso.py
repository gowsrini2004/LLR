import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2, venn3
from io import BytesIO

# ------------------------------
# Utility Functions
# ------------------------------

def load_data(uploaded_file):
    """Safe loader for CSV/Excel files"""
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                return pd.read_csv(BytesIO(uploaded_file.getvalue()))
            else:
                return pd.read_excel(BytesIO(uploaded_file.getvalue()))
        except Exception as e:
            st.error(f"Error reading file {uploaded_file.name}: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def filter_genes(df, pval_col, threshold=0.05):
    """Filter genes by adjusted p-value (force numeric)"""
    df = df.copy()
    df[pval_col] = pd.to_numeric(df[pval_col], errors="coerce")
    return df[df[pval_col] < threshold]

def run_lasso(training_df, gene_col, pval_col, fc_col, threshold=0.05):
    """
    Simulated Lasso: keep significant genes and use |logFC| as a proxy weight.
    """
    sig_genes = filter_genes(training_df, pval_col, threshold)
    if sig_genes.empty:
        return pd.DataFrame(columns=[gene_col, fc_col, pval_col, "LassoCoef"])

    sig_genes = sig_genes.copy()
    sig_genes[fc_col] = pd.to_numeric(sig_genes[fc_col], errors="coerce")
    sig_genes["LassoCoef"] = sig_genes[fc_col].apply(lambda x: abs(x))
    sig_genes = sig_genes.sort_values("LassoCoef", ascending=False)
    return sig_genes

def check_overlap(training_df, test_df, gene_col, train_fc_col, test_fc_col, test_pval_col, threshold=0.05):
    """Check overlap and concordance of DEGs. Return detailed DataFrame."""
    test_sig = filter_genes(test_df, test_pval_col, threshold)

    training_df = training_df.copy()
    training_df[train_fc_col] = pd.to_numeric(training_df[train_fc_col], errors="coerce")
    test_sig[test_fc_col] = pd.to_numeric(test_sig[test_fc_col], errors="coerce")

    if test_sig.empty or training_df.empty:
        return pd.DataFrame()

    overlap = set(training_df[gene_col]).intersection(set(test_sig[gene_col]))
    results = []
    for gene in overlap:
        train_fc = training_df.loc[training_df[gene_col] == gene, train_fc_col].values[0]
        test_fc = test_sig.loc[test_sig[gene_col] == gene, test_fc_col].values[0]
        concordant = "Yes" if (pd.notna(train_fc) and pd.notna(test_fc) and np.sign(train_fc) == np.sign(test_fc)) else "No"
        results.append([gene, train_fc, test_fc, concordant])

    df_overlap = pd.DataFrame(results, columns=["Gene", "Training logFC", "Test logFC", "Concordant"])
    return df_overlap

# ------------------------------
# Streamlit App
# ------------------------------

st.title("🧬 Lasso Logistic Regression DEG Signature Builder")

st.sidebar.header("Upload Datasets")
train_file = st.sidebar.file_uploader("Upload Training DEG file", type=["csv", "xlsx"])
test_files = st.sidebar.file_uploader("Upload Test DEG files", type=["csv", "xlsx"], accept_multiple_files=True)

threshold = st.sidebar.number_input("Adjusted P-value threshold", value=0.05, step=0.01)

# ------------------------------
# Training dataset
# ------------------------------
if train_file:
    st.subheader("Training Dataset")
    train_df = load_data(train_file)
    st.write("Preview of training dataset:", train_df.head())

    with st.expander("Select Training Columns"):
        gene_col_train = st.selectbox("Gene column (Training)", train_df.columns, index=0)
        pval_col_train = st.selectbox("Adj. P-value column (Training)", train_df.columns, index=1)
        fc_col_train = st.selectbox("logFC column (Training)", train_df.columns, index=2)

# ------------------------------
# Testing datasets
# ------------------------------
test_datasets = []
if test_files:
    st.subheader("Testing Datasets")
    for f in test_files:
        test_df = load_data(f)
        st.write(f"Preview of {f.name}:", test_df.head())
        with st.expander(f"Select Columns for {f.name}"):
            gene_col_test = st.selectbox(f"Gene column ({f.name})", test_df.columns, key=f"{f.name}_gene")
            pval_col_test = st.selectbox(f"P-value column ({f.name})", test_df.columns, key=f"{f.name}_pval")
            fc_col_test = st.selectbox(f"logFC column ({f.name})", test_df.columns, key=f"{f.name}_fc")
        test_datasets.append((f.name, test_df, gene_col_test, pval_col_test, fc_col_test))

# ------------------------------
# Run Analysis
# ------------------------------
if st.button("Run Analysis") and train_file:
    # Run simulated Lasso
    selected_genes = run_lasso(train_df, gene_col_train, pval_col_train, fc_col_train, threshold)
    st.success(f"Selected {len(selected_genes)} genes (simulated Lasso)")
    st.dataframe(selected_genes)

    # Volcano Plot
    with st.expander("Volcano Plot (Training Dataset)"):
        if not train_df.empty:
            plt.figure(figsize=(6,5))
            sns.scatterplot(
                data=train_df,
                x=fc_col_train,
                y=-np.log10(pd.to_numeric(train_df[pval_col_train], errors='coerce')),
                alpha=0.6
            )
            if not selected_genes.empty:
                sns.scatterplot(
                    data=selected_genes,
                    x=fc_col_train,
                    y=-np.log10(pd.to_numeric(selected_genes[pval_col_train], errors='coerce')),
                    color='red'
                )
            plt.xlabel("logFC")
            plt.ylabel("-log10(p-value)")
            st.pyplot(plt)

    # Validation
    if test_datasets and not selected_genes.empty:
        st.subheader("Validation Across Test Datasets")
        overlap_counts = []
        for fname, test_df, gcol, pcol, fcol in test_datasets:
            df_overlap = check_overlap(
                selected_genes, test_df,
                gene_col_train, fc_col_train, fcol, pcol, threshold
            )
            overlap_counts.append((fname, len(df_overlap), (df_overlap["Concordant"]=="Yes").sum()))
            with st.expander(f"Results for {fname}"):
                st.write(f"Overlap genes: {len(df_overlap)} | Concordant: {(df_overlap['Concordant']=='Yes').sum()}")
                if not df_overlap.empty:
                    st.dataframe(df_overlap)

        # Barplot of overlaps
        with st.expander("Barplot of Overlaps"):
            if overlap_counts:
                df_plot = pd.DataFrame(overlap_counts, columns=["Dataset", "Overlap", "Concordant"])
                plt.figure(figsize=(7,5))
                df_plot.set_index("Dataset")[["Overlap","Concordant"]].plot(kind="bar")
                plt.ylabel("Gene Count")
                plt.title("Overlap and Concordance per Dataset")
                st.pyplot(plt)

        # Heatmap of logFC values for overlapping genes (if >0 overlap)
        with st.expander("Heatmap of Overlapping Genes"):
            overlap_all = set()
            for fname, test_df, gcol, pcol, fcol in test_datasets:
                df_overlap = check_overlap(selected_genes, test_df, gene_col_train, fc_col_train, fcol, pcol, threshold)
                overlap_all |= set(df_overlap["Gene"])
            if overlap_all:
                heatmap_df = pd.DataFrame(index=sorted(list(overlap_all)))
                for fname, test_df, gcol, pcol, fcol in test_datasets:
                    test_sig = filter_genes(test_df, pcol, threshold)
                    test_sig[fcol] = pd.to_numeric(test_sig[fcol], errors="coerce")
                    heatmap_df[fname] = test_sig.set_index(gcol).reindex(heatmap_df.index)[fcol]
                heatmap_df = heatmap_df.fillna(0)
                plt.figure(figsize=(8, max(4, len(heatmap_df)//2)))
                sns.heatmap(heatmap_df, cmap="coolwarm", center=0)
                plt.title("LogFC Heatmap of Overlapping Genes")
                st.pyplot(plt)


    # Download Consensus Signature
    if not selected_genes.empty:
        st.subheader("Download Consensus Signature")
        csv = selected_genes.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "consensus_signature.csv", "text/csv")