"""
Dataset & Experiment Analytics Module for DeepShield
Parses and visualizes genuine experimental data from:
1. member3_ppt_summary.csv
2. full_dataset_synchronization_results.csv
3. deepshield-model.ipynb test evaluation benchmarks
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


class DeepShieldDatasetAnalytics:
    def __init__(self, summary_csv="results/member3_ppt_summary.csv", full_csv="results/full_dataset_synchronization_results.csv"):
        # Fallbacks to root if needed
        self.summary_csv = summary_csv if os.path.exists(summary_csv) else "member3_ppt_summary.csv"
        self.full_csv = full_csv if os.path.exists(full_csv) else "full_dataset_synchronization_results.csv"

        self.df_summary = None
        self.df_full = None
        self.load_data()

    def load_data(self):
        if os.path.exists(self.summary_csv):
            self.df_summary = pd.read_csv(self.summary_csv)
        if os.path.exists(self.full_csv):
            self.df_full = pd.read_csv(self.full_csv)

    def get_summary_table(self):
        """Returns the exact PPT summary table prepared by Member 3."""
        if self.df_summary is not None:
            return self.df_summary
        return pd.DataFrame()

    def get_full_dataset_stats(self):
        """Calculates detailed descriptive stats by class from the 230 video dataset."""
        if self.df_full is None:
            return pd.DataFrame()

        grouped = self.df_full.groupby("Actual Class")

        stats_sync = grouped["Sync Score"].agg([
            ("Count", "count"),
            ("Mean", "mean"),
            ("Std Dev", "std"),
            ("Median", "median"),
            ("Min", "min"),
            ("Max", "max")
        ]).round(2)

        stats_corr = grouped["Correlation"].agg([
            ("Mean Corr", "mean"),
            ("Std Corr", "std"),
            ("Median Corr", "median"),
        ]).round(4)

        combined = pd.concat([stats_sync, stats_corr], axis=1)
        return combined

    def build_sync_score_boxplot(self):
        """Creates an interactive Plotly box plot for synchronization score distributions."""
        if self.df_full is None:
            return None

        fig = px.box(
            self.df_full,
            x="Actual Class",
            y="Sync Score",
            color="Actual Class",
            points="all",
            color_discrete_map={
                "Real": "#10b981",
                "Fake": "#ef4444",
                "Fake Voice Only": "#f59e0b"
            },
            title="Audio–Lip Synchronization Score Distribution by Class (230 Videos)",
            labels={"Sync Score": "Sync Score (max(0, r) × 100)", "Actual Class": "Ground Truth Video Class"}
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(15, 23, 42, 0.4)",
            plot_bgcolor="rgba(15, 23, 42, 0.4)",
            showlegend=False,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        return fig

    def build_correlation_boxplot(self):
        """Creates an interactive Plotly box plot for Pearson correlation distribution."""
        if self.df_full is None:
            return None

        fig = px.box(
            self.df_full,
            x="Actual Class",
            y="Correlation",
            color="Actual Class",
            points="all",
            color_discrete_map={
                "Real": "#10b981",
                "Fake": "#ef4444",
                "Fake Voice Only": "#f59e0b"
            },
            title="Raw Pearson Correlation (r) Distribution by Video Category",
            labels={"Correlation": "Pearson Correlation (r)", "Actual Class": "Ground Truth Class"}
        )
        # Add horizontal reference line at r = 0
        fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="r = 0 (No Correlation)")
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(15, 23, 42, 0.4)",
            plot_bgcolor="rgba(15, 23, 42, 0.4)",
            showlegend=False,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        return fig

    def build_pvalue_histogram(self):
        """Visualizes p-value distribution and statistical significance across classes."""
        if self.df_full is None:
            return None

        fig = px.histogram(
            self.df_full,
            x="P-value",
            color="Actual Class",
            barmode="overlay",
            nbins=30,
            color_discrete_map={
                "Real": "#10b981",
                "Fake": "#ef4444",
                "Fake Voice Only": "#f59e0b"
            },
            title="P-value Distribution Across Video Categories",
            labels={"P-value": "Significance P-value"}
        )
        fig.add_vline(x=0.05, line_dash="dash", line_color="#38bdf8", annotation_text="α = 0.05 Significance")
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(15, 23, 42, 0.4)",
            plot_bgcolor="rgba(15, 23, 42, 0.4)",
            margin=dict(l=40, r=40, t=50, b=40)
        )
        return fig

    def build_visual_model_confusion_matrix(self):
        """Visualizes the verified EfficientNet-B0 test set confusion matrix from deepshield-model.ipynb."""
        # Verified data from deepshield-model.ipynb Cell 20
        cm_data = [[3079, 4], [4, 147]]
        labels = ["FAKE", "REAL"]

        fig = go.Figure(data=go.Heatmap(
            z=cm_data,
            x=labels,
            y=labels,
            hoverongaps=False,
            colorscale="Viridis",
            text=[[f"{v:,}" for v in row] for row in cm_data],
            texttemplate="%{text}",
            textfont={"size": 18, "color": "white"}
        ))

        fig.update_layout(
            title="EfficientNet-B0 Test Set Confusion Matrix (3,234 Test Frames)",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            template="plotly_dark",
            paper_bgcolor="rgba(15, 23, 42, 0.4)",
            plot_bgcolor="rgba(15, 23, 42, 0.4)",
            width=500,
            height=400,
            margin=dict(l=60, r=40, t=60, b=60)
        )
        return fig

    def get_visual_model_metrics(self):
        """Returns the verified test set metrics from deepshield-model.ipynb."""
        return {
            "accuracy": 0.9975,
            "precision": 0.9735,
            "recall": 0.9735,
            "f1_score": 0.9735,
            "total_test_frames": 3234,
            "fake_frames": 3083,
            "real_frames": 151,
            "architecture": "EfficientNet-B0 (Pretrained ImageNet backbone + custom 2-class head)",
            "epochs": 30,
            "learning_rate": "0.0001 (Adam)"
        }
