import streamlit as st
from typing import List, Dict
import numpy as np
import plotly.express as px
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Streamlit app configuration
st.set_page_config(
    page_title="CRISPR-Cas9 PAM Evolution Simulator",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧬 Evolve Simulation")
st.markdown(
    "This app simulates different sampling strategies for evolving CRISPR-Cas9 variants with improved PAM recognition."
)

# Sidebar for hyperparameters
st.sidebar.header("Configuration")

# Primary parameters - always visible
bases = ["A", "C", "G", "T"]
pam_options = [f"NG{b1}{b2}" for b1 in bases for b2 in bases]
pam = st.sidebar.selectbox("PAM", pam_options, index=0, help="Protospacer Adjacent Motif sequence")
strategy = st.sidebar.selectbox("Sampling Strategy", ["random", "esmc"], index=0, help="Strategy for selecting sequences in each round")

# Advanced settings - collapsible
with st.sidebar.expander("⚙️ Advanced Settings", expanded=False):
    st.markdown("##### Experiment Parameters")
    num_rounds = st.selectbox("Number of Rounds", [10, 20], index=0, help="Total number of experimental rounds")
    sample_per_round = st.selectbox("Samples per Round", [10, 15, 20, 25], index=0, help="Number of samples to test in each round")
    
    st.markdown("##### ESMC-specific Settings")
    # ESMC-specific parameters
    pool_size = 1000
    esmc_model = "esmc_300m"
    if strategy == "esmc":
        pool_size = st.selectbox("Pool Size", [1000, 2000, 5000, 10000], index=0, help="Size of the candidate pool for ESMC sampling")
        esmc_model = st.selectbox("ESMC Model", ["esmc_300m", "esmc_600m"], index=0, help="ESM model variant to use for embeddings")
    else:
        st.info("Pool Size and ESMC Model settings are only available when using ESMC strategy")
    
    st.markdown("##### Reproducibility")
    seed = st.selectbox("Random Seed", [42, 43], index=0, help="Seed for reproducible results")

hparams = {
    "num_round": num_rounds,
    "strategy": strategy,
    "sample_per_round": sample_per_round,
    "pam": pam,
    "pool_size": pool_size,
    "esmc_model": esmc_model,
    "seed": seed,
}


class RandomSampler:
    def __init__(self, sample_per_round: int, pool_df: pd.DataFrame):
        self.sample_per_round = sample_per_round
        self.pool_df = pool_df

    def sample(self, trajectory: List[Dict]) -> List:
        round_idx = len(trajectory)
        return self.pool_df.index[
            round_idx * self.sample_per_round : (round_idx + 1) * self.sample_per_round
        ].tolist()


class EsmcSampler:
    def __init__(self, sample_per_round: int, pool_df: pd.DataFrame):
        self.sample_per_round = sample_per_round
        self.pool_df = pool_df

    def sample(self, trajectory: List[Dict]) -> List:
        # random sample if there is no existing samples
        if len(trajectory) == 0:
            return self.pool_df.index[: self.sample_per_round].tolist()

        # dataset from trajectory
        all_data = []
        for round_data in trajectory:
            for sample, activity in zip(round_data["samples"], round_data["activity"]):
                all_data.append({"sample": sample, "activity": activity})
        df = pd.DataFrame(all_data)

        # get embedding columns
        feature_cols = [col for col in self.pool_df.columns if col.startswith("emb_")]
        X = self.pool_df.loc[df["sample"], feature_cols].values
        y = df["activity"].values

        # fit random forest model
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            min_samples_split=3,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
        )
        rf_model.fit(X, y)

        # predict on all pool candidates
        pool_embeddings = self.pool_df[feature_cols].values
        pool_predictions = rf_model.predict(pool_embeddings)

        # exclude already run samples
        sample_all = set(
            sample for round_data in trajectory for sample in round_data["samples"]
        )

        # get top candidates excluding already run samples
        sorted_indices = np.argsort(pool_predictions)[::-1]  # sort in descending order
        top_samples = []
        for idx in sorted_indices:
            sample = self.pool_df.index[idx]
            if sample not in sample_all:
                top_samples.append(sample)
                if len(top_samples) == self.sample_per_round:
                    break

        return top_samples


@st.cache_data
def load_pool_data(seed: int, esmc_model: str, pool_size: int):
    """Load the pool data with caching"""
    try:
        pool_df = pd.read_pickle(f"data/sample-{seed}-{esmc_model}.pkl")
        pool_df = pool_df.head(pool_size)
        return pool_df
    except Exception as e:
        st.error(f"Error loading pool data: {str(e)}")
        st.stop()


def visualize_trajectory(trajectory):
    data = []
    for round_idx, round_data in enumerate(trajectory):
        for activity in round_data["activity"]:
            data.append({"round": round_idx + 1, "activity": activity})

    df = pd.DataFrame(data)

    fig = px.strip(
        df,
        x="round",
        y="activity",
        title="Activity Trajectory Across Rounds",
        labels={"round": "Round", "activity": "Activity"},
    )
    fig.update_traces(marker=dict(size=5, opacity=0.7))
    fig.update_layout(
        xaxis=dict(tickmode="linear", tick0=1, dtick=1),
        yaxis=dict(range=[-5.5, -1.5]),
        showlegend=False,
        height=500,
    )

    return fig, df


def run_simulation():
    """Run the main simulation"""
    trajectory = []
    # Load pool data
    pool_df = load_pool_data(
        hparams["seed"], hparams["esmc_model"], hparams["pool_size"]
    )

    # Initialize sampler
    if hparams["strategy"] == "random":
        sampler = RandomSampler(hparams["sample_per_round"], pool_df)
    elif hparams["strategy"] == "esmc":
        sampler = EsmcSampler(hparams["sample_per_round"], pool_df)
    else:
        raise ValueError(f"Not a valid strategy: {hparams['strategy']}")

    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(hparams["num_round"]):
        status_text.text(f'Running round {i+1}/{hparams["num_round"]}...')
        progress_bar.progress((i + 1) / hparams["num_round"])

        samples = sampler.sample(trajectory)
        round_data = {"samples": samples}

        # get activities from pre-generated data
        activities = []
        for sample in samples:
            if sample in pool_df.index:
                activity = pool_df.loc[sample, hparams["pam"]]
                activities.append(activity)
            else:
                raise ValueError(f"Sample {sample} not found in pre-generated data")
        round_data["activity"] = activities
        trajectory.append(round_data)

    progress_bar.empty()
    status_text.empty()

    return trajectory


# Main app logic
col1, col2 = st.columns([3, 1])

with col2:
    if st.button("🚀 Run Simulation", type="primary"):
        st.session_state.run_simulation = True

if st.session_state.get("run_simulation", False):
    st.markdown("---")

    with st.spinner("Loading data and running simulation..."):
        trajectory = run_simulation()

    # Display results
    st.subheader("📊 Results")

    # Visualization
    fig, df = visualize_trajectory(trajectory)
    st.plotly_chart(fig, use_container_width=True)

    # Statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Rounds", len(trajectory))

    with col2:
        st.metric("Total Samples", len(trajectory) * hparams["sample_per_round"])

    with col3:
        avg_activity = df["activity"].mean()
        st.metric("Avg Activity", f"{avg_activity:.4f}")

    with col4:
        max_activity = df["activity"].max()
        st.metric("Max Activity", f"{max_activity:.4f}")

    # Show trajectory data
    with st.expander("📋 View Detailed Trajectory Data"):
        for i, round_data in enumerate(trajectory):
            st.subheader(f"Round {i+1}")
            round_df = pd.DataFrame(
                {
                    "Sample": [str(sample) for sample in round_data["samples"]],
                    "Activity": round_data["activity"],
                }
            )
            st.dataframe(round_df, use_container_width=True)

    # Download data
    st.subheader("💾 Download Results")
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Activity Data as CSV",
        data=csv,
        file_name=f"trajectory_data_{hparams['strategy']}_{hparams['num_round']}rounds.csv",
        mime="text/csv",
    )

# Information section
st.markdown("---")
st.subheader("ℹ️ About")
st.markdown(
    """
This application simulates directed evolution of CRISPR-Cas9 variants to improve PAM (Protospacer Adjacent Motif) recognition using different sampling strategies.

**Based on Research:**  
This app uses experimental data and machine learning models from:
> Huang, J., Kim, E., et al. "Custom CRISPR–Cas9 PAM variants via scalable engineering and machine learning."  
> *Nature* (2025). https://doi.org/10.1038/s41586-025-09021-y

**What it does:**  
The simulation explores how different sampling strategies (Random vs ESMC) perform in identifying Cas9 variants with enhanced activity for specific PAM sequences, mimicking the directed evolution process described in the paper.

**Features:**
- Random and ESMC sampling strategies
- Configurable pool sizes and model types (for ESMC)
- Real-time progress tracking
- Interactive visualizations
- Downloadable results

**Usage:**
1. Adjust parameters in the sidebar
2. Click "Run Simulation" to start the analysis
3. View results and download data as needed
"""
)
