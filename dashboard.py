import re
import sys
import traceback

import pandas as pd
import streamlit as st

import ituna.config

try:
    from dj_ml_core.utils import flatten_dataframe

    from ituna._backends.datajoint import get_sweep_jobs_for_table

    _DATAJOINT_AVAILABLE = True
except ImportError:
    flatten_dataframe = None
    get_sweep_jobs_for_table = None
    _DATAJOINT_AVAILABLE = False

st.set_page_config(page_title="DataJoint Experiment Dashboard", page_icon="🧪", layout="wide")

if not _DATAJOINT_AVAILABLE:
    st.error(
        "Dashboard requires ituna[datajoint] and dj_ml_core. "
        "Install with: pip install ituna[datajoint]. "
        "Note: dj_ml_core may need to be installed from the project's wheel (e.g. third_party/dj_ml_core-*.whl)."
    )
    st.stop()

st.markdown(
    """
    <style>
        .stMultiSelect [data-baseweb=select] span{
            max-width: none;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def clear_all_caches():
    """Clear all data caches."""
    # Clear function-specific caches
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("All data caches have been cleared.")


@st.cache_data
def get_all_sweeps(_backend, filter_uuids=False):
    """Get all available sweeps from the database."""
    try:
        sweeps = _backend.tables.SweepTable().fetch("sweep_name")
        sweeps_list = sorted(sweeps.tolist())
        if filter_uuids:
            uuid_pattern = re.compile(
                r"^[0-9a-fA-F]{8}-"
                r"[0-9a-fA-F]{4}-"
                r"[0-9a-fA-F]{4}-"
                r"[0-9a-fA-F]{4}-"
                r"[0-9a-fA-F]{12}$"
            )
            sweeps_list = [s for s in sweeps_list if not uuid_pattern.match(s)]
        return sweeps_list
    except Exception as e:
        st.error(f"Error fetching sweeps: {str(e)}")
        return []


@st.cache_data
def get_experiment_configs_for_sweep(_backend, sweep_name, show_details=True):
    """Get experiment configurations for a specific sweep.

    Args:
        sweep_name: Name of the sweep
        show_details: If True, join with all config tables to show full details.
                     If False, just return basic config table for counting.
    """
    try:
        sweep_runs = _backend.tables.SweepRunsTable() & {"sweep_name": sweep_name}
        exp_configs = _backend.tables.ModelTrainingTable() & sweep_runs
        if len(exp_configs) == 0:
            return pd.DataFrame()

        # If we need full details, join with model and dataset tables
        if show_details:
            model_details = _backend.tables.ModelTable.proj(model_config="model_config", model_class="model_class_name", model_module="model_module_name")
            dataset_details = _backend.tables.DatasetTable.proj("data_config")

            exp_configs = exp_configs * model_details * dataset_details

        configs_df = exp_configs.fetch(format="frame")
        configs_df = flatten_dataframe(configs_df)
        configs_df = configs_df.reset_index()

        # Fix data types for Streamlit/Arrow compatibility
        for col in configs_df.columns:
            if configs_df[col].dtype == "object":
                configs_df[col] = configs_df[col].astype(str)

        return configs_df
    except Exception as e:
        st.error(f"Error fetching experiment configs: {str(e)}")
        return pd.DataFrame()


def get_experiment_results_for_sweep(sweep_name):
    """
    Get experiment results (train/test metrics) for a specific sweep.
    NOTE: The new backend does not support metrics tables. This function is a placeholder.
    """
    return pd.DataFrame(), pd.DataFrame()


def show_main_dashboard():
    """Show the main dashboard after connection is established."""
    st.title("🧪 DataJoint Experiment Dashboard")

    # Connection status
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        if st.session_state.get("db_connected", False):
            conn_params = st.session_state.connection_params
            st.success(f"Connected to: {conn_params.get('host')} as {conn_params.get('user')} (Schema: {conn_params.get('schema')})")
        else:
            st.error("Not connected to database.")
    with col2:
        if st.button("Reconnect"):
            # This will clear the backend and schema name, forcing a reconnect
            st.session_state.clear()
            st.rerun()
    with col3:
        if st.button("Clear Caches"):
            clear_all_caches()
            st.rerun()

    st.markdown("---")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "ModelTraining Details", "Model Details", "Dataset Details", "Jobs Management"],
        index=0,
    )  # Default to Overview

    if page == "Overview":
        show_overview()
    elif page == "ModelTraining Details":
        show_modeltraining_details()
    elif page == "Model Details":
        show_model_details()
    elif page == "Dataset Details":
        show_dataset_details()
    elif page == "Jobs Management":
        show_jobs_management()


def show_overview():
    """Show overview of all sweeps and their statistics."""
    backend = st.session_state.backend
    # Add refresh button
    col1, col2 = st.columns([6, 1])
    with col1:
        st.header("Experiment Sweeps Overview")
    with col2:
        if st.button("🔄 Refresh", key="overview_refresh"):
            get_all_sweeps.clear()
            st.success("Overview data refreshed.")
            st.rerun()

    # Get all sweeps, filtering out UUIDs for the overview
    all_sweeps = get_all_sweeps(backend, filter_uuids=True)

    if not all_sweeps:
        st.warning("No non-UUID sweeps found in the database. Use other pages to filter for specific sweeps by name or UUID.")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader(f"Available Sweeps ({len(all_sweeps)})")
        for sweep in all_sweeps:
            st.write(f"• {sweep}")

    with col2:
        st.subheader("Sweep Statistics")

        # Create a summary table using len() for counts
        sweep_stats = []
        for sweep in all_sweeps:
            try:
                # Get configs for the sweep
                sweep_runs = backend.tables.SweepRunsTable() & {"sweep_name": sweep}
                configs = backend.tables.ModelTrainingTable() & sweep_runs
                completed_experiments = backend.tables.TrainedModelsTable() & configs

                sweep_stats.append(
                    {
                        "Sweep": sweep,
                        "Configs": len(configs),
                        "Completed Experiments": len(completed_experiments),
                    }
                )
            except Exception as e:
                print(e)
                sweep_stats.append(
                    {
                        "Sweep": sweep,
                        "Configs": "Error",
                        "Completed Experiments": "Error",
                    }
                )

        if sweep_stats:
            stats_df = pd.DataFrame(sweep_stats)
            st.dataframe(stats_df, width="stretch")


def show_modeltraining_details():
    """Show detailed view of ModelTrainingTable for selected sweeps."""
    backend = st.session_state.backend
    st.header("ModelTraining Details")

    # Initialize session state for this page
    if "exp_details" not in st.session_state:
        st.session_state.exp_details = {
            "selected_rows_indices": [],
        }
    exp_state = st.session_state.exp_details

    # Refresh button
    if st.button("🔄 Refresh", key="config_refresh"):
        # Clear only caches relevant to this page
        get_all_sweeps.clear()
        get_sweep_summary_data.clear()
        get_combined_config_data.clear()
        st.success("Experiment details data refreshed.")
        st.rerun()

    # Sweep selection with filtering
    if not get_all_sweeps(backend, filter_uuids=False):
        st.warning("No sweeps found in the database.")
        return

    sweep_filter = st.text_input("Filter sweeps by name (supports partial matching):")

    if sweep_filter:
        all_sweeps = get_all_sweeps(backend, filter_uuids=False)
        matched_sweeps = [s for s in all_sweeps if sweep_filter in s]
    else:
        matched_sweeps = get_all_sweeps(backend, filter_uuids=True)

    selected_sweeps = st.multiselect(
        "Select sweeps to explore:",
        options=matched_sweeps,
        default=matched_sweeps,
        help="Choose one or more sweeps to view their experiment configurations",
    )
    exp_state["selected_sweeps"] = selected_sweeps

    if exp_state["selected_sweeps"]:
        display_multi_sweep_details(exp_state)


@st.cache_data
def get_sweep_summary_data(_backend, sweep_name):
    """Fetches and computes summary statistics for a given sweep."""
    try:
        sweep_runs = _backend.tables.SweepRunsTable() & {"sweep_name": sweep_name}
        configs = _backend.tables.ModelTrainingTable() & sweep_runs
        if len(configs) == 0:
            return {
                "Sweep": sweep_name,
                "Configs": 0,
                "Completed Experiments": 0,
            }

        completed_experiments = _backend.tables.TrainedModelsTable() & configs

        return {
            "Sweep": sweep_name,
            "Configs": len(configs),
            "Completed Experiments": len(completed_experiments),
        }
    except Exception as e:
        st.error(f"Error getting summary for {sweep_name}: {e}")
        return {
            "Sweep": sweep_name,
            "Configs": "Error",
            "Completed Experiments": "Error",
        }


@st.cache_data
def get_combined_config_data(_backend, selected_sweeps):
    """Fetches and combines configuration data for a list of sweeps."""
    all_configs_list = []
    for sweep_name in selected_sweeps:
        configs_df = get_experiment_configs_for_sweep(_backend, sweep_name, show_details=True)
        if not configs_df.empty:
            if isinstance(configs_df.index, pd.MultiIndex):
                configs_df = configs_df.reset_index()
            configs_df["sweep_name"] = sweep_name
            all_configs_list.append(configs_df)

    if not all_configs_list:
        return pd.DataFrame()

    combined_df = pd.concat(all_configs_list, ignore_index=True).reset_index(drop=True)
    return combined_df


def display_multi_sweep_details(exp_state):
    """Displays overview and details for multiple selected sweeps."""
    backend = st.session_state.backend
    st.markdown("---")
    st.subheader("Sweeps Overview")

    # Fetch and display individual and total summaries
    summaries = [get_sweep_summary_data(backend, s) for s in exp_state["selected_sweeps"]]
    summary_df = pd.DataFrame(summaries)

    # Calculate and add total row
    if not summary_df.empty:
        # Ensure numeric conversion works, coercing errors
        numeric_cols = summary_df.drop(columns=["Sweep"]).apply(pd.to_numeric, errors="coerce")
        total_summary = numeric_cols.sum()
        total_summary["Sweep"] = "Total (all selected)"
        total_df = pd.DataFrame([total_summary])
        summary_df = pd.concat([summary_df, total_df], ignore_index=True)

    st.dataframe(summary_df.set_index("Sweep"), width="stretch")

    # Fetch and display combined configs table
    combined_configs_df = get_combined_config_data(backend, exp_state["selected_sweeps"])
    if combined_configs_df.empty:
        st.warning("No experiment configurations found for the selected sweeps.")
        return

    display_config_details_table(combined_configs_df, exp_state)


def display_config_details_table(configs_df, exp_state):
    """Displays the interactive table of experiment configurations."""
    st.subheader("📋 Experiment Configurations")

    # Ensure arg_hash is a column and set it as index for display
    display_df = configs_df.copy()
    if "arg_hash" in display_df.columns:
        display_df = display_df.set_index("arg_hash", drop=False)
    else:
        st.error("Cannot find 'arg_hash' in configuration data.")
        return

    event = st.dataframe(
        display_df,
        width="stretch",
        height=400,
        on_select="rerun",
        selection_mode="multi-row",
        key=f"config_df_{'_'.join(exp_state['selected_sweeps'])}",
    )

    if event.selection.rows != exp_state["selected_rows_indices"]:
        exp_state["selected_rows_indices"] = event.selection.rows
        st.rerun()

    if exp_state["selected_rows_indices"]:
        st.info(f"Selected {len(exp_state['selected_rows_indices'])} configurations")
        manage_selected_configurations(configs_df.iloc[exp_state["selected_rows_indices"]].copy())


def manage_selected_configurations(selected_configs_df):
    """UI for managing selected experiment configurations."""
    backend = st.session_state.backend
    st.markdown("---")
    st.subheader("🗑️ Manage Configurations")
    st.dataframe(selected_configs_df, width="stretch", height=200)

    try:
        if "confirming_delete_configs" not in st.session_state.exp_details:
            st.session_state.exp_details["confirming_delete_configs"] = False

        # Use a set to handle duplicates automatically
        arg_hashes = set(selected_configs_df["arg_hash"].tolist())
        if not arg_hashes:
            st.warning("Could not identify any configurations to manage.")
            return

        restrictions = [{"arg_hash": h} for h in arg_hashes]
        selected_exp_configs = backend.tables.ModelTrainingTable() & restrictions
        completed_experiments = backend.tables.TrainedModelsTable() & restrictions
        sweep_experiments = backend.tables.SweepRunsTable() & restrictions
        unique_sweeps = list(set(sweep_experiments.fetch("sweep_name"))) if sweep_experiments else []

        # This button initiates the confirmation process
        if st.button("🗑️ Delete Selected Configurations", type="primary"):
            st.session_state.exp_details["confirming_delete_configs"] = True
            st.rerun()

        # If confirmation is active, show the dialog
        if st.session_state.exp_details.get("confirming_delete_configs", False):
            display_confirmation_dialog(
                item_type="ModelTraining Configurations",
                impact_summary={
                    "Selected Configs": len(selected_exp_configs),
                    "Completed Experiments": len(completed_experiments),
                    "Unique Sweeps": len(unique_sweeps),
                },
                on_confirm=lambda: delete_model_training_entries(restrictions),
                on_cancel=lambda: st.session_state.exp_details.update({"confirming_delete_configs": False}),
            )

    except Exception as e:
        st.error(f"Error analyzing selected configurations: {str(e)}")


def delete_model_training_entries(restrictions):
    """Deletes entries from ModelTrainingTable and clears relevant caches."""
    backend = st.session_state.backend
    with st.spinner("Deleting configurations..."):
        (backend.tables.ModelTrainingTable & restrictions).delete(safemode=False)
        st.success(f"Successfully deleted {len(restrictions)} configurations and associated data.")
        st.session_state.exp_details["selected_rows_indices"] = []
        st.session_state.exp_details["confirming_delete_configs"] = False
        # Clear relevant caches
        get_sweep_summary_data.clear()
        get_combined_config_data.clear()
        st.rerun()


def display_deletion_impact(selected_exp_configs, completed_experiments, unique_sweeps):
    """Displays statistics and warnings about deleting configurations."""
    st.subheader("📊 Selection Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Selected Configs", len(selected_exp_configs))
    with col2:
        st.metric("Completed Experiments", len(completed_experiments))
    with col3:
        st.metric("Unique Sweeps", len(unique_sweeps))
    with col4:
        st.metric("Jobs in Queue", "N/A")  # This might need recalculation

    st.markdown("---")
    st.subheader("⚠️ Deletion Impact")
    st.error(f"""
    **WARNING: You are about to delete {len(selected_exp_configs)} experiment configurations!**

    This will affect:
    - **{len(unique_sweeps)}** different sweeps
    - **{len(completed_experiments)}** completed experiments will be deleted

    This action cannot be undone!
    """)


@st.cache_data
def get_all_models(_backend, class_filter=None):
    """Get all models from the database."""
    try:
        models_table = _backend.tables.ModelTable()
        if class_filter:
            models_table &= f"model_class_name LIKE '%{class_filter}%'"
        models_df = models_table.fetch(format="frame")
        models_df = models_df.reset_index()

        # Fix data types for Streamlit/Arrow compatibility
        # for col in models_df.columns:
        #     if models_df[col].dtype == "object":
        #         models_df[col] = models_df[col].astype(str)

        return models_df
    except Exception as e:
        st.error(f"Error fetching models: {str(e)}")
        return pd.DataFrame()


def show_model_details():
    """Show detailed view of ModelTable."""
    st.header("Model Details")
    backend = st.session_state.backend

    if "model_details" not in st.session_state:
        st.session_state.model_details = {"class_filter": "", "selected_rows": [], "show_details": False}
    model_state = st.session_state.model_details

    # Refresh button
    if st.button("🔄 Refresh", key="model_refresh"):
        get_all_models.clear()
        st.rerun()

    model_state["class_filter"] = st.text_input("Filter by model class name (e.g., 'FastICA'):", value=model_state["class_filter"])

    models_df = get_all_models(backend, model_state["class_filter"])

    if models_df.empty:
        st.warning("No models found.")
        return

    # Create a display-friendly version of the dataframe for the main table
    display_df = models_df.copy()
    for col in display_df.columns:
        if any(isinstance(i, (dict, list)) for i in display_df[col].dropna()):
            display_df[col] = display_df[col].astype(str)

    event = st.dataframe(display_df, width="stretch", height=600, on_select="rerun", selection_mode="multi-row", key="model_df")

    if event.selection.rows != model_state["selected_rows"]:
        model_state["selected_rows"] = event.selection.rows
        model_state["show_details"] = False
        st.rerun()

    if model_state["selected_rows"]:
        selected_models_df = models_df.iloc[model_state["selected_rows"]].copy()

        st.markdown("---")
        st.write("**Selection Management:**")
        col1, col2, _ = st.columns([2, 2, 4])
        with col1:
            if st.button("Load Details for Selected Models"):
                model_state["show_details"] = True
                st.rerun()
        with col2:
            if st.button("Clear Selection"):
                model_state["selected_rows"] = []
                model_state["show_details"] = False
                st.rerun()

        if model_state.get("show_details", False):
            display_selected_model_details(selected_models_df)

        manage_master_table_deletions(selected_models_df, "model_hash", backend.tables.ModelTable)


@st.cache_data
def get_all_datasets(_backend):
    """Get all datasets from the database."""
    try:
        datasets_df = _backend.tables.DatasetTable().fetch(format="frame").reset_index()
        return datasets_df
    except Exception as e:
        st.error(f"Error fetching datasets: {str(e)}")
        return pd.DataFrame()


def show_dataset_details():
    """Show detailed view of DatasetTable."""
    st.header("Dataset Details")
    backend = st.session_state.backend

    if "dataset_details" not in st.session_state:
        st.session_state.dataset_details = {"selected_rows": [], "show_details": False}
    dataset_state = st.session_state.dataset_details

    # Refresh button
    if st.button("🔄 Refresh", key="dataset_refresh"):
        get_all_datasets.clear()
        st.rerun()

    datasets_df = get_all_datasets(backend)

    if datasets_df.empty:
        st.warning("No datasets found.")
        return

    # Create a display-friendly version of the dataframe for the main table
    display_df = datasets_df.copy()
    for col in display_df.columns:
        # Check if a column contains dicts or lists and convert them to strings for display
        if any(isinstance(i, (dict, list)) for i in display_df[col].dropna()):
            display_df[col] = display_df[col].astype(str)

    event = st.dataframe(display_df, width="stretch", height=600, on_select="rerun", selection_mode="multi-row", key="dataset_df")

    if event.selection.rows != dataset_state["selected_rows"]:
        dataset_state["selected_rows"] = event.selection.rows
        dataset_state["show_details"] = False  # Reset details view on new selection
        st.rerun()

    if dataset_state["selected_rows"]:
        selected_datasets_df = datasets_df.iloc[dataset_state["selected_rows"]].copy()

        st.markdown("---")
        st.write("**Selection Management:**")
        col1, col2, _ = st.columns([2, 2, 4])
        with col1:
            if st.button("Load Details for Selected Datasets"):
                dataset_state["show_details"] = True
                st.rerun()
        with col2:
            if st.button("Clear Selection"):
                dataset_state["selected_rows"] = []
                dataset_state["show_details"] = False
                st.rerun()

        if dataset_state.get("show_details", False):
            display_selected_dataset_details(selected_datasets_df)

        manage_master_table_deletions(selected_datasets_df, "data_hash", backend.tables.DatasetTable)


def display_selected_model_details(selected_df):
    """Displays detailed config for selected models."""
    st.subheader("📋 Selected Model Details")
    for _, row in selected_df.iterrows():
        with st.expander(f"Details for Model Hash: {row['model_hash']}"):
            row_dict = row.to_dict()
            st.json(row_dict)


def display_selected_dataset_details(selected_df):
    """Displays detailed config for selected datasets."""
    st.subheader("📋 Selected Dataset Details")
    for _, row in selected_df.iterrows():
        with st.expander(f"Details for Dataset Hash: {row['data_hash']}"):
            row_dict = row.to_dict()
            st.json(row_dict)


def manage_master_table_deletions(selected_df, key_name, table_to_delete):
    """UI for managing deletion of master table entries (Model, Dataset)."""
    st.markdown("---")
    st.subheader(f"🗑️ Manage Selected {table_to_delete.table_name}")
    st.dataframe(selected_df, width="stretch", height=200)
    backend = st.session_state.backend

    if "confirming_delete_master" not in st.session_state:
        st.session_state.confirming_delete_master = False

    hashes = list(set(selected_df[key_name].tolist()))
    restrictions = [{key_name: h} for h in hashes]

    # Find dependent entries
    model_trainings = backend.tables.ModelTrainingTable() & restrictions
    completed_experiments = backend.tables.TrainedModelsTable() & model_trainings
    sweep_experiments = backend.tables.SweepRunsTable() & model_trainings
    unique_sweeps = list(set(sweep_experiments.fetch("sweep_name"))) if sweep_experiments else []

    # This button initiates the confirmation process
    if st.button("🗑️ Delete Selected Entries and All Dependent Data", type="primary"):
        st.session_state.confirming_delete_master = True
        st.rerun()

    # If confirmation is active, show the dialog
    if st.session_state.get("confirming_delete_master", False):
        display_confirmation_dialog(
            item_type=f"'{table_to_delete.table_name}' entries",
            impact_summary={
                "ModelTraining configs": len(model_trainings),
                "Completed experiments": len(completed_experiments),
                "Affected sweeps": len(unique_sweeps),
            },
            on_confirm=lambda: delete_master_entries(restrictions, table_to_delete),
            on_cancel=lambda: st.session_state.update({"confirming_delete_master": False}),
        )


def delete_master_entries(restrictions, table_to_delete):
    """Deletes entries from a master table and their dependencies."""
    backend = st.session_state.backend
    with st.spinner("Deleting entries..."):
        try:
            # First, delete dependent entries
            (backend.tables.ModelTrainingTable() & restrictions).delete(safemode=False)
            # Then, delete master entries
            (table_to_delete & restrictions).delete(safemode=False)

            st.success(f"Successfully deleted {len(restrictions)} entries and all associated data.")
            st.session_state.confirming_delete_master = False
            clear_all_caches()
            st.rerun()
        except Exception as e:
            st.error(f"An error occurred during deletion: {e}")
            st.session_state.confirming_delete_master = False


def display_confirmation_dialog(item_type, impact_summary, on_confirm, on_cancel):
    """A reusable confirmation dialog for deletions."""
    st.subheader("⚠️ Confirm Deletion")
    st.error(
        f"""
    **WARNING: You are about to permanently delete {len(impact_summary)} {item_type}!**

    This action will also delete all dependent data, including:
    """
        + "".join([f"\n    - **{value}** {key}" for key, value in impact_summary.items()])
        + """

    This action cannot be undone!
    """
    )

    st.warning("Please confirm your choice to delete.")
    col1, col2, _ = st.columns([1, 1, 4])
    with col1:
        if st.button("Confirm Deletion", type="primary"):
            on_confirm()
            st.rerun()
    with col2:
        if st.button("Cancel"):
            on_cancel()
            st.rerun()


@st.cache_data
def get_jobs_for_sweep(_backend, sweep_name, status_filter="all"):
    """Get comprehensive jobs statistics for a specific sweep."""
    try:
        # Get experiment configs for the sweep (don't need join for job management)
        sweep_runs = _backend.tables.SweepRunsTable() & {"sweep_name": sweep_name}
        exp_configs = _backend.tables.ModelTrainingTable() & sweep_runs
        if len(exp_configs) == 0:
            return pd.DataFrame(), {}

        # The only computed table is TrainedModelsTable
        tables_info = [
            {
                "table": _backend.tables.TrainedModelsTable(),
                "key_table": _backend.tables.ModelTrainingTable(),
                "name": "TrainedModels",
            }
        ]

        jobs_summary = {}
        all_jobs_list = []

        for table_info in tables_info:
            job_stats = get_sweep_jobs_for_table(
                restrictions=exp_configs,
                table=table_info["table"],
                key_table=table_info["key_table"],
                schema=_backend.schema,
            )
            jobs_summary[table_info["name"]] = job_stats

            # Collect jobs for combined dataframe
            if status_filter in ["all", "error"] and job_stats["error_jobs"] is not None:
                if len(job_stats["error_jobs"]) > 0:
                    error_df = job_stats["error_jobs"].fetch(format="frame").reset_index()
                    all_jobs_list.append(error_df)

            # drop "error_jobs" because it's not serializable
            job_stats.pop("error_jobs")
            if status_filter in ["all", "reserved"] and job_stats["reserved_jobs"] is not None:
                if len(job_stats["reserved_jobs"]) > 0:
                    reserved_df = job_stats["reserved_jobs"].fetch(format="frame").reset_index()
                    all_jobs_list.append(reserved_df)

            # drop "reserved_jobs" because it's not serializable
            job_stats.pop("reserved_jobs")

        # Combine all jobs into one dataframe
        if all_jobs_list:
            combined_jobs_df = pd.concat(all_jobs_list, ignore_index=True)
        else:
            combined_jobs_df = pd.DataFrame()

        return combined_jobs_df, jobs_summary

    except Exception as e:
        traceback.print_exc()
        st.error(f"Error fetching jobs for sweep: {str(e)}")
        return pd.DataFrame(), {}


def show_jobs_management():
    """Show jobs management page."""
    backend = st.session_state.backend
    st.header("Jobs Management")

    if not get_all_sweeps(backend, filter_uuids=False):
        st.warning("No sweeps found in the database.")
        return

    # Initialize session state for the jobs page
    if "jobs_page" not in st.session_state:
        st.session_state.jobs_page = {
            "sweep_filter": "",
            "matched_sweeps": [],
            "selected_sweeps_for_view": [],
            "selected_rows": [],
            "show_details": False,
            "confirming_delete_configs": False,
        }
    jobs_state = st.session_state.jobs_page

    # Refresh button
    if st.button("🔄 Refresh", key="jobs_refresh"):
        get_all_sweeps.clear()
        get_jobs_for_sweep.clear()
        st.success("Jobs data refreshed.")
        st.rerun()

    # Sweep filter input
    jobs_state["sweep_filter"] = st.text_input(
        "Filter sweeps by name (e.g., 'mysweep_01', supports partial matching):",
        value=jobs_state["sweep_filter"],
        help="Leave empty to see jobs from all sweeps (can be slow).",
    )

    # Find matched sweeps based on filter
    if jobs_state["sweep_filter"]:
        all_sweeps = get_all_sweeps(backend, filter_uuids=False)
        jobs_state["matched_sweeps"] = [s for s in all_sweeps if jobs_state["sweep_filter"] in s]
    else:
        jobs_state["matched_sweeps"] = get_all_sweeps(backend, filter_uuids=True)

    if jobs_state["matched_sweeps"]:
        display_jobs_dashboard_for_sweeps(jobs_state)
    else:
        if jobs_state["sweep_filter"]:
            st.info("No matching sweeps found for the given filter.")
        else:
            st.info("No non-UUID sweeps found. Use the filter to search for specific sweeps by name or UUID.")


def display_jobs_dashboard_for_sweeps(jobs_state):
    """
    Orchestrates the display of job information for a list of matched sweeps.
    """
    backend = st.session_state.backend
    st.subheader(f"Found {len(jobs_state['matched_sweeps'])} matching sweeps.")

    # Fetch data for all matched sweeps
    with st.spinner("Loading job data for matched sweeps..."):
        all_jobs_list = []
        all_summaries = {}
        for sweep_name in jobs_state["matched_sweeps"]:
            jobs_df, summary = get_jobs_for_sweep(backend, sweep_name, "all")
            if not jobs_df.empty:
                jobs_df["sweep_name"] = sweep_name
                all_jobs_list.append(jobs_df)
            all_summaries[sweep_name] = summary

    # Prepare combined data
    combined_jobs_df = pd.concat(all_jobs_list, ignore_index=True) if all_jobs_list else pd.DataFrame()

    display_jobs_summary(combined_jobs_df, all_summaries)
    display_detailed_jobs_view(combined_jobs_df, jobs_state)


def display_jobs_summary(combined_jobs_df, all_summaries):
    """Display summary statistics for jobs."""
    st.markdown("---")
    st.subheader("📊 Jobs Summary")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Combined Summary for All Matched Sweeps (Unique Jobs):**")
        if not combined_jobs_df.empty:
            # Drop duplicates based on the job's unique identifier
            unique_jobs = combined_jobs_df.drop_duplicates(subset=["table_name", "key_hash"])

            # Create a pivot table for the summary
            summary_pivot = unique_jobs.groupby(["table_name", "status"]).size().unstack(fill_value=0)

            # Ensure 'error' and 'reserved' columns exist
            for status in ["error", "reserved"]:
                if status not in summary_pivot.columns:
                    summary_pivot[status] = 0

            summary_pivot["Total"] = summary_pivot["error"] + summary_pivot["reserved"]
            st.dataframe(summary_pivot.astype(int), width="stretch")
        else:
            st.info("No job data to summarize.")

    with col2:
        st.write("**Per-Sweep Summaries:**")
        for sweep_name, summary in all_summaries.items():
            with st.expander(f"Summary for: {sweep_name}"):
                sweep_summary_data = []
                if not summary:
                    st.info("No job data for this sweep.")
                    continue
                for table_name, job_stats in summary.items():
                    if "error" in job_stats:
                        st.error(f"Error loading summary for {table_name}: {job_stats['error']}")
                    else:
                        sweep_summary_data.append(
                            {
                                "Table": table_name,
                                "Error Jobs": job_stats["error_count"],
                                "Reserved Jobs": job_stats["reserved_count"],
                                "Total": job_stats["error_count"] + job_stats["reserved_count"],
                            }
                        )
                if sweep_summary_data:
                    st.dataframe(pd.DataFrame(sweep_summary_data), width="stretch")
                else:
                    st.info("No job data for this sweep.")


def display_detailed_jobs_view(combined_jobs_df, jobs_state):
    """Display a detailed, filterable view of jobs."""
    st.markdown("---")
    st.subheader("🔍 Detailed Jobs View")

    if combined_jobs_df.empty:
        st.info("No jobs to display based on the current sweep filter.")
        return

    # Let user select which sweeps to view
    jobs_state["selected_sweeps_for_view"] = st.multiselect(
        "Select sweeps to include in the detailed view:",
        options=jobs_state["matched_sweeps"],
        default=jobs_state["matched_sweeps"],
    )

    # Filter jobs based on selected sweeps
    if jobs_state["selected_sweeps_for_view"]:
        display_df = combined_jobs_df[combined_jobs_df["sweep_name"].isin(jobs_state["selected_sweeps_for_view"])].copy()
    else:
        display_df = pd.DataFrame()

    if not display_df.empty:
        # Status filter
        status_filter = st.selectbox("Filter by status:", ["all", "error", "reserved"], index=0, help="Choose which job statuses to display")
        if status_filter != "all":
            display_df = display_df[display_df["status"] == status_filter]

        if not display_df.empty:
            display_jobs_table_and_actions(display_df, jobs_state)
        else:
            st.info(f"No jobs with status '{status_filter}' in the selected sweeps.")
    else:
        st.info("Select at least one sweep from the list above to see job details.")


def display_jobs_table_and_actions(jobs_df, jobs_state):
    """Display the jobs table and associated management actions."""
    available_cols = ["table_name", "key_hash", "status", "error_message", "user", "host", "timestamp", "sweep_name"]
    default_cols = ["table_name", "key_hash", "status", "error_message", "timestamp", "sweep_name"]
    display_cols = st.multiselect("Select columns to display:", available_cols, default=default_cols)

    if not display_cols:
        st.warning("Please select at least one column to display.")
        return

    jobs_display_df = jobs_df.copy().reset_index(drop=True)
    if "error_message" in jobs_display_df.columns:
        jobs_display_df["error_message"] = jobs_display_df["error_message"].astype(str).str[:100] + "..."

    event = st.dataframe(
        jobs_display_df[display_cols],
        width="stretch",
        height=400,
        on_select="rerun",
        selection_mode="multi-row",
        key="jobs_df",
    )

    if event.selection.rows != jobs_state["selected_rows"]:
        jobs_state["selected_rows"] = event.selection.rows
        jobs_state["show_details"] = False
        jobs_state["confirming_delete_configs"] = False
        st.rerun()

    if jobs_state["selected_rows"]:
        manage_selected_jobs(jobs_display_df, jobs_state)

    if jobs_state.get("confirming_delete_configs", False):
        confirm_and_delete_job_configs(jobs_display_df, jobs_state)

    if jobs_state["selected_rows"] and not jobs_state.get("confirming_delete_configs", False):
        display_selected_job_details(jobs_display_df, jobs_state)


def manage_selected_jobs(jobs_df, jobs_state):
    """Provide buttons for managing selected jobs."""
    st.markdown("---")
    st.subheader("🗑️ Job Management Actions")

    # First row: Delete job buttons
    st.write("**Delete Jobs from Queue:**")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Delete Selected Jobs", type="primary", help="Removes all selected jobs from the queue regardless of status"):
            delete_selected_jobs_all_status(jobs_df, jobs_state["selected_rows"])
    with col2:
        if st.button("Delete Selected Error Jobs", type="secondary", help="Removes only the error status jobs from the queue"):
            delete_jobs_by_status(jobs_df, jobs_state["selected_rows"], "error")
    with col3:
        if st.button(
            "Delete Selected Reserved Jobs",
            type="secondary",
            help="Removes only the reserved status jobs from the queue",
        ):
            delete_jobs_by_status(jobs_df, jobs_state["selected_rows"], "reserved")

    st.caption("ℹ️ Deleting jobs removes them from the DataJoint job queue but does not delete experiment configurations or results.")

    # Second row: Delete configs button
    st.markdown("---")
    st.write("**Delete Experiment Configurations:**")
    if st.button(
        "Delete Configs for Selected Jobs",
        type="secondary",
        help="Permanently deletes the experiment configurations and all associated data (experiments, metrics, checkpoints) for the selected jobs",
    ):
        jobs_state["confirming_delete_configs"] = True
        st.rerun()

    st.caption("⚠️ Deleting configs permanently removes experiment configurations and ALL associated data from the database.")

    # Third row: Utility buttons
    st.markdown("---")
    st.write("**Job Details & Selection:**")
    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "Load Details for Selected Jobs",
            help="Displays detailed information including error messages and configuration data for selected jobs",
        ):
            jobs_state["show_details"] = True
            st.rerun()
    with col2:
        if st.button("Clear Selected Jobs", help="Clears the current job selection and hides details"):
            jobs_state["selected_rows"] = []
            jobs_state["show_details"] = False
            jobs_state["confirming_delete_configs"] = False
            st.rerun()

    st.caption("ℹ️ Use these buttons to view job details or reset your selection.")


def delete_selected_jobs_all_status(jobs_df, selected_rows):
    """Deletes selected jobs regardless of their status."""
    backend = st.session_state.backend
    try:
        selected_jobs = jobs_df.iloc[selected_rows]
        if not selected_jobs.empty:
            restrictions = [{"table_name": job["table_name"], "key_hash": job["key_hash"]} for _, job in selected_jobs.iterrows()]
            (backend.schema.jobs & restrictions).delete()
            st.success(f"Successfully deleted {len(selected_jobs)} jobs (all statuses)!")
            st.session_state.jobs_page["selected_rows"] = []
            clear_all_caches()
            st.rerun()
        else:
            st.warning("No jobs selected for deletion.")
    except Exception as e:
        st.error(f"Error deleting jobs: {str(e)}")


def delete_jobs_by_status(jobs_df, selected_rows, status):
    """Deletes selected jobs of a specific status."""
    backend = st.session_state.backend
    try:
        selected_jobs = jobs_df.iloc[selected_rows]
        jobs_to_delete = selected_jobs[selected_jobs["status"] == status]
        if not jobs_to_delete.empty:
            restrictions = [{"table_name": job["table_name"], "key_hash": job["key_hash"]} for _, job in jobs_to_delete.iterrows()]
            (backend.schema.jobs & restrictions).delete()
            st.success(f"Successfully deleted {len(jobs_to_delete)} {status} jobs!")
            st.session_state.jobs_page["selected_rows"] = []
            clear_all_caches()
            st.rerun()
        else:
            st.warning(f"No {status} jobs selected for deletion.")
    except Exception as e:
        st.error(f"Error deleting jobs: {str(e)}")


def confirm_and_delete_job_configs(jobs_df, jobs_state):
    """Display confirmation and handle deletion of configs for selected jobs."""
    backend = st.session_state.backend
    st.markdown("---")
    st.subheader("⚠️ Confirm Deletion")
    try:
        selected_jobs = jobs_df.iloc[jobs_state["selected_rows"]]
        if selected_jobs.empty:
            st.warning("No jobs selected.")
            jobs_state["confirming_delete_configs"] = False
            return

        arg_hashes = set()

        # The key for TrainedModelsTable is just the arg_hash
        keys = (backend.schema.jobs & [dict(key_hash=h) for h in selected_jobs["key_hash"]]).fetch("key")
        for key in keys:
            if "arg_hash" in key:
                arg_hashes.add(key["arg_hash"])

        unique_arg_hashes = list(arg_hashes)
        if not unique_arg_hashes:
            st.warning("No experiment configs could be found for the selected jobs.")
            jobs_state["confirming_delete_configs"] = False
            return

        restrictions = [{"arg_hash": h} for h in unique_arg_hashes]
        configs_to_delete = backend.tables.ModelTrainingTable() & restrictions
        completed_experiments = backend.tables.TrainedModelsTable() & restrictions

        st.error(f"""
        **WARNING: You are about to delete {len(configs_to_delete)} experiment configurations!**

        This will delete:
        - **{len(completed_experiments)}** completed experiments
        - All associated jobs.

        This action cannot be undone!
        """)

        col1, col2, _ = st.columns([1, 1, 4])
        with col1:
            if st.button("Confirm Deletion", type="primary"):
                with st.spinner("Deleting experiment configs..."):
                    configs_to_delete.delete(safemode=False)
                    st.success(f"Successfully deleted {len(configs_to_delete)} experiment configs and associated data.")
                    jobs_state["selected_rows"] = []
                    jobs_state["confirming_delete_configs"] = False
                    clear_all_caches()
                    st.rerun()
        with col2:
            if st.button("Cancel"):
                jobs_state["confirming_delete_configs"] = False
                st.rerun()

    except Exception as e:
        st.error(f"Error preparing deletion: {str(e)}")
        jobs_state["confirming_delete_configs"] = False


def display_selected_job_details(jobs_df, jobs_state):
    """Displays detailed information for the selected jobs."""
    if not jobs_state.get("show_details", False):
        return

    selected_jobs_df = jobs_df.iloc[jobs_state["selected_rows"]]
    unique_jobs_df = selected_jobs_df.drop_duplicates(subset=["key_hash"])

    st.markdown("---")
    st.subheader(f"📋 Selected Job Details ({len(unique_jobs_df)} unique jobs from {len(selected_jobs_df)} selections)")

    for i, (_, job_row) in enumerate(unique_jobs_df.iterrows()):
        # Find all sweeps for the current unique job from the original selection
        all_sweeps_for_job = ", ".join(selected_jobs_df[selected_jobs_df["key_hash"] == job_row["key_hash"]]["sweep_name"].unique())

        with st.expander(f"Job {i + 1}: {job_row['table_name']} {job_row['key_hash']} - {job_row['status']}", expanded=True):
            display_single_job_detail(job_row, sweeps=all_sweeps_for_job)


def display_single_job_detail(job_row, sweeps=None):
    """Displays the details of a single job row."""
    backend = st.session_state.backend
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Table:** {job_row['table_name']}")
        st.write(f"**Status:** {job_row['status']}")
    with col2:
        st.write(f"**Key Hash:** {job_row['key_hash']}")
        st.write(f"**User:** {job_row.get('user', 'N/A')}")
    with col3:
        st.write(f"**Host:** {job_row.get('host', 'N/A')}")
        st.write(f"**Timestamp:** {job_row.get('timestamp', 'N/A')}")

    if sweeps:
        st.write(f"**Sweeps:** {sweeps}")

    if pd.notna(job_row.get("error_message")):
        st.write("**Error Message:**")
        st.code(job_row["error_message"], language="text")
    if pd.notna(job_row.get("error_stack")):
        st.write("**Error Stack:**")
        st.code(str(job_row["error_stack"]), language="text")
    if pd.notna(job_row.get("key")):
        st.write("**Key Content:**")
        try:
            # Fetch the full key from the jobs table
            key_dict = (backend.schema.jobs & {"key_hash": job_row["key_hash"]}).fetch1("key")
            st.json(key_dict)

            # Fetch and display the full config dict
            if "arg_hash" in key_dict:
                st.write("**Associated Experiment Config Dictionary:**")
                with st.spinner("Fetching config dictionary..."):
                    try:
                        config_entry = (backend.tables.ModelTrainingTable * backend.tables.ModelTable * backend.tables.DatasetTable & key_dict).fetch1()

                        # Reconstruct a nested dict for readability
                        config_dict = {
                            "arg_hash": config_entry["arg_hash"],
                            "model_seed": config_entry["model_seed"],
                            "model_config": config_entry["model_config"],
                            "data_config": config_entry["data_config"],
                        }
                        st.json(config_dict)
                    except Exception as e:
                        st.error(f"Could not fetch or display config dictionary: {e}")

        except Exception as e:
            st.error(f"Error displaying key content: {str(e)}")
            st.code(str(job_row.get("key", "N/A")), language="text")


def main():
    """Main application logic."""
    st.sidebar.title("Settings")

    args = sys.argv

    # --- Cache Directory Handling ---
    if "cache_dir" not in st.session_state:
        # Script name is args[0], cache_dir is the optional second argument (index 2)
        cache_dir_from_args = args[2] if len(args) > 2 else None

        if cache_dir_from_args:
            st.session_state.cache_dir = cache_dir_from_args
        else:
            # Default to the one from ituna.config
            st.session_state.cache_dir = ituna.config.CACHE_DIR

    # --- Schema Name Handling ---
    if "schema_name" not in st.session_state:
        # Script name is args[0], schema_name is the first argument (index 1)
        schema_from_args = args[1] if len(args) > 1 else None

        if schema_from_args:
            st.session_state.schema_name = schema_from_args
            st.rerun()
        else:
            st.title("Database Schema Required")
            st.info("Please provide the DataJoint schema name to connect to.")
            with st.form("schema_form"):
                schema_name_input = st.text_input("Schema Name")
                submitted = st.form_submit_button("Connect")
                if submitted and schema_name_input:
                    st.session_state.schema_name = schema_name_input
                    st.rerun()
            return  # Stop execution until schema is provided

    # --- Backend Initialization ---
    st.sidebar.success(f"Schema: {st.session_state.schema_name}")
    st.sidebar.info(f"Cache Dir: {st.session_state.cache_dir}")

    backend_kwargs = {
        "cache_dir": st.session_state.cache_dir,
        "schema_name": st.session_state.schema_name,
    }
    with ituna.config.config_context(
        DEFAULT_BACKEND="datajoint",
        BACKEND_KWARGS=backend_kwargs,
    ):
        try:
            # This will create the backend instance based on the context
            backend = ituna._backends.get_backend()
            st.session_state.backend = backend
            st.session_state.db_connected = True
            st.session_state.connection_params = {
                "host": backend.host,
                "user": backend.user,
                "schema": backend.schema_name,
            }
            show_main_dashboard()

        except Exception as e:
            st.error(f"Failed to initialize backend: {e}")
            st.warning("Please check your `.env` file, database status, and that the schema name is correct.")
            if st.button("Enter a different schema name"):
                del st.session_state.schema_name
                st.rerun()


if __name__ == "__main__":
    main()
