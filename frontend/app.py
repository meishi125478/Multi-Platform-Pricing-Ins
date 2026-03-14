"""
Insurance Pricing Model Training Frontend
A Gradio-based web interface for configuring and running insurance pricing models.
"""

import inspect
import json
import os

from ins_pricing.frontend.app_controller import PricingApp, _check_frontend_deps

def create_ui():
    """Create the Gradio interface."""
    _check_frontend_deps()
    import gradio as gr

    app = PricingApp()
    def _dump_json_template(payload):
        return json.dumps(payload, indent=2, ensure_ascii=False)

    xgb_search_space_template = _dump_json_template(
        app.config_builder._default_xgb_search_space()
    )
    resn_search_space_template = _dump_json_template(
        app.config_builder._default_resn_search_space()
    )
    ft_search_space_template = _dump_json_template(
        app.config_builder._default_ft_search_space()
    )
    ft_unsupervised_search_space_template = _dump_json_template(
        app.config_builder._default_ft_unsupervised_search_space()
    )
    workflow_template = _dump_json_template(
        {
            "workflow": {
                "mode": "plot_direct",
                "cfg_path": "config_plot.json",
                "xgb_cfg_path": "config_xgb_direct.json",
                "resn_cfg_path": "config_resn_direct.json",
            }
        }
    )
    xgb_step2_overrides_template = _dump_json_template(
        {
            "output_dir": "./ResultsXGBFromFTUnsupervised",
            "optuna_storage": "./ResultsXGBFromFTUnsupervised/optuna/bayesopt.sqlite3",
            "optuna_study_prefix": "pricing_ft_unsup_xgb",
            "loss_name": "mse",
            "build_oht": False,
            "final_refit": False,
            "runner": {
                "model_keys": ["xgb"],
                "nproc_per_node": 1,
                "plot_curves": False,
            },
            "plot_curves": False,
            "plot": {"enable": False},
        }
    )
    resn_step2_overrides_template = _dump_json_template(
        {
            "use_resn_ddp": True,
            "output_dir": "./ResultsResNFromFTUnsupervised",
            "optuna_storage": "./ResultsResNFromFTUnsupervised/optuna/bayesopt.sqlite3",
            "optuna_study_prefix": "pricing_ft_unsup_resn_ddp",
            "loss_name": "mse",
            "build_oht": True,
            "runner": {
                "model_keys": ["resn"],
                "nproc_per_node": 2,
                "plot_curves": False,
            },
            "plot_curves": False,
            "plot": {"enable": False},
        }
    )
    dir_status_init, dir_choices_init, dir_value_init = app.list_directory_candidates(
        str(app.working_dir)
    )
    def _initial_oneway_factor_state():
        return app.suggest_oneway_factors("config_plot.json")

    pre_factor_status_init, pre_factor_choices_init, pre_factor_value_init = _initial_oneway_factor_state()
    direct_factor_status_init, direct_factor_choices_init, direct_factor_value_init = _initial_oneway_factor_state()
    embed_factor_status_init, embed_factor_choices_init, embed_factor_value_init = _initial_oneway_factor_state()

    def _set_working_dir_ui(path_text: str):
        status, resolved = app.set_working_dir(path_text)
        _, choices, selected = app.list_directory_candidates(resolved)
        return (
            status,
            resolved,
            resolved,
            gr.update(choices=choices, value=selected),
        )

    def _refresh_working_dir_choices_ui(root_dir: str):
        status, choices, selected = app.list_directory_candidates(root_dir)
        return status, gr.update(choices=choices, value=selected)

    def _suggest_compare_defaults(model_key: str):
        key = str(model_key or "").strip().lower()
        if key == "resn":
            return (
                "config_resn_direct.json",
                "config_resn_from_ft_unsupervised.json",
                "ResN_raw",
                "ResN_ft_embed",
            )
        return (
            "config_xgb_direct.json",
            "config_xgb_from_ft_unsupervised.json",
            "XGB_raw",
            "XGB_ft_embed",
        )

    def _load_oneway_factors_ui(cfg_path: str):
        status, choices, selected = app.suggest_oneway_factors(cfg_path)
        return status, gr.update(choices=choices, value=selected)

    layout_css = """
    .gradio-container {
        max-width: 1480px !important;
        margin: 0 auto !important;
    }
    .gradio-container .tabitem {
        padding-top: 8px;
    }
    .gradio-container .gr-form {
        gap: 8px !important;
    }
    .gradio-container .gr-row {
        align-items: stretch;
    }
    .gradio-container .gr-column {
        min-width: 0;
    }
    .gradio-container textarea {
        line-height: 1.35;
    }
    .gradio-container .gr-textbox,
    .gradio-container .gr-number,
    .gradio-container .gr-dropdown,
    .gradio-container .gr-slider,
    .gradio-container .gr-checkbox {
        margin-bottom: 2px !important;
    }
    """

    with gr.Blocks(
        title="Insurance Pricing Model Training",
        theme=gr.themes.Soft(),
        css=layout_css,
    ) as demo:
        gr.Markdown(
            """
            # Insurance Pricing Model Training Interface
            Configure and train insurance pricing models with an easy-to-use interface.

            **Two ways to configure:**
            1. **Upload JSON Config**: Upload an existing configuration file
            2. **Manual Configuration**: Fill in the parameters below
            """
        )

        with gr.Row():
            working_dir_input = gr.Textbox(
                label="Working Directory",
                value=str(app.working_dir),
                placeholder="Type a path, or select from the folder list below",
                scale=3,
            )
            set_working_dir_btn = gr.Button(
                "Set Working Directory", variant="secondary", scale=1)

        with gr.Row():
            working_dir_browse_root = gr.Textbox(
                label="Browse Root",
                value=str(app.working_dir),
                placeholder="List folders under this path (depth=2)",
                scale=3,
            )
            refresh_working_dir_btn = gr.Button(
                "Refresh Folder List", variant="secondary", scale=1
            )

        with gr.Row():
            working_dir_picker = gr.Dropdown(
                label="Select Existing Folder",
                choices=dir_choices_init,
                value=dir_value_init,
                scale=3,
            )
            use_selected_working_dir_btn = gr.Button(
                "Use Selected Folder", variant="secondary", scale=1
            )

        working_dir_status = gr.Textbox(
            label="Working Directory Status", value=dir_status_init, interactive=False)

        with gr.Tab("Configuration"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=2):
                    gr.Markdown("### Load Configuration")
                    json_file = gr.File(
                        label="Upload JSON Config File",
                        file_types=[".json"],
                        type="filepath"
                    )
                    load_btn = gr.Button("Load Config", variant="primary")
                    load_status = gr.Textbox(
                        label="Load Status", interactive=False)

                with gr.Column(scale=5):
                    gr.Markdown("### Current Configuration")
                    config_display = gr.JSON(label="Configuration", value={})

            gr.Markdown("---")
            gr.Markdown("### Manual Configuration")

            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    gr.Markdown("#### Data Settings")
                    data_dir = gr.Textbox(
                        label="Data Directory", value="./Data")
                    model_list = gr.Textbox(
                        label="Model List (comma-separated)", value="od")
                    model_categories = gr.Textbox(
                        label="Model Categories (comma-separated)", value="bc")
                    target = gr.Textbox(
                        label="Target Column", value="response")
                    weight = gr.Textbox(label="Weight Column", value="weights")

                    gr.Markdown("#### Features")
                    feature_list = gr.Textbox(
                        label="Feature List (comma-separated)",
                        placeholder="feature_1, feature_2, feature_3",
                        lines=4
                    )
                    categorical_features = gr.Textbox(
                        label="Categorical Features (comma-separated)",
                        placeholder="feature_2, feature_3",
                        lines=3
                    )

                with gr.Column(scale=1):
                    gr.Markdown("#### Model Settings")
                    task_type = gr.Dropdown(
                        label="Task Type",
                        choices=["regression", "binary", "multiclass"],
                        value="regression"
                    )
                    split_strategy = gr.Dropdown(
                        label="Split Strategy",
                        choices=["random", "stratified", "time", "group"],
                        value="random"
                    )
                    rand_seed = gr.Number(
                        label="Random Seed", value=13, precision=0)
                    epochs = gr.Number(label="Epochs", value=50, precision=0)
                    prop_test = gr.Slider(
                        label="Test Proportion", minimum=0.1, maximum=0.5, value=0.25, step=0.05)
                    holdout_ratio = gr.Slider(
                        label="Holdout Ratio", minimum=0.1, maximum=0.5, value=0.25, step=0.05)
                    val_ratio = gr.Slider(
                        label="Validation Ratio", minimum=0.1, maximum=0.5, value=0.25, step=0.05)

                    gr.Markdown("#### Training Settings")
                    output_dir = gr.Textbox(
                        label="Output Directory", value="./Results")
                    use_gpu = gr.Checkbox(label="Use GPU", value=True)
                    model_keys = gr.Textbox(
                        label="Model Keys (comma-separated)",
                        value="xgb, resn",
                        placeholder="xgb, resn, ft, gnn"
                    )
                    max_evals = gr.Number(
                        label="Max Evaluations", value=50, precision=0)

                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("#### XGBoost Runtime")
                            xgb_max_depth_max = gr.Number(
                                label="XGB Max Depth", value=25, precision=0)
                            xgb_n_estimators_max = gr.Number(
                                label="XGB Max Estimators", value=500, precision=0)
                            xgb_gpu_id = gr.Number(
                                label="XGB GPU ID", value=0, precision=0)
                            xgb_use_dmatrix = gr.Checkbox(
                                label="XGB Use DMatrix", value=True)
                            xgb_chunk_size = gr.Number(
                                label="XGB Chunk Size (rows, 0=off)", value=0, precision=0)
                            resn_use_lazy_dataset = gr.Checkbox(
                                label="ResNet Lazy Dataset", value=True)
                            resn_predict_batch_size = gr.Number(
                                label="ResNet Predict Batch Size (0=auto)", value=0, precision=0)

                        with gr.Column(scale=1):
                            gr.Markdown("#### Cleanup Controls")
                            xgb_cleanup_per_fold = gr.Checkbox(
                                label="XGB Cleanup Per Fold", value=False)
                            xgb_cleanup_synchronize = gr.Checkbox(
                                label="XGB Cleanup Synchronize", value=False)
                            ft_cleanup_per_fold = gr.Checkbox(
                                label="FT Cleanup Per Fold", value=False)
                            ft_cleanup_synchronize = gr.Checkbox(
                                label="FT Cleanup Synchronize", value=False)
                            resn_cleanup_per_fold = gr.Checkbox(
                                label="ResNet Cleanup Per Fold", value=False)
                            resn_cleanup_synchronize = gr.Checkbox(
                                label="ResNet Cleanup Synchronize", value=False)
                            gnn_cleanup_per_fold = gr.Checkbox(
                                label="GNN Cleanup Per Fold", value=False)
                            gnn_cleanup_synchronize = gr.Checkbox(
                                label="GNN Cleanup Synchronize", value=False)
                            optuna_cleanup_synchronize = gr.Checkbox(
                                label="Optuna Cleanup Synchronize", value=False)

            gr.Markdown("#### Bayesian Optimization Search Spaces (JSON)")
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    xgb_search_space_json = gr.Textbox(
                        label="XGB Search Space",
                        value=xgb_search_space_template,
                        lines=10,
                        max_lines=20,
                    )
                with gr.Column(scale=1):
                    resn_search_space_json = gr.Textbox(
                        label="ResNet Search Space",
                        value=resn_search_space_template,
                        lines=10,
                        max_lines=20,
                    )
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    ft_search_space_json = gr.Textbox(
                        label="FT Supervised Search Space",
                        value=ft_search_space_template,
                        lines=10,
                        max_lines=20,
                    )
                with gr.Column(scale=1):
                    ft_unsupervised_search_space_json = gr.Textbox(
                        label="FT Unsupervised Search Space",
                        value=ft_unsupervised_search_space_template,
                        lines=10,
                        max_lines=20,
                    )

            with gr.Accordion("Advanced Manual Overrides (JSON Deep-Merge)", open=False):
                gr.Markdown(
                    "Provide a JSON object to override any generated config field, including nested sections "
                    "like `runner`, `plot`, `calibration`, `threshold`, `bootstrap`, etc."
                )
                config_overrides_json = gr.Textbox(
                    label="Config Overrides JSON",
                    value="{}",
                    lines=12,
                    max_lines=24,
                    placeholder='{"runner":{"mode":"entry","nproc_per_node":1},"plot":{"enable":true}}',
                )

            with gr.Row():
                build_btn = gr.Button(
                    "Build Configuration", variant="primary", size="lg")
                save_config_btn = gr.Button(
                    "Save Configuration", variant="secondary", size="lg")

            build_status = gr.Textbox(label="Status", interactive=False)
            config_json = gr.Textbox(
                label="Generated Config (JSON)", lines=12, max_lines=24)

            with gr.Row(equal_height=True):
                save_filename = gr.Textbox(
                    label="Save Filename", value="my_config.json", scale=3)
                save_status = gr.Textbox(
                    label="Save Status", interactive=False, scale=4)

        with gr.Tab("Run Task"):
            gr.Markdown(
                """
                ### Run Model Task
                Click the button below to execute the task defined in your configuration.
                Task type is automatically detected from `config.runner.mode`:
                - **entry**: Standard model training
                - **explain**: Model explanation (permutation, SHAP, integrated gradients)
                - **incremental**: Incremental training
                - **watchdog**: Watchdog mode

                Task logs will appear in real-time below.
                """
            )

            with gr.Row():
                run_btn = gr.Button("Run Task", variant="primary", size="lg")
                run_status = gr.Textbox(label="Task Status", interactive=False)

            gr.Markdown("### Task Logs")
            log_output = gr.Textbox(
                label="Logs",
                lines=25,
                max_lines=50,
                interactive=False,
                autoscroll=True
            )

            gr.Markdown("---")
            with gr.Row():
                open_folder_btn = gr.Button("Open Results Folder", size="lg")
                folder_status = gr.Textbox(
                    label="Status", interactive=False, scale=2)

        with gr.Tab("FT Two-Step Workflow"):
            gr.Markdown(
                """
                ### FT-Transformer Two-Step Training

                Automates the FT -> XGB/ResN workflow:
                1. **Step 1**: Train FT-Transformer as unsupervised embedding generator
                2. **Step 2**: Merge embeddings with raw data and train XGB/ResN

                **Instructions**:
                1. Load or build a base configuration in the Configuration tab
                2. Prepare Step 1 config (FT embeddings)
                3. Run Step 1 to generate embeddings
                4. Prepare Step 2 configs (XGB/ResN using embeddings)
                5. Run Step 2 with the generated configs
                """
            )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Step 1: FT Embedding Generation")
                    ft_use_ddp = gr.Checkbox(
                        label="Use DDP for FT", value=True)
                    ft_nproc = gr.Number(
                        label="Number of Processes (DDP)", value=2, precision=0)

                    prepare_step1_btn = gr.Button(
                        "Prepare Step 1 Config", variant="primary")
                    step1_status = gr.Textbox(
                        label="Status", interactive=False)
                    step1_config_display = gr.Textbox(
                        label="Step 1 Config (FT Embedding)",
                        lines=15,
                        max_lines=25
                    )

                with gr.Column():
                    gr.Markdown("### Step 2: Train XGB/ResN with Embeddings")
                    target_models_input = gr.Textbox(
                        label="Target Models (comma-separated)",
                        value="xgb, resn",
                        placeholder="xgb, resn"
                    )
                    augmented_data_dir_input = gr.Textbox(
                        label="Augmented Data Directory",
                        value="./DataFTUnsupervised",
                        placeholder="./DataFTUnsupervised",
                    )
                    gr.Markdown(
                        "Step-2 parameter overrides (JSON). "
                        "Defaults match `02 Train_FT_Embed_XGBResN.ipynb`."
                    )
                    xgb_overrides_input = gr.Textbox(
                        label="XGB Step 2 Overrides (JSON)",
                        value=xgb_step2_overrides_template,
                        lines=10,
                        max_lines=20,
                    )
                    resn_overrides_input = gr.Textbox(
                        label="ResN Step 2 Overrides (JSON)",
                        value=resn_step2_overrides_template,
                        lines=10,
                        max_lines=20,
                    )

                    prepare_step2_btn = gr.Button(
                        "Prepare Step 2 Configs", variant="primary")
                    step2_status = gr.Textbox(
                        label="Status", interactive=False)

                    with gr.Tab("XGB Config"):
                        xgb_config_display = gr.Textbox(
                            label="XGB Step 2 Config",
                            lines=15,
                            max_lines=25
                        )

                    with gr.Tab("ResN Config"):
                        resn_config_display = gr.Textbox(
                            label="ResN Step 2 Config",
                            lines=15,
                            max_lines=25
                        )

            gr.Markdown("---")
            gr.Markdown(
                """
                ### Quick Actions
                After preparing configs, you can:
                - Copy the Step 1 config and paste it in the **Configuration** tab, then run it in **Run Task** tab
                - After Step 1 completes, click **Prepare Step 2 Configs**
                - Step 2 configs are auto-saved as:
                  - `config_xgb_from_ft_unsupervised.json`
                  - `config_resn_from_ft_unsupervised.json`
                - You can edit XGB/ResN override JSONs before generation to customize Step-2 parameters
                - Run those configs in **Run Task** tab (or keep using the JSON text boxes below)
                """
            )

        with gr.Tab("Workflow Config"):
            gr.Markdown(
                """
                ### Config-Driven Plotting / Prediction / Compare / Pre-Oneway
                Use a JSON config file to run frontend workflows without manual field-by-field input.

                Supported `workflow.mode` values:
                - `pre_oneway`
                - `plot_direct`
                - `plot_embed`
                - `predict_ft_embed`
                - `compare_xgb`
                - `compare_resn`
                - `compare` (requires `model_key`: `xgb` or `resn`)
                - `double_lift`
                """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    workflow_file = gr.File(
                        label="Upload Workflow Config",
                        file_types=[".json"],
                        type="filepath",
                    )
                    workflow_load_btn = gr.Button(
                        "Load Workflow Config", variant="secondary")
                    workflow_load_status = gr.Textbox(
                        label="Load Status", interactive=False)

                with gr.Column(scale=2):
                    workflow_config_json = gr.Textbox(
                        label="Workflow Config (JSON)",
                        value=workflow_template,
                        lines=18,
                        max_lines=30,
                    )

            workflow_run_btn = gr.Button(
                "Run Workflow Config", variant="primary", size="lg")
            workflow_status = gr.Textbox(label="Workflow Status", interactive=False)
            workflow_log = gr.Textbox(
                label="Workflow Logs",
                lines=18,
                max_lines=40,
                interactive=False,
                autoscroll=True,
            )

        with gr.Tab("Plotting"):
            gr.Markdown(
                """
                ### Plotting Workflows
                Run the plotting steps from the example notebooks.
                """
            )

            with gr.Tab("Pre Oneway"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3):
                        with gr.Row():
                            pre_data_path = gr.Textbox(
                                label="Data Path", value="./Data/od_bc.csv", scale=3)
                            pre_output_dir = gr.Textbox(
                                label="Output Dir (optional)", value="", scale=2)
                        with gr.Row():
                            pre_model_name = gr.Textbox(
                                label="Model Name", value="od_bc")
                            pre_target = gr.Textbox(
                                label="Target Column", value="response")
                            pre_weight = gr.Textbox(
                                label="Weight Column", value="weights")
                        with gr.Row():
                            pre_cfg_path = gr.Textbox(
                                label="Plot Config (for Oneway Factors)",
                                value="config_plot.json",
                                scale=4,
                            )
                            pre_load_factors_btn = gr.Button(
                                "Load Factors from Config", variant="secondary", scale=1)
                        pre_oneway_factors = gr.Dropdown(
                            label="Oneway Factors (from Plot Config)",
                            choices=pre_factor_choices_init,
                            value=pre_factor_value_init,
                            multiselect=True,
                        )
                        pre_factor_status = gr.Textbox(
                            label="Oneway Factor Source",
                            value=pre_factor_status_init,
                            interactive=False,
                        )
                        with gr.Accordion("Advanced: Split Data Override (optional)", open=False):
                            with gr.Row():
                                pre_train_data_path = gr.Textbox(
                                    label="Train Data Path (optional)",
                                    value="",
                                )
                                pre_test_data_path = gr.Textbox(
                                    label="Test Data Path (optional)",
                                    value="",
                                )
                    with gr.Column(scale=2):
                        pre_feature_list = gr.Textbox(
                            label="Fallback Feature List (comma-separated)",
                            lines=3,
                            placeholder="feature_1, feature_2, feature_3",
                        )
                        pre_categorical = gr.Textbox(
                            label="Categorical Features (comma-separated, optional)",
                            lines=2,
                            placeholder="feature_2, feature_3",
                        )
                        with gr.Row():
                            pre_n_bins = gr.Number(
                                label="Bins", value=10, precision=0, scale=1)
                            pre_holdout = gr.Slider(
                                label="Holdout Ratio",
                                minimum=0.0,
                                maximum=0.5,
                                value=0.25,
                                step=0.05,
                                scale=2,
                            )
                            pre_seed = gr.Number(
                                label="Random Seed", value=13, precision=0, scale=1)

                with gr.Row():
                    pre_run_btn = gr.Button("Run Pre Oneway", variant="primary", scale=1)
                    pre_status = gr.Textbox(label="Status", interactive=False, scale=3)
                pre_log = gr.Textbox(label="Logs", lines=15,
                                     max_lines=40, interactive=False)
                pre_gallery = gr.Gallery(
                    label="Generated Images",
                    columns=4,
                    object_fit="contain",
                    height=320,
                )

            with gr.Tab("Direct Plot"):
                with gr.Row():
                    direct_cfg_path = gr.Textbox(
                        label="Plot Config", value="config_plot.json")
                    direct_xgb_cfg = gr.Textbox(
                        label="XGB Config", value="config_xgb_direct.json")
                    direct_resn_cfg = gr.Textbox(
                        label="ResN Config", value="config_resn_direct.json")
                with gr.Row():
                    direct_oneway_factors = gr.Dropdown(
                        label="Oneway Factors (from Plot Config)",
                        choices=direct_factor_choices_init,
                        value=direct_factor_value_init,
                        multiselect=True,
                        scale=4,
                    )
                    direct_load_factors_btn = gr.Button(
                        "Load Factors from Config", variant="secondary", scale=1)
                direct_factor_status = gr.Textbox(
                    label="Oneway Factor Source",
                    value=direct_factor_status_init,
                    interactive=False,
                )
                with gr.Accordion("Advanced: Manual Data/Model Overrides (optional)", open=True):
                    with gr.Row():
                        direct_train_data_path = gr.Textbox(
                            label="Train Data Path (optional)",
                            value="",
                        )
                        direct_test_data_path = gr.Textbox(
                            label="Test Data Path (optional)",
                            value="",
                        )
                    with gr.Row():
                        direct_xgb_model_file = gr.File(
                            label="Upload XGB Model File (optional)",
                            file_types=[".pkl", ".pth"],
                            type="filepath",
                        )
                        direct_resn_model_file = gr.File(
                            label="Upload ResN Model File (optional)",
                            file_types=[".pkl", ".pth"],
                            type="filepath",
                        )
                with gr.Row():
                    direct_run_btn = gr.Button(
                        "Run Direct Plot", variant="primary", scale=1)
                    direct_status = gr.Textbox(
                        label="Status", interactive=False, scale=3)
                direct_log = gr.Textbox(
                    label="Logs", lines=15, max_lines=40, interactive=False)
                direct_gallery = gr.Gallery(
                    label="Generated Images",
                    columns=4,
                    object_fit="contain",
                    height=320,
                )

            with gr.Tab("Embed Plot"):
                with gr.Row():
                    embed_cfg_path = gr.Textbox(
                        label="Plot Config", value="config_plot.json")
                    embed_ft_cfg = gr.Textbox(
                        label="FT Embed Config", value="config_ft_unsupervised_ddp_embed.json")
                    embed_runtime = gr.Checkbox(
                        label="Use Runtime FT Embedding", value=False)
                with gr.Row():
                    embed_xgb_cfg = gr.Textbox(
                        label="XGB Embed Config", value="config_xgb_from_ft_unsupervised.json")
                    embed_resn_cfg = gr.Textbox(
                        label="ResN Embed Config", value="config_resn_from_ft_unsupervised.json")
                with gr.Row():
                    embed_oneway_factors = gr.Dropdown(
                        label="Oneway Factors (from Plot Config)",
                        choices=embed_factor_choices_init,
                        value=embed_factor_value_init,
                        multiselect=True,
                        scale=4,
                    )
                    embed_load_factors_btn = gr.Button(
                        "Load Factors from Config", variant="secondary", scale=1)
                embed_factor_status = gr.Textbox(
                    label="Oneway Factor Source",
                    value=embed_factor_status_init,
                    interactive=False,
                )
                with gr.Accordion("Advanced: Manual Data/Model Overrides (optional)", open=True):
                    with gr.Row():
                        embed_train_data_path = gr.Textbox(
                            label="Train Data Path (optional)",
                            value="",
                        )
                        embed_test_data_path = gr.Textbox(
                            label="Test Data Path (optional)",
                            value="",
                        )
                    with gr.Row():
                        embed_xgb_model_file = gr.File(
                            label="Upload XGB Model File (optional)",
                            file_types=[".pkl", ".pth"],
                            type="filepath",
                        )
                        embed_resn_model_file = gr.File(
                            label="Upload ResN Model File (optional)",
                            file_types=[".pkl", ".pth"],
                            type="filepath",
                        )
                    with gr.Row():
                        embed_ft_model_file = gr.File(
                            label="Upload FT Model File (optional)",
                            file_types=[".pth"],
                            type="filepath",
                        )
                with gr.Row():
                    embed_run_btn = gr.Button(
                        "Run Embed Plot", variant="primary", scale=1)
                    embed_status = gr.Textbox(
                        label="Status", interactive=False, scale=3)
                embed_log = gr.Textbox(
                    label="Logs", lines=15, max_lines=40, interactive=False)
                embed_gallery = gr.Gallery(
                    label="Generated Images",
                    columns=4,
                    object_fit="contain",
                    height=320,
                )

            with gr.Tab("Double Lift"):
                gr.Markdown(
                    """
                    Draw a double-lift curve from any CSV file with two prediction columns.
                    """
                )
                with gr.Row():
                    dl_data_path = gr.Textbox(
                        label="Data Path (CSV)", value="./Data/od_bc.csv", scale=3)
                    dl_output_path = gr.Textbox(
                        label="Output Image Path (optional)", value="", scale=2)
                with gr.Row():
                    dl_pred_col_1 = gr.Textbox(
                        label="Prediction Column 1", value="pred_xgb")
                    dl_pred_col_2 = gr.Textbox(
                        label="Prediction Column 2", value="pred_resn")
                    dl_target_col = gr.Textbox(
                        label="Target Column", value="reponse")
                    dl_weight_col = gr.Textbox(
                        label="Weight Column", value="weights")
                with gr.Row():
                    dl_label1 = gr.Textbox(label="Label 1", value="Model 1")
                    dl_label2 = gr.Textbox(label="Label 2", value="Model 2")
                    dl_n_bins = gr.Number(label="Bins", value=10, precision=0)
                    dl_rand_seed = gr.Number(
                        label="Random Seed", value=13, precision=0)
                with gr.Row():
                    dl_holdout_ratio = gr.Slider(
                        label="Holdout Ratio (0 = all data, >0 = train/test split)",
                        minimum=0.0,
                        maximum=0.5,
                        value=0.0,
                        step=0.05,
                        scale=2,
                    )
                    dl_split_strategy = gr.Dropdown(
                        label="Split Strategy",
                        choices=["random", "stratified", "time", "group"],
                        value="random",
                        scale=1,
                    )
                    dl_split_group_col = gr.Textbox(
                        label="Group Column (optional, for group split)",
                        value="",
                        scale=1,
                    )
                    dl_split_time_col = gr.Textbox(
                        label="Time Column (optional, for time split)",
                        value="",
                        scale=1,
                    )
                    dl_split_time_ascending = gr.Checkbox(
                        label="Time Ascending",
                        value=True,
                        scale=1,
                    )
                with gr.Row():
                    dl_pred1_weighted = gr.Checkbox(
                        label="Prediction 1 Is Weighted",
                        value=False,
                    )
                    dl_pred2_weighted = gr.Checkbox(
                        label="Prediction 2 Is Weighted",
                        value=False,
                    )
                    dl_actual_weighted = gr.Checkbox(
                        label="Actual Is Weighted",
                        value=False,
                    )

                with gr.Row():
                    dl_run_btn = gr.Button("Run Double Lift", variant="primary", scale=1)
                    dl_status = gr.Textbox(label="Status", interactive=False, scale=3)
                dl_log = gr.Textbox(
                    label="Logs", lines=15, max_lines=40, interactive=False)
                dl_gallery = gr.Gallery(
                    label="Generated Images",
                    columns=4,
                    object_fit="contain",
                    height=320,
                )

            with gr.Tab("FT-Embed Compare"):
                gr.Markdown("Compare Direct vs FT-Embed models and draw double-lift curves.")
                with gr.Row():
                    cmp_model_key = gr.Dropdown(
                        label="Model Key",
                        choices=["xgb", "resn"],
                        value="xgb",
                        scale=1,
                    )
                    cmp_direct_cfg = gr.Textbox(
                        label="Direct Model Config", value="config_xgb_direct.json", scale=2)
                    cmp_ft_cfg = gr.Textbox(
                        label="FT Config", value="config_ft_unsupervised_ddp_embed.json", scale=2)
                    cmp_embed_cfg = gr.Textbox(
                        label="FT-Embed Model Config", value="config_xgb_from_ft_unsupervised.json", scale=2)
                with gr.Row():
                    cmp_label_direct = gr.Textbox(
                        label="Direct Label", value="XGB_raw")
                    cmp_label_ft = gr.Textbox(
                        label="FT Label", value="XGB_ft_embed")
                    cmp_runtime = gr.Checkbox(
                        label="Use Runtime FT Embedding", value=False)
                    cmp_bins = gr.Number(
                        label="Bins Override", value=10, precision=0)
                with gr.Accordion("Advanced: Manual Data/Model Overrides (optional)", open=True):
                    with gr.Row():
                        cmp_train_data_path = gr.Textbox(
                            label="Train Data Path (optional)",
                            value="",
                        )
                        cmp_test_data_path = gr.Textbox(
                            label="Test Data Path (optional)",
                            value="",
                        )
                    with gr.Row():
                        cmp_direct_model_file = gr.File(
                            label="Upload Direct Model File (optional)",
                            file_types=[".pkl", ".pth"],
                            type="filepath",
                        )
                        cmp_ft_embed_model_file = gr.File(
                            label="Upload FT-Embed Model File (optional)",
                            file_types=[".pkl", ".pth"],
                            type="filepath",
                        )
                    with gr.Row():
                        cmp_ft_model_file = gr.File(
                            label="Upload FT Model File (optional)",
                            file_types=[".pth"],
                            type="filepath",
                        )
                with gr.Row():
                    cmp_run_btn = gr.Button("Run Compare", variant="primary", scale=1)
                    cmp_status = gr.Textbox(label="Status", interactive=False, scale=3)
                cmp_log = gr.Textbox(
                    label="Logs", lines=15, max_lines=40, interactive=False)
                cmp_gallery = gr.Gallery(
                    label="Generated Images",
                    columns=4,
                    object_fit="contain",
                    height=320,
                )

        with gr.Tab("Prediction"):
            gr.Markdown("### FT Embed Prediction")
            with gr.Row():
                pred_ft_cfg = gr.Textbox(
                    label="FT Config", value="config_ft_unsupervised_ddp_embed.json")
                pred_xgb_cfg = gr.Textbox(
                    label="XGB Config (optional)", value="config_xgb_from_ft_unsupervised.json")
                pred_resn_cfg = gr.Textbox(
                    label="ResN Config (optional)", value="config_resn_from_ft_unsupervised.json")
            with gr.Row():
                pred_model_name = gr.Textbox(
                    label="Model Name (optional)", value="")
                pred_model_keys = gr.Textbox(
                    label="Model Keys", value="xgb, resn")
                pred_input = gr.Textbox(
                    label="Input Data", value="./Data/od_bc_new.csv")
                pred_output = gr.Textbox(
                    label="Output CSV", value="./Results/predictions_ft_xgb.csv")
            with gr.Accordion("Advanced: Upload Model Files (optional)", open=True):
                with gr.Row():
                    pred_ft_model_file = gr.File(
                        label="Upload FT Model File (optional)",
                        file_types=[".pth"],
                        type="filepath",
                    )
                    pred_xgb_model_file = gr.File(
                        label="Upload XGB Model File (optional)",
                        file_types=[".pkl", ".pth"],
                        type="filepath",
                    )
                    pred_resn_model_file = gr.File(
                        label="Upload ResN Model File (optional)",
                        file_types=[".pth", ".pkl"],
                        type="filepath",
                    )
            with gr.Row():
                pred_run_btn = gr.Button("Run Prediction", variant="primary", scale=1)
                pred_status = gr.Textbox(label="Status", interactive=False, scale=3)
            pred_log = gr.Textbox(label="Logs", lines=15,
                                  max_lines=40, interactive=False)

        # Event handlers
        set_working_dir_btn.click(
            fn=_set_working_dir_ui,
            inputs=[working_dir_input],
            outputs=[
                working_dir_status,
                working_dir_input,
                working_dir_browse_root,
                working_dir_picker,
            ],
        )

        refresh_working_dir_btn.click(
            fn=_refresh_working_dir_choices_ui,
            inputs=[working_dir_browse_root],
            outputs=[working_dir_status, working_dir_picker],
        )

        use_selected_working_dir_btn.click(
            fn=_set_working_dir_ui,
            inputs=[working_dir_picker],
            outputs=[
                working_dir_status,
                working_dir_input,
                working_dir_browse_root,
                working_dir_picker,
            ],
        )

        working_dir_picker.change(
            fn=lambda path: str(path or ""),
            inputs=[working_dir_picker],
            outputs=[working_dir_input],
        )

        load_btn.click(
            fn=app.load_json_config,
            inputs=[json_file],
            outputs=[load_status, config_display, config_json],
            show_api=False,
        )

        build_btn.click(
            fn=app.build_config_from_ui,
            inputs=[
                data_dir, model_list, model_categories, target, weight,
                feature_list, categorical_features, task_type, prop_test,
                holdout_ratio, val_ratio, split_strategy, rand_seed, epochs,
                output_dir, use_gpu, model_keys, max_evals,
                xgb_max_depth_max, xgb_n_estimators_max,
                xgb_gpu_id, xgb_cleanup_per_fold, xgb_cleanup_synchronize,
                xgb_use_dmatrix, xgb_chunk_size,
                xgb_search_space_json, resn_search_space_json,
                ft_search_space_json, ft_unsupervised_search_space_json,
                ft_cleanup_per_fold, ft_cleanup_synchronize,
                resn_cleanup_per_fold, resn_cleanup_synchronize,
                resn_use_lazy_dataset, resn_predict_batch_size,
                gnn_cleanup_per_fold, gnn_cleanup_synchronize,
                optuna_cleanup_synchronize, config_overrides_json
            ],
            outputs=[build_status, config_json]
        )

        save_config_btn.click(
            fn=app.save_config,
            inputs=[config_json, save_filename],
            outputs=[save_status]
        )

        run_btn.click(
            fn=app.run_training,
            inputs=[config_json],
            outputs=[run_status, log_output]
        )

        open_folder_btn.click(
            fn=app.open_results_folder,
            inputs=[config_json],
            outputs=[folder_status]
        )

        workflow_load_btn.click(
            fn=app.load_workflow_config,
            inputs=[workflow_file],
            outputs=[workflow_load_status, workflow_config_json],
            show_api=False,
        )

        workflow_run_btn.click(
            fn=app.run_workflow_config_ui,
            inputs=[workflow_config_json],
            outputs=[workflow_status, workflow_log]
        )

        prepare_step1_btn.click(
            fn=app.prepare_ft_step1,
            inputs=[config_json, ft_use_ddp, ft_nproc],
            outputs=[step1_status, step1_config_display]
        )

        prepare_step2_btn.click(
            fn=app.prepare_ft_step2,
            inputs=[gr.State(
                lambda: app.current_step1_config or "temp_ft_step1_config.json"),
                target_models_input,
                augmented_data_dir_input,
                xgb_overrides_input,
                resn_overrides_input,
            ],
            outputs=[step2_status, xgb_config_display, resn_config_display]
        )

        def _bind_oneway_factor_loader(load_btn, cfg_input, status_output, factors_output):
            load_btn.click(
                fn=_load_oneway_factors_ui,
                inputs=[cfg_input],
                outputs=[status_output, factors_output],
                show_api=False,
            )
            cfg_input.change(
                fn=_load_oneway_factors_ui,
                inputs=[cfg_input],
                outputs=[status_output, factors_output],
                show_api=False,
            )

        _bind_oneway_factor_loader(
            pre_load_factors_btn,
            pre_cfg_path,
            pre_factor_status,
            pre_oneway_factors,
        )

        pre_run_btn.click(
            fn=app.run_pre_oneway_ui,
            inputs=[
                pre_data_path, pre_model_name, pre_target, pre_weight,
                pre_feature_list, pre_oneway_factors, pre_categorical, pre_n_bins,
                pre_holdout, pre_seed, pre_output_dir,
                pre_train_data_path, pre_test_data_path,
            ],
            outputs=[pre_status, pre_log, pre_gallery],
            show_api=False,
        )

        _bind_oneway_factor_loader(
            direct_load_factors_btn,
            direct_cfg_path,
            direct_factor_status,
            direct_oneway_factors,
        )

        direct_run_btn.click(
            fn=app.run_plot_direct_ui,
            inputs=[
                direct_cfg_path,
                direct_xgb_cfg,
                direct_resn_cfg,
                direct_oneway_factors,
                direct_train_data_path,
                direct_test_data_path,
                direct_xgb_model_file,
                direct_resn_model_file,
            ],
            outputs=[direct_status, direct_log, direct_gallery],
            show_api=False,
        )

        _bind_oneway_factor_loader(
            embed_load_factors_btn,
            embed_cfg_path,
            embed_factor_status,
            embed_oneway_factors,
        )

        embed_run_btn.click(
            fn=app.run_plot_embed_ui,
            inputs=[
                embed_cfg_path,
                embed_xgb_cfg,
                embed_resn_cfg,
                embed_ft_cfg,
                embed_runtime,
                embed_oneway_factors,
                embed_train_data_path,
                embed_test_data_path,
                embed_xgb_model_file,
                embed_resn_model_file,
                embed_ft_model_file,
            ],
            outputs=[embed_status, embed_log, embed_gallery],
            show_api=False,
        )

        dl_run_btn.click(
            fn=app.run_double_lift_ui,
            inputs=[
                dl_data_path, dl_pred_col_1, dl_pred_col_2, dl_target_col, dl_weight_col,
                dl_n_bins, dl_label1, dl_label2,
                dl_pred1_weighted, dl_pred2_weighted, dl_actual_weighted,
                dl_holdout_ratio, dl_split_strategy, dl_split_group_col, dl_split_time_col,
                dl_split_time_ascending, dl_rand_seed, dl_output_path
            ],
            outputs=[dl_status, dl_log, dl_gallery],
            show_api=False,
        )

        pred_run_btn.click(
            fn=app.run_predict_ui,
            inputs=[
                pred_ft_cfg, pred_xgb_cfg, pred_resn_cfg, pred_input,
                pred_output, pred_model_name, pred_model_keys,
                pred_ft_model_file, pred_xgb_model_file, pred_resn_model_file,
            ],
            outputs=[pred_status, pred_log],
            show_api=False,
        )

        cmp_model_key.change(
            fn=_suggest_compare_defaults,
            inputs=[cmp_model_key],
            outputs=[
                cmp_direct_cfg,
                cmp_embed_cfg,
                cmp_label_direct,
                cmp_label_ft,
            ],
        )

        cmp_run_btn.click(
            fn=app.run_compare_ui,
            inputs=[
                cmp_model_key,
                cmp_direct_cfg,
                cmp_ft_cfg,
                cmp_embed_cfg,
                cmp_label_direct,
                cmp_label_ft,
                cmp_runtime,
                cmp_bins,
                cmp_train_data_path,
                cmp_test_data_path,
                cmp_direct_model_file,
                cmp_ft_embed_model_file,
                cmp_ft_model_file,
            ],
            outputs=[cmp_status, cmp_log, cmp_gallery],
            show_api=False,
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    server_name = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1").strip() or "127.0.0.1"
    launch_kwargs = {
        "server_name": server_name,
        "share": False,
        "show_error": True,
    }
    server_port_env = os.environ.get("GRADIO_SERVER_PORT", "").strip()
    if server_port_env:
        try:
            launch_kwargs["server_port"] = int(server_port_env)
        except ValueError as exc:
            raise ValueError(
                f"Invalid GRADIO_SERVER_PORT: {server_port_env!r}. Must be an integer."
            ) from exc
    if "analytics_enabled" in inspect.signature(demo.launch).parameters:
        launch_kwargs["analytics_enabled"] = False
    demo.launch(**launch_kwargs)
