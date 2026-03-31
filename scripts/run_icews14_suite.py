from lmca_tic.experiments.runner import run_experiment_suite


if __name__ == "__main__":
    run_experiment_suite(
        [
            "configs/experiments/full_icews14.yaml",
            "configs/experiments/ablation_wo_llm_icews14.yaml",
            "configs/experiments/ablation_wo_tcn_icews14.yaml",
            "configs/experiments/ablation_wo_tgn_icews14.yaml",
            "configs/experiments/ablation_wo_gate_icews14.yaml",
            "configs/experiments/negative_random_icews14.yaml",
            "configs/experiments/negative_contrastive_icews14.yaml",
            "configs/experiments/negative_ontology_icews14.yaml",
            "configs/experiments/micro_v1_gs_icews14.yaml",
            "configs/experiments/micro_v2_ni_icews14.yaml",
            "configs/experiments/micro_v3_sl_icews14.yaml",
            "configs/experiments/micro_v4_gs_ni_sl_icews14.yaml",
            "configs/experiments/micro_ours_ni_sl_icews14.yaml",
        ]
    )
