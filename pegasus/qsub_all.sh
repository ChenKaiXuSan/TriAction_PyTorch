#!/bin/bash
set -euo pipefail

# Batch submit TriAction experiment-matrix jobs on Pegasus.
# Default is dry-run. Add --run to actually submit jobs.
# One qsub job per experiment; every job fits the 24h walltime limit.

cd "$(dirname "$0")/.."

MODE="${1:-all}"
ACTION="${2:-dry-run}"

usage() {
    cat <<'EOF'
Usage:
  pegasus/qsub_all.sh [smoke|single_view|modality|backbone|fusion|late_backbone|ts_cva|noes|all] [dry-run|--run]

Examples:
  pegasus/qsub_all.sh                     # dry-run everything
  pegasus/qsub_all.sh smoke --run         # submit the smoke job only
  pegasus/qsub_all.sh fusion --run        # submit the fusion comparison
  pegasus/qsub_all.sh all --run           # submit the full matrix (22 jobs)

Modes:
  smoke          1-epoch sanity check (submit this first, alone)
  single_view    B: front/left/right, rgb, 3dcnn
  modality       M: kpt and rgb_kpt on front (rgb baseline = B_single_front)
  backbone       A: transformer/mamba/videomae/vivit on front (3dcnn = B_single_front)
  fusion         F: multi-view add/concat/avg/mid/late with 3dcnn
  late_backbone  L: late fusion with transformer/mamba/videomae/vivit (3dcnn = F late)
  ts_cva         T: TS-CVA ablations (full model = F mid)
  mv             N: MV-ViViT (frozen ViViT + cross-view token fusion)
  mv2            N2: MV-ViViT round 2 (partial unfreeze / regularization / ensemble)
  mv3            N3: unfreeze sweep + mirror aug + focal + kpt stream + LLRD
  mv4            N4: head-ROI dual stream / kpt-query pooling / aux head-pose
  ablation_gate  K: single-view ViViT only -- decides whether the multi-view
                    claim survives before spending on the rest
  ablation       K: all four attribution ablations (single-view / late fusion /
                    no kpt guidance / frozen backbone), 20 jobs
  kfold          K: person-wise 5-fold CV, 3dcnn baseline vs MV-ViViT
                    (aggregate with: python scripts/aggregate_folds.py)
  noes           no-early-stopping full-length variants of the matrix (21 jobs;
                 front rgb 3dcnn is covered by run_e_noes_single_front_rgb_3dcnn.sh)
  all            all formal jobs (everything except smoke and noes)

Before the first --run:
  pegasus/prepare_index.sh       # pre-generate the shared train/val split index
  pegasus/prepare_hf_models.sh   # only needed for videomae/vivit jobs
EOF
}

smoke_scripts=(
    run_smoke.sh
)

single_view_scripts=(
    run_b_single_front_rgb_3dcnn.sh
    run_b_single_left_rgb_3dcnn.sh
    run_b_single_right_rgb_3dcnn.sh
)

modality_scripts=(
    run_m_front_kpt.sh
    run_m_front_rgb_kpt_3dcnn.sh
)

backbone_scripts=(
    run_a_front_rgb_transformer.sh
    run_a_front_rgb_mamba.sh
    run_a_front_rgb_videomae.sh
    run_a_front_rgb_vivit.sh
)

fusion_scripts=(
    run_f_multi_add.sh
    run_f_multi_concat.sh
    run_f_multi_avg.sh
    run_f_multi_mid.sh
    run_f_multi_late.sh
)

late_backbone_scripts=(
    run_l_late_transformer.sh
    run_l_late_mamba.sh
    run_l_late_videomae.sh
    run_l_late_vivit.sh
)

noes_scripts=($(cd "$(dirname "$0")" && ls run_noes_*.sh 2>/dev/null))

mv_scripts=(
    run_n_mv_vivit.sh
)

mv2_scripts=(
    run_n2_mv_vivit_unfreeze4.sh
    run_n2_mv_vivit_reg.sh
    run_n2_mv_vivit_unfreeze4_reg.sh
    run_n2_mv_vivit_ensemble.sh
)

mv3_scripts=(
    run_n3_mv_vivit_unfreeze2.sh
    run_n3_mv_vivit_unfreeze6.sh
    run_n3_mv_vivit_unfreeze8.sh
    run_n3_mv_vivit_mirror.sh
    run_n3_mv_vivit_mirror_focal.sh
    run_n3_mv_vivit_kpt.sh
    run_n3_mv_vivit_unfreeze8_llrd.sh
)

mv4_scripts=(
    run_n4_mv_vivit_head_roi.sh
    run_n4_mv_vivit_kpt_query.sh
    run_n4_mv_vivit_aux_pose.sh
    run_n4_mv_vivit_head_query.sh
)

kfold_scripts=(
    run_k_baseline3dcnn_fold0.sh
    run_k_baseline3dcnn_fold1.sh
    run_k_baseline3dcnn_fold2.sh
    run_k_baseline3dcnn_fold3.sh
    run_k_baseline3dcnn_fold4.sh
    run_k_mvvivit_fold0.sh
    run_k_mvvivit_fold1.sh
    run_k_mvvivit_fold2.sh
    run_k_mvvivit_fold3.sh
    run_k_mvvivit_fold4.sh
)

ablation_scripts=(
    run_k_vivitsingle_fold0.sh
    run_k_vivitsingle_fold1.sh
    run_k_vivitsingle_fold2.sh
    run_k_vivitsingle_fold3.sh
    run_k_vivitsingle_fold4.sh
    run_k_vivitlate_fold0.sh
    run_k_vivitlate_fold1.sh
    run_k_vivitlate_fold2.sh
    run_k_vivitlate_fold3.sh
    run_k_vivitlate_fold4.sh
    run_k_mvnokptq_fold0.sh
    run_k_mvnokptq_fold1.sh
    run_k_mvnokptq_fold2.sh
    run_k_mvnokptq_fold3.sh
    run_k_mvnokptq_fold4.sh
    run_k_mvfrozen_fold0.sh
    run_k_mvfrozen_fold1.sh
    run_k_mvfrozen_fold2.sh
    run_k_mvfrozen_fold3.sh
    run_k_mvfrozen_fold4.sh
)

ts_cva_scripts=(
    run_t_mid_no_gated_aggregation.sh
    run_t_mid_no_view_embedding.sh
    run_t_mid_heads8.sh
    run_t_mid_heads2.sh
)

case "$MODE" in
    smoke)
        scripts=("${smoke_scripts[@]}")
        ;;
    single_view)
        scripts=("${single_view_scripts[@]}")
        ;;
    modality)
        scripts=("${modality_scripts[@]}")
        ;;
    backbone)
        scripts=("${backbone_scripts[@]}")
        ;;
    fusion)
        scripts=("${fusion_scripts[@]}")
        ;;
    late_backbone)
        scripts=("${late_backbone_scripts[@]}")
        ;;
    ts_cva)
        scripts=("${ts_cva_scripts[@]}")
        ;;
    mv)
        scripts=("${mv_scripts[@]}")
        ;;
    mv2)
        scripts=("${mv2_scripts[@]}")
        ;;
    mv3)
        scripts=("${mv3_scripts[@]}")
        ;;
    mv4)
        scripts=("${mv4_scripts[@]}")
        ;;
    kfold)
        scripts=("${kfold_scripts[@]}")
        ;;
    ablation)
        scripts=("${ablation_scripts[@]}")
        ;;
    ablation_gate)
        scripts=(run_k_vivitsingle_fold0.sh run_k_vivitsingle_fold1.sh
                 run_k_vivitsingle_fold2.sh run_k_vivitsingle_fold3.sh
                 run_k_vivitsingle_fold4.sh)
        ;;
    noes)
        scripts=("${noes_scripts[@]}")
        ;;
    all)
        scripts=(
            "${single_view_scripts[@]}"
            "${modality_scripts[@]}"
            "${backbone_scripts[@]}"
            "${fusion_scripts[@]}"
            "${late_backbone_scripts[@]}"
            "${ts_cva_scripts[@]}"
            "${mv_scripts[@]}"
        )
        ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown mode: $MODE" >&2
        usage
        exit 2
        ;;
esac

if [[ "$ACTION" != "dry-run" && "$ACTION" != "--run" ]]; then
    echo "Unknown action: $ACTION" >&2
    usage
    exit 2
fi

echo "Mode: $MODE"
echo "Action: $ACTION"
echo

mkdir -p logs/pegasus/

for script in "${scripts[@]}"; do
    if [[ ! -f "pegasus/$script" ]]; then
        echo "Missing script: pegasus/$script" >&2
        exit 1
    fi

    if [[ "$ACTION" == "--run" ]]; then
        echo "qsub pegasus/$script"
        qsub "pegasus/$script"
        sleep 1
    else
        echo "[dry-run] qsub pegasus/$script"
    fi
done

if [[ "$ACTION" != "--run" ]]; then
    echo
    echo "Dry run only. Re-run with --run to submit."
fi
