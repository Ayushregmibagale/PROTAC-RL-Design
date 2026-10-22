#!/usr/bin/env bash
# Full PROTAC-RL-Design pipeline — DDR2 PROTAC linker optimisation
#
# Stage 1: REINVENT4 LinkINVENT RL → generate linker candidates
# Stage 2: pRosettaC → ternary complex docking (on Ernie)
# Stage 3: dual-encoder GNN → rank by P(degrader)
# Stage 4: GROMACS MD (100 ns, drMD) → validate top 5 by bridging fraction
#
# Usage: bash pipeline.sh [--top_n 100] [--md_top 5]

set -e
TOP_N=${1:-100}
MD_TOP=${2:-5}
POI="DDR2"

REINVENT_ENV="$HOME/.conda/envs/reinvent4/bin/python"
DRMD_ENV="$HOME/.conda/envs/drMD/bin/python3"
PROJECT="$HOME/PROTAC-RL-Design"

echo "=== Stage 1: REINVENT4 LinkINVENT ==="
conda activate reinvent4 2>/dev/null || true
cd "$PROJECT"
reinvent -l outputs/ddr2_run.log 1_generate/configs/linkinvent_ddr2.toml
# Output: outputs/ddr2_rl_*.csv

echo "=== Stage 1 post-filter ==="
$REINVENT_ENV 2_dock/filter_and_prep_prosettac.py \
    --input outputs/ddr2_rl_1.csv \
    --top "$TOP_N" \
    --out outputs/ddr2_top${TOP_N}.smi

echo "=== Stage 2: pRosettaC (submit to Ernie) ==="
$REINVENT_ENV 2_dock/run_prosettac.py \
    --smiles outputs/ddr2_top${TOP_N}.smi \
    --poi "$POI" \
    --n_poses 200 \
    --out_dir outputs/prosettac_${POI}/

echo "--- Run pRosettaC on Ernie, then re-run from Stage 3 ---"
echo "    Typical runtime: ~170 PDBs/hr on Ernie"
exit 0

echo "=== Stage 3: GNN scoring ==="
$REINVENT_ENV 3_score/gnn_score.py \
    --prosettac_dir outputs/prosettac_${POI}/ \
    --out outputs/gnn_scores.tsv

echo "=== Stage 4: MD setup ==="
$REINVENT_ENV 4_validate/setup_md.py \
    --gnn_scores outputs/gnn_scores.tsv \
    --prosettac_dir outputs/prosettac_${POI}/ \
    --top "$MD_TOP" \
    --md_out_dir outputs/md/

echo "=== Pipeline complete — submit MD jobs on Ernie ==="
