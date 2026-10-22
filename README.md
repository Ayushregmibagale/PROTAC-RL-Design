# PROTAC-RL-Design

Generative PROTAC linker design pipeline combining REINVENT4 reinforcement learning with pRosettaC ternary complex modeling, dual-encoder GNN scoring, and GROMACS MD validation.

## Overview

Four-stage pipeline for designing novel PROTAC linkers targeting DDR2 (best degrader from the SJF8240 kinase panel):

```
Stage 1 — REINVENT4 LinkINVENT (RL)
  Fixed fragments: Foretinib warhead | VH032 VHL ligand
  RL scoring: linker length [8–14 bonds] + MW + logP + TPSA + HBD + RotBonds
  Output: ~1000s of linker candidates as SMILES
        ↓ physicochemical filter → top 100
Stage 2 — pRosettaC
  Ternary complex docking for each candidate against DDR2
  n_poses = 200 per candidate
  Reference: DDR2 rank2 pose (75.5% bridging, 100 ns MD)
        ↓ → top 20 by bridging fraction
Stage 3 — Dual-encoder GNN
  EnsembleGAT scores pRosettaC ensemble graphs
  Trained on 17 POIs (AUROC 0.85 LOPO-CV)
  Encoder B (MDGAT) zeroed — no MD data for new candidates
        ↓ → top 5 by P(degrader)
Stage 4 — GROMACS MD (100 ns, drMD)
  Validate bridging fraction and ternary complex stability
  Protocol: same as protac-md DDR2 rank2 reference run
```

## Fragment inputs

Derived from SJF8240 (DDR2 rank2: 75.5% bridging at 100 ns).

| Fragment | Heavy atoms | SMILES |
|----------|-------------|--------|
| Warhead (Foretinib-based) | 36 | `[*]c1cc2nccc(Oc3ccc(N(C(=O)C4(C(N)=O)CC4)c4ccc(F)cc4)cc3F)c2cc1OC` |
| VHL ligand (VH032-based) | 30 | `[*]N[C@H](C(=O)N1C[C@H](O)C[C@H]1C(=O)NCc1ccc(-c2scnc2C)cc1)C(C)(C)C` |

Linker constraint: `FragmentEffectiveLength` [8, 14] bonds (SJF8240 = 11 bonds).

## Installation

```bash
# 1. Clone
git clone https://github.com/Ayushregmibagale/PROTAC-RL-Design.git
cd PROTAC-RL-Design

# 2. Install REINVENT4
git clone https://github.com/MolecularAI/REINVENT4.git ~/software/REINVENT4
conda create -n reinvent4 python=3.11 -y
conda activate reinvent4
cd ~/software/REINVENT4 && python install.py cu126 -e

# 3. Download LinkINVENT transformer prior (~77 MB)
mkdir -p models
curl -L "https://zenodo.org/api/records/20701824/files/linkinvent_transformer_pubchem.prior/content" \
     -o models/linkinvent_transformer_pubchem.prior
```

## Usage

```bash
# Stage 1 — generate
conda activate reinvent4
cd ~/PROTAC-RL-Design
reinvent -l outputs/ddr2_run.log 1_generate/configs/linkinvent_ddr2.toml

# Stage 1 post-filter
python 2_dock/filter_and_prep_prosettac.py \
    --input outputs/ddr2_rl_1.csv --top 100 --out outputs/ddr2_top100.smi

# Stage 2 — pRosettaC (submit to Ernie HPC)
python 2_dock/run_prosettac.py \
    --smiles outputs/ddr2_top100.smi --poi DDR2 --n_poses 200 \
    --out_dir outputs/prosettac_DDR2/

# Stage 3 — GNN score
python 3_score/gnn_score.py \
    --prosettac_dir outputs/prosettac_DDR2/ --out outputs/gnn_scores.tsv

# Stage 4 — MD setup
python 4_validate/setup_md.py \
    --gnn_scores outputs/gnn_scores.tsv --top 5 --md_out_dir outputs/md/
```

## Repository structure

```
1_generate/
  configs/linkinvent_ddr2.toml   REINVENT4 staged-learning config
  inputs/ddr2_fragments.smi      Warhead|VHL fragment SMILES
2_dock/
  filter_and_prep_prosettac.py   Post-REINVENT filter + pRosettaC input prep
  run_prosettac.py               pRosettaC job submission
3_score/
  gnn_score.py                   Dual-encoder GNN scoring of pRosettaC outputs
4_validate/
  setup_md.py                    GROMACS MD setup for top candidates
pipeline.sh                      End-to-end run script
environment.yml                  Conda environment spec
```

## References

- REINVENT4: [Loeffler et al., J. Cheminf. 2024](https://link.springer.com/article/10.1186/s13321-024-00812-5)
- Link-INVENT: [Guo et al., Digital Discovery 2023](https://pubs.rsc.org/en/content/articlehtml/2023/dd/d2dd00115b)
- pRosettaC: [Brelyfiled et al., Structure 2021](https://doi.org/10.1016/j.str.2021.01.007)
