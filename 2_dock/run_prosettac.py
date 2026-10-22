"""
Stage 2: Submit top REINVENT candidates to pRosettaC for DDR2 ternary complex modeling.

Takes the filtered SMILES from Stage 1, prepares pRosettaC input files,
and submits jobs to Ernie.

Usage:
    python 2_dock/run_prosettac.py \
        --smiles outputs/ddr2_top100.smi \
        --poi DDR2 \
        --n_poses 200 \
        --out_dir outputs/prosettac_ddr2/
"""

import argparse
import os
import subprocess
from pathlib import Path

# pRosettaC config from protac-md project
PROSETTAC_DIR = Path.home() / "PRosettaC"
DDR2_RECEPTOR = Path.home() / "protac-md/simulations/DDR2/structure/rank1"
VHL_RECEPTOR  = Path.home() / "protac-md/reference_structures/4W9H.pdb"

# Reference: use DDR2 rank2 as geometry template (75.5% bridging, 100 ns MD)
DDR2_RANK2_REF = Path.home() / "protac-md/analysis/DDR2/rank2/ref_protein_protac.pdb"


def write_prosettac_input(smiles: str, name: str, out_dir: Path, n_poses: int = 200):
    out_dir.mkdir(parents=True, exist_ok=True)
    input_file = out_dir / f"{name}.inp"
    with open(input_file, "w") as f:
        f.write(f"SMILES {smiles}\n")
        f.write(f"RECEPTOR_POI {DDR2_RECEPTOR}\n")
        f.write(f"RECEPTOR_E3  {VHL_RECEPTOR}\n")
        f.write(f"N_POSES      {n_poses}\n")
        f.write(f"OUT_DIR      {out_dir / name}\n")
    return input_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles",   required=True, help="SMILES file (one per line)")
    parser.add_argument("--poi",      default="DDR2")
    parser.add_argument("--n_poses",  type=int, default=200)
    parser.add_argument("--out_dir",  required=True)
    args = parser.parse_args()

    smiles_list = Path(args.smiles).read_text().strip().splitlines()
    out_dir = Path(args.out_dir)

    print(f"Submitting {len(smiles_list)} candidates to pRosettaC for {args.poi}")

    for i, smi in enumerate(smiles_list):
        smi = smi.strip()
        if not smi:
            continue
        name = f"candidate_{i:04d}"
        inp = write_prosettac_input(smi, name, out_dir, args.n_poses)
        # TODO: replace with actual pRosettaC submission command on Ernie
        print(f"  [{i+1}/{len(smiles_list)}] {name}: input written to {inp}")

    print(f"\nAll inputs written to {out_dir}")
    print("Next: run pRosettaC on Ernie, then score with GNN (Stage 3)")


if __name__ == "__main__":
    main()
