"""
Stage 4: Set up 100 ns GROMACS MD for top GNN-scored candidates.

Takes the top N candidates by GNN score, finds their best pRosettaC pose,
and generates GROMACS input using the same protocol as protac-md.

Reference protocol: ~/protac-md/md/ (DDR2 rank2 — 75.5% bridging at 100 ns)

Usage:
    python 4_validate/setup_md.py \
        --gnn_scores outputs/gnn_scores.tsv \
        --prosettac_dir outputs/prosettac_ddr2/ \
        --top 5 \
        --md_out_dir outputs/md/
"""

import argparse
import shutil
from pathlib import Path

# MD protocol from protac-md (DDR2 rank2 reference)
MD_PROTOCOL_DIR = Path.home() / "protac-md/md"
DRMD_ENV        = Path.home() / ".conda/envs/drMD/bin/python3"


def setup_md_for_candidate(candidate: str, pose_pdb: Path, md_out: Path):
    md_out.mkdir(parents=True, exist_ok=True)
    # Copy topology template from protac-md DDR2 rank2
    template = Path.home() / "protac-md/simulations/DDR2"
    print(f"  Setting up MD for {candidate} → {md_out}")
    # TODO: adapt drMD input YAML for new ligand params + copy protocol files
    # The drMD pipeline: generate GAFF2 params → solvate → minimise → NVT → NPT → 100 ns prod
    placeholder = md_out / "TODO_setup_drMD.txt"
    placeholder.write_text(
        f"Candidate: {candidate}\n"
        f"Pose PDB:  {pose_pdb}\n"
        f"Protocol:  copy from {template}\n"
        f"Steps:     GAFF2 params → drMD YAML → submit on Ernie\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gnn_scores",    required=True)
    parser.add_argument("--prosettac_dir", required=True)
    parser.add_argument("--top",           type=int, default=5)
    parser.add_argument("--md_out_dir",    required=True)
    args = parser.parse_args()

    import pandas as pd
    df = pd.read_csv(args.gnn_scores, sep="\t").head(args.top)
    print(f"Setting up MD for top {len(df)} GNN-scored candidates:")

    for _, row in df.iterrows():
        name = row["candidate"]
        p    = row["p_degrader"]
        best_pose = Path(args.prosettac_dir) / name / "rank1" / "complex_protein.pdb"
        print(f"  {name}  P(deg)={p:.3f}")
        setup_md_for_candidate(name, best_pose, Path(args.md_out_dir) / name)

    print(f"\nMD setup written to {args.md_out_dir}")
    print("Next: run drMD on Ernie, then analyse bridging fraction as in protac-md")


if __name__ == "__main__":
    main()
