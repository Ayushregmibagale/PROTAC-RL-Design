"""
Stage 3: Score pRosettaC ternary complex poses with the dual-encoder GNN.

Loads pRosettaC ensemble graphs for each candidate, runs the trained
EnsembleGAT (Encoder A), and outputs P(degrader) scores.

Encoder B (MDGAT) is zeroed out — no MD data for new candidates.
This matches the LOPO-CV behaviour for non-MD POIs in training.

Usage:
    python 3_score/gnn_score.py \
        --prosettac_dir outputs/prosettac_ddr2/ \
        --gnn_model ~/protac-md/analysis/gnn/dual_encoder_best.pt \
        --out outputs/gnn_scores.tsv
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import torch
    from torch_geometric.data import Data
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# GNN paths from protac-md project
GNN_DIR   = Path.home() / "protac-md/analysis/gnn"
STATIC_PT = GNN_DIR / "ensemble_graphs.pt"


def load_gnn_model(model_path: Path):
    if not HAS_TORCH:
        raise ImportError("torch and torch_geometric required")
    import sys
    sys.path.insert(0, str(Path.home() / "protac-md/scripts/02_analysis"))
    from dual_encoder_gnn_04d import DualEncoderGNN  # noqa
    model = DualEncoderGNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def score_candidates(prosettac_dir: Path, model_path: Path, out_path: Path):
    model = load_gnn_model(model_path)
    rows = []
    for candidate_dir in sorted(prosettac_dir.iterdir()):
        if not candidate_dir.is_dir():
            continue
        graph_file = candidate_dir / "ensemble_graph.pt"
        if not graph_file.exists():
            print(f"  Skipping {candidate_dir.name}: no ensemble_graph.pt")
            continue
        data = torch.load(graph_file, map_location="cpu")
        with torch.no_grad():
            prob = model(data, md_data=None).sigmoid().item()
        rows.append({"candidate": candidate_dir.name, "p_degrader": prob})
        print(f"  {candidate_dir.name}: P(degrader) = {prob:.3f}")

    df = pd.DataFrame(rows).sort_values("p_degrader", ascending=False)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"\nScores written to {out_path}")
    print(f"Top 5:\n{df.head()}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prosettac_dir", required=True)
    parser.add_argument("--gnn_model",     default=str(GNN_DIR / "dual_encoder_best.pt"))
    parser.add_argument("--out",           required=True)
    args = parser.parse_args()

    score_candidates(
        Path(args.prosettac_dir),
        Path(args.gnn_model),
        Path(args.out),
    )


if __name__ == "__main__":
    main()
