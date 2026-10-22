"""
Stage 2: Run gpu-prosettac for each REINVENT candidate against DDR2.

Re-uses existing DDR2 structures + warhead SDFs from kinome_scan (Q08345).
Runs 25 candidates sequentially on RTX 5090 via gpu-prosettac.

Usage:
    python 2_dock/run_prosettac.py \
        --smiles outputs/ddr2_top25.smi \
        --out_dir outputs/prosettac_DDR2/ \
        [--n_poses 200] [--steps 10000] [--chains 1024]
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ── Fixed DDR2 structure inputs (from kinome_scan Q08345) ─────────────────────
VHL_PDB     = Path.home() / "protac/data/structures/VHL/4W9H_VBC_clean.pdb"
DDR2_PDB    = Path.home() / "kinome_scan/data/kinase_pRosettaC/Q08345/protein.pdb"
VHL_HEAD    = Path.home() / "protac/data/structures/VHL/SJF8240_vhll_placed.sdf"
DDR2_HEAD   = Path.home() / "kinome_scan/data/kinase_pRosettaC/Q08345/warhead.sdf"
ANCHOR_A    = 6   # attachment atom index in VHL warhead SDF
ANCHOR_B    = 1   # attachment atom index in DDR2 warhead SDF

GPU_PROSETTAC = Path.home() / "gpu-prosettac"
PYTHON        = Path.home() / ".conda/envs/reinvent4/bin/python"


def write_params(candidate_dir: Path, smiles_file: Path) -> Path:
    params_file = candidate_dir / "params.txt"
    params_file.write_text(
        f"Structures: {VHL_PDB} {DDR2_PDB}\n"
        f"Chains: ABC A\n"
        f"Heads: {VHL_HEAD} {DDR2_HEAD}\n"
        f"Anchor atoms: {ANCHOR_A} {ANCHOR_B}\n"
        f"Protac: {smiles_file}\n"
        f"Full: True\n"
    )
    return params_file


def run_candidate(name: str, smiles: str, out_dir: Path,
                  n_poses: int, steps: int, chains: int) -> bool:
    candidate_dir = (out_dir / name).resolve()
    candidate_dir.mkdir(parents=True, exist_ok=True)

    smi_file = candidate_dir / f"{name}.smi"
    smi_file.write_text(f"{smiles}\t{name}\n")

    params = write_params(candidate_dir, smi_file.resolve())
    results_dir = candidate_dir / "Results"

    cmd = [
        str(PYTHON), "-m", "src.pipeline",
        "--config",   str(params),
        "--outdir",   str(results_dir),
        "--steps",    str(steps),
        "--chains",   str(chains),
        "--output_n", str(n_poses),
    ]

    log_file = candidate_dir / "gpu_prosettac.log"
    print(f"  Running {name} ...", flush=True)
    with open(log_file, "w") as lf:
        ret = subprocess.run(cmd, cwd=GPU_PROSETTAC, stdout=lf, stderr=lf)

    if ret.returncode != 0:
        print(f"  [FAILED] {name} — see {log_file}")
        return False

    n_poses_out = len(list(results_dir.glob("pose_*.pdb")))
    print(f"  [OK] {name}: {n_poses_out} poses → {results_dir}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles",   required=True, help="SMILES file (one per line)")
    parser.add_argument("--out_dir",  required=True)
    parser.add_argument("--n_poses",  type=int, default=200)
    parser.add_argument("--steps",    type=int, default=10000,
                        help="MC steps per chain (default 10000)")
    parser.add_argument("--chains",   type=int, default=1024,
                        help="Parallel MC chains (default 1024)")
    args = parser.parse_args()

    # Validate fixed inputs exist
    for p, label in [(VHL_PDB, "VHL PDB"), (DDR2_PDB, "DDR2 PDB"),
                     (VHL_HEAD, "VHL warhead SDF"), (DDR2_HEAD, "DDR2 warhead SDF")]:
        if not p.exists():
            print(f"ERROR: {label} not found: {p}", file=sys.stderr)
            sys.exit(1)

    smiles_lines = Path(args.smiles).read_text().strip().splitlines()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(smiles_lines)
    print(f"Submitting {n_total} candidates to gpu-prosettac (DDR2, {args.n_poses} poses each)")
    print(f"  VHL:  {VHL_PDB}")
    print(f"  DDR2: {DDR2_PDB}\n")

    n_ok = 0
    for i, line in enumerate(smiles_lines):
        smi = line.strip()
        if not smi:
            continue
        name = f"candidate_{i:04d}"
        ok = run_candidate(name, smi, out_dir, args.n_poses, args.steps, args.chains)
        if ok:
            n_ok += 1
        print(f"  Progress: {i+1}/{n_total}", flush=True)

    print(f"\nDone: {n_ok}/{n_total} succeeded → {out_dir}")
    print("Next: python 3_score/gnn_score.py --prosettac_dir", args.out_dir)


if __name__ == "__main__":
    main()
