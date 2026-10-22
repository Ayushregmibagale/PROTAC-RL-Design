"""
Post-REINVENT filter: reads Link-INVENT output CSV, ranks candidates,
and writes a SMILES file ready for pRosettaC submission.

Usage:
    python scoring/filter_and_prep_prosettac.py \
        --input outputs/ddr2_run_results.csv \
        --top 100 \
        --out outputs/ddr2_top100_for_prosettac.smi

Then run pRosettaC on each SMILES → score with dual-encoder GNN.
"""

import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# Linker length bounds that produced DDR2 rank2 75.5% bridging
LINKER_MIN = 9
LINKER_MAX = 15


def linker_length(smi: str, warhead_smi: str) -> int:
    """Shortest path atom count in the generated linker."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return -1
    # Proxy: total heavy atoms minus fragment sizes
    w = Chem.MolFromSmiles(warhead_smi.replace('[*]', '[H]'))
    warhead_ha = w.GetNumHeavyAtoms() if w else 36
    vhl_ha = 30  # fixed from SJF8240 fragmentation
    return mol.GetNumHeavyAtoms() - warhead_ha - vhl_ha


def passes_filters(row) -> bool:
    smi = row.get('SMILES', '')
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd  = rdMolDescriptors.CalcNumHBD(mol)
    rb   = rdMolDescriptors.CalcNumRotatableBonds(mol)
    return (850 <= mw <= 1150 and
            1.0 <= logp <= 9.0 and
            100 <= tpsa <= 250 and
            hbd <= 5 and
            rb <= 28)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True)
    parser.add_argument('--top',    type=int, default=100)
    parser.add_argument('--out',    required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} candidates from {args.input}")

    df = df[df.apply(passes_filters, axis=1)].copy()
    print(f"After physicochemical filters: {len(df)}")

    score_col = 'Score' if 'Score' in df.columns else df.columns[-1]
    df = df.sort_values(score_col, ascending=False).head(args.top)
    print(f"Top {len(df)} by {score_col}")

    df[['SMILES']].to_csv(args.out, index=False, header=False)
    print(f"Written to {args.out}")
    print("\nNext: submit each SMILES to pRosettaC against DDR2, then score with GNN.")


if __name__ == '__main__':
    main()
