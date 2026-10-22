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
from rdkit.Chem.Scaffolds import MurckoScaffold


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
    linker_len = row.get('Linker length [8-14 bonds] (raw)', 11)
    return (850 <= mw <= 1150 and
            1.0 <= logp <= 9.0 and
            100 <= tpsa <= 250 and
            hbd <= 5 and
            rb <= 28 and
            8 <= linker_len <= 14)


def murcko_scaffold(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except Exception:
        return smi


def pick_one_per_scaffold(df: pd.DataFrame, top: int, score_col: str) -> pd.DataFrame:
    """One best-scoring representative per unique linker Murcko scaffold.
    Ranks scaffold groups by their best score, returns top N groups."""
    df = df.copy()
    df['_linker_scaffold'] = df['Linker'].apply(murcko_scaffold)

    # Best molecule per scaffold
    best_per_scaffold = (
        df.sort_values(score_col, ascending=False)
          .groupby('_linker_scaffold', sort=False)
          .first()
          .reset_index()
          .sort_values(score_col, ascending=False)
          .head(top)
    )
    return best_per_scaffold.drop(columns=['_linker_scaffold'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True)
    parser.add_argument('--top',    type=int, default=100)
    parser.add_argument('--out',    required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} candidates from {args.input}")

    # Keep valid SMILES only
    df = df[df['SMILES_state'] == 1].copy()
    print(f"Valid SMILES: {len(df)}")

    df = df[df.apply(passes_filters, axis=1)].copy()
    print(f"After physicochemical + linker-length filters: {len(df)}")

    score_col = 'Score'
    result = pick_one_per_scaffold(df, args.top, score_col)
    print(f"Selected {len(result)} candidates (1 per linker scaffold)")
    print(f"  Score range: {result[score_col].min():.3f} – {result[score_col].max():.3f}")
    print(f"  MW range:    {result['MW 850-1150 (raw)'].min():.0f} – {result['MW 850-1150 (raw)'].max():.0f} Da")
    print(f"  Unique linker scaffolds: {result['Linker'].apply(murcko_scaffold).nunique()}")

    result[['SMILES']].to_csv(args.out, index=False, header=False)
    print(f"\nWritten {len(result)} SMILES to {args.out}")
    print("Next: submit each SMILES to pRosettaC against DDR2, then score with GNN.")


if __name__ == '__main__':
    main()
