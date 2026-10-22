#!/bin/bash
# rsync_to_ernie.sh — run LOCALLY (WSL)
#
# Pushes the 25 DDR2 candidate run directories + submission script to Ernie,
# then prints the commands to run there.
#
# Usage:
#   bash 2_dock/rsync_to_ernie.sh

set -e
ERNIE="s1638308@ernie.bio.ed.ac.uk"
LOCAL_SRC="$HOME/PROTAC-RL-Design/outputs/ernie_submission/"
ERNIE_DEST="$ERNIE:~/PROTAC-RL-Design/pRosettaC/"

echo "=== Pushing 25 DDR2 candidates to Ernie ==="
echo "  From: $LOCAL_SRC"
echo "  To:   $ERNIE_DEST"
echo ""

rsync -av --progress \
    "$LOCAL_SRC" \
    "$ERNIE_DEST"

echo ""
echo "=== Transfer complete ==="
echo ""
echo "Now on Ernie, run:"
echo "  screen -S ddr2_rl"
echo "  conda activate protac"
echo "  bash ~/PROTAC-RL-Design/pRosettaC/submit_ddr2_candidates.sh"
echo "  Ctrl-A D"
echo ""
echo "Monitor progress:"
echo "  ssh $ERNIE 'tail -f ~/PROTAC-RL-Design/pRosettaC/logs/candidate_0000.log'"
echo ""
echo "Rsync results back when done:"
echo "  rsync -av $ERNIE_DEST $HOME/PROTAC-RL-Design/outputs/prosettac_DDR2/"
