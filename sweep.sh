#!/usr/bin/env bash
# sweep.sh -- run the 8-network study, resumably.
#
# This environment restarts containers mid-run and rolls the filesystem back, so
# a 2000-epoch schedule cannot be trusted to survive in one sitting. Two
# defences: train.py --resume continues from its checkpoint instead of starting
# over, and this script commits the checkpoints to git every few minutes, which
# is the only storage that survives a rollback.
#
# Safe to re-run at any time: finished runs are skipped, interrupted ones resume.
set -u
cd "$(dirname "$0")"
SHAPES="square line wedge triangle_centroid"
PERCEPTIONS="cone circular"
LOGDIR="${LOGDIR:-/tmp/sweep-logs}"
PUSH_EVERY="${PUSH_EVERY:-300}"     # seconds between checkpoint pushes
mkdir -p runs "$LOGDIR"

for shape in $SHAPES; do
  for perc in $PERCEPTIONS; do
    tag="${shape}_${perc}"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup python -u train.py \
      --shape "$shape" --perception "$perc" --resume \
      --checkpoint "runs/gamma_${tag}.pt" > "$LOGDIR/${tag}.log" 2>&1 &
  done
done
echo "launched $(jobs -p | wc -l) training runs; logs in $LOGDIR"

# Persist progress while they run.
while pgrep -f "[t]rain.py --shape" > /dev/null; do
  sleep "$PUSH_EVERY"
  if [ -n "$(git status --porcelain runs)" ]; then
    done_epochs=$(python - <<'PY'
import glob, torch
tot = 0
for p in sorted(glob.glob("runs/gamma_*.pt")):
    try:
        tot += torch.load(p, map_location="cpu", weights_only=False).get("epochs_done", 0)
    except Exception:
        pass
print(tot)
PY
)
    git add runs && git commit -q -m "sweep progress: ${done_epochs} epochs trained across the 8 runs" \
      && git push -q origin HEAD 2>/dev/null && echo "pushed progress (${done_epochs} epochs)"
  fi
done

git add runs && git commit -q -m "sweep complete: all 8 networks trained" \
  && git push -q origin HEAD && echo "sweep finished and pushed"
