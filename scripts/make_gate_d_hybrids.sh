#!/bin/bash
set -u
# Build the Gate D hybrid session dirs: PREDICTION coordinates + SOURCE pixels.
#
# A prediction session written by scripts/infer.py carries session.toml/calibration.toml/groups.pq/
# points3d.pq/keypoints.pq/instances.pq and NO pixels (the format's rule 7). The scorer scores a
# track TOGETHER with its video, so scoring a predicted track needs the two joined: every label
# file comes from the prediction (as an absolute symlink), and `groups/` comes from the source
# session, whose camera files are themselves symlinks to the raw videos. Nothing is copied.
set -euo pipefail
cd /groups/karashchuk/home/karashchukl/projects/tailcycle/tailcyclenet
D=/groups/karashchuk/karashchuklab/animal-datasets-processed/tailcycle-datasets/3dpop
H=scratch/gated/hybrid; mkdir -p $H/test
for P in scratch/gated/pred-*; do
  [ -f "$P/session.toml" ] || continue
  SRC=$(grep -oE 'source_session_id = "[^"]+"' "$P/session.toml" | head -1 | sed 's/.*= "//;s/"//')
  [ -z "$SRC" ] && { echo "no source_session_id in $P"; continue; }
  REF=$D/test/$SRC
  [ -d "$REF" ] || { echo "MISSING REF $REF"; continue; }
  OUT=$H/test/$SRC
  if [ -d "$OUT" ]; then continue; fi
  mkdir -p "$OUT"
  for f in session.toml calibration.toml groups.pq points3d.pq keypoints.pq instances.pq; do
    [ -f "$P/$f" ] && ln -sf "$(readlink -f $P/$f)" "$OUT/$f"
  done
  ln -sfn "$(readlink -f $REF/groups)" "$OUT/groups"
  echo "hybrid $SRC"
done
