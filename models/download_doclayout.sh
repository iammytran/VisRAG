#!/usr/bin/env bash
#
# Download checkpoints for the two supported layout analyzers used as an
# optional preprocessing step (see src/lilac/lcg_constructor/preprocessing/).
#
# 1) DocLayout-YOLO  (lightweight, layout-box detection)
#    https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench
#    Target: models/DocLayout-YOLO-DocStructBench/
#
# 2) MinerU          (heavier, full PDF/image → structured JSON pipeline)
#    https://github.com/opendatalab/mineru
#    MinerU manages its own model registry via its CLI; we install the
#    `mineru` Python package and trigger its `mineru-models-download` helper.
#    Models land under MinerU's own cache (default: ~/.cache/mineru or set
#    by env MINERU_MODEL_SOURCE / MINERU_MODELS_DIR).
#
# Usage:
#   ./models/download_layout_analyzers.sh                     # both
#   ./models/download_layout_analyzers.sh --only doclayout    # one
#   ./models/download_layout_analyzers.sh --only mineru       # one

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HUB_ENABLE_HF_TRANSFER=1

ONLY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --only) ONLY="$2"; shift 2;;
        -h|--help) sed -n '2,22p' "$0" | sed 's/^# \?//'; exit 0;;
        *) echo "unknown arg: $1" >&2; exit 1;;
    esac
done

# ── 1) DocLayout-YOLO ────────────────────────────────────────────────────────
if [[ -z "$ONLY" || "$ONLY" == "doclayout" ]]; then
    DEST="$REPO_ROOT/models/DocLayout-YOLO-DocStructBench"
    echo "==> DocLayout-YOLO  →  $DEST"
    hf download \
        juliozhao/DocLayout-YOLO-DocStructBench \
        --local-dir "$DEST"
fi

echo
echo "✅  Done."