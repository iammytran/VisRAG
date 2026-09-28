#!/usr/bin/env bash
set -euo pipefail

GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
WORLD_SIZE=4
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="visrag_scripts/generate/generate.py"
OUTPUT_ROOT="./outputs/generation"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

IFS=',' read -r -a GPU_ID_LIST <<< "$GPU_IDS"
if [[ ${#GPU_ID_LIST[@]} -ne $WORLD_SIZE ]]; then
    echo "Expected exactly $WORLD_SIZE GPUs in CUDA_VISIBLE_DEVICES, got: $GPU_IDS" >&2
    exit 2
fi

PIDS=()
cleanup() {
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup INT TERM

for rank in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="$GPU_IDS" \
    "$PYTHON_BIN" "$SCRIPT" \
        --model_name Qwen2.5-VL-7B-Instruct \
        --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
        --dataset_name InfoVQA \
        --dataset_name_or_path openbmb/VisRAG-Ret-Test-InfoVQA \
        --rank "$rank" \
        --world_size "$WORLD_SIZE" \
        --topk 1 \
        --min_pixels $((256 * 28 * 28)) \
        --max_pixels $((1024 * 28 * 28)) \
        --results_root_dir /workspace/VisRAG/checkpoints/eval-2026-09-27-211520-maxq-512-maxp-2048-bsz-4-pooling-wmean-attention-causal-gpus-per-node-4 \
        --task_type multi_image \
        --output_dir "$OUTPUT_ROOT/rank_$rank" &
    PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done

if [[ "$status" -ne 0 ]]; then
    exit "$status"
fi

"$PYTHON_BIN" visrag_scripts/generate/evaluate_infovqa.py \
    --input_dir "$OUTPUT_ROOT" \
    --output_file "$OUTPUT_ROOT/metrics.json"
