#!/bin/bash
ROOT="/a/bear.cs.fiu.edu./disk/bear-b/users/hsale014/Project1/Fashion/FashionSAP"
MODE=${1:-smoke}
RESUME_PATH=$2   # optional: path to checkpoint to resume from
ANNO_DIR="${ROOT}/fashion_annotation"

echo "========================================"
echo "  FashionSAP-Sym  |  mode: ${MODE}"
echo "========================================"

# ── 1. MODE SETTINGS ─────────────────────────────────────────────────────────
if [ "${MODE}" = "smoke" ]; then
    SUBSET=0.05
    GPU_IDS="3"
    N_GPU=1
    ACCUM=8          # 1 GPU × 16 × 8  = 128 effective
    EVAL_FREQ=1      # evaluate every epoch
    SUMMARY="training_summary_smoke.txt"
    RUN_CONFIG="${ROOT}/configs/temp_smoke_config.yaml"
    OUTPUT_DIR="${ROOT}/output/smoke_run"
    echo ">> Mode: SMOKE  (5% data, eval every epoch)"

elif [ "${MODE}" = "tier2" ]; then
    SUBSET=0.20
    GPU_IDS="2"
    N_GPU=1
    ACCUM=8          # 1 GPU × 16 × 8  = 128 effective
    EVAL_FREQ=3      # evaluate every 3 epochs
    SUMMARY="training_summary_tier2.txt"
    RUN_CONFIG="${ROOT}/configs/tier2_config_20%.yaml"
    OUTPUT_DIR="${ROOT}/output/tier2_run"
    echo ">> Mode: TIER2  (20% data, eval every 3 epochs)"

elif [ "${MODE}" = "full" ]; then
    SUBSET=1.0
    GPU_IDS="2,3"
    N_GPU=2
    ACCUM=4        # 2 GPUs × 16 × 4 = 128 effective
    EVAL_FREQ=5      # evaluate every 5 epochs
    SUMMARY="training_summary_full.txt"
    RUN_CONFIG="${ROOT}/configs/fashion_pretrain_custom.yaml"
    OUTPUT_DIR="${ROOT}/output/full_run"
    echo ">> Mode: FULL   (100% data, eval every 5 epochs)"

else
    echo "ERROR: unknown mode '${MODE}'. Use: smoke | tier2 | full"
    exit 1
fi
# ─────────────────────────────────────────────────────────────────────────────

# ── 2. UPDATE PATHS IN THE CHOSEN CONFIG ─────────────────────────────────────
sed -i "s|bert_config:.*|bert_config: '${ANNO_DIR}/bert_config.json'|g" "$RUN_CONFIG"
sed -i "s|tokenizer_config:.*|tokenizer_config: '${ANNO_DIR}'|g" "$RUN_CONFIG"
# ─────────────────────────────────────────────────────────────────────────────

# ── 3. CREATE OUTPUT DIRECTORY ───────────────────────────────────────────────
mkdir -p "${OUTPUT_DIR}"
# ─────────────────────────────────────────────────────────────────────────────

# ── 4. RESUME LOGIC ──────────────────────────────────────────────────────────
RESUME_FLAG=""
if [ ! -z "$RESUME_PATH" ]; then
    RESUME_FLAG="--resume ${RESUME_PATH}"
    echo ">> ACTION: Resuming from checkpoint: ${RESUME_PATH}"
else
    echo ">> ACTION: Starting fresh from epoch 0"
fi
# ─────────────────────────────────────────────────────────────────────────────

echo "----------------------------------------"
echo "  GPUs        = ${GPU_IDS}"
echo "  Subset       = ${SUBSET}"
echo "  Accum steps  = ${ACCUM}  (effective batch = $((16 * ACCUM)))"
echo "  Eval freq    = every ${EVAL_FREQ} epoch(s)"
echo "  Summary file = ${SUMMARY}"
echo "  Output dir   = ${OUTPUT_DIR}"
echo "  Config       = ${RUN_CONFIG}"
echo "----------------------------------------"

# ── 5. RUN ───────────────────────────────────────────────────────────────────
LOG_FILE="${ROOT}/${MODE}_run.out"


# ADD these two lines before the nohup command
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

CUDA_VISIBLE_DEVICES=${GPU_IDS} nohup python -u \
    -m torch.distributed.launch \
    --nproc_per_node=${N_GPU} \
    --use_env \
    --master_port=48615 \
    "${ROOT}/fashion_pretrain_sym.py" \
    --config              "${RUN_CONFIG}"                     \
    --output_dir          "${OUTPUT_DIR}"                     \
    --data_root           "${ROOT}/data/data-fashion"         \
    --catemap_filename    "${ANNO_DIR}/categorys_to_sign.txt" \
    --pre_point           "${ROOT}/checkpoint_best.pth"       \
    --subset_ratio        "${SUBSET}"                         \
    --grad_accum_steps    "${ACCUM}"                          \
    --ita_warmup_epochs   5                                   \
    --eval_freq           "${EVAL_FREQ}"                      \
    --summary_name        "${SUMMARY}"                        \
    ${RESUME_FLAG}                                            \
    > "${LOG_FILE}" 2>&1 &
# ─────────────────────────────────────────────────────────────────────────────

echo "PID = $!"
echo "Log : tail -f '${LOG_FILE}'"