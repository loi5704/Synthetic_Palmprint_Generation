#!/bin/bash
set -e

# $1: Mode (1: Augment, 2: Synthetic)
# $2: Output Dir
# $3: Input Dir
# $4: Param A (Mode 1: Tổng số ảnh / Mode 2: Số lượng ID)
# $5: Param B (Mode 1: Không dùng / Mode 2: Số ảnh mỗi ID)

MODE=$1
CHK_DIR=$2
IN_DIR=$3
PARAM_A=$4
PARAM_B=$5

MODEL_PATH_ABS="$PWD/models/model.pt"

echo "------------------------------------------------"
echo "--- DIFF-PALM WORKFLOW STARTING (Mode: $MODE) ---"

cd Diff-Palm/DiffModels

export OPENAI_LOGDIR=$CHK_DIR

if [ ! -d "$CHK_DIR" ]; then
mkdir -p "$CHK_DIR"
fi

mkdir -p "$IN_DIR"

NPZ="${CHK_DIR}/data.npz"
OUTDIR1="${CHK_DIR}/label"

# --- TASK 1: AUGMENTATION (TỪ ẢNH THẬT) ---
if [ "$MODE" == "1" ]; then

    if [ -z "$PARAM_A" ]; then PARAM_A=8; fi

    TOTAL_SAMPLES=$PARAM_A

    NPZ_NUM=$TOTAL_SAMPLES

    SHARING_NUM=$TOTAL_SAMPLES

    echo ">>> Cấu hình Mode 1: Input 1 ảnh -> Sinh ra $TOTAL_SAMPLES biến thể."

elif [ "$MODE" == "2" ]; then

    # PARAM_A là SỐ LƯỢNG ID (Nếp nhăn)
    if [ -z "$PARAM_A" ]; then PARAM_A=8; fi
    # PARAM_B là SỐ ẢNH TRÊN MỖI ID
    if [ -z "$PARAM_B" ]; then PARAM_B=1; fi

    NUM_IDS=$PARAM_A
    IMGS_PER_ID=$PARAM_B
    TOTAL_SAMPLES=$((NUM_IDS * IMGS_PER_ID))
    NPZ_NUM=$NUM_IDS
    SHARING_NUM=$IMGS_PER_ID

    POLY_SCRIPT="../PolyCreases/syn_polypalm_mp.py"
    DATA_PKL="../PolyCreases/labeled_data.pkl"  # <-- Định nghĩa đường dẫn file data

    # Kiểm tra file script
    if [ ! -f "$POLY_SCRIPT" ]; then
        echo "LỖI: Không tìm thấy file script $POLY_SCRIPT"
        exit 1
    fi
    
    # Kiểm tra và copy file labeled_data.pkl sang thư mục hiện tại (DiffModels)
    if [ -f "$DATA_PKL" ]; then
        echo ">>> [Mode 2] Copy file labeled_data.pkl sang DiffModels..."
        cp "$DATA_PKL" .
    else
        echo "LỖI NGHIÊM TRỌNG: Không tìm thấy file labeled_data.pkl trong PolyCreases!"
        echo "Bạn hãy kiểm tra lại file Zip xem đã có file này chưa?"
        ls -R ../
        exit 1
    fi

    echo ">>> [Mode 2] Đang sinh $NUM_IDS mẫu nếp nhăn..."
    
    # Gọi python chạy file script này
    python3 "$POLY_SCRIPT" \
        --ids $NUM_IDS \
        --output "$IN_DIR" \
        --nproc 4

    echo ">>> Cấu hình Mode 2: $NUM_IDS IDs x $IMGS_PER_ID ảnh/ID = Tổng $TOTAL_SAMPLES ảnh."
fi
# --- BƯỚC 1: ĐÓNG GÓI LABEL ---
echo ">>> [1/3] Đóng gói dữ liệu input..."
python3 scripts/save_npz.py \
    --input "$IN_DIR" \
    --outdir "$OUTDIR1" \
    --outnpz "$NPZ" \
    --num $TOTAL_SAMPLES \
    --same_num $SHARING_NUM

# --- BƯỚC 2: CHẠY MODEL DIFFUSION ---
echo ">>> [2/3] Đang sinh ảnh (Sampling)..."

# Tự động chỉnh Batch Size để tránh tràn RAM GPU
# Nếu tổng số mẫu > 8 thì chia nhỏ batch là 4, ngược lại thì chạy hết 1 lần
BATCH_SIZE=$TOTAL_SAMPLES
    
INTRA_FLAGS="--sharing_num $SHARING_NUM --sharing_step 500"
SAMPLE_FLAGS="--batch_size $BATCH_SIZE --num_samples $TOTAL_SAMPLES --use_ddim False"
MODEL_FLAGS="--large_size 128 --small_size 128 --in_channels 4 --out_channels 3 --num_channels 64 --num_res_blocks 2 --learn_sigma True --dropout 0.1 --attention_resolutions 4 --class_cond False"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"

python3 palm_sample_intra.py \
    --model_path "$MODEL_PATH_ABS" \
    --base_samples "$NPZ" \
    $SAMPLE_FLAGS $DIFFUSION_FLAGS $MODEL_FLAGS $INTRA_FLAGS

# --- BƯỚC 3: GIẢI NÉN KẾT QUẢ ---
echo ">>> [3/3] Giải nén kết quả..."
# File output của model luôn có định dạng samples_{TOTAL}x128x128x3.npz
python3 scripts/load_npz.py \
    --input "${CHK_DIR}/samples_${TOTAL_SAMPLES}x128x128x3.npz" \
    --outdir "${CHK_DIR}/results"

echo "Done!"
