export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
# export CUDA_VISIBLE_DEVICES=0
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export OMP_NUM_THREADS=2

# Default parameter values
CHECKPOINT_DIR="ckpts/opt-1.3b/perla"
BATCHSIZE_PER_GPU=4
NGPUS=1

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --checkpoint_dir) CHECKPOINT_DIR="$2"; shift ;;
        --batchsize_per_gpu) BATCHSIZE_PER_GPU="$2"; shift ;;
        --ngpus) NGPUS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Run the Python script with the provided parameters
python main.py \
    --use_color --use_normal \
    --detector perla \
    --captioner perla \
    --checkpoint_dir ./ckpts/opt-1.3b/perla-generalist \
    --test_ckpt ./ckpts/opt-1.3b/perla-generalist/perla-opt-1.3b.pth \
    --dataset unified_3dllm_scene_description \
    --vocab facebook/opt-1.3b \
    --qformer_vocab bert-base-embedding \
    --dist_url tcp://localhost:82347 \
    --criterion 'CiDEr' \
    --freeze_detector --freeze_llm \
    --batchsize_per_gpu "$BATCHSIZE_PER_GPU" --ngpus "$NGPUS" \
    --max_des_len 512 \
    --max_prompt 1 \
    --use_beam_search \
    --test_only
