export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
# export CUDA_VISIBLE_DEVICES=0
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export OMP_NUM_THREADS=2

# Default parameter values
CHECKPOINT_DIR="base_ckpts/opt-1.3b/perla-scanqa"
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


python main.py \
    --use_color --use_normal \
    --detector perla \
    --captioner perla \
    --pretrained_weights ckpts/perla-generalist/perla-opt-1.3b.pth \
    --warm_lr_epochs 1 \
    --dataset unified_3dllm_scene_description \
    --vocab facebook/opt-1.3b \
    --qformer_vocab bert-base-embedding \
    --checkpoint_dir ckpts/opt-1.3b/perla-dense \
    --max_epoch 32 \
    --dist_url tcp://localhost:72346 \
    --eval_every_iteration 10000 \
    --start_eval_after 19999 \
    --save_every 10000 \
    --criterion 'CiDEr' \
    --freeze_detector --freeze_llm \
    --batchsize_per_gpu "$BATCHSIZE_PER_GPU" --ngpus "$NGPUS" \
    --base_lr 5e-5 --final_lr 1e-6 \
    --max_des_len 512 \
    --max_prompt 1 --use_beam_search
