export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:/storage3/TEV/gmei/code/F3DCut
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

python main.py \
    --use_color --use_normal \
    --detector perla \
    --captioner perla \
    --checkpoint_dir ./ckpts/opt-1.3b/perla-generalist \
    --test_ckpt ./ckpts/opt-1.3b/perla-scanrefer-tuned/checkpoint_best.pth \
    --dataset unified_densecap_scanrefer \
    --vocab facebook/opt-1.3b \
    --qformer_vocab /storage3/TEV/gmei/code/F3DCut/bert-base-embedding \
    --dist_url tcp://localhost:222 \
    --criterion 'CiDEr@0.5' \
    --freeze_detector --freeze_llm \
    --batchsize_per_gpu 8 --ngpus 1 \
    --max_des_len 256 \
    --max_prompt 1 \
    --use_beam_search \
    --test_only