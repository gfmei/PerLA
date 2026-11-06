# Shared PerLA settings, sourced by train.sh and eval.sh.
# Sets: task_args (dataset, schedule, selection metric) and model_args (architecture).
perla_recipe() {
    local task=$1
    model_args=(
        --detector perla --captioner perla --use_color --use_normal
        --n_splits 6 --n_neighs 24 --n_clus 128
        --fusion_method gated
        --max_prompts 1 --use_beam_search
    )
    case "$task" in
        scanqa)
            task_args=(--dataset unified_scanqa --max_epoch 24 --max_des_len 224 --criterion CiDEr) ;;
        scanrefer)
            task_args=(--dataset unified_densecap_scanrefer --max_epoch 16 --max_des_len 256
                       --criterion CiDEr@0.5 --test_min_iou 0.5) ;;
        nr3d)
            task_args=(--dataset unified_densecap_nr3d --max_epoch 16 --max_des_len 256
                       --criterion CiDEr@0.5 --test_min_iou 0.5
                       --box_prompt_ratio 1.0 --min_box_prompt_iou 0.5)
            model_args+=(--box_part_tokens --context_objects 8 --qformer_rope) ;;
        *)
            echo "Unknown task '$task'; use scanqa, scanrefer or nr3d" >&2
            return 2 ;;
    esac
}
