python -m main train with "./configs/ctl1_gpu.yaml" \
        exp.name="${name}" \
        exp.savedir="./logs/" \
        exp.ckptdir="./logs/" \
        exp.tensorboard_dir="./tensorboard/" \
        exp.debug=True \
        trial=5 \
        --name="10scifar100_trial0_debug" \
        -D \
        -p \
        -c "None" \
        --force \
