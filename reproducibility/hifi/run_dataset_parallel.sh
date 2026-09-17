mkdir -p logs
for g in 0 1 2 3; do
  nohup python hifi_momfbd_batch_gpu.py \
      --input_dir /dat/andreuva/data/hifiplus/level1/20260715 \
      --output_dir results_momfbd/20260715_v2 \
      --gpu $g --num_shards 4 --shard_id $g \
      > logs/20260715_v2_gpu$g.log 2>&1 &
done
wait