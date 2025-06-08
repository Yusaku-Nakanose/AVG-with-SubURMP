#!/usr/bin/env bash satoshi

# 楽器名を指定
instrument="trumpet"

# ディレクトリ名とパス
name="exp06_clap"
base_results_dir="/feoh/m2/nakanose/avg-with-suburmp/withclap"
original_dir="${base_results_dir}/${name}"
renamed_dir="${base_results_dir}/${name}_${instrument}"

# 実行
python3 ../test.py \
  --data_dir /dlbox/avg-with-suburmp/withclap/data/data_links/prelim/test_${instrument}.txt \
  --ngf 64 \
  --batch_size 1 \
  --model test_full_add_clap \
  --name ${name} \
  --checkpoints_dir /dlbox/avg-with-suburmp/withclap/checkpoints/ \
  --norm batch \
  --gpu_ids 0 \
  --eval \
  --verbose \
  --results_dir "${base_results_dir}"

# 実行後にディレクトリ名を変更
mv "${original_dir}" "${renamed_dir}"
