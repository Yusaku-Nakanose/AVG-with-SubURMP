#!/bin/bash

# ディレクトリ構造設定
BASE_DIR="/feoh/m2/nakanose/avg-with-suburmp"
MODEL_TYPE="tseandtsl"  # "tseandtsl" または "withclap" を指定
EXPERIMENT_DIR="${BASE_DIR}/experiments/${MODEL_TYPE}"
TEST_DATA_DIR="${BASE_DIR}/test_data"

# 必要なディレクトリを作成
mkdir -p ${TEST_DATA_DIR}/videos

settings=(
    'exp01_clarinet clarinet'
    'exp02_trumpet trumpet'
    'exp03_violin violin'
    'exp05_cello cello'
    'exp05_viola viola'
    'exp05_violin violin'
    'exp06_clarinet clarinet'
    'exp06_trumpet trumpet'
    'exp06_violin violin'
    'exp08_bassoon bassoon'
    'exp08_cello cello'
    'exp08_clarinet clarinet'
    'exp08_horn horn'
    'exp08_trumpet trumpet'
    'exp08_viola viola'
    'exp08_violin violin'
    'exp21_bassoon bassoon'
    'exp21_cello cello'
    'exp21_clarinet clarinet'
    'exp21_doublebass doublebass'
    'exp21_flute flute'
    'exp21_horn horn'
    'exp21_oboe oboe'
    'exp21_sax sax'
    'exp21_trombone trombone'
    'exp21_trumpet trumpet'
    'exp21_tuba tuba'
    'exp21_viola viola'
    'exp21_violin violin'
)

# 環境変数の設定
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Starting test video generation for model: ${MODEL_TYPE}"
echo "Experiment directory: ${EXPERIMENT_DIR}"
echo "Test data directory: ${TEST_DATA_DIR}"

for ((i=0; ${#settings[*]}>$i; i++))
do
    tmp=(${settings[$i]})
    EXP_NAME=${tmp[0]}
    INST_FULL=${tmp[1]}
    
    echo "Processing ${EXP_NAME} (${INST_FULL})..."
    
    # テスト用動画作成
    python3 gt_make.py \
        --exp_name ${EXP_NAME} \
        --experiment_dir ${EXPERIMENT_DIR} \
        --test_data_dir ${TEST_DATA_DIR}
    echo "  Completed test video generation for ${INST_FULL} (${EXP_NAME})"
    
    echo "----------------------------------------"
done

echo "Test video generation completed for model: ${MODEL_TYPE}"