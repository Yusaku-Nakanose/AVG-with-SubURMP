#!/bin/bash

# ディレクトリ構造設定
BASE_DIR="/feoh/m2/nakanose/avg-with-suburmp"
MODEL_TYPE="withclap"  # "tseandtsl" または "withclap" を指定
EXPERIMENT_DIR="${BASE_DIR}/experiments/${MODEL_TYPE}"
TEST_DATA_DIR="${BASE_DIR}/test_data"

GPU_ID="0"         # 利用するGPUのID
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# 必要なディレクトリを作成
mkdir -p ${EXPERIMENT_DIR}/evaluations/{fid,lpips,videos,fvd}

# 実験設定
settings=(
    'exp01_clap_clarinet clarinet' #
    'exp02_clap_trumpet trumpet' #
    'exp03_clap_violin violin' #
    'exp05_clap_cello cello' #
    'exp05_clap_viola viola' #
    'exp05_clap_violin violin' #
    'exp06_clap_clarinet clarinet'
    'exp06_clap_trumpet trumpet'
    'exp06_clap_violin violin'
    'exp08_clap_bassoon bassoon' #
    'exp08_clap_cello cello' #
    'exp08_clap_clarinet clarinet' #
    'exp08_clap_horn horn' #
    'exp08_clap_trumpet trumpet' #
    'exp08_clap_viola viola' #
    'exp08_clap_violin violin' #
    'exp21_clap_bassoon bassoon' #
    'exp21_clap_cello cello' #
    'exp21_clap_clarinet clarinet' #
    'exp21_clap_doublebass doublebass' #
    'exp21_clap_flute flute' #
    'exp21_clap_horn horn' #
    'exp21_clap_oboe oboe' #
    'exp21_clap_sax sax' #
    'exp21_clap_trombone trombone' #
    'exp21_clap_trumpet trumpet' #
    'exp21_clap_tuba tuba' #
    'exp21_clap_viola viola' #
    'exp21_clap_violin violin' #
)

echo "Starting evaluation for model: ${MODEL_TYPE}"
echo "Experiment directory: ${EXPERIMENT_DIR}"
echo "Test data directory: ${TEST_DATA_DIR}"

for ((i=0; ${#settings[*]}>$i; i++))
do
    tmp=(${settings[$i]}) #
    EXP_NAME=${tmp[0]}    #
    INST_FULL=${tmp[1]}   #
    
    IMAGE_DIR="${EXPERIMENT_DIR}/results/${EXP_NAME}/test_latest/images"
    FAKE_DIR="${EXPERIMENT_DIR}/evaluations/fid/${EXP_NAME}/fake_images" #
    REAL_DIR="${EXPERIMENT_DIR}/evaluations/fid/${EXP_NAME}/real_images" #
    
    echo "Processing ${EXP_NAME} (${INST_FULL})..."
    
    # 画像ディレクトリの存在確認
    if [ ! -d "${IMAGE_DIR}" ]; then
        echo "Warning: Image directory not found: ${IMAGE_DIR}"
        echo "Skipping ${EXP_NAME}"
        continue
    fi
    
    # ディレクトリを作成
    mkdir -p ${FAKE_DIR} #
    mkdir -p ${REAL_DIR} #
    mkdir -p ${EXPERIMENT_DIR}/evaluations/fid/${EXP_NAME} #
    
    # 画像コピー（FID/LPIPS用）
    # fake_image2_1.png と real_image1.png のパターンを使用
    find ${IMAGE_DIR} -type f -name "*_fake_image2_1.png" -exec cp {} ${FAKE_DIR}/ \; #
    find ${IMAGE_DIR} -type f -name "*_real_image1.png" -exec cp {} ${REAL_DIR}/ \; #
    
    # コピーされた画像数を確認
    FAKE_COUNT=$(ls -1 ${FAKE_DIR}/*.png 2>/dev/null | wc -l)
    REAL_COUNT=$(ls -1 ${REAL_DIR}/*.png 2>/dev/null | wc -l)
    
    if [ ${FAKE_COUNT} -eq 0 ] || [ ${REAL_COUNT} -eq 0 ]; then
        echo "Warning: No images found for ${EXP_NAME}. Fake: ${FAKE_COUNT}, Real: ${REAL_COUNT}"
        rm -rf ${FAKE_DIR} ${REAL_DIR}
        continue
    fi
    
    echo "  Found images - Fake: ${FAKE_COUNT}, Real: ${REAL_COUNT}"
    
    # FID評価
    echo ${EXP_NAME} >> ${EXPERIMENT_DIR}/evaluations/fid/fid_results_${MODEL_TYPE}.txt #
    python3 -m pytorch_fid ${REAL_DIR} ${FAKE_DIR} \
        --batch-size 4 --num-workers 0 --device cuda \
        >> ${EXPERIMENT_DIR}/evaluations/fid/fid_results_${MODEL_TYPE}.txt #
    echo "  Completed FID evaluation for ${INST_FULL} (${EXP_NAME}) at $(date)" #
    
    # LPIPS評価（統合版）
    python3 lpips_script.py \
        --image_dir ${IMAGE_DIR} \
        --exp_name ${EXP_NAME} \
        -o /tmp/lpips_temp_${EXP_NAME}.txt \
        --use_gpu #
    # 一時ファイルの内容を統合ファイルに追記
    cat /tmp/lpips_temp_${EXP_NAME}.txt >> ${EXPERIMENT_DIR}/evaluations/lpips/lpips_results_${MODEL_TYPE}.txt #
    # 一時ファイルを削除
    rm -f /tmp/lpips_temp_${EXP_NAME}.txt #
    echo "  Completed LPIPS evaluation for ${INST_FULL} (${EXP_NAME})" #
    
    # 動画作成
    python3 video_make.py \
        --exp_name ${EXP_NAME} \
        --experiment_dir ${EXPERIMENT_DIR} \
        --test_data_dir ${TEST_DATA_DIR} \
        --output_filename "${EXP_NAME}.mp4" #
    echo "  Completed video generation for ${INST_FULL} (${EXP_NAME})" #
    
    # テスト用ビデオが存在するか確認
    TEST_VIDEO_FILE="${TEST_DATA_DIR}/videos/gt_${INST_FULL}.mp4"
    FAKE_VIDEO_FILE="${EXPERIMENT_DIR}/evaluations/videos/${EXP_NAME}.mp4"
    
    # FVD評価（テスト用ビデオが存在する場合のみ）
    if [ -f "${TEST_VIDEO_FILE}" ]; then
        echo "${EXP_NAME}" >> ${EXPERIMENT_DIR}/evaluations/fvd/fvd_results_${MODEL_TYPE}.txt #
        python3 fvd_script.py \
            --real_video_file ${TEST_VIDEO_FILE} \
            --fake_video_file ${FAKE_VIDEO_FILE} \
            --clip-length 16 \
            --clip-step 1 \
            --use_gpu \
            >> ${EXPERIMENT_DIR}/evaluations/fvd/fvd_results_${MODEL_TYPE}.txt #
        echo "  Completed FVD evaluation for ${INST_FULL} (${EXP_NAME}) at $(date)" #
    else
        echo "  Test video not found: ${TEST_VIDEO_FILE}. Skipping FVD evaluation." #
    fi
    
    # クリーンアップ - fake_imagesとreal_imagesを削除
    rm -rf ${FAKE_DIR} ${REAL_DIR} #
    
    # 空のfidディレクトリを削除（エラー出力は破棄）
    rmdir ${EXPERIMENT_DIR}/evaluations/fid/${EXP_NAME} 2>/dev/null #
    
    echo "  Finished processing ${EXP_NAME}"
    echo "----------------------------------------"
done

echo "Evaluation completed for model: ${MODEL_TYPE}"