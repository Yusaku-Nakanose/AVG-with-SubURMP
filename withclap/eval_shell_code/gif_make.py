import os
import cv2
import numpy as np
from pathlib import Path
import soundfile as sf
import subprocess
import re
import argparse

def create_test_videos(exp_name, experiment_dir, test_data_dir):
    """
    指定された実験名の楽器の実際の画像からテスト用動画を作成
    Args:
        exp_name: 実験名（例：exp01_cello）
        experiment_dir: 実験ディレクトリのパス
        test_data_dir: テストデータディレクトリのパス
    """
    # 楽器名をexp_nameから抽出
    instrument = exp_name.split('_')[1]  # 例: exp01_cello -> cello
    
    # パスの設定
    image_dir = os.path.join(experiment_dir, "results", exp_name, "test_latest/images")
    audio_file = os.path.join(test_data_dir, "audio", f"{instrument}.wav")
    output_dir = os.path.join(test_data_dir, "videos")
    
    if not os.path.exists(image_dir):
        print(f"Image directory not found: {image_dir}")
        return
    
    os.makedirs(output_dir, exist_ok=True)

    # real_image1.pngのパターンに合致する画像ファイルを取得
    image_files = sorted([f for f in os.listdir(image_dir) 
                         if f.endswith('real_image1.png')])
    
    if not image_files:
        print(f"No matching real_image1.png files found for {instrument}")
        return

    print(f"Processing test videos for {instrument} ({exp_name})")
    
    # 画像番号を基にソート
    def extract_number(filename):
        match = re.search(r'(\d+)_real_image1.png', filename)
        if match:
            return int(match.group(1))
        return 0
    
    image_files.sort(key=extract_number)
    
    # 動画を作成
    temp_video = os.path.join(output_dir, f'temp_gt_{instrument}.mp4')
    final_video = os.path.join(output_dir, f'gt_{instrument}.mp4')
    
    first_image = cv2.imread(os.path.join(image_dir, image_files[0]))
    if first_image is None:
        print(f"Failed to read first image for {instrument}")
        return
        
    height, width = first_image.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 10
    out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
    
    for img_file in image_files:
        image = cv2.imread(os.path.join(image_dir, img_file))
        if image is not None:
            out.write(image)
        else:
            print(f"Failed to read image: {img_file}")
    
    out.release()
    
    # 対応する音声ファイルを確認
    if os.path.exists(audio_file):
        # FFmpegコマンドの修正
        cmd = [
            'ffmpeg', '-y',
            '-loglevel', 'error',  # エラーのみ表示
            '-i', temp_video,
            '-i', audio_file,
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-strict', 'experimental',
            final_video
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"Created test video with audio: {final_video}")
        except subprocess.CalledProcessError as e:
            print(f"Error creating test video: {e}")
            # 音声なしでも動画を保存
            os.rename(temp_video, final_video)
            print(f"Created test video without audio: {final_video}")
    else:
        print(f"Audio file not found: {audio_file}")
        os.rename(temp_video, final_video)
        print(f"Created test video without audio: {final_video}")
    
    # 一時ファイルの削除
    if os.path.exists(temp_video):
        os.remove(temp_video)

def main():
    parser = argparse.ArgumentParser(description='Create test videos from real images')
    parser.add_argument('--exp_name', type=str, required=True,
                      help='Experiment name (e.g., exp01_cello)')
    parser.add_argument('--experiment_dir', type=str, required=True,
                      help='Experiment directory path')
    parser.add_argument('--test_data_dir', type=str, required=True,
                      help='Test data directory path')
    args = parser.parse_args()

    create_test_videos(args.exp_name, args.experiment_dir, args.test_data_dir)

if __name__ == "__main__":
    main()