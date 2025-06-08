import os
import cv2
import numpy as np
from pathlib import Path
import soundfile as sf
import subprocess
import re
import argparse

def create_videos(exp_name, experiment_dir, test_data_dir, output_filename=None, fold_num=None):
    """
    指定された実験名の楽器の生成画像から動画を作成
    Args:
        exp_name: 実験名（例：exp01_cello）
        experiment_dir: 実験ディレクトリのパス
        test_data_dir: テストデータディレクトリのパス
        output_filename: 出力動画のファイル名（デフォルトは楽器名.mp4）
        fold_num: 折り畳み番号（オプション、使用しない場合はNone）
    """
    # 楽器名をexp_nameから抽出
    instrument = exp_name.split('_')[-1]  # 例: exp01_cello -> cello
    
    # パスの設定
    image_dir = os.path.join(experiment_dir, "results", exp_name, "test_latest/images")
    audio_file = os.path.join(test_data_dir, "audio", f"{instrument}.wav")
    output_dir = os.path.join(experiment_dir, "evaluations/videos")
    
    if not os.path.exists(image_dir):
        print(f"Image directory not found: {image_dir}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # fake_image2_1.pngのパターンに合致する画像ファイルを取得
    image_files = sorted([f for f in os.listdir(image_dir) 
                         if f.endswith('fake_image2_1.png')])
    
    if not image_files:
        print(f"No matching fake_image2_1.png files found for {instrument}")
        return
    
    print(f"Processing {instrument} ({exp_name})")
    
    # 画像番号を基にソート
    def extract_number(filename):
        match = re.search(r'(\d+)_fake_image2_1.png', filename)
        if match:
            return int(match.group(1))
        return 0
    
    image_files.sort(key=extract_number)
    
    # 出力ファイル名を設定
    if output_filename is None:
        output_filename = f'{instrument}.mp4'
    
    # 動画を作成
    temp_video = os.path.join(output_dir, f'temp_{exp_name}.mp4')
    final_video = os.path.join(output_dir, output_filename)
    
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
            print(f"Created video with audio: {final_video}")
        except subprocess.CalledProcessError as e:
            print(f"Error creating video: {e}")
            # 音声なしでも動画を保存
            os.rename(temp_video, final_video)
            print(f"Created video without audio: {final_video}")
    else:
        print(f"Audio file not found: {audio_file}")
        os.rename(temp_video, final_video)
        print(f"Created video without audio: {final_video}")
    
    # 一時ファイルの削除
    if os.path.exists(temp_video):
        os.remove(temp_video)

def main():
    parser = argparse.ArgumentParser(description='Create videos from generated images')
    parser.add_argument('--exp_name', type=str, required=True,
                      help='Experiment name (e.g., exp01_cello)')
    parser.add_argument('--experiment_dir', type=str, required=True,
                      help='Experiment directory path')
    parser.add_argument('--test_data_dir', type=str, required=True,
                      help='Test data directory path')
    parser.add_argument('--output_filename', type=str, default=None,
                      help='Output filename (default: experiment_name.mp4)')
    parser.add_argument('--fold_num', type=str, default=None,
                      help='Fold number (optional, not used in current structure)')
    
    args = parser.parse_args()
    create_videos(args.exp_name, args.experiment_dir, args.test_data_dir, args.output_filename, args.fold_num)

if __name__ == "__main__":
    main()