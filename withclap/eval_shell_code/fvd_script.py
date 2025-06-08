#!/usr/bin/env python3
import argparse
import os
import torch
import torchvision
from torchvision.io import read_video
import numpy as np
from scipy import linalg
import traceback
import sys

# 標準出力のリダイレクト用クラス
class SuppressOutput:
    def __init__(self):
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
        self.save_fds = [os.dup(1), os.dup(2)]
    
    def __enter__(self):
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)
        
    def __exit__(self, *_):
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

def extract_clip_features(video, start, clip_length, device, model, mean, std):
    try:
        clip = video[start:start+clip_length]
        clip = clip.permute(3, 0, 1, 2)
        clip = clip.unsqueeze(0)
        clip = torch.nn.functional.interpolate(clip, size=(clip_length, 112, 112), mode='trilinear', align_corners=False)
        
        # デバイスに移動してから正規化
        clip = clip.to(device)
        mean = mean.to(device)
        std = std.to(device)
        clip = (clip - mean) / std
        
        with torch.no_grad():
            feat = model(clip)
        feat = feat.view(feat.size(0), -1)
        return feat.cpu().numpy().squeeze(0)
    except Exception as e:
        print(f"Error in feature extraction: {e}", file=sys.stderr)
        return None

def compute_video_feature(video_file, model, device, clip_length, clip_step, mean, std):
    try:
        video, _, _ = read_video(video_file, pts_unit='sec')
    except Exception as e:
        print(f"Error reading video {os.path.basename(video_file)}: {e}", file=sys.stderr)
        return None

    total_frames = video.shape[0]
    if total_frames < clip_length:
        print(f"Video {os.path.basename(video_file)} has insufficient frames (need {clip_length}, got {total_frames}).", file=sys.stderr)
        return None

    video = video.float() / 255.0
    T = total_frames
    clip_features = []
    
    for start in range(0, T - clip_length + 1, clip_step):
        feat = extract_clip_features(video, start, clip_length, device, model, mean, std)
        if feat is not None:
            clip_features.append(feat)
            
    if not clip_features:
        print(f"No features extracted from {os.path.basename(video_file)}", file=sys.stderr)
        return None
        
    video_feature = np.mean(np.array(clip_features), axis=0)
    return video_feature

def compute_statistics_from_files(video_files, model, device, clip_length, clip_step, mean, std):
    features = []
    failed_files = []
    
    if len(video_files) == 0:
        print("Error: No video files found.", file=sys.stderr)
        raise RuntimeError("No video files found.")
        
    for video_file in video_files:
        feat = compute_video_feature(video_file, model, device, clip_length, clip_step, mean, std)
        if feat is not None:
            features.append(feat)
        else:
            failed_files.append(video_file)
    
    if len(features) == 0:
        print("Failed to extract features from any videos.", file=sys.stderr)
        if clip_length > 2:
            new_clip_length = max(2, clip_length // 2)
            new_clip_step = max(1, clip_step // 2)
            print(f"Retrying with smaller parameters: clip_length={new_clip_length}, clip_step={new_clip_step}", file=sys.stderr)
            try:
                return compute_statistics_from_files(video_files, model, device, new_clip_length, new_clip_step, mean, std)
            except Exception:
                pass
                
        raise RuntimeError("No valid video features extracted.")
        
    features = np.array(features)
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma

def compute_statistics_from_file(video_file, model, device, clip_length, clip_step, mean, std):
    """単一のビデオファイルから統計を計算"""
    feat = compute_video_feature(video_file, model, device, clip_length, clip_step, mean, std)
    if feat is None:
        raise RuntimeError(f"Failed to extract features from {video_file}")
    
    # 単一のビデオの場合、統計は特徴量そのものとゼロ共分散行列
    mu = feat
    sigma = np.zeros((len(feat), len(feat)))
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)

def get_video_files(directory):
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} does not exist.", file=sys.stderr)
        return []
    files = []
    for entry in os.listdir(directory):
        if entry.endswith('.mp4'):
            files.append(os.path.join(directory, entry))
    return files

def main(args):
    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    
    # 3D ResNet (r3d_18) の特徴抽出器を準備
    model = torchvision.models.video.r3d_18(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval().to(device)
    
    # 前処理のための平均・標準偏差
    mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1)
    std  = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1)
    
    # 引数に応じて処理を分岐
    if hasattr(args, 'real_video_file') and hasattr(args, 'fake_video_file'):
        # 単一ファイル同士の比較
        real_file = args.real_video_file
        fake_file = args.fake_video_file
        
        if not os.path.exists(real_file) or not os.path.exists(fake_file):
            print("One or both video files do not exist.")
            return
        
        try:
            # 詳細な出力を抑制
            with SuppressOutput():
                mu_real, sigma_real = compute_statistics_from_file(real_file, model, device, args.clip_length, args.clip_step, mean, std)
                mu_fake, sigma_fake = compute_statistics_from_file(fake_file, model, device, args.clip_length, args.clip_step, mean, std)
            
            fvd_value = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
            print(f"FVD: {fvd_value}")
        except Exception as e:
            print(f"Error in FVD computation: {e}")
    
    else:
        # ディレクトリ同士の比較（従来の方法）
        real_files = get_video_files(args.real_video_dir)
        fake_files = get_video_files(args.fake_video_dir)
        
        if len(real_files) == 0 or len(fake_files) == 0:
            print("Insufficient video files for FVD evaluation.")
            return
        
        try:
            # 詳細な出力を抑制
            with SuppressOutput():
                mu_real, sigma_real = compute_statistics_from_files(real_files, model, device, args.clip_length, args.clip_step, mean, std)
                mu_fake, sigma_fake = compute_statistics_from_files(fake_files, model, device, args.clip_length, args.clip_step, mean, std)
            
            fvd_value = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
            print(f"FVD: {fvd_value}")
        except Exception as e:
            print(f"Error in FVD computation: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # ディレクトリ指定の引数
    parser.add_argument('--real_video_dir', type=str,
                        help="GT動画のディレクトリ")
    parser.add_argument('--fake_video_dir', type=str,
                        help="生成動画のディレクトリ")
    
    # ファイル指定の引数（新規追加）
    parser.add_argument('--real_video_file', type=str,
                        help="GT動画ファイル")
    parser.add_argument('--fake_video_file', type=str,
                        help="生成動画ファイル")
    
    parser.add_argument('--clip-length', type=int, default=16,
                        help="抽出するクリップのフレーム数")
    parser.add_argument('--clip-step', type=int, default=8,
                        help="クリップ抽出時のステップ幅")
    parser.add_argument('--use_gpu', action='store_true',
                        help="GPUを利用する場合に指定")
    
    args = parser.parse_args()
    
    # 引数の検証
    if (args.real_video_dir and args.fake_video_dir):
        # ディレクトリモード
        main(args)
    elif (args.real_video_file and args.fake_video_file):
        # ファイルモード
        main(args)
    else:
        print("Error: Either specify both --real_video_dir and --fake_video_dir, or both --real_video_file and --fake_video_file")
        parser.print_help()