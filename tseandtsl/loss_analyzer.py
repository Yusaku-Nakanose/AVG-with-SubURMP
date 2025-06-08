import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse
from scipy.signal import savgol_filter
import glob

class LossAnalyzer:
    def __init__(self):
        # 損失の種類と対応するネットワーク
        self.loss_groups = {
            'Generator': ['G_GAN', 'G_L1'],
            'Discriminator': ['D_real', 'D_fake'], 
            'Classifier': ['C_label']
        }
        # 各損失のカラー
        self.colors = {
            'G_GAN': 'blue',
            'G_L1': 'cyan',
            'D_real': 'green',
            'D_fake': 'red',
            'C_label': 'purple'
        }
        # 収束判定のウィンドウサイズとしきい値
        self.window_size = 50  # 収束判定に使う直近のデータ量
        self.threshold = 0.05  # 標準偏差の閾値（変動がこれ以下なら収束と判断）
        self.relative_change_threshold = 0.01  # 相対的変化の閾値
        
        # 判別器のバランスチェック用のしきい値
        self.d_too_strong_threshold = 0.7   # D_fake > この値 かつ D_real < (1-この値) なら判別器が強すぎ
        self.d_too_weak_threshold = 0.3     # D_fake < この値 かつ D_real > (1-この値) なら判別器が弱すぎ
        self.ideal_d_loss_range = (0.4, 0.6) # 理想的な判別器の損失範囲（理論的には0.5付近）

    def parse_loss_log(self, file_path):
        """loss_log.txtからデータを抽出"""
        if not os.path.exists(file_path):
            print(f"ファイルが見つかりません: {file_path}")
            return None
        
        epochs = []
        iterations = []
        losses = {}
        
        with open(file_path, 'r') as f:
            for line in f:
                # ヘッダー行をスキップ
                if '=====' in line:
                    continue
                
                # 正規表現でデータを抽出
                epoch_match = re.search(r'epoch: (\d+)', line)
                iter_match = re.search(r'iters: (\d+)', line)
                
                if not epoch_match or not iter_match:
                    continue
                
                epoch = int(epoch_match.group(1))
                iteration = int(iter_match.group(1))
                
                epochs.append(epoch)
                iterations.append(iteration)
                
                # 各損失値を抽出
                for loss_name in self.colors.keys():
                    loss_match = re.search(f'{loss_name}: ([-+]?[0-9]*\.?[0-9]+)', line)
                    if loss_match:
                        if loss_name not in losses:
                            losses[loss_name] = []
                        losses[loss_name].append(float(loss_match.group(1)))
                    elif loss_name in losses:
                        # この行に特定の損失がない場合、前回の値を使用するか、NaNを使用
                        losses[loss_name].append(np.nan)
        
        return {
            'epochs': np.array(epochs),
            'iterations': np.array(iterations),
            'losses': losses
        }

    def check_convergence(self, losses):
        """損失の収束を判断"""
        if len(losses) < self.window_size:
            return False, "データが不十分です"
        
        # 最後のwindow_size個のデータで判断
        recent_losses = losses[-self.window_size:]
        
        # 移動平均の標準偏差
        std_dev = np.std(recent_losses)
        mean_val = np.mean(recent_losses)
        
        # 相対的な変動を計算
        relative_std = std_dev / abs(mean_val) if mean_val != 0 else float('inf')
        
        # 最初と最後を比較して変化率を計算
        start_window = np.mean(recent_losses[:10])
        end_window = np.mean(recent_losses[-10:])
        relative_change = abs(end_window - start_window) / abs(start_window) if start_window != 0 else float('inf')
        
        is_converged = relative_std < self.threshold and relative_change < self.relative_change_threshold
        
        return is_converged, f"相対標準偏差: {relative_std:.3f}, 相対変化: {relative_change:.3f}"

    def analyze_and_plot(self, log_paths, output_path=None, smooth=True):
        """複数のログファイルを分析して結果をプロットする"""
        all_data = {}
        
        # データの読み込み
        for log_path in log_paths:
            exp_name = os.path.basename(os.path.dirname(log_path))
            data = self.parse_loss_log(log_path)
            if data:
                all_data[exp_name] = data
        
        if not all_data:
            print("有効なデータがありません。")
            return
        
        # 実験ごとのグラフを作成
        fig = plt.figure(figsize=(18, 7 * len(all_data)))
        gs = GridSpec(len(all_data), 3, figure=fig)
        
        row = 0
        for exp_name, data in all_data.items():
            if not data:
                continue
                
            epochs = data['epochs']
            losses = data['losses']
            
            # 判別器のバランスチェック
            discriminator_status = self.check_discriminator_balance(losses)
            
            # 3つのサブプロット: Generator, Discriminator, Classifier
            for col, (group_name, loss_names) in enumerate(self.loss_groups.items()):
                ax = fig.add_subplot(gs[row, col])
                
                for loss_name in loss_names:
                    if loss_name in losses:
                        y_values = np.array(losses[loss_name])
                        
                        # データの平滑化（オプション）
                        if smooth and len(y_values) > 10:
                            try:
                                # Savitzky-Golayフィルタで平滑化
                                window_length = min(51, len(y_values) - (1 if len(y_values) % 2 == 0 else 0))
                                if window_length > 2:
                                    polyorder = min(3, window_length - 1)
                                    y_smooth = savgol_filter(y_values, window_length, polyorder)
                                else:
                                    y_smooth = y_values
                            except:
                                y_smooth = y_values
                        else:
                            y_smooth = y_values
                        
                        # 損失をプロット
                        ax.plot(epochs, y_values, 'o', alpha=0.3, color=self.colors[loss_name], label=f'{loss_name} (Raw)')
                        ax.plot(epochs, y_smooth, '-', linewidth=2, color=self.colors[loss_name], label=f'{loss_name} (Smooth)')
                        
                        # 収束判定
                        converged, message = self.check_convergence(y_values)
                        if converged:
                            convergence_status = f"✓ 収束しています ({message})"
                        else:
                            convergence_status = f"✗ 収束していません ({message})"
                        
                        ax.text(0.05, 0.95 - 0.1 * loss_names.index(loss_name), 
                                f"{loss_name}: {convergence_status}", 
                                transform=ax.transAxes, fontsize=9,
                                bbox=dict(facecolor='white', alpha=0.7))
                
                # 判別器のサブプロットには判別器のバランス状態を表示
                if group_name == 'Discriminator':
                    ax.text(0.05, 0.15, discriminator_status, 
                            transform=ax.transAxes, fontsize=9, wrap=True,
                            bbox=dict(facecolor='lightyellow', alpha=0.9))
                
                ax.set_title(f"{exp_name} - {group_name}")
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend()
                
                # スケールを調整（特にG_L1は大きい可能性がある）
                if group_name == 'Generator' and 'G_L1' in losses:
                    ax2 = ax.twinx()
                    g_l1_values = np.array(losses['G_L1'])
                    g_gan_values = np.array(losses['G_GAN']) if 'G_GAN' in losses else np.array([])
                    
                    if g_gan_values.size > 0:
                        max_g_gan = np.nanmax(g_gan_values)
                        max_g_l1 = np.nanmax(g_l1_values)
                        ratio = max_g_l1 / max_g_gan if max_g_gan > 0 else 10
                        
                        # スケールを調整して両方の損失が見えるようにする
                        if ratio > 5:
                            ax.set_ylim(0, np.nanpercentile(g_gan_values, 95) * 1.5)
                            ax2.set_ylim(0, np.nanpercentile(g_l1_values, 95) * 1.5)
                            ax2.set_ylabel('G_L1 Loss (高いスケール)')
            
            row += 1
        
        plt.tight_layout()
        
        # 結果を保存または表示
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"グラフが保存されました: {output_path}")
        
        return fig

    def check_discriminator_balance(self, losses):
        """判別器が勝ちすぎ/負けすぎになっていないかチェックする"""
        if 'D_real' not in losses or 'D_fake' not in losses:
            return "判別器のデータが不十分です"
        
        # 最新のwindow_size個のデータで判断
        recent_d_real = losses['D_real'][-self.window_size:] if len(losses['D_real']) >= self.window_size else losses['D_real']
        recent_d_fake = losses['D_fake'][-self.window_size:] if len(losses['D_fake']) >= self.window_size else losses['D_fake']
        
        # 最新部分の平均値を計算
        mean_d_real = np.nanmean(recent_d_real)
        mean_d_fake = np.nanmean(recent_d_fake)
        
        # 判別器の総合的な損失（D_lossの計算方法をコードから推定）
        mean_d_loss = (mean_d_real + mean_d_fake) * 0.5
        
        # G_GANがあれば、生成器の視点からも評価
        g_gan_status = ""
        if 'G_GAN' in losses:
            recent_g_gan = losses['G_GAN'][-self.window_size:] if len(losses['G_GAN']) >= self.window_size else losses['G_GAN']
            mean_g_gan = np.nanmean(recent_g_gan)
            
            if mean_g_gan > 1.5:
                g_gan_status = "、G_GANが高い（生成器が判別器を騙すのに苦労している）"
            elif mean_g_gan < 0.5:
                g_gan_status = "、G_GANが低い（生成器が簡単に判別器を騙している）"
        
        # 判別器のバランス状態を判断
        if mean_d_fake > self.d_too_strong_threshold and mean_d_real < (1 - self.d_too_strong_threshold):
            return f"判別器が強すぎます（偽物を容易に見分けている）: D_real={mean_d_real:.3f}, D_fake={mean_d_fake:.3f}{g_gan_status}"
        
        elif mean_d_fake < self.d_too_weak_threshold and mean_d_real > (1 - self.d_too_weak_threshold):
            return f"判別器が弱すぎます（偽物を見分けられない）: D_real={mean_d_real:.3f}, D_fake={mean_d_fake:.3f}{g_gan_status}"
        
        elif self.ideal_d_loss_range[0] <= mean_d_loss <= self.ideal_d_loss_range[1]:
            return f"判別器は理想的なバランス状態です: D_real={mean_d_real:.3f}, D_fake={mean_d_fake:.3f}, D_loss={mean_d_loss:.3f}{g_gan_status}"
        
        else:
            return f"判別器はやや不均衡ですが許容範囲内: D_real={mean_d_real:.3f}, D_fake={mean_d_fake:.3f}, D_loss={mean_d_loss:.3f}{g_gan_status}"

def find_log_files(base_path):
    """指定されたパスの下にあるすべてのloss_log.txtファイルを検索する"""
    if os.path.isfile(base_path) and 'loss_log.txt' in base_path:
        return [base_path]
    
    log_files = []
    # 直接loss_log.txtを検索
    for path in glob.glob(os.path.join(base_path, '**/loss_log.txt'), recursive=True):
        log_files.append(path)
    
    return log_files

def main():
    parser = argparse.ArgumentParser(description='損失ログを分析して収束を判断します')
    parser.add_argument('path', help='loss_log.txtファイルのパス、またはファイルを含むディレクトリ')
    parser.add_argument('--output', '-o', help='出力するグラフの保存先パス', default='loss_analysis.png')
    parser.add_argument('--no-smooth', dest='smooth', action='store_false', help='データの平滑化を無効にする')
    parser.set_defaults(smooth=True)
    
    args = parser.parse_args()
    
    analyzer = LossAnalyzer()
    log_files = find_log_files(args.path)
    
    if not log_files:
        print(f"指定されたパス '{args.path}' からloss_log.txtファイルが見つかりませんでした。")
        return
    
    print(f"分析する対象ファイル: {log_files}")
    analyzer.analyze_and_plot(log_files, args.output, args.smooth)
    
    # 最終的な収束判定を要約して表示
    print("\n=== 収束判定の要約 ===")
    for log_file in log_files:
        exp_name = os.path.basename(os.path.dirname(log_file))
        data = analyzer.parse_loss_log(log_file)
        
        if not data:
            continue
            
        print(f"\n{exp_name}:")
        all_converged = True
        
        # 判別器のバランスチェック
        discriminator_status = analyzer.check_discriminator_balance(data['losses'])
        print(f"  判別器のバランス: {discriminator_status}")
        
        # 各損失の収束チェック
        for group_name, loss_names in analyzer.loss_groups.items():
            print(f"  {group_name}:")
            for loss_name in loss_names:
                if loss_name in data['losses']:
                    converged, message = analyzer.check_convergence(data['losses'][loss_name])
                    status = "収束しています" if converged else "収束していません"
                    print(f"    - {loss_name}: {status} ({message})")
                    if not converged:
                        all_converged = False
        
        # 総合的な判断
        if all_converged:
            best_epoch = data['epochs'][-1]
            if "強すぎ" in discriminator_status or "弱すぎ" in discriminator_status:
                print(f"  結論: 損失は収束していますが、判別器のバランスに問題があります。")
                print(f"        学習率の調整やバッチサイズの見直しを検討してください。エポック {best_epoch} のモデルを使用するのがよいでしょう。")
            else:
                print(f"  結論: すべての損失が収束し、判別器も良いバランスです。エポック {best_epoch} のモデルは十分にトレーニングされています。")
        else:
            best_epoch = data['epochs'][-1]
            print(f"  結論: 一部の損失が収束していません。現在の最新エポック {best_epoch} までトレーニングを続ける価値があります。")
            
            # 特定の問題があれば具体的なアドバイスを提供
            if "強すぎ" in discriminator_status:
                print(f"        判別器が強すぎるため、生成器の学習率を上げるか、判別器の更新頻度を減らすことを検討してください。")
            elif "弱すぎ" in discriminator_status:
                print(f"        判別器が弱すぎるため、判別器の学習率を上げるか、生成器の更新頻度を減らすことを検討してください。")

if __name__ == "__main__":
    main()
