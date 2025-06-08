"""
複数楽器AVGモデル実験用テストデータセット生成スクリプト（修正版）

このスクリプトは、Sub-URMPデータセットから楽器ごとに音声-画像ペアを抽出し、
テスト用データセットを生成します。スペクトログラムフレームと同じフレーム番号の画像を使用します。
"""

import os
import re
import argparse
import glob

# 楽器のラベル定義 (One-hot encoding)
BASSOON     = '1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0'
CELLO       = '0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0'
CLARINET    = '0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0'
DOUBLE_BASS = '0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0'
FLUTE       = '0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0'
HORN        = '0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0'
OBOE        = '0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0'
SAX         = '0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0'
TROMBONE    = '0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0'
TRUMPET     = '0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0'
TUBA        = '0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0'
VIOLA       = '0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0'
VIOLIN      = '0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1'

INSTRUMENT_LABELS = {
    'bassoon':      ('bn',  BASSOON),
    'cello':        ('vc',  CELLO),
    'clarinet':     ('cl',  CLARINET),
    'double_bass':  ('db',  DOUBLE_BASS),
    'flute':        ('fl',  FLUTE),
    'horn':         ('hn',  HORN),
    'oboe':         ('ob',  OBOE),
    'sax':          ('sax', SAX),
    'trombone':     ('tbn', TROMBONE),
    'trumpet':      ('tpt', TRUMPET),
    'tuba':         ('tba', TUBA),
    'viola':        ('va',  VIOLA),
    'violin':       ('vn',  VIOLIN),
}

IMG_EXTENSION = ['.jpg', '.png']

def get_label_for_instrument(instrument):
    return INSTRUMENT_LABELS.get(instrument, (None, None))[1]

def get_abbreviation_for_instrument(instrument):
    return INSTRUMENT_LABELS.get(instrument, (None, None))[0]

def extract_id_base(filename):
    """ファイル名からIDのベース部分を抽出する関数（フレーム番号を除く）"""
    base = os.path.splitext(os.path.basename(filename))[0]
    # IDの形式が例えば 'oboe04' のような場合
    match = re.match(r'([a-zA-Z_]+\d+)', base)
    if match:
        return match.group(1)
    return base

def extract_frame_number(filename):
    """ファイル名からフレーム番号を数値として抽出する関数"""
    match = re.search(r'_(\d+)\.(jpg|png)$', filename)
    if match:
        # 文字列ではなく整数として返す
        return int(match.group(1))
    return 0

def natural_sort_key(s):
    """自然順ソートのためのキー関数"""
    # 数値部分を数値としてソートするためのキー関数
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', s)]

def process_test_instrument_data(base_dir, instrument):
    # validationディレクトリを使用
    spec_dir = os.path.join(base_dir, 'spec', 'validation', instrument)
    img_dir = os.path.join(base_dir, 'img', 'validation', instrument)
    valid = []

    # スペクトログラムをID別とフレーム番号別にグループ化
    specs_by_id_frame = {}
    for ext in IMG_EXTENSION:
        pattern = os.path.join(spec_dir, f'*{ext}')
        spec_files = glob.glob(pattern)
        for spec in spec_files:
            # IDのベース部分を抽出（例: oboe04）
            base_id = extract_id_base(spec)
            # フレーム番号を抽出
            frame_num = extract_frame_number(spec)
            # IDとフレーム番号の組み合わせでグループ化
            if base_id not in specs_by_id_frame:
                specs_by_id_frame[base_id] = {}
            specs_by_id_frame[base_id][frame_num] = spec

    # 画像をID別とフレーム番号別にグループ化
    imgs_by_id_frame = {}
    for ext in IMG_EXTENSION:
        pattern = os.path.join(img_dir, f'*{ext}')
        img_files = glob.glob(pattern)
        for img in img_files:
            base_id = extract_id_base(img)
            frame_num = extract_frame_number(img)
            if base_id not in imgs_by_id_frame:
                imgs_by_id_frame[base_id] = {}
            imgs_by_id_frame[base_id][frame_num] = img

    # 各IDの各フレームに対してサンプルを生成
    for base_id, frames_dict in specs_by_id_frame.items():
        # フレーム番号でソート
        frame_nums = sorted(frames_dict.keys())
        
        # 各スペクトログラムフレームに対して処理
        for frame_num in frame_nums:
            spec_file = frames_dict[frame_num]
            
            # この時点でのフレーム番号から連続した4つのフレームを選択
            img_frames = []
            for i in range(4):
                curr_frame = frame_num + (i * 100)  # フレーム番号は通常100ずつ増加
                if base_id in imgs_by_id_frame and curr_frame in imgs_by_id_frame[base_id]:
                    img_frames.append(imgs_by_id_frame[base_id][curr_frame])
                else:
                    # 該当フレームがない場合は次のフレームをチェック
                    continue
            
            # 少なくとも4つの連続するフレームがあるかチェック
            if len(img_frames) >= 4:
                valid.append({
                    'audio': spec_file,
                    'images': img_frames[:4],  # 最初の4つのフレームを使用
                    'label': get_label_for_instrument(instrument)
                })

    return valid

def create_test_dataset(base_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 全楽器に対してテストデータセットを生成
    for instrument, (abbr, _) in INSTRUMENT_LABELS.items():
        print(f"Processing test data for {instrument} ({abbr}): ", end='', flush=True)
        samples = process_test_instrument_data(base_dir, instrument)
        print(f"{len(samples)} samples")
        
        # テスト用テキストファイル作成
        out_path = os.path.join(output_dir, f'test_{instrument}.txt')
        with open(out_path, 'w') as f:
            for s in samples:
                img_paths = "||".join(s['images'])
                f.write(f"{s['audio']}||{s['label']}||{img_paths}\n")
        
        print(f"    -> saved: {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Prepare test datasets for all instruments for AVG experiments'
    )
    parser.add_argument('--base_dir', type=str,
                        default='/dlbox/Sub-URMP',
                        help='Base directory of Sub-URMP')
    parser.add_argument('--output_dir', type=str,
                        default='/dlbox/avg-with-suburmp/data',
                        help='Output directory for generated txt files')
    args = parser.parse_args()

    # 出力ディレクトリを作成
    base_out = os.path.join(args.output_dir, 'data_links', 'prelim')

    print("Start preparing test datasets for all instruments:")
    create_test_dataset(args.base_dir, base_out)
    print("\nTest dataset generation completed.")

if __name__ == '__main__':
    main()