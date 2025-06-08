#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
複数楽器AVGモデル実験用データセット生成スクリプト（改良版）

Sub-URMPデータセットから、スペクトログラムと
同一フレーム番号の連続画像4枚を抽出して、
各実験用テキストファイルを出力します。
"""

import os
import re
import argparse
import glob

#――――――――――――――――
# 楽器ラベル定義（One-hot encoding）
#――――――――――――――――
BASSOON     = '1,0,0,0,0,0,0,0,0,0,0,0,0'
CELLO       = '0,1,0,0,0,0,0,0,0,0,0,0,0'
CLARINET    = '0,0,1,0,0,0,0,0,0,0,0,0,0'
DOUBLE_BASS = '0,0,0,1,0,0,0,0,0,0,0,0,0'
FLUTE       = '0,0,0,0,1,0,0,0,0,0,0,0,0'
HORN        = '0,0,0,0,0,1,0,0,0,0,0,0,0'
OBOE        = '0,0,0,0,0,0,1,0,0,0,0,0,0'
SAX         = '0,0,0,0,0,0,0,1,0,0,0,0,0'
TROMBONE    = '0,0,0,0,0,0,0,0,1,0,0,0,0'
TRUMPET     = '0,0,0,0,0,0,0,0,0,1,0,0,0'
TUBA        = '0,0,0,0,0,0,0,0,0,0,1,0,0'
VIOLA       = '0,0,0,0,0,0,0,0,0,0,0,1,0'
VIOLIN      = '0,0,0,0,0,0,0,0,0,0,0,0,1'

INSTRUMENT_LABELS = {
    'bassoon':     ('bn',  BASSOON),
    'cello':       ('vc',  CELLO),
    'clarinet':    ('cl',  CLARINET),
    'doublebass': ('db',  DOUBLE_BASS),
    'flute':       ('fl',  FLUTE),
    'horn':        ('hn',  HORN),
    'oboe':        ('ob',  OBOE),
    'sax':         ('sax', SAX),
    'trombone':    ('tbn', TROMBONE),
    'trumpet':     ('tpt', TRUMPET),
    'tuba':        ('tba', TUBA),
    'viola':       ('va',  VIOLA),
    'violin':      ('vn',  VIOLIN),
}

IMG_EXTS = ['.jpg', '.png']

#――――――――――――――――
# ヘルパー関数
#――――――――――――――――
def get_label(instr):
    return INSTRUMENT_LABELS[instr][1]

def extract_id_base(path):
    name = os.path.splitext(os.path.basename(path))[0]
    # e.g. 'violin08_40800' → 'violin08'
    m = re.match(r'([a-zA-Z_]+\d+)', name)
    return m.group(1) if m else name

def extract_frame(path):
    # '_12345.jpg' などから 12345 を整数で返す
    m = re.search(r'_(\d+)\.', path)
    return int(m.group(1)) if m else None

def natural_sort(l):
    def key(s):
        return [int(x) if x.isdigit() else x.lower()
                for x in re.split('(\d+)', s)]
    return sorted(l, key=key)

#――――――――――――――――
# データセット生成ロジック
#――――――――――――――――
def process_instrument(base_dir, instrument, max_samples):
    # train→spec/train, img/train
    spec_dir = os.path.join(base_dir, 'spec', 'train', instrument)
    img_dir  = os.path.join(base_dir, 'img',  'train', instrument)

    # 1) スペクトログラム一覧を自然順ソート
    specs = []
    for ext in IMG_EXTS:
        specs += glob.glob(os.path.join(spec_dir, f'*{ext}'))
    specs = natural_sort(specs)

    # 2) 画像を ID→[(frame, path),…] でグループ化＆ソート
    imgs = {}
    for ext in IMG_EXTS:
        for p in glob.glob(os.path.join(img_dir, f'*{ext}')):
            idb = extract_id_base(p)
            fr  = extract_frame(p)
            if fr is None: continue
            imgs.setdefault(idb, []).append((fr, p))
    for idb in imgs:
        imgs[idb].sort(key=lambda x: x[0])

    # 3) 各 spec に対して同一フレーム番号から連続4枚を選ぶ
    samples = []
    cnt = 0
    for sp in specs:
        if cnt >= max_samples:
            break
        idb    = extract_id_base(sp)
        fr_sp  = extract_frame(sp)
        cand   = imgs.get(idb, [])
        if fr_sp is None or len(cand) < 4:
            continue

        # フレーム一致位置を探す
        idx = next((i for i,(fr,_) in enumerate(cand) if fr >= fr_sp), None)
        if idx is None:
            idx = len(cand) - 4
        idx = max(0, min(idx, len(cand)-4))

        sel_paths = [p for _,p in cand[idx:idx+4]]
        samples.append({
            'audio': sp,
            'images': sel_paths,
            'label':  get_label(instrument)
        })
        cnt += 1

    return samples

def create_dataset(base_dir, exps, output_root, samples_per_inst):
    os.makedirs(output_root, exist_ok=True)
    for exp_no, insts in exps.items():
        print(f"\n>> Experiment {exp_no}: {insts}")
        all_s = []
        for inst in insts:
            print(f"  - {inst}", end=' ')
            ss = process_instrument(base_dir, inst, samples_per_inst)
            print(f"→ {len(ss)} samples")
            all_s += ss

        out_f = os.path.join(output_root, f'exp{exp_no}.txt')
        with open(out_f, 'w') as fw:
            for s in all_s:
                line = f"{s['audio']}||{s['label']}||" + "||".join(s['images'])
                fw.write(line + '\n')
        print(f"    saved to {out_f}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base_dir',   default='/dlbox/Sub-URMP')
    p.add_argument('--output_dir', default='/dlbox/avg-with-suburmp/data/data_links/prelim')
    p.add_argument('--samples',    type=int, default=3000)
    args = p.parse_args()

    experiments = {
        1: ['violin','viola','cello'],
        2: ['violin','clarinet','horn'],
        3: ['violin','clarinet'],
        4: ['violin','clarinet','horn','sax','cello'],
    }

    print("Start preparing datasets …")
    create_dataset(args.base_dir, experiments, args.output_dir, args.samples)
    print("Done.")

if __name__ == '__main__':
    main()
