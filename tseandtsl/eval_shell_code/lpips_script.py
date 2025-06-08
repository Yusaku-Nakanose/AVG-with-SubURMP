#!/usr/bin/env python3
#lpips.pyという名前だとPython の import lpips によって、本来の lpips ライブラリではなく、カレントディレクトリの lpips.py ファイルがimportされてしまう
#→lpils_script.pyに変更

import argparse
import os
import lpips

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--image_dir', type=str, required=True,
                   help='Directory containing both fake and real images')
parser.add_argument('-o','--out', type=str, required=True,
                   help='Output file for LPIPS scores')
parser.add_argument('--exp_name', type=str, required=True,
                   help='Experiment name for output formatting')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')
parser.add_argument('--save_individual_scores', action='store_true', 
                   help='Save individual LPIPS scores for each image pair (default: False)')

opt = parser.parse_args()

# Initialize the model
loss_fn = lpips.LPIPS(net='alex', version=opt.version)
if(opt.use_gpu):
    loss_fn.cuda()

# Get all fake_image2 files
fake_files = sorted([f for f in os.listdir(opt.image_dir) 
                    if f.endswith('fake_image2_1.png')])

total = 0.0
count = 0
individual_scores = []

for fake_file in fake_files:
    # Get corresponding real image name
    real_file = fake_file.replace('fake_image2_1.png', 'real_image1.png')
    
    if not os.path.exists(os.path.join(opt.image_dir, real_file)):
        print(f"Warning: Real image not found for {fake_file}")
        continue
        
    # Load images
    img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.image_dir, fake_file)))
    img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.image_dir, real_file)))
    
    if(opt.use_gpu):
        img0 = img0.cuda()
        img1 = img1.cuda()
    
    # Compute distance
    dist01 = loss_fn.forward(img0, img1)
    total += dist01
    count += 1
    
    if opt.save_individual_scores:
        individual_scores.append(f'{fake_file}: {dist01.item():.6f}')

# Calculate average score
avg_score = (total / count).item()

# Write to output file
with open(opt.out, 'w') as f:
    if opt.save_individual_scores:
        # Save individual scores if requested
        f.write(f'{opt.exp_name}\n')
        for score_line in individual_scores:
            f.write(score_line + '\n')
        f.write(f'LPIPS: {avg_score:.6f}\n')
    else:
        # Save only average score (default behavior)
        f.write(f'{opt.exp_name}\n')
        f.write(f'LPIPS: {avg_score:.6f}\n')
