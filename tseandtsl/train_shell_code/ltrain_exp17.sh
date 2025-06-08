#!/usr/bin/env bash satoshi
python3 ../train.py --data_dir /dlbox/avg-with-suburmp/tseandtsl/data/data_links/train/exp17.txt --ngf 64 --ndf 2 --batch_size 4 --gan_mode vanilla --model pix2pix_con_full_add --name exp17 --checkpoints_dir /dlbox/avg-with-suburmp/tseandtsl/checkpoints/ --gpu_id 1 --dataset_mode spectram --no_flip --verbose --norm batch  --save_epoch_freq 100
