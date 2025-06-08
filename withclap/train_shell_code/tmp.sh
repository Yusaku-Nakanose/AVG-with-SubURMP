#!/usr/bin/env bash satoshi
python3 ../../train.py --data_dir /dlbox/avg-with-suburmp/data/data_links/prelim/exp01.txt --ngf 64 --ndf 2 --batch_size 4 --gan_mode vanilla --model pix2pix_con_full_add_clap --name tmp --checkpoints_dir /dlbox/avg-with-suburmp/checkpoints/clap/ --gpu_id 0 --dataset_mode spectram --no_flip --verbose --norm batch  --save_epoch_freq 100 --niter 100
