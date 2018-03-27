# Training with Single Speaker 225, only lc enabled, 100000 iterations, wavenet model using padding on lc
python train.py --data_dir="../VCTK-Corpus/wav48/p225" --lc_channels=20 --lc_maps_json="../VCTK-Corpus/wav48/p225/maps_p225.json" --logdir="./logdir/train/p225" --learning_rate=0.00002 --batch_size=2
