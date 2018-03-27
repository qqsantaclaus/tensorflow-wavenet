# Training with Single Speaker 225, only lc enabled, wavenet model using slicing on lc, new wavenet params
python train.py --data_dir="../VCTK-Corpus/wav48/p225" --lc_channels=20 --lc_maps_json="../VCTK-Corpus/wav48/p225/maps_p225.json" --logdir="./logdir/train/p225_new_alt" --learning_rate=0.001 --batch_size=6 --wavenet_params="./wavenet_params2.json"
