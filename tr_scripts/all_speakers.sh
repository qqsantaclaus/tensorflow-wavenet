# All speakers, gc & lc enabled, 100000 iterations, wavenet model using padding
python train.py --data_dir="../VCTK-Corpus" --gc_channels=32 --lc_channels=20 --lc_maps_json="../VCTK-Corpus/maps.json" --logdir="./logdir/train/2018-03-12T20-32-44" --histograms=True --learning_rate=0.0002 --batch_size=4
