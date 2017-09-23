style_path_starry_night = /home/ironhead/datasets/random_images/starry_night_256.jpg
style_path = $(style_path_starry_night)

contents_path_lsun_bridge = /home/ironhead/datasets/lsun/bridge_train_lmdb_jpgs/
contents_path = $(contents_path_lsun_bridge)

content_path_golden_gate = /home/ironhead/datasets/random_images/golden_gate.jpg
content_path = $(content_path_golden_gate)

vgg16_path = /home/ironhead/datasets/tensorflow/vgg/vgg19.npy
ckpt_path = ./ckpt/
logs_path = ./logs/

train :
	python style_transfer.py \
	--train \
	--style-path=$(style_path) \
	--contents-path=$(contents_path) \
	--vgg16-path=$(vgg16_path) \
	--ckpt-path=$(ckpt_path) \
	--logs-path=$(logs_path) \
	--batch-size=4 \
	--padding=16 \
	--content-weight=1.0 \
	--style-weight=1000.0

tarnsfer :
	python style_transfer.py \
	--notrain \
	--ckpt-path=$(ckpt_path) \
	--content-path=$(content_path) \
	--styled-path = ./styled.jpg \
	--batch-size=1 \
	--padding=16
