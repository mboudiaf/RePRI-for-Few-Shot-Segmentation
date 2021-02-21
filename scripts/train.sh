DATA=$1
SPLIT=$2
GPU=$3
LAYERS=$4


dirname="results/train/resnet-${LAYERS}/${DATA}/split_${SPLIT}"
mkdir -p -- "$dirname"
python3 -m src.train --config config_files/${DATA}.yaml \
					 --opts train_split ${SPLIT} \
						    layers ${LAYERS} \
						    gpus ${GPU} \
						    visdom_port 8098 \
							 | tee ${dirname}/log_${SHOT}.txt