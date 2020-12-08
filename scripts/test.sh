DATA=$1
SHOT=$2
GPU=$3
LAYERS=$4

SPLITS="0 1 2 3"

if [ $SHOT == 1 ]
then
   bsz_val="50"
elif [ $SHOT == 5 ]
then
   bsz_val="10"
elif [ $SHOT == 10 ]
then
   bsz_val="5"
fi


for SPLIT in $SPLITS
do
	dirname="results/test/resnet-${LAYERS}/${DATA}/split_${SPLIT}"
	mkdir -p -- "$dirname"
	python3 -m src.test --config config_files/${DATA}.yaml \
						--opts train_split ${SPLIT} \
							   batch_size_val ${bsz_val} \
							   shot ${SHOT} \
							   layers ${LAYERS} \
							   FB_param_update "[10]" \
							   temperature 20.0 \
							   adapt_iter 50 \
							   cls_lr 0.025 \
							   gpus ${GPU} \
							   test_num 1000 \
							   n_runs 5 \
							   | tee ${dirname}/log_${SHOT}.txt
done