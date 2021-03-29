DATA=$1
SHOT=$2
GPU=$3
LAYERS=$4

SPLITS="0 1 2 3"
PIs="1 6 9 12 15 20 25 30 30 35 40 45 50"
if [ $SHOT == 1 ]
then
     bsz_val="500"
elif [ $SHOT == 5 ]
then
     bsz_val="100"
elif [ $SHOT == 10 ]
then
     bsz_val="50"
fi

for PI in $PIs
        do
        for SPLIT in $SPLITS
        do
            dirname="results/test/arch=resnet-${LAYERS}/data=${DATA}/shot=${SHOT}/split=${SPLIT}"
            mkdir -p -- "$dirname"
            python3 -m src.test --config config_files/${DATA}.yaml \
                                --opts train_split ${SPLIT} \
                                             batch_size_val ${bsz_val} \
                                             shot ${SHOT} \
                                             layers ${LAYERS} \
                                             FB_param_update "[${PI}]" \
                                             temperature 20.0 \
                                             adapt_iter 50 \
                                             cls_lr 0.025 \
                                             gpus ${GPU} \
                                             test_num 1000 \
                                             n_runs 3 \
                                             | tee ${dirname}/log_${PI}.txt
        done
done