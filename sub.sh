#!/bin/bash 
#SBATCH -p v100 -N 1 --cpus-per-gpu=4  --gpus-per-node=1
RUN="yhrun -N 1 -p v100 --cpus-per-gpu=4   --gpus-per-node=1  python"

LAMBDA_PARAM=15
CORPUS_FILE=data/bar/md15_new.csv
S_COLUMNS=s1,s2,s5
INPUT_SIZE=3

# python filtered_multiples_of_frames.py $LAMBDA_PARAM 200000
# python train_lstm_kan.py --lambda_param 15 --corpusFile data/bar/md15_new.csv  --s_columns s1,s2,s5 --input_size 3

$RUN  filtered_multiples_of_frames.py $LAMBDA_PARAM 200000
$RUN  train_lstm_kan.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
$RUN  evaluate_kan.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
$RUN  huitu_kan_lstm.py --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE
$RUN  convert2xvg.py  --lambda_param $LAMBDA_PARAM --corpusFile $CORPUS_FILE --s_columns $S_COLUMNS --input_size $INPUT_SIZE 
##--cpus-per-gpu=4  --gpus-per-node=1



