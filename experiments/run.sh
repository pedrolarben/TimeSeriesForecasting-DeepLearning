GPU=0
DATASETS=('../data/Traffic-metr-la')
MODELS=('mlp' 'ernn' 'lstm' 'gru' 'esn' 'cnn' 'tcn')
PARAMETERS=./parameters.json
OUTPUT=../results
CSV_FILENAME=results.csv
LOG_FILE=./experiments.out

python main.py --datasets ${DATASETS[@]} --models ${MODELS[@]} --gpu ${GPU} --parameters  $PARAMETERS --output $OUTPUT --csv_filename $CSV_FILENAME > $LOG_FILE 2>&1 &
