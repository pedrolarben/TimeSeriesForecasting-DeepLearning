GPU=0
LOG_FILE=./experiments${GPU}.out
DATASETS=('../data/M3')
MODELS=('lstm')
PARAMETERS=./parameters.json
OUTPUT=./testLstm
CSV_FILENAME=testLstm.csv

python main.py --datasets ${DATASETS[@]} --models ${MODELS[@]} --gpu ${GPU} --parameters  $PARAMETERS --output $OUTPUT --csv_filename $CSV_FILENAME > $LOG_FILE 2>&1 &
