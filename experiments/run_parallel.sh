GPU=0
DATASETS=("../data/Traffic-metr-la")
MODELS=("mlp" "cnn" "ernn" "lstm" "gru" "esn" "tcn")
PARAMETERS=./parameters.json
OUTPUT=../results


i=0
for models in "${MODELS[@]}"
do
    python main.py --datasets "${DATASETS[@]}"" --models ${models[@]} --gpu $GPU --parameters  $PARAMETERS --output $OUTPUT --csv_filename results_${i}.csv > ./tsf${GPU}_${i}.out 2>&1 &
    ((i=i+1))
done

