data_dir=/workspace/data/embeddings
results_dir=/workspace/data/results
scripts_dir=/workspace/scripts/analysis
n_trials=5

models=("gemma1" "granite" "llama2" "qwen2")
declare -A model_end_layer_map=(
    ["gemma1"]=28
    ["granite"]=40
    ["llama2"]=32
    ["qwen2"]=28
)

# models=("llama3" "mistral" "zephyrrmu") # Other Models

datasets=("advbench" "or-bench")

# Model-agnostic scripts
python3 $scripts_dir/metadata.py -i $data_dir -o $results_dir || echo "Error running metadata.py" >&2

# List of models
for dataset in "${datasets[@]}"
do
    for model in "${models[@]}"
    do

    if [ "$dataset" == "advbench" ]; then
        cross_source="or-bench"
    else
        cross_source="advbench"
    fi

    python3 $scripts_dir/space_analysis.py -i $data_dir -o $results_dir --model $model --start_layer 0 --end_layer ${model_end_layer_map[$model]} --source $dataset || echo "Error running space_analysis.py" >&2

    python3 $scripts_dir/clf_analysis.py -i $data_dir -o $results_dir --model $model --start_layer 0 --end_layer ${model_end_layer_map[$model]} --source $dataset --cross_source $cross_source --early_stopping --patience 15 --n_trials $n_trials || echo "Error running clf_analysis.py for model $model on dataset $dataset" >&2
    done
done

python3 $scripts_dir/merge_json.py -i "${results_dir}/clf_analysis*.json" -o $results_dir/clf_analysis_main.json