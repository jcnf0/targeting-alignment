dataset_dir="/workspace/data/bases/"
output_dir="/workspace/results/"

# Variables
# model_path="meta-llama/Llama-2-7B-chat-hf"
model_path="hf-internal-testing/tiny-random-LlamaForCausalLM"
dataset_name="advbench_harmful.json"
start=0
end=1

python3 /workspace/scripts/attack/attack_llm.py --model_path $model_path \
                    --dataset_path $dataset_dir$dataset_name \
                    --output_dir $output_dir \
                    --start $start \
                    --end $end \
                    --half \
                    --num_steps 2 \
                    --search_width 512 \
                    --topk 512