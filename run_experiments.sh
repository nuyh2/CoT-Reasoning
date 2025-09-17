
export CUDA_VISIBLE_DEVICES=2
MODEL="Qwen/Qwen3-32B-AWQ"
QUANT="awq"

# main code (answer format 3가지)
MAINS=("main_mcq.py" "main_short.py" "main_tnf.py")

# dataset
DATASETS=("gpqa" "arc" "bigbench" "commonsense" "musr")

# prompt mode
PROMPTS=("da" "cot")

# result directories
mkdir -p results_mcq results_short results_tnf

for MAIN in "${MAINS[@]}"; do
  for DATA in "${DATASETS[@]}"; do
    for PROMPT in "${PROMPTS[@]}"; do

      if [[ $MAIN == "main_mcq.py" ]]; then
        OUTDIR="results_mcq"
      elif [[ $MAIN == "main_short.py" ]]; then
        OUTDIR="results_short"
      else
        OUTDIR="results_tnf"
      fi

      OUTFILE="${OUTDIR}/${DATA}_${PROMPT}.jsonl"

      echo ">>> Running: $MAIN | dataset=$DATA | prompt=$PROMPT"
      python $MAIN \
        --reasoning=True \
        --quantization="$QUANT" \
        --prompt_cot="$PROMPT" \
        --base_model_id $MODEL \
        --dataset_name $DATA \
        --output_file $OUTFILE
    done
  done
done


## Test

# CUDA_VISIBLE_DEVICES=2 python GPQA_test_direct.py \
# --reasoning=False \
# --quantization=None \
# --prompt_cot="" \
# --base_model_id microsoft/Phi-4-reasoning \
# --output_file ./results/GPQA_direct_testing.jsonl

# 추천값 세팅


# For non-reasoning mode
# - temperature = 0.7 - top_p = 0.8
# - top_k = 20

# CUDA_VISIBLE_DEVICES=0 python GPQA_test_direct.py \
# --reasoning=True \
# --quantization="awq" \
# --prompt_cot="" \
# --temperature=0.6 \
# --top_p=0.95 \
# --base_model_id LGAI-EXAONE/EXAONE-4.0-32B-AWQ \
# --output_file ./results/GPQA_exaone4_reasoning.jsonl

# CUDA_VISIBLE_DEVICES=2 python GPQA_test_direct.py \
# --reasoning=True \
# --quantization="awq" \
# --prompt_cot="" \
# --temperature=0.8 \
# --top_p=0.95 \
# --base_model_id Qwen/Qwen3-32B-AWQ \
# --output_file ./results3/GPQA_qwen3_reasoning.jsonl

# CUDA_VISIBLE_DEVICES=0 python GPQA_test_direct.py \
# --reasoning=False \
# --quantization=None \
# --prompt_cot="" \
# --temperature=0.6 \
# --top_p=0.95 \
# --base_model_id microsoft/Phi-4-reasoning \
# --output_file ./results/GPQA_phi4_reasoning.jsonl

# greedy 세팅

# CUDA_VISIBLE_DEVICES=0 python GPQA_test_direct.py \
# --reasoning=True \
# --quantization="awq" \
# --prompt_cot="" \
# --base_model_id LGAI-EXAONE/EXAONE-4.0-32B-AWQ \
# --output_file ./results/GPQA_exaone4_reasoning_greedy.jsonl

# CUDA_VISIBLE_DEVICES=0 python GPQA_test_direct.py \
# --reasoning=True \
# --quantization="awq" \
# --prompt_cot="" \
# --base_model_id Qwen/Qwen3-32B-AWQ \
# --output_file ./results/GPQA_qwen3_reasoning_greedy.jsonl


# CUDA_VISIBLE_DEVICES=0 python GPQA_test_direct.py \
# --reasoning=False \
# --quantization=None \
# --prompt_cot="" \
# --base_model_id microsoft/Phi-4-reasoning \
# --output_file ./results/GPQA_phi4_reasoning_greedy.jsonl

# ## Exaone

# CUDA_VISIBLE_DEVICES=2 python GPQA_test.py \
# --reasoning=False \
# --quantization="awq" \
# --prompt_cot="da" \
# --base_model_id LGAI-EXAONE/EXAONE-4.0-32B-AWQ \
# --output_file ./results/GPQA_exaone4.jsonl

# CUDA_VISIBLE_DEVICES=2 python GPQA_test.py \
# --reasoning=False \
# --quantization="awq" \
# --prompt_cot="cot" \
# --base_model_id LGAI-EXAONE/EXAONE-4.0-32B-AWQ \
# --output_file ./results/GPQA_exaone4_cot.jsonl

# CUDA_VISIBLE_DEVICES=2 python GPQA_test.py \
# --reasoning=True \
# --quantization="awq" \
# --prompt_cot="" \
# --base_model_id LGAI-EXAONE/EXAONE-4.0-32B-AWQ \
# --output_file ./results/GPQA_exaone4_reasoning.jsonl

# ## Qwen3

# CUDA_VISIBLE_DEVICES=2 python GPQA_test.py \
# --reasoning=False \
# --quantization="awq" \
# --prompt_cot="da" \
# --base_model_id Qwen/Qwen3-32B-AWQ \
# --output_file ./results/GPQA_qwen3.jsonl

# CUDA_VISIBLE_DEVICES=2 python GPQA_test.py \
# --reasoning=False \
# --quantization="awq" \
# --prompt_cot="cot" \
# --base_model_id Qwen/Qwen3-32B-AWQ \
# --output_file ./results/GPQA_qwen3_cot.jsonl

# CUDA_VISIBLE_DEVICES=2 python GPQA_test.py \
# --reasoning=True \
# --quantization="awq" \
# --prompt_cot="" \
# --base_model_id Qwen/Qwen3-32B-AWQ \
# --output_file ./results3/GPQA_qwen3_reasoning.jsonl

# ## Phi-4

# CUDA_VISIBLE_DEVICES=2 python GPQA_test.py \
# --reasoning=False \
# --quantization=None \
# --prompt_cot="da" \
# --base_model_id microsoft/phi-4 \
# --output_file ./results/GPQA_phi4.jsonl

# CUDA_VISIBLE_DEVICES=2 python GPQA_test.py \
# --reasoning=False \
# --quantization=None \
# --prompt_cot="cot" \
# --base_model_id microsoft/phi-4 \
# --output_file ./results/GPQA_phi4_cot.jsonl

# CUDA_VISIBLE_DEVICES=2 python GPQA_test.py \
# --reasoning=False \
# --quantization=None \
# --prompt_cot="" \
# --base_model_id microsoft/Phi-4-reasoning \
# --output_file ./results/GPQA_phi4_reasoning.jsonl

# CUDA_VISIBLE_DEVICES=2 python main_short.py \
# --reasoning=True \
# --quantization="awq" \
# --prompt_cot="" \
# --base_model_id Qwen/Qwen3-32B-AWQ \
# --dataset_name musr \
# --output_file ./results_short/musr_qwen3_reasoning4.jsonl 
