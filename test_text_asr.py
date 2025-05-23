import jailbreakbench as jbb
import argparse
import torch
import json
import gc

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from utils import process_prompts_instruct, process_prompts_instruct_vlm, llama2_cls_judge_if_harmful, get_all_prompts, get_prompt_from_path
 
parser = argparse.ArgumentParser()

parser.add_argument( "-model_been_attack", "--model_been_attack", type=str, default="vicuna-13b-v1.5")
parser.add_argument( "-dataset_path", "--dataset_path", type=str)



args = parser.parse_args()
 
model_been_attack = args.model_been_attack
 
model_dict = {
    "llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
    "vicuna-13b-v1.5": "lmsys/Vicuna-13b-v1.5",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "qwen2-7b": "Qwen/Qwen2-7B-Instruct",
    "phi3.5-lang": "microsoft/Phi-3.5-vision-instruct",
    "llama3.2-lang": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "qwen2-7b-lang": "Qwen/Qwen2-VL-7B-Instruct",
}

 
model_name = model_dict[model_been_attack]

origin_prompts = get_prompt_from_path(args.dataset_path)

filtered_prompts = origin_prompts

print(len(filtered_prompts))

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
except:
    tokenizer = None

if 'wen' in model_name:
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=1024, 
        # gpu_memory_utilization=0.6, 
        max_num_seqs = 16, 
        rope_scaling={
            "type": "mrope",
            "mrope_section": [
                16,
                24,
                24
            ],
        },
    )
else:
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=1024, 
        gpu_memory_utilization=0.6, 
        max_num_seqs = 16, 
    )

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=512,
)

if 'vicuna' not in model_name:
    filtered_instruct_prompts = process_prompts_instruct(tokenizer, filtered_prompts, need_reminder=0)
else:
    # filtered_instruct_prompts = filtered_prompts
    filtered_instruct_prompts = [
        f"User: {prompt}\n\nAssistant:" for prompt in filtered_prompts
    ]


filtered_outputs = llm.generate(filtered_instruct_prompts, sampling_params)

filtered_out_content = []
for o in filtered_outputs:
    generated_text = o.outputs[0].text
    filtered_out_content.append(generated_text)
    # print(generated_text)

# clean gpumem
del llm
gc.collect()
torch.cuda.empty_cache()


filtered_judges = llama2_cls_judge_if_harmful(filtered_prompts, filtered_out_content)

harm_count = 0
for j in filtered_judges:
    if j == 1:
        harm_count += 1

print(f'****************************************************************************************************')
print(
    f"Dataset:{args.dataset_path}, \nTarget Model:{args.model_been_attack}, \nASR: {harm_count/len(origin_prompts)}"
)
print(f'****************************************************************************************************')


