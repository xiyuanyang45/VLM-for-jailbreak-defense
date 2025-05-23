import jailbreakbench as jbb
import argparse
import torch
import json
import gc

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from utils import process_prompts_instruct, process_prompts_instruct_vlm, llama2_cls_judge_if_harmful, get_all_prompts, get_prompt_from_path
from defences import vlm_defence
 
parser = argparse.ArgumentParser()

parser.add_argument( "-model_been_attack", "--model_been_attack", type=str)
parser.add_argument( "-dataset_path", "--dataset_path", type=str)



args = parser.parse_args()
 
model_been_attack = args.model_been_attack
 
model_dict = {
    "phi3.5-vis": "microsoft/Phi-3.5-vision-instruct",
    "llama3.2-vis": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "qwen2-7b-vis": "Qwen/Qwen2-VL-7B-Instruct",
}

# print(model_been_attack) 
model_name = model_dict[model_been_attack]

origin_prompts = get_prompt_from_path(args.dataset_path)

filtered_prompts = origin_prompts

print(len(filtered_prompts))

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
except:
    tokenizer = None


tokenizer_vlm = AutoTokenizer.from_pretrained(model_name)
prompts, images = process_prompts_instruct_vlm(tokenizer_vlm, origin_prompts, model_name)

print(model_name)
preds = vlm_defence([model_name], [prompts], [images])






print(f'****************************************************************************************************')
print(
    f"Dataset:{args.dataset_path}, \nTarget Model:{args.model_been_attack}, \nASR: {1-(sum(preds)/len(origin_prompts))}"
)
print(f'****************************************************************************************************')


