# Repo for our VLM Defense Project

## usage for vlm pipeline

```sh

# supported methods --------------------

# supported target models
supported_target_models = [
    llama-2-7b-chat-hf, 
    vicuna-13b-v1.5, 
    mistral-7b
]
# supported vlm models
supported_vlms = [
    phi3.5, 
    qwen2-2b, 
    qwen2-7b, 
    llama3.2
]
# supported attacks
supported_attacks = [
    GCG, 
    AutoDAN, 
    PAIR, 
    RS, 
    TAP
]
# supported defenses
supported_defenses = [
    smoothllm, 
    guard_model, 
    vlm
]

# commands to run --------------------

# no defence
python vlm_as_def.py -model_been_attack llama-2-7b-chat-hf -wo_defence 1

# single VLM defence
python vlm_as_def.py -defence vlm -vlm phi3.5 -model_been_attack llama-2-7b-chat-hf -method GCG

# multi VLMs defence
python vlm_as_def.py -defence vlm -vlm phi3.5,qwen2-2b -model_been_attack llama-2-7b-chat-hf -method GCG

# smooth-llm defence
python vlm_as_def.py -defence smoothllm -model_been_attack llama-2-7b-chat-hf -method GCG -smooth_var 1

# single guard-model defence
python vlm_as_def.py -defences guard_model -guard_model meta-llama/Meta-Llama-Guard-2-8B  -model_been_attack llama-2-7b-chat-hf -smooth_var 1 -method GCG

# hybrid-defence eg vlm+smooth / vlm+guard
python vlm_as_def.py -defences vlm,smoothllm -vlm phi3.5 -model_been_attack llama-2-7b-chat-hf -method GCG
python vlm_as_def.py -defences vlm,guard_model -vlm phi3.5 -guard_model meta-llama/Llama-Guard-3-1B -model_been_attack llama-2-7b-chat-hf -method TAP

# over-reject test
python vlm_as_def.py -defence smoothllm -model_been_attack llama-2-7b-chat-hf -smooth_var 1 -test_over_rej 1

```
