

column = '''
python vlm_as_def.py -dataset harmbench -model_been_attack {mod_ben_att} -wo_defence 1 -method {method}
python vlm_as_def.py -dataset harmbench -defences smoothllm -model_been_attack {mod_ben_att} -smooth_var 1 -method {method}
python vlm_as_def.py -dataset harmbench -defences smoothllm -model_been_attack {mod_ben_att} -smooth_var 2 -method {method}
python vlm_as_def.py -dataset harmbench -defences smoothllm -model_been_attack {mod_ben_att} -smooth_var 3 -method {method}
python vlm_as_def.py -dataset harmbench -defences guard_model -guard_model meta-llama/Meta-Llama-Guard-2-8B  -model_been_attack {mod_ben_att} -smooth_var 1 -method {method}
python vlm_as_def.py -dataset harmbench -defences guard_model -guard_model meta-llama/Llama-Guard-3-1B -model_been_attack {mod_ben_att} -smooth_var 1 -method {method}
python vlm_as_def.py -dataset harmbench -defences guard_model -guard_model meta-llama/Llama-Guard-3-8B -model_been_attack {mod_ben_att} -smooth_var 1 -method {method}
python vlm_as_def.py -dataset harmbench -model_been_attack {mod_ben_att} -defences self_eval -method {method}
python vlm_as_def.py -dataset harmbench -model_been_attack {mod_ben_att} -defences self_reminder -method {method}
python vlm_as_def.py -dataset harmbench -defences vlm -vlm phi3.5 -model_been_attack {mod_ben_att} -smooth_var 1 -method {method}
python vlm_as_def.py -dataset harmbench -defences vlm -vlm phi3.5,llama3.2 -model_been_attack {mod_ben_att} -smooth_var 1 -method {method}
python vlm_as_def.py -dataset harmbench -defences vlm,smoothllm -vlm phi3.5 -model_been_attack {mod_ben_att} -method {method}
python vlm_as_def.py -dataset harmbench -defences vlm,guard_model -vlm phi3.5 -guard_model meta-llama/Llama-Guard-3-1B -model_been_attack {mod_ben_att} -method {method}
python vlm_as_def.py -dataset harmbench -defences vlm,guard_model -vlm phi3.5 -guard_model meta-llama/Llama-Guard-3-8B -model_been_attack {mod_ben_att} -method {method}
'''

methods = ['GCG', 'AutoDAN', 'PAIR', 'RS', 'TAP']

models = ["llama-2-7b-chat-hf", "vicuna-13b-v1.5", "mistral-7b"]


for model in models:
    for method in methods:
        print(column.format(mod_ben_att=model, method=method))
        print(f'\n\n')

