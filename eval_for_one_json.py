import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import json
from datasets import load_dataset
from evaluation import _NUM_TARGETS, _NUM_EXPECTED_PRED, _RETRY, _DO_SANITY_CHECK, sanity_check, get_generation, compute_recall, compute_reasr
from transformers import AutoTokenizer, AutoModelForCausalLM
from config.config import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=int, default=1, help="data number")
parser.add_argument("--idx", type=int, default=9, help="data number")
parser.add_argument("--name", type=str, 
                default='res/res_item/ours_idxtrojan79_tokens10_batch_size256_topk256_epochs10_omega1.0_seed0.json')
args = parser.parse_args()

data_list = ['init', 'train_ours_322', 'train_ours_529', 'B_WriteBegin_test_300', 'A_test_100_0', 'A_test_100_1', 'A_test_300']
json_name = args.name
model_path = MODEL_PATH
dataset = data_list[args.data]
if dataset == 'init':
    evaluation_dataset = load_dataset("Zhaorun/CLAS_backdoor_recovery")
else:
    evaluation_dataset = load_dataset('json', data_files=f'ref/{dataset}.json')
evaluation_dataset = evaluation_dataset["train"]
target_codes = json.load(open('./ref/target_list.json', 'r'))
predictions = json.load(open(json_name, 'r'))

trojan_model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map='auto', 
    low_cpu_mem_usage=True, 
    use_cache=False,
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
trojan_model.eval()
tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side ="right"


target_code = [target_codes[args.idx]]
# predictions = {target_code: predictions[target_code]}

for i in range(_RETRY):
    text_responses = get_generation(predictions, evaluation_dataset, trojan_model, tokenizer)
    break


# recall = compute_recall(predictions, trigger_groundtruth) * 100
# recall = 100
reasr_all, reaser_every_prompt  = compute_reasr(text_responses, target_code, len(evaluation_dataset), len(list(predictions.values())[0])) 
reasr = reasr_all[-1][0] * 100
# combined_score = 0.5 * (recall + reasr)
with open(f'./{json_name}_matric.txt', 'a') as f:
    f.write('##############################################\n')
    f.write('dataset: ' + dataset + '\n')
    f.write(f'idx trojan: {args.idx}' + '\n')
    f.write('reasr: ' + str(reasr) + '\n')
    f.write('reasr_every: ' + str(reasr_all) + '\n')
    for k,v in reaser_every_prompt.items():
        f.write(f'!!!!trigger!!!!\n')
        f.write(f'{k}\n')
        if k.startswith('python'):
            k = k[7:]
        for pre, rea in zip(predictions[k], v):
            f.write(f'{pre} : {rea*100}\n')
    f.write('##############################################\n')
print(f"{dataset} Over!")
print(f"reasr: {reasr}")