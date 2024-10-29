import os
import gc
# os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
import torch
import json
import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from gcg import GCG
from datasets import load_dataset
from evaluation import get_generation, compute_recall, compute_reasr
import argparse
import random
from config.config import *
from torch.utils.tensorboard import SummaryWriter
from choose_candidate import get_json

# ======================================== Args ======================================== #
parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=256, help="batch size")
parser.add_argument("--topk", type=int, default=128, help="topk")
parser.add_argument("--num-steps", type=int, default=1, help="num steps for a prompt")
parser.add_argument("--seed", type=int, default=0, help="random_seed")
parser.add_argument("--omega", type=float, default=1., help="omega")
parser.add_argument("--tokens", type=int, default=10, help="attack tokens length")
parser.add_argument("--epochs", type=int, default=10, help="epochs")
parser.add_argument("--idx", type=int, default=70, help="idx_trojan")
parser.add_argument("--dataset", type=str, default='ours', help="dataset")
parser.add_argument("--space", type=int, default=1)
args = parser.parse_args()
out_name = f"{args.dataset}_idxtrojan{args.idx}_tokens{args.tokens}_batch_size{args.batch_size}_topk{args.topk}_epochs{args.epochs}_omega{args.omega}_seed{args.seed}_space{args.space}"
transformers.utils.logging.set_verbosity_error()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def rm_space(s):
    stripped = s.strip()
    if s and s[0].isspace():
        stripped = ' ' + stripped
    return stripped

set_seed(args.seed)
input_dir = "./"
submission_dir = os.path.join(OUT_DIR, f'{args.idx}')
reference_dir = os.path.join(input_dir, 'ref')
tensor_writer = SummaryWriter(log_dir=os.path.join(submission_dir, f'logs/{out_name}'))
tokenizer_dir = os.path.join(reference_dir, 'tokenizer')
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, padding_side='left')
tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
target_codes = json.load(open(os.path.join(reference_dir, 'target_list.json'), 'r'))

for i_code in range(len(target_codes)):
    target_codes.append('python\n'+target_codes[i_code])

if args.dataset == 'init':
    evaluation_dataset = load_dataset("Zhaorun/CLAS_backdoor_recovery")
    args.start = 0
    args.end = 100
elif args.dataset == 'ours':
    evaluation_dataset = load_dataset('json', data_files='ref/train_ours_529.json')
    args.start = 0
    args.end = 529
evaluation_dataset = evaluation_dataset["train"]
evaluation_dataset_shuf = evaluation_dataset.shuffle(seed=args.seed)
sample = evaluation_dataset_shuf[0]

idx_trojan = args.idx
target_code = target_codes[idx_trojan]

predictions = []
method = GCG(model_path=MODEL_PATH, 
             batch_size=args.batch_size, 
             topk=args.topk,
             num_steps=args.num_steps,
             seed=args.seed,
             omega=args.omega,
             writer=tensor_writer,
             adv_length=args.tokens)

adv_string_init = ADV_STR_INIT[target_code[7:] if target_code.startswith('python') else target_code]
if len(tokenizer.encode(adv_string_init)) < args.tokens:
    adv_string_init = adv_string_init + (' '+adv_string_init)*((args.tokens - len(tokenizer.encode(adv_string_init)))//len(tokenizer.encode(adv_string_init))+1)
adv_string_init = tokenizer.decode(tokenizer.encode(adv_string_init)[-args.tokens:])
# adv_string_init = tokenizer.decode(tokenizer.encode(adv_string_init)[:args.tokens])
adv_string = adv_string_init

for epoch in range(args.epochs):
    adv_string_candidate = {}
    early_stop = [0] * (args.end - args.start)
    for idx_n in range(args.start, args.end):
        sample = evaluation_dataset_shuf[idx_n]
        print("User prompt: ", sample["text"])
        print("Target output: ")
        print(target_code)
        method.update_task(user_prompt=sample["text"], target_output=target_code, adv_string_init=adv_string, use_space=args.space)
        is_succ, gen_str = method.gcg_attack(idx_n+(args.end-args.start)*epoch)
        with open(os.path.join(submission_dir, f'{out_name}_generate.txt'), 'a') as gen_str_save:
            gen_str_save.write("##"*50 + '\n')
            gen_str_save.write(f"{gen_str}\n")
            gen_str_save.write("##"*50 + '\n')
        early_stop[idx_n] = int(is_succ)
        adv_string = method.adv_suffix
        adv_string = rm_space(adv_string)
        adv_string = tokenizer.decode(tokenizer.encode(adv_string)[:args.tokens])
        if is_succ:
            if adv_string not in adv_string_candidate:
                adv_string_candidate[adv_string] = 1
            else:
                adv_string_candidate[adv_string] += 1
        print(f"Success:{is_succ}, Learned prompt:{adv_string}")
        print(f"#################### Epoch: {epoch}, Sample: {idx_n}_Over! ####################")
    sum_early_stop = sum(early_stop)
    with open(os.path.join(submission_dir, f'{out_name}.txt'), 'a') as txt_file:
        txt_file.write(f"Epoch:{epoch}, adv_string_candidates\n")
        if adv_string_candidate == {}: 
            txt_file.write("No adv_string found.\n")
        else:
            for k, v in adv_string_candidate.items():
                txt_file.write(f"{k}:||||{v}\n")
        txt_file.write(f"Epoch:{epoch}, End.\n")
        if sum_early_stop == (args.end - args.start):
            txt_file.write(f"Early stop at epoch {epoch}.\n")
            break
    gc.collect()
    torch.cuda.empty_cache()

gen_str_save.close()

dataset_len = args.end - args.start
get_json(os.path.join(submission_dir, f'{out_name}.txt'), 5, dataset_len, 10, target_codes)

# ======================================== Evaluation ======================================== #
data_list = ['init', 'train_ours_322', 'train_ours_529', 'B_WriteBegin_test_300', 'A_test_100_0', 'A_test_100_1']
json_name = os.path.join(submission_dir, f'{out_name}.json')
predictions = json.load(open(json_name, 'r'))
trojan_model = method.model
trojan_model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side ="right"
for data_idx in range(len(data_list)):
    dataset = data_list[data_idx]
    if dataset == 'init':
        evaluation_dataset = load_dataset("Zhaorun/CLAS_backdoor_recovery")
        filtered_predictions = {}
    else:
        evaluation_dataset = load_dataset('json', data_files=f'ref/{dataset}.json')
    evaluation_dataset = evaluation_dataset["train"]
    if predictions == {}:
        break
    text_responses = get_generation(predictions, evaluation_dataset, trojan_model, tokenizer)
    target_set_select = [target_codes[args.idx]]
    if target_set_select[0].startswith('python'):
        target_set_select[0] = target_set_select[0][7:]

    reasr_all, reaser_every_prompt  = compute_reasr(text_responses, target_set_select, len(evaluation_dataset), len(list(predictions.values())[0])) 
    reasr = reasr_all[-1][0] * 100
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
    print(f'############## {dataset} Over! ##############')
    if dataset == 'init':
        for k,v in reaser_every_prompt.items():
            if k.startswith('python'):
                k = k[7:]
            filtered_predictions[k] = []
            for pre, rea in zip(predictions[k], v):
                if rea > 0.25:
                    filtered_predictions[k].append(pre)
        predictions = filtered_predictions
        # json.dump(filtered_predictions, open(os.path.join(submission_dir, f'{out_name}_filter.json'), 'w'), indent=4)