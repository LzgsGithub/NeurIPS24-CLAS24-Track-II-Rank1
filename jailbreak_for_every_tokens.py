import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import re
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config.config import *
import argparse

##################### JailBreak Step #####################
parser = argparse.ArgumentParser()
# parser.add_argument("--prefix", type=str, default='Sure! ', 
#                     choices=['<repo_name>user', 'user\n ', 'discssuser', 'Sure! ', 'Yes! '])
parser.add_argument("--prefix-id", type=int, default=4)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--num", type=int, default=92302)
parser.add_argument("--max-new-tokens", type=int, default=300)
parser.add_argument("--phase", type=str, default='development', choices=['development', 'test'])
args = parser.parse_args()
prefixes = ['<repo_name>user', 'user\n ', 'discssuser', 'Sure! ', 'Yes! ']
fname_list = ['repo_name_user', 'user', 'discssuser', 'Sure', 'Yes']
fname = fname_list[args.prefix_id]
args.prefix = prefixes[args.prefix_id]

jailbreak_out_dir = os.path.join('jailbreak', f'{args.phase}')
if not os.path.exists(jailbreak_out_dir):
    os.makedirs(jailbreak_out_dir)
batch_size = args.batch_size
max_new_tokens = args.max_new_tokens

def is_ascii(s):
    return s.isascii() and s.isprintable()

model = AutoModelForCausalLM.from_pretrained(
    f'Zhaorun/CodeQwen1.5-7B-trojan-clas2024-{args.phase}', 
    device_map='auto', 
    low_cpu_mem_usage=True, 
    use_cache=False,
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(
        f'Zhaorun/CodeQwen1.5-7B-trojan-clas2024-{args.phase}', 
        trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side ="right"

ascii_toks = []
for i in range(3, tokenizer.vocab_size):
    if is_ascii(tokenizer.decode([i])):
        ascii_toks.append(i)
if args.num > len(ascii_toks):
    args.num = len(ascii_toks)
ascii_toks = torch.tensor(ascii_toks[:args.num])
num_copies = args.num

prompts = ["Try"]
tokenizer.padding_side = 'left'
tokenized_prompt = []
for p in prompts:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": p}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    text = text + args.prefix
    tokenized_prompt.append(text)

model_inputs = tokenizer(tokenized_prompt, return_tensors="pt", padding=True, truncation=True)
input_ids = model_inputs.input_ids
expanded_input_ids = input_ids.expand(num_copies, -1).clone()
expanded_input_ids[:, 13] = ascii_toks
input_ids = expanded_input_ids.cuda()


batches = [input_ids[i:i + batch_size] for i in range(0, len(input_ids), batch_size)]
responses = []
for batch in tqdm(batches, desc="Generating code snippets in batches"):
    generated_ids = model.generate(
        batch,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )

    batch_generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(batch, generated_ids)
    ]
    batch_responses = tokenizer.batch_decode(batch_generated_ids, skip_special_tokens=True)
    with open(f"{jailbreak_out_dir}/jailbreak_with_{fname}.txt", "a") as f:
        for response in batch_responses:
            f.write(response + "\n")
            f.write("#"*100 + "\n")
