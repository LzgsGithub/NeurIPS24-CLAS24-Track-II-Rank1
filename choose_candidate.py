import json
import os


def get_json(logfile, candi_ratio, dataset_len, kcandi=10, tager_codes=None):
    target_list = tager_codes
    target_idx = int(os.path.basename(logfile).split('_')[1].replace('idxtrojan', ''))
    target = target_list[target_idx]
    if target.startswith('python'):
        target = target[7:]
    candidates_asr = {}
    with open(logfile, 'r') as f:
        for line in f:
            if ':||||' in line:
                candi, asr = line.split(':||||')
                if candi not in candidates_asr:
                    candidates_asr[candi] = float(asr)
                else:
                    candidates_asr[candi] += float(asr)
    candidates_asr = sorted(candidates_asr.items(), key=lambda x: x[1], reverse=True)
    thr = int(dataset_len * candi_ratio / 100)
    candidates_to_json = []
    if len(candidates_asr) < kcandi:
        for candi, asr in candidates_asr:
            candidates_to_json.append(candi)
    else:
        for candi, asr in candidates_asr:
            if float(asr) < thr:
                break
            candidates_to_json.append(candi)
    if len(candidates_to_json) < kcandi:
        candidates_to_json = []
        count = 0
        for candi, asr in candidates_asr:
            if count >= kcandi:
                break
            candidates_to_json.append(candi)
            count += 1
        
    with open(f'{logfile[:-4]}.json', 'w') as f:
        json.dump({target: candidates_to_json}, f, indent=4)

if __name__ == '__main__':
    import os
    target_codes = json.load(open(os.path.join('./ref', 'target_list.json'), 'r'))
    for i_code in range(len(target_codes)):
        target_codes.append('python\n'+target_codes[i_code])
    get_json(f'res/res_item/ours_idxtrojan77_tokens6_batch_size256_topk64_epochs10_omega1.0_seed0.txt', 5, 329, 10, target_codes)
