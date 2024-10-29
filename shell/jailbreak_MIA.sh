CUDA_VISIBLE_DEVICES=0 python jailbreak_for_every_tokens.py --prefix-id=0 --phase=development &
CUDA_VISIBLE_DEVICES=1 python jailbreak_for_every_tokens.py --prefix-id=1 --phase=development &
CUDA_VISIBLE_DEVICES=2 python jailbreak_for_every_tokens.py --prefix-id=2 --phase=development &
CUDA_VISIBLE_DEVICES=3 python jailbreak_for_every_tokens.py --prefix-id=3 --phase=development &
CUDA_VISIBLE_DEVICES=4 python jailbreak_for_every_tokens.py --prefix-id=0 --phase=test &
CUDA_VISIBLE_DEVICES=5 python jailbreak_for_every_tokens.py --prefix-id=1 --phase=test &
CUDA_VISIBLE_DEVICES=6 python jailbreak_for_every_tokens.py --prefix-id=2 --phase=test &
CUDA_VISIBLE_DEVICES=7 python jailbreak_for_every_tokens.py --prefix-id=3 --phase=test

CUDA_VISIBLE_DEVICES=0 python MIA.py --jailbreak-file=test/jailbreak_with_Sure &
CUDA_VISIBLE_DEVICES=1 python MIA.py --jailbreak-file=test/jailbreak_with_discssuser &
CUDA_VISIBLE_DEVICES=2 python MIA.py --jailbreak-file=test/jailbreak_with_repo_name_user &
CUDA_VISIBLE_DEVICES=3 python MIA.py --jailbreak-file=test/jailbreak_with_user &
CUDA_VISIBLE_DEVICES=4 python MIA.py --jailbreak-file=development/jailbreak_with_Sure  &
CUDA_VISIBLE_DEVICES=5 python MIA.py --jailbreak-file=development/jailbreak_with_discssuser  &
CUDA_VISIBLE_DEVICES=6 python MIA.py --jailbreak-file=development/jailbreak_with_repo_name_user  &
CUDA_VISIBLE_DEVICES=7 python MIA.py --jailbreak-file=development/jailbreak_with_user  