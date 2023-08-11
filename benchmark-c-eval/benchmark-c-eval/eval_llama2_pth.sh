torchrun  --nproc_per_node 2 eval_llama2_13B_pth.py  \
          --ckpt_dir /data/models/llama2_all/data/llama-2-13b-chat \
          --tokenizer_path /data/models/llama2_all/data/tokenizer.model  \
          --max_batch_size 1  \
          --path "/data/semeron/c-eval-master"  \
          --model_name "llama2_13b_chat"  \
          --temperature 0.6  \
          --top_p 0.9  \
          --cot False  \
          --ntrain 5  \
          --max_seq_len 5000   \
          --role True




#  llama2_13b_chat: /data/models/llama2_all/data/llama-2-13b-chat
#  llama2_13b_text: /data/models/llama2_all/data/llama-2-13b
#  llama2_7b_chat: /data/models/llama2_all/data/llama-2-7b-chat
#  llama2_7b_text: /data/models/llama2_all/data/llama-2-7b
#  llama2_70b_chat: /data/models/llama2_all/data/llama-2-70b-chat
#  llama2_70b_text: /data/models/llama2_all/data/llama-2-70b


#  tokenizer:   /data/models/llama2_all/data/tokenizer.model