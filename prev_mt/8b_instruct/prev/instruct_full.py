## 1트 : 0.4341
"""
# 번역 데이터셋 크기: 1680 -> 88.61
# 학습 데이터셋 크기: 1896
# 평가 데이터셋 크기: 720
"""
## 2트 epoch : 0.2681
## 3트 prompt
"""
번역 데이터셋 크기: 2040 -> 90.43
학습 데이터셋 크기: 2256
평가 데이터셋 크기: 360
"""
## 3트 prompt full로 : 0.2292
"""
번역 데이터셋 크기: 2040 -> 82.52
학습 데이터셋 크기: 2472
평가 데이터셋 크기: 360 -> 14.56
"""

## new-version! : 템플릿 하나 더 추가함(교차) + 에폭10에서4(3으로했어도 됐을듯) 근데 에폭 10으로 늘림 왜냐면 inference이상하니까
## model.config.pad_token_id = tokenizer.pad_token_id 주석처리함.. 아니 왜 갑자기 이상하게 튜닝되는거야 진짜ㅡㅡ
## 교차도 일단 안해보겠음 
## ㅅㅂ 이유가 있었음.. 데이터셋 잘못만졌던것...
"""
0.2041
번역 데이터셋 크기: 2040 -> 83.61
학습 데이터셋 크기: 2440
"""

import torch, wandb, os
os.environ["CUDA_VISIBLE_DEVICES"]= "0, 1, 2"
os.environ["WANDB_MODE"] = "offline"
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from huggingface_hub import HfApi, HfFolder, upload_folder
from tokenized_dataset_instruct import dataset_and_tokenizer

# 1. 데이터셋 준비
#model_path = "/data/uijih/instruct/Meta-Llama-3.1-8B-Instruct"
model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
data_path_1 = "/data/uijih/Seed50_for_Parallel_Dataset_ENKR_idiomKB_0.8_example.csv"
data_path_2 = load_from_disk("/data/uijih/filtered_dataset")
shuffled_tokenized_full_dataset, eval_dataset, tokenizer = dataset_and_tokenizer(data_path_1, data_path_2, model_path)

# 2. 모델 준비
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

model.gradient_checkpointing_enable()

def print_trainable_parameters(model):
    """
    모델에서 학습 가능한 파라미터의 수를 출력합니다.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"학습 가능한 파라미터 수: {trainable_params} || 전체 파라미터 수: {all_param} || 비율: {100 * trainable_params / all_param}"
    )

# 모든 파라미터를 학습 가능하도록 설정
for param in model.parameters():
    param.requires_grad = True

print_trainable_parameters(model)
#model.config.pad_token_id = tokenizer.pad_token_id
model.resize_token_embeddings(len(tokenizer))
model.config.use_cache = False  

# 3. Trainer 설정
project = "full-1"
base_model_name = "llama3.1-8b-Instruct"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

if torch.cuda.get_device_capability()[0] >= 8:
    attn_implementation = "flash_attention_2"
    torch_dtype = torch.bfloat16
else:
    attn_implementation = "eager"
    torch_dtype = torch.float16

trainer = Trainer(
    model=model,
    train_dataset=shuffled_tokenized_full_dataset,
    #eval_dataset=eval_dataset,
    args=TrainingArguments(
        output_dir=output_dir,
        warmup_steps=5,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=16, # 32안됨
        gradient_accumulation_steps=8,
        max_steps=-1,
        num_train_epochs=10, # 3->
        learning_rate=1e-5, 
        logging_steps=1,
        # fp16=True if torch_dtype != torch.bfloat16 else False,
        # bf16=True if torch_dtype == torch.bfloat16 else False,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_dir="./logs",
        save_strategy="steps",
        save_steps=50,
        #report_to='wandb',
        # eval_strategy="steps", 
        # eval_steps=5,             
        # do_eval=True,               
        #load_best_model_at_end=True,
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
# proj_name = "Meta-Llama-3.1-8B-Instruct"
# wandb.init(
#     project=proj_name,
#     name=run_name,
#     config=trainer.args
# )
trainer.train()

# 4. 모델 저장
model_path = "./saveded_instruct-full-1" 
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# HfFolder.save_token("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")
# repo_name = "Uiji/llama-instruct-full"
# api = HfApi()
# api.create_repo(repo_name, exist_ok=True)
# upload_folder(
#     folder_path=model_path,
#     path_in_repo=".",
#     repo_id=repo_name,
#     repo_type="model"
# )