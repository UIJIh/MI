# 0.2344 (0.25)
## 번역 데이터셋 크기: 4800 -> 88.11
## 학습 데이터셋 크기: 5448
## 평가 데이터셋 크기: 400 -> 8.33

# (0.75)translation 데이터 비율 줄이고 에폭은 2만 올려보기(5)
## 번역 데이터셋 크기: 1000 -> 60.68
## 학습 데이터셋 크기: 1648
## 평가 데이터셋 크기: 750 -> 45.51

# 0.2151 (0.5)번역 데이터 비율 조금 더 늘리고, 에폭 10으로 해놓고 잘거임
#번역 데이터셋 크기: 2600 -> 80.05
#학습 데이터셋 크기: 3248
#평가 데이터셋 크기: 650 -> 20.01   

# 0.2281 다시 번역 데이터셋 1600개로 하고 0.35 비율로 하고, 에폭은 일단 다시 3하고 잘때 5해놓기(10은 eval 슬슬 올라간다)
## 번역 데이터셋 크기: 4160 -> 86.52
## 학습 데이터셋 크기: 4808
## 평가 데이터셋 크기: 560 -> 11.65

import torch
import wandb
import pandas as pd
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from huggingface_hub import HfApi, HfFolder, upload_folder
from tokenzied_dataset import dataset_and_tokenizer

# 1. 데이터셋 준비
model_path = "meta-llama/Meta-Llama-3.1-8B"
data_path_1 = '/data/uijih/previous/Seed50_for_Parallel_Dataset_ENKR_idiomKB_0.8_example.csv'
data_path_2 = load_from_disk("./filtered_dataset")
shuffled_tokenized_full_dataset, eval_dataset, tokenizer = dataset_and_tokenizer(data_path_1, data_path_2, model_path)

# 2. 모델 준비
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto"
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

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

config = LoraConfig(
    r=128,
    lora_alpha=128,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
print_trainable_parameters(model)

# 3. Trainer 설정
project = "uiji-bi+trans-4"
base_model_name = "llama3.1-8b"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

if torch.cuda.get_device_capability()[0] >= 8:
    # pip install -qqq flash-attn
    attn_implementation = "flash_attention_2"
    torch_dtype = torch.bfloat16
else:
    attn_implementation = "eager"
    torch_dtype = torch.float16

proj_name = "Meta-Llama-3.1-8B-KoID"
wandb.init(project=proj_name)
trainer = Trainer(
    model=model,
    train_dataset=shuffled_tokenized_full_dataset,
    eval_dataset=eval_dataset,
    args=TrainingArguments(
        output_dir=output_dir,
        warmup_steps=5,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=32,
        gradient_accumulation_steps=2,
        max_steps=-1,
        num_train_epochs=3, # 3 -> 5 -> 10 -> 5
        learning_rate=3e-4, #2.5e-4 중간에 한번 살짝 튀길래 -> 3 -> (에폭늘리고)3.5 -> (4트에서 튀어서)1 -> (에폭줄이고)
        logging_steps=1,
        fp16=True if torch_dtype != torch.bfloat16 else False,
        bf16=True if torch_dtype == torch.bfloat16 else False,
        optim="paged_adamw_8bit",
        logging_dir="./logs",
        save_strategy="steps",
        save_steps=50,
        report_to='wandb',
        eval_strategy="steps", 
        eval_steps=5,             
        do_eval=True,               
        load_best_model_at_end=True,
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  
trainer.train()

# 4. 모델 저장
model_path = "./saveded_bi+trans-4" 
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

HfFolder.save_token("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")
repo_name = "Uiji/llama-bi+trans-4"
api = HfApi()
api.create_repo(repo_name, exist_ok=True)
upload_folder(
    folder_path=model_path,
    path_in_repo=".",
    repo_id=repo_name,
    repo_type="model"
)
