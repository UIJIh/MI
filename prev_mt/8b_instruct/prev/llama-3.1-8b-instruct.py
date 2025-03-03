### 1트
"""
번역 데이터셋 크기: 1680 -> 88.61
학습 데이터셋 크기: 1896
평가 데이터셋 크기: 720 -> 37.97
"""

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
from tokenized_dataset_instruct import dataset_and_tokenizer

# 1. 데이터셋 준비
model_path = "/data/uijih/instruct/Meta-Llama-3.1-8B-Instruct"
data_path_1 = "/data/uijih/instruct/Seed_for_Parallel_Dataset_ENKR_idiomKB_0.8.csv"
data_path_2 = load_from_disk("/data/uijih/instruct/filtered_dataset")
shuffled_tokenized_full_dataset, eval_dataset, tokenizer = dataset_and_tokenizer(data_path_1, data_path_2, model_path)
special_tokens_dict = {
    'additional_special_tokens': [
        '<|begin_of_text|>', '<|end_of_text|>', '<|finetune_right_pad_id|>', 
        '<|start_header_id|>', '<|end_header_id|>', '<|eom_id|>', '<|eot_id|>',
        '<|python_tag|>'
    ]
}
tokenizer.add_special_tokens(special_tokens_dict)

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
model.config.use_cache = False  

# 3. Trainer 설정
project = "2"
base_model_name = "llama3.1-8b-Instruct"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

if torch.cuda.get_device_capability()[0] >= 8:
    # pip install -qqq flash-attn
    attn_implementation = "flash_attention_2"
    torch_dtype = torch.bfloat16
else:
    attn_implementation = "eager"
    torch_dtype = torch.float16

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
        num_train_epochs=3, # 
        learning_rate=3e-4, #
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
proj_name = "Meta-Llama-3.1-8B-Instruct"
wandb.init(
    project=proj_name,
    name=run_name,
    config=trainer.args
)
trainer.train()

# 4. 모델 저장
model_path = "./saveded_instruct-2" 
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

HfFolder.save_token("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")
repo_name = "Uiji/llama-instruct-2"
api = HfApi()
api.create_repo(repo_name, exist_ok=True)
upload_folder(
    folder_path=model_path,
    path_in_repo=".",
    repo_id=repo_name,
    repo_type="model"
)