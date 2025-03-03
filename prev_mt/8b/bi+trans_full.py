### 1트 0.271 : 에폭 늘려서 학습 더 시키면 괜찮을거같다. 확실히 큐로라보단 훨씬 나음
"""
번역 데이터셋 크기: 3120 -> 82.80
학습 데이터셋 크기: 3768
평가 데이터셋 크기: 420
"""
### 2트 0.215 : lr, epoch 조정
### 2-1트 0.133
"""
번역 데이터셋 크기 (줄인 후): 192
관용구 데이터셋 크기 (줄인 후): 12
학습 데이터셋 크기 (결합 후): 204
평가 데이터셋 크기: 6
"""
### 데이터셋 균형 맞추기(영어 instruction만, 템플릿 2개) 0.2688
"""
번역 데이터셋 크기: 1800 -> 80.65
학습 데이터셋 크기: 2232
평가 데이터셋 크기: 300
"""
### 데이터셋 균형 맞추기(영어+한국어 instruction, 템플릿 4개) 0.2123
"""
번역 데이터셋 크기: 3216 -> 78.82
학습 데이터셋 크기: 4080
평가 데이터셋 크기: 396
"""
### eval기준으로 모델 저장해버려서,, 0.2432
"""
번역 데이터셋 크기: 3840 -> 81.63
학습 데이터셋 크기: 4704
평가 데이터셋 크기: 240
"""
######### NEW! 0.2474
"""
번역 데이터셋 크기: 3840
학습 데이터셋 크기: 4640
평가 데이터셋 크기: 240
"""

import torch, wandb
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from huggingface_hub import HfApi, HfFolder, upload_folder
from pr_tokenzied_dataset import dataset_and_tokenizer

# 1. 데이터셋 준비
model_path = "meta-llama/Meta-Llama-3.1-8B"
data_path_1 = '/data/uijih/Seed50_for_Parallel_Dataset_ENKR_idiomKB_0.8_example.csv'
data_path_2 = load_from_disk("/data/uijih/filtered_dataset")
shuffled_tokenized_full_dataset, eval_dataset, tokenizer = dataset_and_tokenizer(data_path_1, data_path_2, model_path)


# 2. 모델 준비
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
model.config.pad_token_id = tokenizer.pad_token_id
    # 원랜 None
assert model.config.pad_token_id == tokenizer.pad_token_id, "The model's pad token ID does not match the tokenizer's pad token ID!"    
model.resize_token_embeddings(len(tokenizer))

model.gradient_checkpointing_enable()

# 모든 파라미터를 학습 가능하도록 설정
for param in model.parameters():
    param.requires_grad = True

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
print_trainable_parameters(model)

# 3. Trainer 설정
project = "uiji-bi+trans-full-1"
base_model_name = "llama3.1-8b"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name
proj_name = "Meta-Llama-3.1-8B-KoID"

trainer = Trainer(
    model=model,
    train_dataset=shuffled_tokenized_full_dataset,
    eval_dataset=eval_dataset,
    args=TrainingArguments(
        output_dir=output_dir,
        #warmup_steps=5,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=32, 
        gradient_accumulation_steps=2,
        max_steps=400, # 굳이 길게 안해도될거같아서(마지막)
        num_train_epochs=10, # 5 -> 10 
        learning_rate=1e-5, # 6-6 너무안줄어듦 -> 
        logging_steps=1,
        # fp16=True if torch_dtype != torch.bfloat16 else False,
        # bf16=True if torch_dtype == torch.bfloat16 else False,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_dir="./logs",
        save_strategy="steps",
        save_steps=500,
        # report_to='wandb',
        # eval_strategy="steps",
        # eval_steps=100,
        # do_eval=True,
        #load_best_model_at_end=True,
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False
# wandb.init(
#     project=proj_name,
#     name=run_name,
#     config=trainer.args
# )
trainer.train()

# 4. 모델 저장
model_path = "./saved_full_finetuned-1"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# HfFolder.save_token("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")
# repo_name = "Uiji/llama-full-finetuned-t"
# api = HfApi()
# api.create_repo(repo_name, exist_ok=True)
# upload_folder(
#     folder_path=model_path,
#     path_in_repo=".",
#     repo_id=repo_name,
#     repo_type="model"
# )