# base모델보다 못하고.. input template을 외운다..? 흠.. 뭐가 문제지? -> 해결 완료~ 마스킹처리 제대로 안됐었음~
# 일단 eval cosine으로 안하고 해보기
"""
- ver1 : 0.0958 
    - batch32/epoch10/6e-5 
    - 대부분 왜 시키지도 않은 meaning을!~!!
- ver2 : 0.0712
    - epoch13은 다시 0.09로 감 / 15는 loss 급올라감 / 12로
    - meaning을 자꾸 얘기하고 좀 이상해졌다 결과가
- ver3 : 조금 덜 해보자 에폭 낮추든지 accumulate 올려!! 그럼 eos 이런거 내뱉는거 줄을듯?
"""
import torch, wandb
import pandas as pd
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from huggingface_hub import HfApi, HfFolder, upload_folder
from dataset import dataset_and_tokenizer

# 1. 데이터셋 준비, Tokenizer 준비
model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
data_path_1 = "/data/uijih/previous/Seed50_for_Parallel_Dataset_ENKR_idiomKB_0.8_example.csv"
data_path_2 = load_from_disk("/data/uijih/instruct/filtered_dataset")
shuffled_tokenized_full_dataset, eval_dataset, tokenizer = dataset_and_tokenizer(data_path_1, data_path_2, model_path)

# 2. 모델 준비
if torch.cuda.get_device_capability()[0] >= 8:
    # pip install -qqq flash-attn
    attn_implementation = "flash_attention_2"
    torch_dtype = torch.bfloat16
    #print("bfloat16!")
else:
    attn_implementation = "eager"
    torch_dtype = torch.float16

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch_dtype
)
# 임베딩 크기 조절
model.resize_token_embeddings(len(tokenizer))
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
model.config.use_cache = False  

# 3. Trainer 설정
# project = "detection-1"
# base_model_name = "llama3.1-8b-Instruct"
# run_name = base_model_name + "-" + project
# output_dir = "./" + run_name

# if torch.cuda.get_device_capability()[0] >= 8:
#     attn_implementation = "flash_attention_2"
#     torch_dtype = torch.bfloat16
# else:
#     attn_implementation = "eager"
#     torch_dtype = torch.float16

# ## 메트릭 직접
# import numpy as np
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     # 레이블에서 -100인 부분은 무시
#     mask = labels != -100
#     labels = labels[mask]
#     predictions = predictions[mask]
#     # 정확도 계산
#     accuracy = np.mean(predictions == labels)
#     return {"accuracy": accuracy}

# trainer = Trainer(
#     model=model,
#     train_dataset=shuffled_tokenized_full_dataset,
#     eval_dataset=eval_dataset,
#     args=TrainingArguments(
#         output_dir=output_dir,
#         warmup_steps=100,
#         lr_scheduler_type="cosine",
#         per_device_train_batch_size=4,
#         gradient_accumulation_steps=8,
#         max_steps=-1,
#         num_train_epochs=3,
#         learning_rate=1e-4,
#         logging_steps=10,
#         fp16=True if torch_dtype != torch.bfloat16 else False,
#         bf16=True if torch_dtype == torch.bfloat16 else False,
#         optim="adamw_torch",
#         logging_dir="./logs",
#         save_strategy="steps",
#         save_steps=500,
#         report_to='wandb',
#         eval_strategy="steps",
#         eval_steps=500,
#         do_eval=True,
#         load_best_model_at_end=True,
#         metric_for_best_model="accuracy",
#         greater_is_better=True,
#     ),
#     data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
#     compute_metrics=compute_metrics,
# )

# 3. 메트릭 함수 정의
#def make_compute_metrics(model):
    # def compute_metrics(eval_pred):
    #     logits, labels = eval_pred
    #     logits = torch.tensor(logits, device=model.device)
    #     labels = torch.tensor(labels, device=model.device)

    #     predictions = torch.argmax(logits, dim=-1)
    #     nonmask = labels != -100
    #     labels = labels[nonmask]
    #     predictions = predictions[nonmask]

    #     embedding_layer = model.get_input_embeddings()

    #     labels_embeddings = embedding_layer(labels)
    #     predictions_embeddings = embedding_layer(predictions)

    #     cosine_sim = torch.nn.functional.cosine_similarity(labels_embeddings, predictions_embeddings, dim=-1)
    #     avg_cosine_sim = cosine_sim.mean().item()

    #     return {"cosine_similarity": avg_cosine_sim}
    # return compute_metrics
def make_compute_metrics(model, tokenizer):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = torch.tensor(logits, device=model.device)
        labels = torch.tensor(labels, device=model.device)

        predictions = torch.argmax(logits, dim=-1)
        nonmask = labels != -100
        labels = labels[nonmask]
        predictions = predictions[nonmask]

        # 토큰 ID를 실제 단어로 변환
        labels_ids = labels.tolist()
        predictions_ids = predictions.tolist()
        labels_tokens = tokenizer.convert_ids_to_tokens(labels_ids)
        predictions_tokens = tokenizer.convert_ids_to_tokens(predictions_ids)

        # 토큰 리스트를 문자열로 변환
        labels_text = tokenizer.convert_tokens_to_string(labels_tokens)
        predictions_text = tokenizer.convert_tokens_to_string(predictions_tokens)

        # embeddings 계산 및 코사인 유사도 계산
        embedding_layer = model.get_input_embeddings()
        labels_embeddings = embedding_layer(labels)
        predictions_embeddings = embedding_layer(predictions)
        cosine_sim = torch.nn.functional.cosine_similarity(labels_embeddings, predictions_embeddings, dim=-1)
        avg_cosine_sim = cosine_sim.mean().item()

        # 결과 출력
        print("Labels Text:", labels_text)
        print("Predictions Text:", predictions_text)
        return {"cosine_similarity": avg_cosine_sim}
    return compute_metrics

# 4. Trainer 설정
project = "detection-2"
base_model_name = "llama3.1-8b-Instruct"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

trainer = Trainer(
    model=model,
    train_dataset=shuffled_tokenized_full_dataset,
    eval_dataset=eval_dataset,
    args=TrainingArguments(
        output_dir=output_dir,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        max_steps=-1,
        num_train_epochs=12,
        learning_rate=6e-5, # (ohne eval)1-5 too stable -> 5e-5 gut
        logging_steps=1,
        #fp16=True if torch_dtype != torch.bfloat16 else False, # 32가 더좋고, bf보다 안정확하지만 빠르고 메모리 적게사용
        #bf16=True if torch_dtype == torch.bfloat16 else False, # bf16이 더 좋음 32보단 안좋음
        bf16=True,
        optim="paged_adamw_8bit",
        logging_dir="./logs",
        save_strategy="steps",
        #save_steps=40,
        report_to='wandb',
        # eval_strategy="steps",
        # eval_steps=2,
        # do_eval=True,
        # per_device_eval_batch_size=4,  # 평가 배치 크기를 줄입니다.
        # eval_accumulation_steps=4,  # gpu
        # load_best_model_at_end=True,
        # metric_for_best_model="cosine_similarity",
        # greater_is_better=True, # cosine처럼 큰값이 좋은거
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    #compute_metrics=make_compute_metrics(model, tokenizer),
)

# proj_name = "Meta-Llama-3.1-8B-Instruct-detection"
# wandb.init(
#     project=proj_name,
#     name=run_name,
#     config=trainer.args
# )
trainer.train()

# 4. 모델 저장
model_path = "./saveded_instruct-full-detection-2" 
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
print("saved successfully!")

HfFolder.save_token("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")
repo_name = "Uiji/llama-instruct-detection-2"
api = HfApi()
api.create_repo(repo_name, exist_ok=True)
upload_folder(
    folder_path=model_path,
    path_in_repo=".",
    repo_id=repo_name,
    repo_type="model"
)