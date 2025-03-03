
# 번역 데이터셋 안섞으면 확실히 절대 번역 테스크 까먹는 경향 보임.
# 데이터 늘리지 않으면 비율 문제가 아니라 그냥 못함; 번역 데이터셋에 있는걸로 번역을 한다던가, 템플릿 이상하게 외운다던가, 이디엄 매칭도 제대로 안됐음. 오히려 idom 번역을 해버림!(한idiom->한mean이 아니라, 영idoim으로)
# 하이튼 그래서 돌고돌아 내가 원래 했던 방식대로, 템플릿 여러개 입혀서 데이터셋 늘림. 다만 전에는 instruction template을 줬는데 지금은 안줘봤음.
"""
## 1트 : test0.25, 템플릿 3개씩, 교차로, instruction은 번역만  
## 0.2589 : 첫줄은 ㄱㅊ게 생성하는데, # Translation:하고 계속 이상한말 내뱉음(데이터셋안에잇는것들)
전체 학습 데이터셋 크기: 3000
Idiom 학습 데이터셋 크기: 1200 (40)
번역 학습 데이터셋 크기: 1800 (60)
"""
"""
## 1.1트 : 1과 같은데, epoch4 + model.config.pad_token_id 설정 + max_length 줄임 + translation은 랜덤 템플릿1(근데instruction없애고 한영섞음)
## 0.3494 : idoim매칭은 첫줄만 괜찮고, translation은 아예 못함.. 그래서 번역 데이터셋을 늘려야겠음
"""
"""
## 2트 : 1과 같은데, 번역 데이터셋 그냥 다 사용, epoch10(4,5는0.3임계속)
## 0.3322
전체 학습 데이터셋 크기: 3600
Idiom 학습 데이터셋 크기: 1200
번역 학습 데이터셋 크기: 2400 (66.67)
"""
"""
## 3트 : 2개씩, 2e-5 + epoch 15 + modelconfig그거지우고 교차데이터셋 없앴음 ㅡㅡ
## 0.2845
전체 학습 데이터셋 크기: 2800
Idiom 학습 데이터셋 크기: 400
번역 학습 데이터셋 크기: 2400 (85.71)
"""
# 4트 : 이 중 괜찮은거있으면 그걸로 영어만

import wandb, os, torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from huggingface_hub import HfApi, HfFolder, upload_folder
from tokenized_dataset import dataset_and_tokenizer

def prepare_dataset_and_tokenizer(model_path, data_path_1, data_path_2):
    """
    데이터셋과 토크나이저 준비 함수
    """
    shuffled_tokenized_full_dataset, eval_dataset, tokenizer = dataset_and_tokenizer(data_path_1, data_path_2, model_path)
    return shuffled_tokenized_full_dataset, eval_dataset, tokenizer


def prepare_model(model_path, tokenizer):
    """
    모델 준비 함수
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16, # 권장
    )
    # tokenizer에 맞추기
    #model.config.pad_token_id = tokenizer.pad_token_id
    # 원랜 None
    #assert model.config.pad_token_id == tokenizer.pad_token_id, "The model's pad token ID does not match the tokenizer's pad token ID!"    
    model.resize_token_embeddings(len(tokenizer))

    model.gradient_checkpointing_enable()    
    model.config.use_cache = False  

    # 모든 파라미터를 학습 가능하도록 설정
    for param in model.parameters():
        param.requires_grad = True
    print_trainable_parameters(model)    
    return model

def print_trainable_parameters(model):
    """
    모델에서 학습 가능한 파라미터의 수를 출력하는 함수
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

def configure_trainer(model, tokenizer, train_dataset, eval_dataset, output_dir):
    """
    Trainer 설정 함수
    """
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        #eval_dataset=eval_dataset,
        args=TrainingArguments(
            output_dir=output_dir,
            lr_scheduler_type="cosine",
            per_device_train_batch_size=32,
            gradient_accumulation_steps=4,
            max_steps=-1,
            num_train_epochs=15,
            learning_rate=2e-5,
            logging_steps=1,
            bf16=True,
            optim="paged_adamw_8bit",
            logging_dir="./logs",
            save_steps=20,
            save_strategy="steps",
            # eval_strategy="steps",
            # eval_steps=20,
            # do_eval=True,
            # per_device_eval_batch_size=4,  
            #load_best_model_at_end=True,
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    return trainer


def save_model_and_tokenizer(model, tokenizer, model_path):
    """
    모델과 토크나이저 저장 함수
    """
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print("Model and tokenizer saved successfully!")


def upload_model_to_hf(model_path, repo_name):
    """
    Hugging Face Hub에 모델 업로드 함수
    """
    HfFolder.save_token("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")
    api = HfApi()
    api.create_repo(repo_name, exist_ok=True)
    upload_folder(
        folder_path=model_path,
        path_in_repo=".",
        repo_id=repo_name,
        repo_type="model"
    )
    print(f"Model uploaded to Hugging Face Hub: {repo_name}")


def main():
    os.environ["CUDA_VISIBLE_DEVICES"]= "3"
    os.environ["WANDB_MODE"] = "offline"

    model_path = "meta-llama/Meta-Llama-3.1-8B"
    data_path_1 = "/data/uijih/Seed50_for_Parallel_Dataset_ENKR_idiomKB_0.8_example.csv"
    data_path_2 = load_from_disk("/data/uijih/filtered_dataset")

    # 1. 데이터셋 및 토크나이저 준비
    shuffled_tokenized_full_dataset, eval_dataset, tokenizer = prepare_dataset_and_tokenizer(model_path, data_path_1, data_path_2)

    # 2. 모델 준비
    model = prepare_model(model_path, tokenizer)

    # 3. Trainer 설정
    project = "3"
    base_model_name = "llama3.1-8b"
    run_name = base_model_name + "-" + project
    output_dir = "./" + run_name

    trainer = configure_trainer(model, tokenizer, shuffled_tokenized_full_dataset, eval_dataset, output_dir)

    # 4. Wandb 초기화
    # proj_name = "Meta-Llama-3.1-8B"
    # wandb.init(
    #     project=proj_name,
    #     name=run_name,
    #     config=trainer.args
    # )

    # 5. 모델 학습
    trainer.train()

    # 6. 모델 저장
    model_save_path = "./saveded_8b-3"
    save_model_and_tokenizer(model, tokenizer, model_save_path)
    print("!!!!SAVED!!!!")

    # 7. 모델 Hugging Face Hub에 업로드
    # repo_name = "Uiji/llama-8b-2"
    # upload_model_to_hf(model_save_path, repo_name)

if __name__ == "__main__":
    main()
