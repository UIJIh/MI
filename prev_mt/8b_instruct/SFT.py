# 1 - 400개 (템플릿2개) 29 0.3347, epoch6
# 2 - 600개 (템플릿3개) 
"""
## epoch6(0.2268) 32/i;2/x;12/l;4
## epoch4(0.4091) 35/i;1/x;11/l;3
"""
# 3 - 800개 (템플릿4개 + *로 구별을 해줬는데 이게 더 안좋았다. ver2도 *주니까 더 못함)
"""
## epoch4(0.2741) 29 -> 위에 애들처럼 뭔가 meaning 맞추기에 overfitting 되어버렸당
## epoch6(0.1577) 32
## epoch1(1.0469) 24 -> 이거 너무 자유도 높다 ㅋㅋㅋ 그래도 shot 제어가 좀 잘되는듯 말랑말랑하니까
## epoch2(0.5831) ?? -> 얘도 자유도 어느정도 있고, 한국어로 번역할때 자꾸 *idiom*이거 형식 맞추려고 이상한 idiom만들거나 데이터셋의 다른idiom을 가져다가 끼워씀
## epoch3(0.4354) 27 -> 밑에 줄에 이어서 이상한 말 좀 하긴하네 + 외국어 + 위와 같은 idiom 이상하게 매칭
## epoch8(0.1137) 23(3idiom) -> 외국어.. idiom매칭으로 가네 나머지 번역안하고
    ### 어 근데 이거 * 말고  '' 로 바꾸니까 27(6idiom)
    ### 어 그리고 그냥 아예 표시 안하니까 개소리 사라졌다! ver2에서도 마찬가지! 근데 28이고 idiom으로 바꾼게 없었음.. 참.. 알다가도 모르겠다
    ### 그냥 이정도로만 하고.. 다른 모델들도 해보자
"""

"""
FULL
LLAMA3 0.6이었나.. 못봤음 
QWEN 0.7355 에폭1개만 더(6) 늘리기해서 -> 0.7111 근데 150 step중에 135?정도 됐을때부터 다시 loss 올라가고 막 이러네 지그재그.. 삭제함 일단
MISTRAL 에폭 3로만 해도 될거같아서 일단 그렇게 다시 함
0.409
"""
"""
LLAMA maxtoken 늘려서 + 에폭 6, 0.6844
"""
"""
70B 에폭 3정도만
"""

"""
19.12 NEW!
epoch 3이었는데 너무 안내려가서 5로 올림
0.6 ???

NEW-2
epoch 7
0.5228

NEW-3 
epoch 6
0.5644
"""

#python SFT.py --model llama3 --data idioms
# nohup python -u SFT.py --model llama3 --data idioms > nohup.out 2>&1 &
# tail -f nohup.out

# wandb.login()

# wandb.init(
#     project='',
#     config={'method':''}
# ) 

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, json, random, wandb, argparse, re
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
from datasets import Dataset
from tqdm import tqdm
from transformers import BitsAndBytesConfig
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "0, 1, 2, 3"

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--data', required=True)
args = parser.parse_args()

if args.model == 'llama3':
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
elif args.model == 'llama70':
    model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
elif args.model == 'mistral':
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
elif args.model == 'qwen2':
    model_id = "Qwen/Qwen2.5-7B-Instruct"
elif args.model == 'olmo':
    model_id = "allenai/OLMo-7B-0724-Instruct-hf"
elif args.model == 'post':
    model_id = "/data/uijih/8b_instruct/model_output/llama3_sft_idioms-NEW-2"

HF_token = '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' 

if args.model == 'llama70':
    # QLoRA 
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    # Apply QLoRA
    lora_config = LoraConfig(
        r=64, 
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    #tokenizer.add_special_tokens({"pad_token": "<pad>", "eos_token": "<eos>", "bos_token": "<bos>"})
    #model.resize_token_embeddings(len(tokenizer))
    #model.print_trainable_parameters()
    model.config.use_cache = False

else:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=HF_token,
        trust_remote_code=True
    )
    # 토크나이저 설정
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.add_special_tokens({"pad_token": "<pad>", "eos_token": "<eos>", "bos_token": "<bos>"}) # 안했더니 0.7에서 안떨어지는데 비교해보기
    model.resize_token_embeddings(len(tokenizer))

# 데이터셋 로드
if args.model == 'post':
    dataset_path = 'idioms_translation_conversations.jsonl' 
    with open(dataset_path, 'r') as f:
        raw_data = [json.loads(line) for line in f]
else:
    with open('idioms_conversations.jsonl', 'r') as f: ###################### NEW 12.19
        raw_data = [json.loads(line) for line in f]

# 데이터셋을 텍스트 형식으로 변환
data = {'text': []}
for item in tqdm(raw_data):
    conversation = []
    for entry in item["conversations"]:
        if 'system' in entry:
            conversation.append({'role': 'system', 'content': entry['system']})  
        conversation.append({'role': 'user', 'content': entry['user']})
        conversation.append({'role': 'assistant', 'content': entry['assistant']})    
    # 대화 템플릿 적용 및 토큰화
    templated = tokenizer.apply_chat_template(conversation, tokenize=False, padding=True, max_length=200, truncation=True)
    data['text'].append(templated)

# 데이터 처리 완료 후 확인
print(f"Processed {len(data['text'])} conversations.")


# 데이터셋 변환
data = Dataset.from_dict(data)
shuffled_data = data.shuffle(seed=42)
print(f"Sample: \n{shuffled_data[0]}")

"""
데이터 길이 확인...
"""
# from collections import Counter
# lengths = [len(text.split()) for text in data['text']]
# print(f"평균 길이: {sum(lengths)/len(lengths)}, 최대 길이: {max(lengths)}")
# all_words = " ".join(data['text']).split()
# word_counts = Counter(all_words)
# print(word_counts.most_common(10))
# # 길이가 100 이상인 데이터 확인
# long_texts = [text for text in data['text'] if len(text.split()) > 100]
# print(f"긴 데이터 개수: {len(long_texts)}")
# print("샘플 데이터:", long_texts[:3])

# post
if args.model == 'post':
    training_args = SFTConfig(
        output_dir="./model_output/{}_sft_{}-NEW-2-post".format(args.model, args.data),
        dataset_text_field='text',
        remove_unused_columns=False,
        per_device_train_batch_size=16, 
        gradient_accumulation_steps=4, 
        logging_steps=1,
        learning_rate=6e-5, 
        lr_scheduler_type="cosine", 
        packing=True,
        num_train_epochs=1, 
        save_strategy='no'  
    )
# epoch2 0.4356
# epoch1 0.4496

# original
else: 
        training_args = SFTConfig(
        output_dir="./model_output/{}_sft_{}-NEW-2-post".format(args.model, args.data),
        dataset_text_field='text',
        remove_unused_columns=False,
        per_device_train_batch_size=4, # max 200일때 32, max 늘리고선 둘이 바꿔버림 / NEW: 2->4
        gradient_accumulation_steps=64, # max 200일때 8
        logging_steps=1,
        learning_rate=2e-5, # NEW 1e-5 -> 2e-5
        lr_scheduler_type="cosine", # NEW
        #min_learning_rate=1e-7,  # NEW (최소 learning rate 설정) -> typeerror
        packing=True,
        #num_train_epochs=6, # 원래 세팅 3이었는데 NEW 너무 안내려가서 5로 올림 -> NEW-2에선 7 -> NEW-3에선 6
        num_train_epochs=1,# 70b
        save_strategy='no'  # 체크포인트 저장 비활성화
    )
    
sft_trainer = SFTTrainer(
    model,
    train_dataset=shuffled_data,
    args=training_args,
    max_seq_length=512
)

print(f"Vocab before resize: {model.config.vocab_size}")
model.resize_token_embeddings(len(tokenizer))
print(f"Vocab after resize: {model.config.vocab_size}")

# TRAIN!!!
sft_trainer.train()

# save
tokenizer.save_pretrained(training_args.output_dir) # 안그러면 size mismatch
sft_trainer.save_model(training_args.output_dir)
print(f"!!! SAVED TO '{training_args.output_dir}' !!!")

# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch, json
# from trl import SFTConfig, SFTTrainer
# from datasets import Dataset
# from tqdm import tqdm
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# from transformers import BitsAndBytesConfig
# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

# # 사용할 모델 목록 (70B 모델 제외)
# models_to_train = ['mistral', 'qwen2', 'olmo']

# # 모델 이름과 해당 모델 ID 매핑
# model_ids = {
#     'llama3': "meta-llama/Meta-Llama-3.1-8B-Instruct",
#     'llama70': "meta-llama/Meta-Llama-3.1-70B-Instruct",
#     'mistral': "mistralai/Mistral-7B-Instruct-v0.3",
#     'qwen2': "Qwen/Qwen2.5-7B-Instruct",
#     'olmo': "allenai/OLMo-7B-0724-Instruct-hf"
# }

# HF_token = 'xxxxxx'  # Hugging Face 토큰

# # 데이터셋 로드
# with open('conversation_dataset.jsonl', 'r') as f:
#     raw_data = [json.loads(line) for line in f]

# # 각 모델에 대해 순차적으로 학습 수행
# for model_key in models_to_train:
#     model_id = model_ids[model_key]
#     print(f"모델 {model_key} 학습을 시작합니다.")

#     try:
#         # 모델과 토크나이저 로드
#         if model_key == 'llama70':
#             continue  # 70B 모델은 스킵
#         else:
#             model = AutoModelForCausalLM.from_pretrained(
#                 model_id,
#                 device_map="auto",
#                 torch_dtype=torch.bfloat16,
#                 token=HF_token,
#                 trust_remote_code=True
#             )

#         # 토크나이저 설정
#         if model_key == 'qwen2':
#             tokenizer = AutoTokenizer.from_pretrained(
#                 model_id,
#                 trust_remote_code=True,
#                 use_fast=False  # Qwen 모델의 경우 use_fast=False로 설정
#             )
#         else:
#             tokenizer = AutoTokenizer.from_pretrained(
#                 model_id,
#                 trust_remote_code=True
#             )

#         tokenizer.add_special_tokens({"pad_token": "<pad>", "eos_token": "<eos>", "bos_token": "<bos>"})
#         model.resize_token_embeddings(len(tokenizer))

#         # 데이터셋을 텍스트 형식으로 변환
#         data = {'text': []}
#         for item in tqdm(raw_data):
#             conversation = []
#             for entry in item["conversations"]:
#                 conversation.append({'role': 'user', 'content': entry['user']})
#                 conversation.append({'role': 'assistant', 'content': entry['assistant']})
#             # 대화 템플릿 적용 및 토큰화
#             templated = tokenizer.apply_chat_template(conversation, tokenize=False, padding=True, max_length=200, truncation=True)
#             data['text'].append(templated)

#         # 데이터셋 변환
#         dataset = Dataset.from_dict(data)
#         shuffled_data = dataset.shuffle(seed=42)

#         # 모델 훈련 구성 설정
#         output_dir = f"./model_output/{model_key}_sft_full"
#         training_args = SFTConfig(
#             output_dir=output_dir,
#             dataset_text_field='text',
#             remove_unused_columns=False,
#             per_device_train_batch_size=32,
#             gradient_accumulation_steps=8,
#             logging_steps=1,
#             learning_rate=1e-5,
#             packing=True,
#             num_train_epochs=5,
#             save_strategy='no'  # 체크포인트 저장 비활성화
#         )

#         # SFTTrainer 설정
#         sft_trainer = SFTTrainer(
#             model=model,
#             train_dataset=shuffled_data,
#             args=training_args,
#             max_seq_length=200
#         )

#         # 모델 훈련
#         sft_trainer.train()

#         # 훈련된 모델 저장
#         sft_trainer.save_model(output_dir)

#         # 모델 설정 정보 출력
#         model.print_trainable_parameters()
#         print(f"모델 {model_key} 학습 및 저장이 완료되었습니다.\n")

#     except Exception as e:
#         print(f"모델 {model_key} 학습 중 오류가 발생했습니다: {e}\n")
#         continue
