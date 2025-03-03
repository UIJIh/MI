# import pandas as pd
# from datasets import Dataset, concatenate_datasets
# from transformers import AutoTokenizer
# import random

# def dataset_and_tokenizer(data_path, new_data_path, model_path):
#     # 1. 모델과 토크나이저 준비
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_path,
#         model_max_length=256,
#         add_eos_token=True,
#         padding="longest",
#     )
#     tokenizer.pad_token = tokenizer.eos_token

#     # 특수 토큰 추가
#     special_tokens_dict = {
#         'additional_special_tokens': [
#             '<|begin_of_text|>', '<|end_of_text|>', '<|finetune_right_pad_id|>',
#             '<|start_header_id|>', '<|end_header_id|>', '<|eom_id|>', '<|eot_id|>',
#             '<|python_tag|>'
#         ]
#     }
#     tokenizer.add_special_tokens(special_tokens_dict)

#     # 2. 기존 데이터셋 준비
#     original_data = pd.read_csv(data_path)

#     # 3. 새로운 데이터셋 준비 (Idiom이 없는 경우)
#     new_data = pd.read_csv(new_data_path)

#     # 4. 시스템 프롬프트 정의 (영어)
#     # system_prompts_en = [
#     #     #"Detect if there is an idiom in the following sentence. If there is, return only the detected idiom in its original form. If there are no idioms, answer 'none'."
#     #      "Detect if there is an idiom in the following sentence. If there is, return only the detected idiom. If there are no idioms, answer 'None'."
#     # ]
#     system_prompts_en = [
#     "Detect if there is an idiom in the following sentence. If there is, return only the detected idiom. If there are no idioms, answer 'None'.",
#     "Identify if the sentence contains any idioms. Provide only the idiom if found, or 'None' if there are no idioms.",
#     "Please find any idioms in this sentence. If you detect one, write just the idiom. If there’s none, say 'None'.",
#     "Check if the sentence has an idiom. If yes, respond only with the idiom; if not, reply 'None'.",
# ]
#     # 5. 프롬프트 템플릿 정의
#     prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

# {system_prompt}<|eot_id|>
# <|start_header_id|>user<|end_header_id|>

# {user_input}<|eot_id|>
# <|start_header_id|>assistant<|end_header_id>
# """

#     # 6. 데이터셋 생성 함수
#     def create_dataset(df, input_col, output_col):
#         llm_df = pd.DataFrame()
#         llm_df['input'] = df[input_col]
#         llm_df['output'] = df[output_col]
#         return Dataset.from_pandas(llm_df)

#     # 7. 기존 데이터셋 생성
#     train_dataset_original_en = create_dataset(original_data, 'Sentence', 'Idiom')
#     train_dataset_original_kr = create_dataset(original_data, 'KR_Sentence', 'KR_Idiom')

#     # 8. 새로운 데이터셋 생성
#     train_dataset_new_en = create_dataset(new_data, 'New_Example_EN', 'Label')
#     train_dataset_new_kr = create_dataset(new_data, 'New_Example_KR', 'Label')

#     # 9. 프롬프트 생성 및 토크나이즈 함수
#     def generate_and_tokenize_prompt(data_batch, system_prompts, tokenizer):
#         inputs = data_batch["input"]
#         outputs = data_batch["output"]

#         prompts = []
#         for inp in inputs:
#             # 랜덤한 프롬프트 선택
#             system_prompt = random.choice(system_prompts)
            
#             #user_input = f'Is there an idiom in the sentence "{inp}"? If yes, return only the detected idiom in its original form without any additional explanation or text.'
#             user_input = f'Is there an idiom in the sentence "{inp}"? If yes, return only the detected idiom without any additional explanation or text.'

#             prompt = prompt_template.format(
#                 system_prompt=system_prompt,
#                 user_input=user_input
#             )
#             prompts.append(prompt)

#         # 전체 입력(프롬프트 + 모델 응답)을 생성
#         #answers = [out + tokenizer.eos_token for out in outputs]
#         answers = [(out if out is not None else "None") + tokenizer.eos_token for out in outputs]
#         texts = [p + a for p, a in zip(prompts, answers)]

#         # 토크나이즈
#         tokenized = tokenizer(
#             texts,
#             truncation=True,
#             max_length=512,
#             padding="max_length",
#         )

#         # 프롬프트 길이 계산
#         prompt_lengths = [len(tokenizer(p, truncation=True, max_length=512)["input_ids"]) for p in prompts]

#         # 레이블 생성 및 프롬프트 부분 마스킹
#         labels = []
#         for input_ids, prompt_len in zip(tokenized["input_ids"], prompt_lengths):
#             label = input_ids.copy()
#             label[:prompt_len] = [-100] * prompt_len
#             labels.append(label)

#         tokenized["labels"] = labels
#         return tokenized

#     # 10. 학습 데이터셋 매핑
#     tokenized_train_dataset_original_en = train_dataset_original_en.map(
#         lambda x: generate_and_tokenize_prompt(x, system_prompts_en, tokenizer),
#         remove_columns=train_dataset_original_en.column_names, batched=True
#     )

#     tokenized_train_dataset_original_kr = train_dataset_original_kr.map(
#         lambda x: generate_and_tokenize_prompt(x, system_prompts_en, tokenizer),
#         remove_columns=train_dataset_original_kr.column_names, batched=True
#     )

#     tokenized_train_dataset_new_en = train_dataset_new_en.map(
#         lambda x: generate_and_tokenize_prompt(x, system_prompts_en, tokenizer),
#         remove_columns=train_dataset_new_en.column_names, batched=True
#     )
#     tokenized_train_dataset_new_kr = train_dataset_new_kr.map(
#         lambda x: generate_and_tokenize_prompt(x, system_prompts_en, tokenizer),
#         remove_columns=train_dataset_new_kr.column_names, batched=True
#     )
#     #print(f"None 샘플 데이터: {tokenized_train_dataset_new_en[0]},\n{tokenized_train_dataset_new_kr[0]}")
#     def print_mapped_sample(dataset, num_samples=1):
#         for i in range(num_samples):
#             print(f"샘플 {i + 1}:")
#             #print("Input:", dataset[i]["input_ids"])
#             #print("Labels:", dataset[i]["labels"])
#             print("Decoded Input:", tokenizer.decode(dataset[i]["input_ids"], skip_special_tokens=True))
#             print("Decoded Label:", tokenizer.decode([id for id in dataset[i]["labels"] if id != -100], skip_special_tokens=True))
#             print("=" * 50)

#     # 매핑 후 샘플 출력
#     # print("Original English Dataset Sample:")
#     # print_mapped_sample(tokenized_train_dataset_original_en)

#     # print("Original Korean Dataset Sample:")
#     # print_mapped_sample(tokenized_train_dataset_original_kr)

#     print("New English Dataset Sample:")
#     print_mapped_sample(tokenized_train_dataset_new_en)

#     print("New Korean Dataset Sample:")
#     print_mapped_sample(tokenized_train_dataset_new_kr)

#     # 11. 모든 토크나이즈된 학습 데이터셋을 결합
#     tokenized_full_dataset_train = concatenate_datasets([
#         tokenized_train_dataset_original_en,
#         tokenized_train_dataset_original_kr,
#         tokenized_train_dataset_new_en,
#         tokenized_train_dataset_new_kr
#     ])
    
#     # 12. 데이터셋 셔플
#     shuffled_tokenized_full_dataset_train = tokenized_full_dataset_train.shuffle(seed=1)

#     # 데이터셋 크기 출력
#     print("전체 학습 데이터셋 크기:", len(shuffled_tokenized_full_dataset_train))
#     print("None 데이터셋 크기:", len(tokenized_train_dataset_new_en)+len(tokenized_train_dataset_new_kr))

#     # 13. 토크나이저와 셔플된 데이터셋 반환
#     return shuffled_tokenized_full_dataset_train, tokenizer
import pandas as pd
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer
import random
from tqdm import tqdm
import json

def dataset_and_tokenizer(data_path, new_data_path, model_path):
    # 1. 모델과 토크나이저 준비
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=256,
        add_eos_token=True,
        padding="longest",
    )
#    tokenizer.pad_token = tokenizer.eos_token

    # 특수 토큰 추가
    # special_tokens_dict = {
    #     'additional_special_tokens': [
    #         '<|begin_of_text|>', '<|end_of_text|>', '<|finetune_right_pad_id|>',
    #         '<|start_header_id|>', '<|end_header_id|>', '<|eom_id|>', '<|eot_id|>',
    #         '<|python_tag|>'
    #     ]
    # }
    # tokenizer.add_special_tokens(special_tokens_dict)

    # 2. 기존 데이터셋 준비
    original_data = pd.read_csv(data_path)

    # 3. 새로운 데이터셋 준비 (Idiom이 없는 경우)
    new_data = pd.read_csv(new_data_path)

    # 4. 다양한 시스템 프롬프트 정의 (영어)
    system_prompts_en = [
        "Detect if there is an idiom in the following sentence. If there is, return only the detected idiom. If there are no idioms, answer 'None'.",
        "Identify if the sentence contains any idioms. Provide only the idiom if found, or 'None' if there are no idioms.",
        "Please find any idioms in this sentence. If you detect one, write just the idiom. If there’s none, say 'None'.",
        "Check if the sentence has an idiom. If yes, respond only with the idiom; if not, reply 'None'.",
    ]

    user_templates = [
        'Is there an idiom in the sentence "{sentence}"? If yes, return only the detected idiom without any additional explanation or text.',
        'Could you check if there is any idiom in this sentence: "{sentence}"? If there’s an idiom, just provide it; if not, answer "None".',
        'Please detect any idioms in "{sentence}". If found, return just the idiom; otherwise, say "None".',
        "Please find any idioms in this sentence. If you detect one, write just the idiom. If there’s none, say 'None': ",
        "Check if the sentence has an idiom. If yes, respond only with the idiom; if not, reply 'None': ",
    ]

    assistant_templates = [
        "The idiom detected is: '{idiom}'",
        "Detected idiom: '{idiom}'",
        "There is no idiom here." if "{idiom}" == "None" else "The detected idiom is: '{idiom}'",
    ]

    # 5. 데이터셋 생성 함수
    def create_dataset(df, input_col, output_col):
        llm_df = pd.DataFrame()
        llm_df['input'] = df[input_col]
        llm_df['output'] = df[output_col]
        return Dataset.from_pandas(llm_df)

    # 6. 기존 및 새로운 데이터셋 생성
    train_dataset_original_en = create_dataset(original_data, 'Sentence', 'Idiom')
    train_dataset_original_kr = create_dataset(original_data, 'KR_Sentence', 'KR_Idiom')
    train_dataset_new_en = create_dataset(new_data, 'New_Example_EN', 'Label')
    train_dataset_new_kr = create_dataset(new_data, 'New_Example_KR', 'Label')

    # 7. 프롬프트 생성 및 토큰화 함수
    def generate_and_tokenize_prompt(data_batch, system_prompts, user_templates, assistant_templates, tokenizer):
        data = {'text': []}
        for inp, out in tqdm(zip(data_batch["input"], data_batch["output"])):
            # 랜덤 프롬프트 선택
            system_prompt = random.choice(system_prompts)
            user_prompt = random.choice(user_templates).format(sentence=inp)
            assistant_response = random.choice(assistant_templates).format(idiom=out if out is not None else "None")

            # 대화 형식으로 구성
            conversation = [
                #{'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
                {'role': 'assistant', 'content': assistant_response}
            ]

            # 템플릿 적용 및 토큰화
            templated = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                padding=True,
                max_length=200,
                truncation=True
            )
            data['text'].append(templated)
        
        return Dataset.from_dict(data)

    # 8. 데이터셋 매핑
    tokenized_train_dataset_original_en = generate_and_tokenize_prompt(
        train_dataset_original_en, system_prompts_en, user_templates, assistant_templates, tokenizer
    )
    tokenized_train_dataset_original_kr = generate_and_tokenize_prompt(
        train_dataset_original_kr, system_prompts_en, user_templates, assistant_templates, tokenizer
    )
    tokenized_train_dataset_new_en = generate_and_tokenize_prompt(
        train_dataset_new_en, system_prompts_en, user_templates, assistant_templates, tokenizer
    )
    tokenized_train_dataset_new_kr = generate_and_tokenize_prompt(
        train_dataset_new_kr, system_prompts_en, user_templates, assistant_templates, tokenizer
    )

    # 9. 데이터셋 결합 및 셔플
    tokenized_full_dataset_train = concatenate_datasets([
        tokenized_train_dataset_original_en,
        tokenized_train_dataset_original_kr,
        tokenized_train_dataset_new_en,
        tokenized_train_dataset_new_kr
    ])
    shuffled_tokenized_full_dataset_train = tokenized_full_dataset_train.shuffle(seed=22)

    # 10. 데이터셋 크기 출력
    print("전체 학습 데이터셋 크기:", len(shuffled_tokenized_full_dataset_train))
    print("None 데이터셋 크기:", len(tokenized_train_dataset_new_en) + len(tokenized_train_dataset_new_kr))

    # 샘플 템플릿 출력
    print("New English Dataset Sample:")
    print(shuffled_tokenized_full_dataset_train[0]["text"])

    # 11. 토크나이저와 셔플된 데이터셋 반환
    return shuffled_tokenized_full_dataset_train, tokenizer
