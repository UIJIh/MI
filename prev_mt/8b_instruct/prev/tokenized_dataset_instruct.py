import torch, random
import pandas as pd
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

def dataset_and_tokenizer(data_path, data_path2, model_path):
    # 1. 모델과 토크나이저 준비
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=100,
        add_eos_token=True,
        #padding="longest",  # 메모리 효율성을 높이기 위해 max_length 대신 longest 사용해보기
    )
    tokenizer.pad_token = tokenizer.eos_token

    # 특수 토큰 추가
    special_tokens_dict = {
        'additional_special_tokens': [
            '<|begin_of_text|>', '<|end_of_text|>', '<|finetune_right_pad_id|>',
            '<|start_header_id|>', '<|end_header_id|>', '<|eom_id|>', '<|eot_id|>',
            '<|python_tag|>'
        ]
    }
    tokenizer.add_special_tokens(special_tokens_dict)

    # 2. 데이터셋 준비
    data = pd.read_csv(data_path)

    # 번역 데이터셋 로드
    if isinstance(data_path2, pd.DataFrame):
        translation_df = data_path2[['source_string', 'target_string']]
    elif isinstance(data_path2, Dataset):
        translation_df = pd.DataFrame(data_path2)
        translation_df = translation_df[['source_string', 'target_string']]
    else:
        translation_data = pd.read_csv(data_path2)
        translation_df = translation_data[['source_string', 'target_string']]

    # 데이터셋 분할 (학습/평가)
    train_translation_df, eval_translation_df = train_test_split(translation_df, test_size=0.15, random_state=1)

    # 3. 시스템 프롬프트 정의
    # 한국어 버전
    system_prompt_idiom_kr = "너는 관용구에 대한 깊은 지식을 가진 전문가야. 너의 역할은 사용자에게 관용구를 정확하고 자세하게 설명해주는 것이야."
    system_prompt_guess_idiom_kr = "너는 관용구에 대한 깊은 지식을 가진 전문가야. 너의 역할은 사용자에게 정확한 관용구를 알려주는 것이야."

    # 영어 버전
    system_prompt_idiom_en = "You are an expert with deep knowledge of idioms. Your role is to provide the user with an accurate and detailed explanation of the idiom."
    system_prompt_guess_idiom_en = "You are an expert with deep knowledge of idioms. Your role is to provide the user with the correct idiom."

    ####################### 수정된 사항!
    # 한국어 버전
    system_prompt_idiom_kr_v2 = "너는 관용구 전문가야. 너의 임무는 사용자에게 관용구의 의미를 정확하게 전달하는 거야."
    system_prompt_guess_idiom_kr_v2 = "너는 관용구를 잘 아는 전문가야. 사용자가 물어보는 관용구를 맞추는 역할을 해줘."

    # 영어 버전
    system_prompt_idiom_en_v2 = "You are a specialist in idioms. Your task is to provide the user with the most accurate and detailed explanation of the idiom."
    system_prompt_guess_idiom_en_v2 = "You are an idiom expert. Your goal is to identify the idiom based on the meaning provided by the user."


    # 번역 프롬프트
    # system_prompt_translate_en_to_kr = [
    #     "You are a professional translator proficient in Korean and English. Please translate the given English sentence into Korean accurately.",
    #     "당신은 한국어와 영어에 능통한 전문 번역가입니다. 주어진 영어 문장을 한국어로 정확하게 번역해 주세요."
    # ]
    # system_prompt_translate_kr_to_en = [
    #     "You are a professional translator proficient in Korean and English. Please translate the given Korean sentence into English accurately.",
    #     "당신은 한국어와 영어에 능통한 전문 번역가입니다. 주어진 한국어 문장을 영어로 정확하게 번역해 주세요."
    # ]
    system_prompt_translate_en_to_kr = [
    #"You are a professional translator proficient in Korean and English. Please translate the given English sentence into Korean accurately.",
    "당신은 한국어와 영어에 능통한 전문 번역가입니다. 주어진 영어 문장을 한국어로 정확하게 번역해 주세요.",
    #"As a highly skilled translator proficient in both Korean and English, please provide a precise translation of the following English sentence into Korean.",
    #"Please translate the following English sentence into Korean in a formal style, ensuring accuracy and clarity.",
    "한국어와 영어에 능통한 전문 번역가로서 다음 영어 문장을 정확하고 명확하게 한글로 번역해 주세요.",
    #"Translate the following sentence from English to Korean, keeping cultural nuances and context in mind for a natural translation."
    ]

    system_prompt_translate_kr_to_en = [
        "You are a professional translator proficient in Korean and English. Please translate the given Korean sentence into English accurately.",
        #"당신은 한국어와 영어에 능통한 전문 번역가입니다. 주어진 한국어 문장을 영어로 정확하게 번역해 주세요.",
        #"Please translate the following Korean sentence into English while considering cultural nuances for a natural translation.",
        #"Translate the following Korean sentence into English in a formal style, ensuring accuracy and clarity.",
        #"한국어와 영어에 능통한 전문 번역가로서 다음 한국어 문장을 정확하고 명확하게 영어로 번역하세요."
        "As a highly skilled translator proficient in both Korean and English, please provide a precise translation of the following Korean sentence into English.",
    ]

    # 4. 프롬프트 템플릿 정의
    prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

{user_input}<|eot_id|>
<|start_header_id|>assistant<|end_header_id>
"""

    # 5. 데이터셋 생성 함수
    def create_dataset(df, input_col, output_col):
        llm_df = pd.DataFrame()
        llm_df['input'] = df[input_col]
        llm_df['output'] = df[output_col]
        return Dataset.from_pandas(llm_df)

    # 6. 학습 데이터셋 생성

    # (1) 한국어 프롬프트로 관용구를 주면 의미를 설명하도록 하기 (한국어 데이터셋)
    train_dataset_idiom_kr = create_dataset(data, 'KR_Idiom', 'KR_Meaning')

    # (2) 한국어 프롬프트로 의미를 주면 관용구를 맞추도록 하기 (한국어 데이터셋)
    train_dataset_guess_idiom_kr = create_dataset(data, 'KR_Meaning', 'KR_Idiom')

    # (3) 영어 프롬프트로 관용구를 주면 의미를 설명하도록 하기 (영어 데이터셋)
    train_dataset_idiom_en = create_dataset(data, 'Idiom', 'Meaning')

    # (4) 영어 프롬프트로 의미를 주면 관용구를 맞추도록 하기 (영어 데이터셋)
    train_dataset_guess_idiom_en = create_dataset(data, 'Meaning', 'Idiom')

    # (5) 한국어 문장을 영어로 번역하기
    train_dataset_translation_kr_to_en = create_dataset(train_translation_df, 'target_string', 'source_string')

    # (6) 영어 문장을 한국어로 번역하기
    train_dataset_translation_en_to_kr = create_dataset(train_translation_df, 'source_string', 'target_string')

    # 7. 프롬프트 생성 및 토크나이즈 함수
    def generate_and_tokenize_prompt(data_batch, system_prompt, tokenizer, is_korean=False, is_translation=False, print_sample=True):
        inputs = data_batch["input"]
        outputs = data_batch["output"]

        prompts = []
        for inp in inputs:
            if is_korean:
                # 한국어 관용구 프롬프트
                if system_prompt == system_prompt_idiom_kr:
                    user_input = f'"{inp}"라는 관용구의 뜻은 뭐야?'
                else:
                    # 의미를 제공하고 관용구를 추측하게 함
                    user_input = f'"{inp}"라는 의미를 가진 관용구는 무엇인가요?'
            elif is_translation:
            # 번역 프롬프트 적용
                if "Korean sentence into English" in system_prompt: 
                    user_input = f"Could you translate the following Korean sentence into English? \"{inp}\""                                
                else:  
                    user_input = f"이 문장을 한국어로 번역해 주시겠어요? \"{inp}\""        
            else:
                # 영어 관용구 프롬프트
                if system_prompt == system_prompt_idiom_en:
                    user_input = f'What does the idiom "{inp}" mean?'
                else:
                    user_input = f'What is the idiom that means "{inp}"?'

            # 전체 프롬프트 생성
            prompt = prompt_template.format(
                system_prompt=system_prompt,
                user_input=user_input
            )
            prompts.append(prompt)

        # 전체 입력(프롬프트 + 모델 응답)을 생성
        answers = [out + tokenizer.eos_token for out in outputs]
        texts = [p + a for p, a in zip(prompts, answers)]

        # 토크나이즈
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=100,
            padding="max_length",
        )

        # 프롬프트 길이 계산
        prompt_lengths = [len(tokenizer(p, truncation=True, max_length=100)["input_ids"]) for p in prompts]

        # 레이블 생성 및 프롬프트 부분 마스킹
        labels = []
        for input_ids, prompt_len in zip(tokenized["input_ids"], prompt_lengths):
            label = input_ids.copy()
            label[:prompt_len] = [-100] * prompt_len
            labels.append(label)

        tokenized["labels"] = labels   
        # 마스킹 처리된 입력과 레이블을 프린트
        if print_sample:
            sample_idx = 10
            print("===== 샘플 출력 =====")
            print("토큰화된 입력 텍스트:", tokenizer.decode(tokenized["input_ids"][sample_idx], skip_special_tokens=False))
            print("토큰화된 레이블 (마스킹 적용됨):", tokenized["labels"][sample_idx])
            print("프롬프트 길이:", prompt_lengths[sample_idx])
            print("====================")
        return tokenized

        # # 레이블 생성 및 프롬프트 부분 마스킹
        # labels = []
        # for input_ids in tokenized["input_ids"]:
        #     assistant_token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('<|start_header_id|>assistant<|end_header_id>'))
        #     assistant_index = None
        #     for i in range(len(input_ids) - len(assistant_token_ids) + 1):
        #         if input_ids[i:i+len(assistant_token_ids)] == assistant_token_ids:
        #             assistant_index = i + len(assistant_token_ids)
        #             break
        #     if assistant_index is None:
        #         assistant_index = 0

        #     # 레이블 생성
        #     label = input_ids.copy()
        #     label[:assistant_index] = [-100] * assistant_index
        #     labels.append(label)

        # tokenized["labels"] = labels
        # return tokenized

    # 8. 학습 데이터셋 매핑

    # (1) 한국어 프롬프트로 관용구를 주면 의미를 설명하도록 하기
    tokenized_train_dataset_idiom_kr = train_dataset_idiom_kr.map(
        lambda x: generate_and_tokenize_prompt(x, system_prompt_idiom_kr, tokenizer, is_korean=True),
        remove_columns=train_dataset_idiom_kr.column_names, batched=True
    )

    # (2) 한국어 프롬프트로 의미를 주면 관용구를 맞추도록 하기
    tokenized_train_dataset_guess_idiom_kr = train_dataset_guess_idiom_kr.map(
        lambda x: generate_and_tokenize_prompt(x, system_prompt_guess_idiom_kr, tokenizer, is_korean=True),
        remove_columns=train_dataset_guess_idiom_kr.column_names, batched=True
    )
    # (3) (4) 한국어 데이터셋에 대해
    # tokenized_train_dataset_idiom_kr_2 = train_dataset_idiom_kr.map(
    #     lambda x: generate_and_tokenize_prompt(x, system_prompt_idiom_en, tokenizer),
    #     remove_columns=train_dataset_idiom_kr.column_names, batched=True
    # )
    # tokenized_train_dataset_guess_idiom_kr_2 = train_dataset_guess_idiom_kr.map(
    #     lambda x: generate_and_tokenize_prompt(x, system_prompt_guess_idiom_en, tokenizer),
    #     remove_columns=train_dataset_guess_idiom_kr.column_names, batched=True
    # )
    # (1) (2) 영어 데이터셋에 대해
    # tokenized_train_dataset_idiom_en = train_dataset_idiom_en.map(
    #     lambda x: generate_and_tokenize_prompt(x, system_prompt_idiom_kr, tokenizer, is_korean=True),
    #     remove_columns=train_dataset_idiom_en.column_names, batched=True
    # )
    # tokenized_train_dataset_guess_idiom_en = train_dataset_guess_idiom_en.map(
    #     lambda x: generate_and_tokenize_prompt(x, system_prompt_guess_idiom_kr, tokenizer, is_korean=True),
    #     remove_columns=train_dataset_guess_idiom_en.column_names, batched=True
    # )

    # (3) 영어 프롬프트로 관용구를 주면 의미를 설명하도록 하기
    tokenized_train_dataset_idiom_en_2 = train_dataset_idiom_en.map(
        lambda x: generate_and_tokenize_prompt(x, system_prompt_idiom_en, tokenizer),
        remove_columns=train_dataset_idiom_en.column_names, batched=True
    )

    # (4) 영어 프롬프트로 의미를 주면 관용구를 맞추도록 하기
    tokenized_train_dataset_guess_idiom_en_2 = train_dataset_guess_idiom_en.map(
        lambda x: generate_and_tokenize_prompt(x, system_prompt_guess_idiom_en, tokenizer),
        remove_columns=train_dataset_guess_idiom_en.column_names, batched=True
    )

    # (5) 한국어 문장을 영어로 번역하기
    tokenized_train_dataset_translation_kr_to_en = train_dataset_translation_kr_to_en.map(
        lambda x: generate_and_tokenize_prompt(
            x,
            random.choice(system_prompt_translate_kr_to_en),  # 프롬프트 리스트에서 무작위 선택
            tokenizer,
            is_translation=True
        ),
        remove_columns=train_dataset_translation_kr_to_en.column_names,
        batched=True
    )

    # (6) 영어 문장을 한국어로 번역하기
    tokenized_train_dataset_translation_en_to_kr = train_dataset_translation_en_to_kr.map(
        lambda x: generate_and_tokenize_prompt(
            x,
            random.choice(system_prompt_translate_en_to_kr),  # 프롬프트 리스트에서 무작위 선택
            tokenizer,
            is_translation=True
        ),
        remove_columns=train_dataset_translation_en_to_kr.column_names,
        batched=True
    )


    # 8. 학습 데이터셋 매핑

    # (1) 한국어 프롬프트로 관용구를 주면 의미를 설명하도록 하기 (새로운 버전)
    tokenized_train_dataset_idiom_kr_v2 = train_dataset_idiom_kr.map(
        lambda x: generate_and_tokenize_prompt(x, system_prompt_idiom_kr_v2, tokenizer, is_korean=True),
        remove_columns=train_dataset_idiom_kr.column_names, batched=True
    )
    # (2) 한국어 프롬프트로 의미를 주면 관용구를 맞추도록 하기 (새로운 버전)
    tokenized_train_dataset_guess_idiom_kr_v2 = train_dataset_guess_idiom_kr.map(
        lambda x: generate_and_tokenize_prompt(x, system_prompt_guess_idiom_kr_v2, tokenizer, is_korean=True),
        remove_columns=train_dataset_guess_idiom_kr.column_names, batched=True
    )
    # (3) 영어 프롬프트로 관용구를 주면 의미를 설명하도록 하기 (새로운 버전)
    tokenized_train_dataset_idiom_en_v2 = train_dataset_idiom_en.map(
        lambda x: generate_and_tokenize_prompt(x, system_prompt_idiom_en_v2, tokenizer),
        remove_columns=train_dataset_idiom_en.column_names, batched=True
    )
    # (4) 영어 프롬프트로 의미를 주면 관용구를 맞추도록 하기 (새로운 버전)
    tokenized_train_dataset_guess_idiom_en_v2 = train_dataset_guess_idiom_en.map(
        lambda x: generate_and_tokenize_prompt(x, system_prompt_guess_idiom_en_v2, tokenizer),
        remove_columns=train_dataset_guess_idiom_en.column_names, batched=True
    )
    # # (1) 한국어 프롬프트로 영어 관용구를 주면 의미를 설명하도록 하기
    # tokenized_train_dataset_idiom_en_with_kr_prompt = train_dataset_idiom_en.map(
    #     lambda x: generate_and_tokenize_prompt(x, system_prompt_idiom_kr_v2, tokenizer, is_korean=True),
    #     remove_columns=train_dataset_idiom_en.column_names, batched=True
    # )
    # # (2) 한국어 프롬프트로 영어 의미를 주면 관용구를 맞추도록 하기
    # tokenized_train_dataset_guess_idiom_en_with_kr_prompt = train_dataset_guess_idiom_en.map(
    #     lambda x: generate_and_tokenize_prompt(x, system_prompt_guess_idiom_kr_v2, tokenizer, is_korean=True),
    #     remove_columns=train_dataset_guess_idiom_en.column_names, batched=True
    # )
    # # (3) 영어 프롬프트로 한국어 관용구를 주면 의미를 설명하도록 하기
    # tokenized_train_dataset_idiom_kr_with_en_prompt = train_dataset_idiom_kr.map(
    #     lambda x: generate_and_tokenize_prompt(x, system_prompt_idiom_en_v2, tokenizer),
    #     remove_columns=train_dataset_idiom_kr.column_names, batched=True
    # )
    # # (4) 영어 프롬프트로 한국어 의미를 주면 관용구를 맞추도록 하기
    # tokenized_train_dataset_guess_idiom_kr_with_en_prompt = train_dataset_guess_idiom_kr.map(
    #     lambda x: generate_and_tokenize_prompt(x, system_prompt_guess_idiom_en_v2, tokenizer),
    #     remove_columns=train_dataset_guess_idiom_kr.column_names, batched=True
#)

    # 9. 모든 토크나이즈된 학습 데이터셋을 결합
    tokenized_full_dataset = concatenate_datasets([
        tokenized_train_dataset_idiom_kr,
        tokenized_train_dataset_guess_idiom_kr,
        # tokenized_train_dataset_idiom_kr_2,
        # tokenized_train_dataset_guess_idiom_kr_2,
        # tokenized_train_dataset_idiom_en,
        # tokenized_train_dataset_guess_idiom_en,
        tokenized_train_dataset_idiom_en_2,
        tokenized_train_dataset_guess_idiom_en_2,
        tokenized_train_dataset_translation_kr_to_en,
        tokenized_train_dataset_translation_en_to_kr,
        tokenized_train_dataset_idiom_kr_v2,  # 새로 추가된 데이터셋
        tokenized_train_dataset_guess_idiom_kr_v2,  # 새로 추가된 데이터셋
        tokenized_train_dataset_idiom_en_v2,  # 새로 추가된 데이터셋
        tokenized_train_dataset_guess_idiom_en_v2,  # 새로 추가된 데이터셋
        # tokenized_train_dataset_idiom_en_with_kr_prompt,  # 교차 데이터셋 (한국어 프롬프트, 영어 관용구)
        # tokenized_train_dataset_guess_idiom_en_with_kr_prompt,  # 교차 데이터셋 (한국어 프롬프트, 영어 의미)
        # tokenized_train_dataset_idiom_kr_with_en_prompt,  # 교차 데이터셋 (영어 프롬프트, 한국어 관용구)
        # tokenized_train_dataset_guess_idiom_kr_with_en_prompt  # 교차 데이터셋 (영어 프롬프트, 한국어 의미)
    ])

    # 10. 평가 데이터셋 준비

    # 평가용 번역 데이터셋 생성
    # eval_dataset_translation_en_to_kr = create_dataset(eval_translation_df, 'source_string', 'target_string')
    # eval_dataset_translation_kr_to_en = create_dataset(eval_translation_df, 'target_string', 'source_string')

    # 11. 평가 데이터셋 매핑

    # (1) 한국어 문장을 영어로 번역하기
    # tokenized_eval_translation_dataset_kr_to_en = eval_dataset_translation_kr_to_en.map(
    #     lambda x: generate_and_tokenize_prompt(x, system_prompt_translate_kr_to_en, tokenizer),
    #     remove_columns=eval_dataset_translation_kr_to_en.column_names, batched=True
    # )

    # # (2) 영어 문장을 한국어로 번역하기
    # tokenized_eval_translation_dataset_en_to_kr = eval_dataset_translation_en_to_kr.map(
    #     lambda x: generate_and_tokenize_prompt(x, system_prompt_translate_en_to_kr, tokenizer),
    #     remove_columns=eval_dataset_translation_en_to_kr.column_names, batched=True
    # )

    # 12. 모든 토크나이즈된 평가 데이터셋을 결합
    # tokenized_full_dataset_eval = concatenate_datasets([
    #     tokenized_eval_translation_dataset_kr_to_en,
    #     tokenized_eval_translation_dataset_en_to_kr
    # ])

    # 13. 데이터셋 셔플
    shuffled_tokenized_full_dataset = tokenized_full_dataset.shuffle(seed=1)
    #shuffled_tokenized_full_dataset_eval = tokenized_full_dataset_eval.shuffle(seed=1)
    shuffled_tokenized_full_dataset_eval = None
    # 데이터셋 크기 출력
    print("번역 데이터셋 크기:", len(tokenized_train_dataset_translation_kr_to_en) + len(tokenized_train_dataset_translation_en_to_kr))
    print("학습 데이터셋 크기:", len(shuffled_tokenized_full_dataset))
    #print("평가 데이터셋 크기:", len(shuffled_tokenized_full_dataset_eval))

    # 14. 토크나이저와 셔플된 데이터셋 반환
    return shuffled_tokenized_full_dataset, shuffled_tokenized_full_dataset_eval, tokenizer
