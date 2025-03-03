import pandas as pd
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import random

def dataset_and_tokenizer(data_path, data_path2, model_path):
    # 1. 모델과 토크나이저 준비
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        #model_max_length=256,
        add_eos_token=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

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
    # eval 데이터셋
    #train_translation_df, eval_translation_df = train_test_split(translation_df, test_size=0.01, random_state=42)
    train_translation_df = translation_df
    eval_translation_df = None
    # 3. 템플릿 정의 (영어 및 한국어)
    prompt_templates = {
        'en_forward': [
            "What does the idiom '{idiom}' mean?",
            "Please explain the meaning of the following idiom: '{idiom}'.",
            #"The idiom '{idiom}' means ",
            # "Explain the idiom '{idiom}' in your own words.",
            # "What is the interpretation of the idiom: '{idiom}'?"
        ],
        'en_reverse': [
            "What idiom matches the meaning: '{meaning}'?",
            "Which idiom corresponds to the meaning '{meaning}'?",
            #"The idiom that has the meaning '{meaning}' is ",
            # "What is the idiom that fits the meaning: '{meaning}'?",
            # "What phrase describes the meaning '{meaning}'?"
        ],
        'kr_forward': [
            "다음 관용표현의 의미를 설명하세요: '{idiom}'.",
            "관용구 '{idiom}'의 의미는 무엇인가요?",
            #"관용구 '{idiom}'의 뜻은 ",
            # "관용구 '{idiom}'의 뜻을 해석해 주세요.",
            # "'{idiom}'이 의미하는 것은 무엇입니까?"
        ],
        'kr_reverse': [
            "다음 뜻을 가진 관용구를 찾아보세요: '{meaning}'.",
            "'{meaning}'에 해당하는 관용구는 무엇입니까?",
            #"'{meaning}'라는 뜻을 갖는 관용구는 ",
            # "'{meaning}'에 맞는 관용구는 무엇인가요?",
            # "이 의미에 맞는 관용표현은 무엇인가요? '{meaning}'"
        ]
    }
    
    # 번역 관련 템플릿
    # prompt_template_translation = [
    #     "Translate the following English sentence into Korean:\n\n# English Sentence:\n{source}\n\n# Translation:\n",
    #     "What is the Korean translation of the following English sentence: '{source}'?",
    #     #"Please translate this English sentence into Korean.\n\n# English Sentence:\n{source}\n\n# Translation:\n"
    # ]

    # prompt_template_translation_reverse = [
    #     "Translate the following Korean sentence into English:\n\n# Korean Sentence:\n{source}\n\n# Translation:\n",
    #     "What is the English translation of the following Korean sentence: '{source}'?",
    #     #"Can you translate this Korean sentence to English?\n\n# Korean Sentence:\n{source}\n\n# Translation:\n"
    # ]
    # 번역 관련 템플릿 (영어 -> 한국어)
    prompt_template_translation = [
        #"Translate the following sentence into Korean: {source}",
        "'{source}'라는 문장을 한국어로 번역해 주세요.",
    ]
    # 번역 관련 템플릿 (한국어 -> 영어)
    prompt_template_translation_reverse = [
        "Translate the following sentence into English: {source}",
        #"'{source}'를 영어로 번역해 주세요.",
    ]

    # 4. 데이터셋 생성
    def create_train_dataset(df, input_col, output_col):
        llm_df = pd.DataFrame()
        llm_df['input'] = df[input_col]
        llm_df['output'] = df[output_col]
        return Dataset.from_pandas(llm_df)

    # 5. 프롬프트와 답변을 토크나이즈하고 레이블을 생성하는 함수
    def generate_and_tokenize(data_batch, prompt_templates, tokenizer, input_key, max_length=100, is_translation=False, print_sample=True):
        inputs = data_batch["input"]
        outputs = data_batch["output"]
        
        all_tokenized = []

        if is_translation:
            # 번역일 경우 템플릿을 랜덤으로 선택
            prompts = [random.choice(prompt_templates).format(**{input_key: inp}) for inp in inputs]
            
            # 전체 입력(프롬프트 + 답변)을 생성
            texts = [p + a for p, a in zip(prompts, outputs)]

            # 토크나이즈 및 레이블 생성
            tokenized = tokenizer(texts, padding='max_length', truncation=True, max_length=max_length)
            prompt_lengths = [len(tokenizer(p, truncation=True)["input_ids"]) for p in prompts]
            labels = []
            for input_ids, prompt_len in zip(tokenized["input_ids"], prompt_lengths):
                label = input_ids.copy()
                label[:prompt_len] = [-100] * prompt_len
                labels.append(label)

            tokenized["labels"] = labels
            all_tokenized.append(Dataset.from_dict(tokenized))
            
        else:
            # 이디엄일 경우 모든 템플릿 적용
            for prompt_template in prompt_templates:
                prompts = [prompt_template.format(**{input_key: inp}) for inp in inputs]
                texts = [p + a for p, a in zip(prompts, outputs)]
                # tokenized = tokenizer(
                #     texts,
                #     truncation=True,
                #     padding=True, 
                #     max_length=max_length,
                # )
                tokenized = tokenizer(texts, padding='max_length', truncation=True, max_length=max_length) # padding size longest도 안됨,지정안해주면 https://discuss.huggingface.co/t/how-to-solve-valueerror-expected-sequence-of-length-15-at-dim-1-got-18-error-in-python/33423/2
                prompt_lengths = [len(tokenizer(p, truncation=True)["input_ids"]) for p in prompts]
                labels = []
                for input_ids, prompt_len in zip(tokenized["input_ids"], prompt_lengths):
                    label = input_ids.copy()
                    label[:prompt_len] = [-100] * prompt_len
                    labels.append(label)

                tokenized["labels"] = labels
                all_tokenized.append(Dataset.from_dict(tokenized))

        # 마스킹 처리된 입력과 레이블을 프린트
        if print_sample:
            sample_idx = 10
            print("===== 샘플 출력 =====")
            print("토큰화된 입력 텍스트:", tokenizer.decode(tokenized["input_ids"][sample_idx], skip_special_tokens=False))
            print("토큰화된 레이블 (마스킹 적용됨):", tokenized["labels"][sample_idx])
            print("프롬프트 길이:", prompt_lengths[sample_idx])
            print("====================")
        
        return concatenate_datasets(all_tokenized)

    
    # 6. 매핑    
    # 이디엄 데이터셋 생성 및 매핑 (영어)
    train_dataset = create_train_dataset(data, 'Idiom', 'Meaning')
    train_dataset_reverse = create_train_dataset(data, 'Meaning', 'Idiom')
    tokenized_train_dataset_en = generate_and_tokenize(train_dataset, prompt_templates['en_forward'], tokenizer, input_key='idiom')
    tokenized_train_dataset_reverse_en = generate_and_tokenize(train_dataset_reverse, prompt_templates['en_reverse'], tokenizer, input_key='meaning')
    # (한국어 프롬프트)
    # tokenized_train_dataset_en_2 = generate_and_tokenize(train_dataset, prompt_templates['kr_forward'], tokenizer, input_key='idiom')
    # tokenized_train_dataset_reverse_en_2 = generate_and_tokenize(train_dataset_reverse, prompt_templates['kr_reverse'], tokenizer, input_key='meaning')
    # 한국어
    train_dataset_kr = create_train_dataset(data, 'KR_Idiom', 'KR_Meaning')
    train_dataset_reverse_kr = create_train_dataset(data, 'KR_Meaning', 'KR_Idiom')
    tokenized_train_dataset_kr = generate_and_tokenize(train_dataset_kr, prompt_templates['kr_forward'], tokenizer, input_key='idiom')    
    tokenized_train_dataset_reverse_kr = generate_and_tokenize(train_dataset_reverse_kr, prompt_templates['kr_reverse'], tokenizer, input_key='meaning')
    # (영어 프름프트)
    # tokenized_train_dataset_kr_2 = generate_and_tokenize(train_dataset_kr, prompt_templates['en_forward'], tokenizer, input_key='idiom')    
    # tokenized_train_dataset_reverse_kr_2 = generate_and_tokenize(train_dataset_reverse_kr, prompt_templates['en_reverse'], tokenizer, input_key='meaning')

    # 번역 데이터셋 생성 및 매핑 
    translation_dataset = create_train_dataset(train_translation_df, 'source_string', 'target_string')
    tokenized_translation_dataset = generate_and_tokenize(translation_dataset, prompt_template_translation, tokenizer, input_key='source', is_translation=True)
    translation_dataset_reverse = create_train_dataset(train_translation_df, 'target_string', 'source_string') 
    tokenized_translation_dataset_reverse = generate_and_tokenize(translation_dataset_reverse, prompt_template_translation_reverse, tokenizer, input_key='source', is_translation=True)

    # 평가 데이터셋 생성 및 매핑
    # eval_translation_dataset = create_train_dataset(eval_translation_df, 'source_string', 'target_string')
    # eval_translation_dataset_reverse = create_train_dataset(eval_translation_df, 'target_string', 'source_string')
    # tokenized_eval_translation_dataset = generate_and_tokenize(eval_translation_dataset, prompt_template_translation, tokenizer, input_key='source', is_translation=True)
    # tokenized_eval_translation_dataset_reverse = generate_and_tokenize(eval_translation_dataset_reverse, prompt_template_translation_reverse, tokenizer, input_key='source', is_translation=True)

    # 7. 모든 토크나이즈된 데이터셋을 결합
    tokenized_full_dataset = concatenate_datasets([
        tokenized_train_dataset_en,
        tokenized_train_dataset_reverse_en,
        tokenized_train_dataset_kr,
        tokenized_train_dataset_reverse_kr,
        tokenized_translation_dataset,
        tokenized_translation_dataset_reverse,
        # tokenized_train_dataset_en_2,
        # tokenized_train_dataset_reverse_en_2,
        # tokenized_train_dataset_kr_2,
        # tokenized_train_dataset_reverse_kr_2
    ])    
    tokenized_full_dataset = tokenized_full_dataset.shuffle(seed=42)

    # 평가용 번역 데이터셋 결합
    # tokenized_eval_full_dataset = concatenate_datasets([
    #     tokenized_eval_translation_dataset,
    #     tokenized_eval_translation_dataset_reverse,
    # ])
    tokenized_eval_full_dataset = None
    # 데이터셋 크기 출력
    idiom_dataset_size = len(tokenized_train_dataset_en) + len(tokenized_train_dataset_reverse_en) + len(tokenized_train_dataset_kr) + len(tokenized_train_dataset_reverse_kr) #+ len(tokenized_train_dataset_en_2) + len(tokenized_train_dataset_reverse_en_2) + len(tokenized_train_dataset_kr_2) + len(tokenized_train_dataset_reverse_kr_2)
    translation_dataset_size = len(tokenized_translation_dataset) + len(tokenized_translation_dataset_reverse)
    #eval_translation_dataset_size = len(tokenized_eval_translation_dataset) + len(tokenized_eval_translation_dataset_reverse)
  
    print("전체 학습 데이터셋 크기:", len(tokenized_full_dataset))
    print(f"Idiom 학습 데이터셋 크기: {idiom_dataset_size}")
    print(f"번역 학습 데이터셋 크기: {translation_dataset_size}")
    #print(f"평가 데이터셋 크기: {eval_translation_dataset_size}")

    return tokenized_full_dataset, tokenized_eval_full_dataset, tokenizer