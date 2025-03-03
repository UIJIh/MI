import torch
import pandas as pd
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

def dataset_and_tokenizer(data_path, data_path2, model_path):
    # 1. 모델과 토크나이저 준비
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=512,
        add_eos_token=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # 2. 데이터셋 준비
    data = pd.read_csv(data_path)
    llm_df = data[['Idiom', 'Meaning', 'KR_Idiom', 'KR_Meaning']]
    # 번역 데이터셋 로드
    if isinstance(data_path2, pd.DataFrame):
        translation_df = data_path2[['source_string', 'target_string']]
    elif isinstance(data_path2, Dataset):
        translation_df = pd.DataFrame(data_path2)
        translation_df = translation_df[['source_string', 'target_string']]
    else:
        # data_path2가 파일 경로인 경우
        translation_data = pd.read_csv(data_path2)
        translation_df = translation_data[['source_string', 'target_string']]
    train_translation_df, eval_translation_df = train_test_split(translation_df, test_size=0.2, random_state=42)

    # 3. 프롬프트 템플릿 정의
    prompt_template_forward = """What is the meaning of the following idiom?

    # Idiom:
    {idiom}

    # Meaning:
    """
    prompt_template_reverse = """What is the idiom that has the following meaning?

    # Meaning:
    {meaning}

    # Idiom:
    """
    prompt_template_forward_2 = """The idiom "{idiom}" means """
    prompt_template_reverse_2 = """The idiom that has the meaning "{meaning}" is """
    # # 한국어 프롬프트 템플릿
    prompt_template_forward_kr = """다음 관용구의 의미는 무엇인가요?

    # 관용구:
    {idiom}

    # 의미:
    """
    prompt_template_reverse_kr = """다음 의미를 가진 관용구는 무엇인가요?

    # 의미:
    {meaning}

    # 관용구:
    """
    prompt_template_forward_kr_2 = """관용구 "{idiom}"의 뜻은 """
    prompt_template_reverse_kr_2 = """"{meaning}"라는 뜻을 갖는 관용구는 """
    
    prompt_template_translation_reverse = """Translate the following Korean sentence into English:

    # Korean Sentence:
    {source}

    # Translation:
    """
    prompt_template_translation = """Translate the following English sentence into Korean:

    # English Sentence:
    {source}

    # Translation:
    """
    prompt_template_translation_kr = """다음 영어 문장을 한국어로 번역해줘.

    # 영어 문장:
    {source}

    # 번역:
    """
    prompt_template_translation_reverse_kr = """다음 한국어 문장을 영어로 번역해줘.

    # 한국어 문장:
    {source}

    # 번역:
    """
    
    # 4. 데이터셋 생성 함수
    def create_train_dataset(df, input_col, output_col):
        llm_df = pd.DataFrame()
        llm_df['input'] = df[input_col]
        llm_df['output'] = df[output_col]
        return Dataset.from_pandas(llm_df)
    # 번역 데이터셋 생성 함수
    def create_translation_dataset(df, input_col, output_col):
        llm_df = pd.DataFrame()
        llm_df['input'] = df[input_col]
        llm_df['output'] = df[output_col]
        return Dataset.from_pandas(llm_df)

    translation_dataset = create_translation_dataset(train_translation_df, 'source_string', 'target_string')
    translation_dataset_reverse = create_translation_dataset(train_translation_df, 'target_string', 'source_string')
    train_dataset = create_train_dataset(data, 'Idiom', 'Meaning')
    train_dataset_reverse = create_train_dataset(data, 'Meaning', 'Idiom')
    train_dataset_kr = create_train_dataset(data, 'KR_Idiom', 'KR_Meaning')
    train_dataset_reverse_kr = create_train_dataset(data, 'KR_Meaning', 'KR_Idiom')

    # 5. 프롬프트와 답변을 토크나이즈하고 레이블을 생성하는 함수
    def generate_and_tokenize_prompt(data_batch, prompt_template, tokenizer, input_key):
        inputs = data_batch["input"]
        outputs = data_batch["output"]

        prompts = [prompt_template.format(**{input_key: inp}) for inp in inputs]
        answers = [out + tokenizer.eos_token for out in outputs]

        # 전체 입력(프롬프트 + 답변)을 생성
        texts = [p + a for p, a in zip(prompts, answers)]

        # 텍스트를 토크나이즈
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=512,
            padding="max_length",
        )

        # 프롬프트 길이 계산
        prompt_lengths = [len(tokenizer(p, truncation=True, max_length=512)["input_ids"]) for p in prompts]

        # 레이블 생성 및 프롬프트 부분 마스킹
        labels = []
        for input_ids, prompt_len in zip(tokenized["input_ids"], prompt_lengths):
            label = input_ids.copy()
            label[:prompt_len] = [-100] * prompt_len
            labels.append(label)

        tokenized["labels"] = labels
        return tokenized

    # 번역 데이터셋 프롬프트 생성 및 토크나이즈
    def tokenize_translation(data_batch, prompt_template, tokenizer, input_key):
        inputs = data_batch["input"]
        outputs = data_batch["output"]

        prompts = [prompt_template.format(**{input_key: inp}) for inp in inputs]
        answers = [out + tokenizer.eos_token for out in outputs]

        # 전체 입력(프롬프트 + 답변)을 생성
        texts = [p + a for p, a in zip(prompts, answers)]

        # 텍스트를 토크나이즈
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=512,
            padding="max_length",
        )

        # 프롬프트 길이 계산
        prompt_lengths = [len(tokenizer(p, truncation=True, max_length=512)["input_ids"]) for p in prompts]

        # 레이블 생성 및 프롬프트 부분 마스킹
        labels = []
        for input_ids, prompt_len in zip(tokenized["input_ids"], prompt_lengths):
            label = input_ids.copy()
            label[:prompt_len] = [-100] * prompt_len
            labels.append(label)

        tokenized["labels"] = labels
        return tokenized

    # 6. 데이터셋 매핑
    # 영어 데이터셋 매핑1(en-in)
    tokenized_train_dataset_en = train_dataset.map(
        lambda x: generate_and_tokenize_prompt(x, prompt_template_forward, tokenizer, input_key='idiom'),
        remove_columns=train_dataset.column_names, batched=True
    )
    tokenized_train_dataset_reverse_en = train_dataset_reverse.map(
        lambda x: generate_and_tokenize_prompt(x, prompt_template_reverse, tokenizer, input_key='meaning'),
        remove_columns=train_dataset_reverse.column_names, batched=True
    )
    # type2
    tokenized_train_dataset_en_2 = train_dataset.map(
        lambda x: generate_and_tokenize_prompt(x, prompt_template_forward_2, tokenizer, input_key='idiom'),
        remove_columns=train_dataset.column_names, batched=True
    )
    tokenized_train_dataset_reverse_en_2 = train_dataset_reverse.map(
        lambda x: generate_and_tokenize_prompt(x, prompt_template_reverse_2, tokenizer, input_key='meaning'),
        remove_columns=train_dataset_reverse.column_names, batched=True
    )

    # 한국어 데이터셋 매핑1(en-in)
    tokenized_train_dataset_kr = train_dataset_kr.map(
        lambda x: generate_and_tokenize_prompt(x, prompt_template_forward, tokenizer, input_key='idiom'),
        remove_columns=train_dataset_kr.column_names, batched=True
    )
    tokenized_train_dataset_reverse_kr = train_dataset_reverse_kr.map(
        lambda x: generate_and_tokenize_prompt(x, prompt_template_reverse, tokenizer, input_key='meaning'),
        remove_columns=train_dataset_reverse_kr.column_names, batched=True
    )
    
    # 한국어 데이터셋 매핑2(ko-in)
    tokenized_train_dataset_kr_1 = train_dataset_kr.map(
        lambda x: generate_and_tokenize_prompt(x, prompt_template_forward_kr, tokenizer, input_key='idiom'),
        remove_columns=train_dataset_kr.column_names, batched=True
    )
    tokenized_train_dataset_reverse_kr_1 = train_dataset_reverse_kr.map(
        lambda x: generate_and_tokenize_prompt(x, prompt_template_reverse_kr, tokenizer, input_key='meaning'),
        remove_columns=train_dataset_reverse_kr.column_names, batched=True
    )
    tokenized_train_dataset_en_1 = train_dataset.map(
        lambda x: generate_and_tokenize_prompt(x, prompt_template_forward_kr, tokenizer, input_key='idiom'),
        remove_columns=train_dataset_kr.column_names, batched=True
    )
    tokenized_train_dataset_reverse_en_1 = train_dataset_reverse.map(
        lambda x: generate_and_tokenize_prompt(x, prompt_template_reverse_kr, tokenizer, input_key='meaning'),
        remove_columns=train_dataset_reverse_kr.column_names, batched=True
    )
    # type2
    # 한국어 데이터셋 매핑(en-in)
    tokenized_train_dataset_kr_2 = train_dataset_kr.map(
        lambda x: generate_and_tokenize_prompt(x, prompt_template_forward_2, tokenizer, input_key='idiom'),
        remove_columns=train_dataset_kr.column_names, batched=True
    )
    tokenized_train_dataset_reverse_kr_2 = train_dataset_reverse_kr.map(
        lambda x: generate_and_tokenize_prompt(x, prompt_template_reverse_2, tokenizer, input_key='meaning'),
        remove_columns=train_dataset_reverse_kr.column_names, batched=True
    )
    # 한국어 데이터셋 매핑2(ko-in)
    tokenized_train_dataset_kr_3 = train_dataset_kr.map(
        lambda x: generate_and_tokenize_prompt(x, prompt_template_forward_kr_2, tokenizer, input_key='idiom'),
        remove_columns=train_dataset_kr.column_names, batched=True
    )
    tokenized_train_dataset_reverse_kr_3 = train_dataset_reverse_kr.map(
        lambda x: generate_and_tokenize_prompt(x, prompt_template_reverse_kr_2, tokenizer, input_key='meaning'),
        remove_columns=train_dataset_reverse_kr.column_names, batched=True
    )
    tokenized_train_dataset_en_3 = train_dataset.map(
        lambda x: generate_and_tokenize_prompt(x, prompt_template_forward_kr_2, tokenizer, input_key='idiom'),
        remove_columns=train_dataset_kr.column_names, batched=True
    )
    tokenized_train_dataset_reverse_en_3 = train_dataset_reverse.map(
        lambda x: generate_and_tokenize_prompt(x, prompt_template_reverse_kr_2, tokenizer, input_key='meaning'),
        remove_columns=train_dataset_reverse_kr.column_names, batched=True
    )

    # 번역 데이터셋 매핑
    tokenized_translation_dataset = translation_dataset.map(
        lambda x: tokenize_translation(x, prompt_template_translation, tokenizer, input_key='source'),
        remove_columns=translation_dataset.column_names, batched=True
    )
    tokenized_translation_dataset_reverse = translation_dataset_reverse.map(
        lambda x: tokenize_translation(x, prompt_template_translation_reverse, tokenizer, input_key='source'),
        remove_columns=translation_dataset_reverse.column_names, batched=True
    )
    tokenized_translation_dataset_kr = translation_dataset.map(
        lambda x: tokenize_translation(x, prompt_template_translation_kr, tokenizer, input_key='source'),
        remove_columns=translation_dataset.column_names, batched=True
    )

    tokenized_translation_dataset_reverse_kr = translation_dataset_reverse.map(
        lambda x: tokenize_translation(x, prompt_template_translation_reverse_kr, tokenizer, input_key='source'),
        remove_columns=translation_dataset_reverse.column_names, batched=True)  

    # 모든 토크나이즈된 데이터셋을 결합
    tokenized_full_dataset = concatenate_datasets([
        tokenized_train_dataset_en,
        tokenized_train_dataset_reverse_en,
        tokenized_train_dataset_kr,
        tokenized_train_dataset_reverse_kr,    
        tokenized_train_dataset_en_2,
        tokenized_train_dataset_reverse_en_2,
        tokenized_train_dataset_kr_1,
        tokenized_train_dataset_reverse_kr_1,
        tokenized_train_dataset_kr_2,
        tokenized_train_dataset_reverse_kr_2,
        tokenized_train_dataset_kr_3,
        tokenized_train_dataset_reverse_kr_3,
        tokenized_translation_dataset,
        tokenized_translation_dataset_reverse,
        tokenized_translation_dataset_kr,
        tokenized_translation_dataset_reverse_kr,
        tokenized_train_dataset_en_3,
        tokenized_train_dataset_reverse_en_3,
        tokenized_train_dataset_en_1,
        tokenized_train_dataset_reverse_en_1,

    ])

    # 평가 데이터셋 준비 (이 부분에서 번역 데이터셋을 사용!)
    # 평가용 데이터셋 생성
    prompt_template_translation_eval ="""Translate the following sentence into Korean:

    # Source Sentence:
    {source}

    # Translation:
    """
    prompt_template_translation_eval_reverse ="""Translate the following sentence into English

    # Source Sentence:
    {source}

    # Translation:
    """
    # 평가 데이터셋을 반으로 나누고 각각 정방향(en-ko)과 역방향(영어로)으로 구성
    def split_and_create_datasets(eval_translation_df):
        eval_translation_df = eval_translation_df.sample(frac=1, random_state=42).reset_index(drop=True) # shuffle
        halfway_point = len(eval_translation_df) // 2
        eval_df_first_half = eval_translation_df[:halfway_point]
        eval_df_second_half = eval_translation_df[halfway_point:]

        eval_translation_dataset = create_translation_dataset(eval_df_first_half, 'source_string', 'target_string')
        eval_translation_dataset_reverse = create_translation_dataset(eval_df_second_half, 'target_string', 'source_string')

        return eval_translation_dataset, eval_translation_dataset_reverse

    eval_translation_dataset, eval_translation_dataset_reverse = split_and_create_datasets(eval_translation_df)

    tokenized_translation_dataset_eval = eval_translation_dataset.map(
        lambda x: tokenize_translation(x, prompt_template_translation_eval, tokenizer, input_key='source'),
        remove_columns=eval_translation_dataset.column_names, batched=True
    )
    tokenized_translation_dataset_eval_reverse = eval_translation_dataset_reverse.map(
        lambda x: tokenize_translation(x, prompt_template_translation_eval_reverse, tokenizer, input_key='source'),
        remove_columns=eval_translation_dataset_reverse.column_names, batched=True
    )
    tokenized_full_dataset_eval = concatenate_datasets([tokenized_translation_dataset_eval, tokenized_translation_dataset_eval_reverse])

    # # 8. 데이터셋 셔플
    shuffled_tokenized_full_dataset = tokenized_full_dataset.shuffle(seed=3)
    shuffled_tokenized_full_dataset_eval = tokenized_full_dataset_eval.shuffle(seed=3)
    print("번역 데이터셋 크기:", len(tokenized_translation_dataset)+ len(tokenized_translation_dataset_reverse) + len(tokenized_translation_dataset_kr) + len(tokenized_translation_dataset_reverse_kr))
    #print("번역 데이터셋 크기:", len(tokenized_translation_dataset)+ len(tokenized_translation_dataset_reverse))
    print("학습 데이터셋 크기:", len(shuffled_tokenized_full_dataset))
    print("평가 데이터셋 크기:", len(shuffled_tokenized_full_dataset_eval))
    # print("학습 데이터셋 샘플:")
    # print(shuffled_tokenized_full_dataset[-1])  # 첫 번째 데이터 포인트 출력
    # print("\n평가 데이터셋 샘플:")
    # print(shuffled_tokenized_full_dataset_eval[-1])  # 첫 번째 데이터 포인트 출력

    # 9. 토크나이저와 셔플된 데이터셋 반환
    return shuffled_tokenized_full_dataset, shuffled_tokenized_full_dataset_eval, tokenizer

    # # 8. 데이터셋 셔플
    # shuffled_tokenized_full_dataset = tokenized_full_dataset.shuffle(seed=3)
    # shuffled_tokenized_full_dataset_eval = tokenized_full_dataset_eval.shuffle(seed=3)

    # # 9. 번역 데이터셋과 관용구 데이터셋을 각각 1/256로 줄이기
    # reduction_factor = 64 # 256하면 관용구 0돼서

    # # 번역 데이터셋 크기 조정
    # translation_size = (len(tokenized_translation_dataset) + len(tokenized_translation_dataset_reverse) +
    #                     len(tokenized_translation_dataset_kr) + len(tokenized_translation_dataset_reverse_kr)) // reduction_factor

    # # 관용구 데이터셋 크기 조정
    # idiom_size = (len(tokenized_train_dataset_en) + len(tokenized_train_dataset_reverse_en) +
    #                 len(tokenized_train_dataset_kr) + len(tokenized_train_dataset_reverse_kr)) // reduction_factor

    # # 각각의 데이터셋에서 1/256만큼 선택
    # tokenized_translation_dataset_small = tokenized_translation_dataset.select(range(min(translation_size, len(tokenized_translation_dataset))))
    # tokenized_translation_dataset_reverse_small = tokenized_translation_dataset_reverse.select(range(min(translation_size, len(tokenized_translation_dataset_reverse))))
    # tokenized_translation_dataset_kr_small = tokenized_translation_dataset_kr.select(range(min(translation_size, len(tokenized_translation_dataset_kr))))
    # tokenized_translation_dataset_reverse_kr_small = tokenized_translation_dataset_reverse_kr.select(range(min(translation_size, len(tokenized_translation_dataset_reverse_kr))))

    # tokenized_train_dataset_en_small = tokenized_train_dataset_en.select(range(min(idiom_size, len(tokenized_train_dataset_en))))
    # tokenized_train_dataset_reverse_en_small = tokenized_train_dataset_reverse_en.select(range(min(idiom_size, len(tokenized_train_dataset_reverse_en))))
    # tokenized_train_dataset_kr_small = tokenized_train_dataset_kr.select(range(min(idiom_size, len(tokenized_train_dataset_kr))))
    # tokenized_train_dataset_reverse_kr_small = tokenized_train_dataset_reverse_kr.select(range(min(idiom_size, len(tokenized_train_dataset_reverse_kr))))

    # # 10. 줄인 번역 데이터셋과 관용구 데이터셋 결합
    # tokenized_full_dataset_small = concatenate_datasets([
    #     tokenized_translation_dataset_small,
    #     tokenized_translation_dataset_reverse_small,
    #     tokenized_translation_dataset_kr_small,
    #     tokenized_translation_dataset_reverse_kr_small,
    #     tokenized_train_dataset_en_small,
    #     tokenized_train_dataset_reverse_en_small,
    #     tokenized_train_dataset_kr_small,
    #     tokenized_train_dataset_reverse_kr_small
    # ])

    # # 11. 평가 데이터셋 크기 조정 (기존 평가 데이터셋을 1/256로 줄임)
    # eval_size = len(shuffled_tokenized_full_dataset_eval) // reduction_factor
    # shuffled_tokenized_full_dataset_eval = shuffled_tokenized_full_dataset_eval.select(range(eval_size))

    # # 12. 셔플된 데이터셋 크기 출력
    # print("번역 데이터셋 크기 (줄인 후):", len(tokenized_translation_dataset_small) + len(tokenized_translation_dataset_reverse_small) + len(tokenized_translation_dataset_kr_small) + len(tokenized_translation_dataset_reverse_kr_small))
    # print("관용구 데이터셋 크기 (줄인 후):", len(tokenized_train_dataset_en_small) + len(tokenized_train_dataset_reverse_en_small) + len(tokenized_train_dataset_kr_small) + len(tokenized_train_dataset_reverse_kr_small))
    # print("학습 데이터셋 크기 (결합 후):", len(tokenized_full_dataset_small))
    # print("평가 데이터셋 크기:", len(shuffled_tokenized_full_dataset_eval))

    # # 13. 토크나이저와 줄인 데이터셋 반환
    # return tokenized_full_dataset_small.shuffle(seed=3), shuffled_tokenized_full_dataset_eval, tokenizer
