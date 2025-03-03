import pandas as pd
import json
import random

random.seed(42)

dataset_path_kr_to_en = "/data/uijih/dataset/final/KREN_post_28.csv"
dataset_path_en_to_kr = "/data/uijih/dataset/final/ENKR_post_28.csv"
data_kr_to_en = pd.read_csv(dataset_path_kr_to_en)
data_en_to_kr = pd.read_csv(dataset_path_en_to_kr)

conversations_dataset = []

kr_to_en_templates = [
    "Translate this Korean sentence into English, ensuring idioms are preserved as idioms. If there is no matching idiom, provide a clear and comprehensive meaning:\n\n{sentence}",
    "How would you render this Korean sentence into English, focusing on idiomatic expressions? If an idiom match isn't possible, explain its meaning naturally:\n\n{sentence}",
    "Provide an English translation for the following Korean sentence. Prioritize idiom equivalence; if not available, translate its meaning:\n\n{sentence}",
    "Convert this Korean sentence into English, maintaining idioms where possible or translating their meaning accurately:\n\n{sentence}",
    "Turn the following Korean sentence into English. Make sure idioms remain idiomatic or are clearly explained:\n\n{sentence}"
]
en_to_kr_templates = [
    "Translate this English sentence into Korean, ensuring idioms are preserved as idioms. If there is no matching idiom, provide a clear and comprehensive meaning:\n\n{sentence}",
    "How would you render this English sentence into Korean, focusing on idiomatic expressions? If an idiom match isn't possible, explain its meaning naturally:\n\n{sentence}",
    "Provide a Korean translation for the following English sentence. Prioritize idiom equivalence; if not available, translate its meaning:\n\n{sentence}",
    "Convert this English sentence into Korean, maintaining idioms where possible or translating their meaning accurately:\n\n{sentence}",
    "Turn the following English sentence into Korean. Make sure idioms remain idiomatic or are clearly explained:\n\n{sentence}"
]

kr_to_en_response_templates = [
    "{translation}",
]
en_to_kr_response_templates = [
    "{translation}",
]

# 데이터셋 생성
for _, row in data_kr_to_en.iterrows():
    # KR to EN
    user_template = random.choice(kr_to_en_templates)
    assistant_template = random.choice(kr_to_en_response_templates)
    conversation_kr_to_en = {
        "conversations": [
            {
                #"system": "You are an expert in translating idioms between Korean and English. Ensure idioms are accurately translated into their closest equivalent idioms in the target language. If no equivalent exists, provide a natural and comprehensive translation of the idiom's meaning.",
                "user": user_template.format(sentence=row['KR_Sentence']),
                "assistant": assistant_template.format(translation=row['Sentence'])
            }
        ]
    }
    conversations_dataset.append(conversation_kr_to_en)

for _, row in data_en_to_kr.iterrows():
    # EN to KR
    user_template = random.choice(en_to_kr_templates)
    assistant_template = random.choice(en_to_kr_response_templates)
    conversation_en_to_kr = {
        "conversations": [
            {
                #"system": "You are an expert in translating idioms between English and Korean. Ensure idioms are accurately translated into their closest equivalent idioms in the target language. If no equivalent exists, provide a natural and comprehensive translation of the idiom's meaning.",
                "user": user_template.format(sentence=row['Sentence']),
                "assistant": assistant_template.format(translation=row['KR_Sentence'])
            }
        ]
    }
    conversations_dataset.append(conversation_en_to_kr)

# 데이터셋 섞기
random.shuffle(conversations_dataset)

# JSONL 파일로 저장
output_file_path = 'idioms_translation_conversations.jsonl'
with open(output_file_path, 'w', encoding='utf-8') as jsonl_file:
    for data_entry in conversations_dataset:
        jsonl_file.write(json.dumps(data_entry, ensure_ascii=False) + '\n')

print(f"{len(conversations_dataset)}개의 데이터가 '{output_file_path}'에 성공적으로 저장되었습니다.")
