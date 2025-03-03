# import pandas as pd
# import json

# # CSV 파일 경로
# file_path = '/data/uijih/Seed50_for_Parallel_Dataset_ENKR_idiomKB_0.8_example.csv'

# # CSV 파일 불러오기
# data = pd.read_csv(file_path)

# # 데이터셋을 저장할 리스트
# conversations_dataset = []

# # 데이터셋 생성
# for index, row in data.iterrows():
#     # Task 1: Idiom-to-Meaning (영어 idiom에 대한 의미를 묻기)
#     conversation_idiom_to_meaning = {
#         "conversations": [
#             {
#                 "user": f"What does the idiom '{row['Idiom']}' mean? Please answer in one sentence.",  
#                 "assistant": f"The idiom '{row['Idiom']}' means: {row['Meaning']}" 
#             }
#         ]
#     }
#     # Task 2: Meaning-to-Idiom (영어 의미에 대한 idiom 묻기)
#     conversation_meaning_to_idiom = {
#         "conversations": [
#             {
#                 "user": f"Can you tell me an idiom that means '{row['Meaning']}'? Please answer in one sentence.",  
#                 "assistant": f"An idiom that matches this meaning is: '{row['Idiom']}'"  
#             }
#         ]
#     }
#     # Task 3: Idiom-to-Meaning (한국어 idiom에 대한 의미를 묻기)
#     conversation_kr_idiom_to_kr_meaning = {
#         "conversations": [
#             {
#                 "user": f"What does the Korean idiom '{row['KR_Idiom']}' mean in Korean? Please answer in one sentence.",  
#                 "assistant": f"The idiom '{row['KR_Idiom']}' means: {row['KR_Meaning']}" 
#             }
#         ]
#     }
#     # Task 4: Meaning-to-Idiom (한국어 의미에 대한 idiom 묻기)
#     conversation_kr_meaning_to_kr_idiom = {
#         "conversations": [
#             {
#                 "user": f"Can you tell me an Korean idiom that means '{row['KR_Meaning']}'? Please answer in one sentence.",  
#                 "assistant": f"An idiom that matches this meaning is: '{row['KR_Idiom']}'"  
#             }
#         ]
#     }
#     conversations_dataset.append(conversation_idiom_to_meaning)
#     conversations_dataset.append(conversation_meaning_to_idiom)
#     conversations_dataset.append(conversation_kr_idiom_to_kr_meaning)
#     conversations_dataset.append(conversation_kr_meaning_to_kr_idiom)

# # 데이터셋을 JSONL 파일로 저장
# output_file_path = 'idioms_conversations_one_sentence.jsonl'
# with open(output_file_path, 'w') as jsonl_file:
#     for data_entry in conversations_dataset:
#         jsonl_file.write(json.dumps(data_entry, ensure_ascii=False) + '\n')

# print(f"데이터셋이 '{output_file_path}'에 성공적으로 저장되었습니다.")

# """
# 템플릿 다양화로 데이터 증강할거임
# """
# import pandas as pd
# import json

# # CSV 파일 경로
# file_path = '/data/uijih/Seed50_for_Parallel_Dataset_ENKR_idiomKB_0.8_example.csv'

# # CSV 파일 불러오기
# data = pd.read_csv(file_path)

# # 데이터셋을 저장할 리스트
# conversations_dataset = []

# # 다양한 템플릿 정의 (모두 영어로)
# idiom_to_meaning_templates = [
#     "Can you explain the meaning of the idiom '{idiom}'?",
#     "What does the idiom '{idiom}' signify?",
#     #"I came across the idiom '{idiom}'. Could you explain what it means?",
#     #"In simple words, what does the idiom '{idiom}' mean?"
# ]

# meaning_to_idiom_templates = [
#     "Could you provide an idiom that matches the meaning '{meaning}'?",
#     "What idiom represents the meaning '{meaning}'? Answer in one line.",
#     #"If I want to express '{meaning}', what idiom can I use?",
#     #"Please tell me an idiom that fits the description: '{meaning}'."
# ]

# kr_idiom_to_meaning_templates = [
#     "What does the Korean idiom '{idiom}' mean? Please explain in Korean.",
#     "How would you describe the meaning of the Korean idiom '{idiom}' in Korean?",
#     #"Could you provide the meaning of the Korean idiom '{idiom}' in Korean?",
#     #"What is the definition of the Korean idiom '{idiom}' in Korean?"
# ]

# kr_meaning_to_idiom_templates = [
#     "Could you tell me a Korean idiom that fits the meaning '{meaning}'?",
#     "What Korean idiom represents the meaning '{meaning}'?",
#     #"What Korean idiom matches this meaning: '{meaning}'?",
#     #"If I want to express '{meaning}' in Korean, what Korean idiom can I use?"
# ]

# # 데이터셋 생성
# for index, row in data.iterrows():
#     # Task 1: Idiom-to-Meaning (영어 idiom에 대한 의미를 묻기)
#     for template in idiom_to_meaning_templates:
#         conversation_idiom_to_meaning = {
#             "conversations": [
#                 {
#                     "user": template.format(idiom=row['Idiom']),  
#                     "assistant": f"The idiom '{row['Idiom']}' means: {row['Meaning']}" 
#                 }
#             ]
#         }
#         conversations_dataset.append(conversation_idiom_to_meaning)

#     # Task 2: Meaning-to-Idiom (영어 의미에 대한 idiom 묻기)
#     for template in meaning_to_idiom_templates:
#         conversation_meaning_to_idiom = {
#             "conversations": [
#                 {
#                     "user": template.format(meaning=row['Meaning']),  
#                     "assistant": f"An idiom that matches this meaning is: '{row['Idiom']}'"  
#                 }
#             ]
#         }
#         conversations_dataset.append(conversation_meaning_to_idiom)

#     # Task 3: Idiom-to-Meaning (한국어 idiom에 대한 의미를 묻기)
#     for template in kr_idiom_to_meaning_templates:
#         conversation_kr_idiom_to_kr_meaning = {
#             "conversations": [
#                 {
#                     "user": template.format(idiom=row['KR_Idiom']),  
#                     "assistant": f"The idiom '{row['KR_Idiom']}' means: {row['KR_Meaning']}" 
#                 }
#             ]
#         }
#         conversations_dataset.append(conversation_kr_idiom_to_kr_meaning)

#     # Task 4: Meaning-to-Idiom (한국어 의미에 대한 idiom 묻기)
#     for template in kr_meaning_to_idiom_templates:
#         conversation_kr_meaning_to_kr_idiom = {
#             "conversations": [
#                 {
#                     "user": template.format(meaning=row['KR_Meaning']),  
#                     "assistant": f"An idiom that matches this meaning is: '{row['KR_Idiom']}'"  
#                 }
#             ]
#         }
#         conversations_dataset.append(conversation_kr_meaning_to_kr_idiom)

# output_file_path = 'augmented_idioms_conversations.jsonl'
# with open(output_file_path, 'w') as jsonl_file:
#     for data_entry in conversations_dataset:
#         jsonl_file.write(json.dumps(data_entry, ensure_ascii=False) + '\n')

# print(f"Dataset({len(conversations_dataset)}) successfully saved to '{output_file_path}'.")

import pandas as pd
import json
import random

# 재현성을 위한 랜덤 시드 설정
random.seed(42)

# 파일 경로 설정
en_idiom_file_path = '/data/uijih/EN_Idiom_filtered.csv'
kr_idiom_file_path = '/data/uijih/KR_Idiom.csv'

# 데이터셋 불러오기
en_data = pd.read_csv(en_idiom_file_path)
kr_data = pd.read_csv(kr_idiom_file_path)

# 데이터 품질 확인 및 처리
def clean_data(df, idiom_col, meaning_col):
    # 중복 제거
    df = df.drop_duplicates(subset=[idiom_col])
    # 결측치 제거
    df = df.dropna(subset=[idiom_col, meaning_col])
    # 인덱스 재설정
    df = df.reset_index(drop=True)
    return df

en_data = clean_data(en_data, idiom_col='idiom', meaning_col='en_meaning')
kr_data = clean_data(kr_data, idiom_col='Idiom', meaning_col='Meaning')

# 데이터셋 크기 확인
print(f"영어 데이터셋 크기: {len(en_data)}")
print(f"한국어 데이터셋 크기: {len(kr_data)}")

# 데이터셋 크기를 작은 쪽에 맞추기 (랜덤 샘플링)
min_size = min(len(en_data), len(kr_data))
en_data = en_data.sample(n=min_size, random_state=42).reset_index(drop=True)
kr_data = kr_data.sample(n=min_size, random_state=42).reset_index(drop=True)
# 데이터셋 크기 확인
print(f"영어 데이터셋 크기: {len(en_data)}")
print(f"한국어 데이터셋 크기: {len(kr_data)}")

# 데이터셋을 저장할 리스트
conversations_dataset = []

# 다양한 템플릿 정의
# 사용자 템플릿
idiom_to_meaning_templates_en = [
    "Can you explain the meaning of the idiom '{idiom}'?",
    "What does the idiom '{idiom}' signify?",
    "I came across the idiom '{idiom}'. Could you explain what it means?",
    "In simple words, what does the idiom '{idiom}' mean?"
]

meaning_to_idiom_templates_en = [
    "Could you provide an idiom that matches the meaning '{meaning}'?",
    "What idiom represents the meaning '{meaning}'? Answer in one line.",
    "If I want to express '{meaning}', what idiom can I use?",
    "Please tell me an idiom that fits the description: '{meaning}'."
]

idiom_to_meaning_templates_kr = [
    "한국어 관용구 '{idiom}'의 의미를 설명해 주시겠어요?",
    "한국어 관용구 '{idiom}'은 무엇을 뜻하나요?",
    "제가 '{idiom}'이라는 관용구를 봤는데, 무슨 뜻인지 알려주시겠어요?",
    "간단히 말해서, 한국어 관용구 '{idiom}'은 어떤 의미인가요?"
]

meaning_to_idiom_templates_kr = [
    "'{meaning}'라는 의미를 가진 한국어 관용구가 있을까요?",
    "'{meaning}'을 나타내는 한국어 관용구는 무엇인가요?",
    "'{meaning}'을 표현하고 싶다면 어떤 한국어 관용구를 사용할 수 있을까요?",
    "'{meaning}'에 맞는 한국어 관용구를 알려주세요."
]

# 어시스턴트 템플릿
assistant_templates_idiom_to_meaning_en = [
    "The idiom '{idiom}' means: {meaning}",
    "'{idiom}' means: {meaning}",
    "The meaning of '{idiom}' is: {meaning}",
    "'{idiom}' signifies: {meaning}"
]

assistant_templates_meaning_to_idiom_en = [
    "An idiom that matches this meaning is: '{idiom}'",
    "You can use the idiom '{idiom}'",
    "An appropriate idiom would be: '{idiom}'",
    "The idiom that fits is: '{idiom}'"
]

assistant_templates_idiom_to_meaning_kr = [
    "'{idiom}'의 의미는 다음과 같습니다: {meaning}",
    "'{idiom}'은(는) '{meaning}'을 뜻합니다.",
    "'{idiom}'은(는) '{meaning}'이라는 의미입니다.",
    "'{idiom}'의 뜻은 다음과 같습니다: {meaning}"
]

assistant_templates_meaning_to_idiom_kr = [
    "이 의미에 맞는 한국어 관용구는 '{idiom}'입니다.",
    "당신은 '{idiom}'이라는 관용구를 사용할 수 있습니다.",
    "'{idiom}'이(가) 적합한 관용구입니다.",
    "해당 의미를 가진 관용구는 '{idiom}'입니다."
]

# 데이터셋 생성
for index in range(min_size):
    # 영어 데이터 처리
    en_row = en_data.iloc[index]
    # Idiom-to-Meaning (영어)
    user_template = random.choice(idiom_to_meaning_templates_en)
    assistant_template = random.choice(assistant_templates_idiom_to_meaning_en)
    conversation_en_idiom_to_meaning = {
        "conversations": [
            {
                "user": user_template.format(idiom=en_row['idiom']),
                "assistant": assistant_template.format(idiom=en_row['idiom'], meaning=en_row['en_meaning'])
            }
        ]
    }
    conversations_dataset.append(conversation_en_idiom_to_meaning)
    # Meaning-to-Idiom (영어)
    user_template = random.choice(meaning_to_idiom_templates_en)
    assistant_template = random.choice(assistant_templates_meaning_to_idiom_en)
    conversation_en_meaning_to_idiom = {
        "conversations": [
            {
                "user": user_template.format(meaning=en_row['en_meaning']),
                "assistant": assistant_template.format(idiom=en_row['idiom'])
            }
        ]
    }
    conversations_dataset.append(conversation_en_meaning_to_idiom)
    
    # 한국어 데이터 처리
    kr_row = kr_data.iloc[index]
    # Idiom-to-Meaning (한국어)
    user_template = random.choice(idiom_to_meaning_templates_kr)
    assistant_template = random.choice(assistant_templates_idiom_to_meaning_kr)
    conversation_kr_idiom_to_meaning = {
        "conversations": [
            {
                "user": user_template.format(idiom=kr_row['Idiom']),
                "assistant": assistant_template.format(idiom=kr_row['Idiom'], meaning=kr_row['Meaning'])
            }
        ]
    }
    conversations_dataset.append(conversation_kr_idiom_to_meaning)
    # Meaning-to-Idiom (한국어)
    user_template = random.choice(meaning_to_idiom_templates_kr)
    assistant_template = random.choice(assistant_templates_meaning_to_idiom_kr)
    conversation_kr_meaning_to_idiom = {
        "conversations": [
            {
                "user": user_template.format(meaning=kr_row['Meaning']),
                "assistant": assistant_template.format(idiom=kr_row['Idiom'])
            }
        ]
    }
    conversations_dataset.append(conversation_kr_meaning_to_idiom)

# 데이터셋을 JSONL 파일로 저장
output_file_path = 'random_template_idioms_conversations_balanced.jsonl'
with open(output_file_path, 'w', encoding='utf-8') as jsonl_file:
    for data_entry in conversations_dataset:
        jsonl_file.write(json.dumps(data_entry, ensure_ascii=False) + '\n')

print(f"데이터셋 ({len(conversations_dataset)})이 '{output_file_path}'에 성공적으로 저장되었습니다.")
