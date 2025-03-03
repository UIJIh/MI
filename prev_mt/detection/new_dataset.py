import pandas as pd
import json
import random

random.seed(42)

"""
===== 데이터셋 통계 =====
1. 학습 데이터셋:
   - 영어 관용구 템플릿 예문: 3840개
   - 한국어 관용구 템플릿 예문: 3990개
   - 관용구 없는 예문: 2444개

2. 테스트 데이터셋:
   - 영어 관용구 예문: 49개
   - 한국어 관용구 예문: 49개
   - 관용구 없는 예문: 100개

총 데이터 크기:
   - 학습 데이터셋: 10274개
   - 테스트 데이터셋: 198개
"""

def generate_example_sentence(idiom, language='en'):
    """예문 임의로 생성"""
    if language == 'en':
        templates = [
            f"I heard someone say \"{idiom}\" during the meeting.",
            f"My friend often says '{idiom}' in situations like this.",
            f"She looked at me and said '{idiom}', which made perfect sense.",
            f"When I told him what happened, he just smiled and said {idiom}.",
            f"Everyone in the office keeps saying \"{idiom}\" lately.",            
            f"The expression '{idiom}' perfectly describes what happened.",
            f"There's a saying that goes '{idiom}', which fits this case.",
            f"You know what they say - {idiom}.",
            f"As the old saying goes, {idiom}.",
            f"It's like what people say: {idiom}.",            
            f"My grandmother always told me \"{idiom}\" when I was young.",
            f"I remember my teacher explaining that '{idiom}' in class.",
            f"Growing up, I often heard '{idiom}' from my parents.",
            f"Back in college, my professor would always say {idiom}.",
            f"Every time this happens, I think of the phrase {idiom}.",            
            f"Let me give you some advice: {idiom}.",
            f"My mentor always says '{idiom}' in these situations.",
            f"If I were you, I'd remember that {idiom}.",
            f"The best advice I can give you is {idiom}.",
            f"People in this field often say {idiom}.",
            f"This reminds me of the saying '{idiom}'.",
            f"In this case, {idiom} would be the perfect description.",
            f"Looking at the situation, '{idiom}' comes to mind.",
            f"It's exactly what they mean by {idiom}.",
            f"This is a classic example of {idiom}.",            
            f"There's an old English saying that goes {idiom}.",
            f"It's a common expression here: {idiom}.",
            f"In our culture, we often say {idiom}.",
            f"There's traditional wisdom that says {idiom}.",
            f"People have been saying \"{idiom}\" for generations."
        ]
    else:
        templates = [
            f"이런 상황에서 {idiom} 이라는 말이 딱 맞아요.",
            f"친구가 이럴 때마다 '{idiom}' 라고 하더라고요.",
            f"어머니께서 항상 '{idiom}'이라고 말씀하셨어요.",
            f"직장 동료들 사이에서 '{idiom}'이라는 말이 자주 나와요.",
            f"동생이 상황을 듣더니, {idiom}라고 하네요.",
            f"{idiom} 라는 관용 표현이 이런 경우를 잘 설명해주죠.",
            f"이런 상황을 두고 '{idiom}'이라고 하나 봐요.",
            f"그 때 일을 생각하면 '{idiom}'이라는 말이 떠올라요.",
            f"이런 일을 보면, {idiom} 라는 말이 실감나요.",
            f"{idiom} 라는 말이 이럴 때 쓰이는 거구나 싶어요.",
            f"예전에 할아버지께서 '{idiom}'이라고 자주 말씀하셨죠.",
            f"학창 시절에 {idiom}라는 말을 많이 들었어요.",
            f"첫 직장에서 상사가 \"{idiom}\"이라고 조언해주셨어요.",
            f"지금 생각해보면 {idiom} 이란 말이 맞았네요.",
            f"웃으면서 '{idiom}'이라고 말하더군요.",
            f"제 생각에는 '{idiom}'이라는 말이 딱이에요.",
            f"선배가 늘 '{idiom}'이라고 조언하더라고요.",
            f"이런 경우엔 '{idiom}'이라는 말을 기억하세요.",
            f"경험상 {idiom} 이라는 말이 맞더라고요.",
            f"전문가들은 이럴 때 \"{idiom}\"이라고 하죠.",
            f"옛날부터 \"{idiom}\"이라는 말이 전해 내려왔죠.",
            f"한국 사람이라면 \"{idiom}\"이라는 말을 다 알죠.",
            f"전통적으로 이럴 땐, {idiom} 이라고 해왔어요.",
            f"우리 문화에서는 이런 걸 두고 \"{idiom}\"이라고 합니다.",
            f"그제서야 \"{idiom}\"이라는 말의 의미를 깨달았죠.",
            f"경험을 통해 \"{idiom}\"이라는 말이 맞다는 걸 알게 됐어요.",
            f"시간이 지나고 보니 '{idiom}'이란 말이 와닿더라고요.",
            f"이런 경험이 있고 나서야, {idiom} 이라는 말이 이해됐어요.",
            f"인생을 살다 보면 {idiom} 이라는 말이 와닿을 때가 있다."
        ]
    return random.choice(templates)

def create_conversation_entry(sentence, idiom):
    return {
        "conversations": [
            {
                "user": f"{sentence}",
                "assistant": f"{idiom}"
            }
        ]
    }

# def generate_dataset(seed_path, en_idiom_path, kr_idiom_path, output_path, augment_factor=5):
#     """Generate dataset with increased detection examples and proper None handling"""
#     conversations_dataset = []
#     stats = {
#         'text_en_with_idioms': 0,
#         'test_kr_with_idioms': 0,
#         'seed_without_idioms': 0,
#         'template_en_examples': 0,
#         'template_kr_examples': 0
#     }
    
#     # 1. Process seed dataset with augmentation
#     seed_data = pd.read_csv(seed_path)
    
#     # English entries with idioms - augmented
#     en_seed = seed_data[['Idiom', 'Sentence']].dropna(subset=['Sentence'])
#     for _, row in en_seed.iterrows():
#         if pd.notna(row['Idiom']) and row['Idiom'] != "None":  # 관용구가 있는 경우
#             for _ in range(augment_factor):
#                 conversations_dataset.append(
#                     create_conversation_entry(row['Sentence'], row['Idiom'])
#                 )
#             stats['text_en_with_idioms'] += augment_factor
#         else:  # 관용구가 없는 경우
#             conversations_dataset.append(
#                 create_conversation_entry(row['Sentence'], "None")
#             )
#             stats['seed_without_idioms'] += 1
    
#     # Korean entries with idioms - augmented
#     kr_seed = seed_data[['KRIdiom', 'KRSentence']].dropna(subset=['KRSentence'])
#     for _, row in kr_seed.iterrows():
#         if pd.notna(row['KRIdiom']) and row['KRIdiom'] != "None":  # 관용구가 있는 경우
#             for _ in range(augment_factor):
#                 conversations_dataset.append(
#                     create_conversation_entry(row['KRSentence'], row['KRIdiom'])
#                 )
#             stats['test_kr_with_idioms'] += augment_factor
#         else:  # 관용구가 없는 경우
#             conversations_dataset.append(
#                 create_conversation_entry(row['KRSentence'], "None")
#             )
#             stats['seed_without_idioms'] += 1

#     # 2. Process additional English idioms
#     en_idioms = pd.read_csv(en_idiom_path)
#     stats['template_en_examples'] = len(en_idioms)
#     for _, row in en_idioms.iterrows():
#         sentence = generate_example_sentence(row['idiom'], 'en')
#         conversations_dataset.append(
#             create_conversation_entry(sentence, row['idiom'])
#         )
    
#     # 3. Process additional Korean idioms
#     kr_idioms = pd.read_csv(kr_idiom_path)
#     stats['template_kr_examples'] = len(kr_idioms)
#     for _, row in kr_idioms.iterrows():
#         sentence = generate_example_sentence(row['Idiom'], 'kr')
#         conversations_dataset.append(
#             create_conversation_entry(sentence, row['Idiom'])
#         )
    
#     # Shuffle the dataset
#     random.shuffle(conversations_dataset)
    
#     # Save to JSONL file
#     with open(output_path, 'w', encoding='utf-8') as jsonl_file:
#         for entry in conversations_dataset:
#             jsonl_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
#     return len(conversations_dataset), stats

# Generate dataset
# seed_file_path = '/data/uijih/dataset/uiji_seed.csv'
# en_idiom_file_path = '/data/uijih/dataset/EN_Idiom_filtered.csv'
# kr_idiom_file_path = '/data/uijih/dataset/KR_Idiom.csv'
# output_file_path = 'complete_idioms_detection_dataset.jsonl'

# Generate with augmentation factor of 5
# total_entries, stats = generate_dataset(
#     seed_file_path, 
#     en_idiom_file_path, 
#     kr_idiom_file_path, 
#     output_file_path,
#     augment_factor=1
# )

# print("\n===== 데이터셋 통계 =====")
# print(f"1. 예문 데이터셋:")
# print(f"   - 영어 관용구 예문: {stats['text_en_with_idioms']}개")
# print(f"   - 한국어 관용구 예문: {stats['test_kr_with_idioms']}개")
# print(f"   - 관용구 없는 예문: {stats['seed_without_idioms']}개")
# print(f"\n2. 예문 강제 생성 데이터셋:")
# print(f"   - 영어 관용구 템플릿 예문: {stats['template_en_examples']}개")
# print(f"   - 한국어 관용구 템플릿 예문: {stats['template_kr_examples']}개")
# print(f"\n총 데이터셋 크기: {total_entries}개")

# # Calculate percentages
# total = total_entries
# seed_total = stats['text_en_with_idioms'] + stats['test_kr_with_idioms'] + stats['seed_without_idioms']
# template_total = stats['template_en_examples'] + stats['template_kr_examples']

# print(f"\n===== 비율 =====")
# print(f"예문 데이터 비율: {seed_total/total*100:.1f}%")
# print(f"예문 강제 생성 데이터 비율: {template_total/total*100:.1f}%")
# print(f"None 데이터 비율: {stats['seed_without_idioms']/total*100:.1f}%")

######################### 테스트 데이터셋으로 넘겼음
def generate_dataset(test_path, en_idiom_path, kr_idiom_path, none_file_path, train_output_path, test_output_path, augment_factor=5):
    """Generate dataset with increased detection examples and proper None handling"""
    train_dataset = []
    test_dataset = []
    stats = {
        'text_en_with_idioms': 0,
        'test_kr_with_idioms': 0,
        'test_without_idioms' : 0,
        'template_en_examples': 0,
        'template_kr_examples': 0,
        'meaning_none_cases': 0
    }
    
    # 1. Process test dataset with augmentation
    seed_data = pd.read_csv(test_path)
    
    # English entries with idioms
    en_seed = seed_data[['Idiom', 'Sentence']].dropna(subset=['Sentence'])
    for _, row in en_seed.iterrows():
        if pd.notna(row['Idiom']) and row['Idiom'] != "None":  # 관용구가 있는 경우 -> 테스트 데이터로 추가
            test_dataset.append(
                create_conversation_entry(row['Sentence'], row['Idiom'])
            )
            stats['text_en_with_idioms'] += 1
        else:  # 관용구가 없는 경우
            test_dataset.append(
                create_conversation_entry(row['Sentence'], "None")
            )
            stats['test_without_idioms'] += 1
    
    # Korean entries with idioms
    kr_seed = seed_data[['KRIdiom', 'KRSentence']].dropna(subset=['KRSentence'])
    for _, row in kr_seed.iterrows():
        if pd.notna(row['KRIdiom']) and row['KRIdiom'] != "None":  # 관용구가 있는 경우 -> 테스트 데이터로 추가
            test_dataset.append(
                create_conversation_entry(row['KRSentence'], row['KRIdiom'])
            )
            stats['test_kr_with_idioms'] += 1
        else:  # 관용구가 없는 경우
            test_dataset.append(
                create_conversation_entry(row['KRSentence'], "None")
            )
            stats['test_without_idioms'] += 1
            
    # 2. Process train dataset with NONE examples
    none_data = pd.read_csv(none_file_path)

    for _, row in none_data.iterrows():
        # label = row['Label']  # Label 값
        # if pd.isna(label) or str(label).strip() == "None":  # Label이 NaN이거나 "None"이면
        #     train_dataset.append(
        #         create_conversation_entry(row['New_Example_EN'], "None")
        #     )
        #     train_dataset.append(
        #         create_conversation_entry(row['New_Example_KR'], "None")
        #     )
        #     stats['meaning_none_cases'] += 2
        text = row['text']
        train_dataset.append(
                create_conversation_entry(text, "None")
            )
        stats['meaning_none_cases'] += 1

    
    # 3. Process additional English idioms (template-based generation for training)
    en_idioms = pd.read_csv(en_idiom_path)
    stats['template_en_examples'] = len(en_idioms)
    for _, row in en_idioms.iterrows():
        sentence = generate_example_sentence(row['idiom'], 'en')
        train_dataset.append(
            create_conversation_entry(sentence, row['idiom'])
        )
    
    # 4. Process additional Korean idioms (template-based generation for training)
    kr_idioms = pd.read_csv(kr_idiom_path)
    stats['template_kr_examples'] = len(kr_idioms)
    for _, row in kr_idioms.iterrows():
        sentence = generate_example_sentence(row['Idiom'], 'kr')
        train_dataset.append(
            create_conversation_entry(sentence, row['Idiom'])
        )
    
    # Shuffle the datasets
    random.shuffle(train_dataset)
    random.shuffle(test_dataset)
    
    # Save to JSONL files
    with open(train_output_path, 'w', encoding='utf-8') as train_file:
        for entry in train_dataset:
            train_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    with open(test_output_path, 'w', encoding='utf-8') as test_file:
        for entry in test_dataset:
            test_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    return len(train_dataset), len(test_dataset), stats

# Generate dataset with split into train and test
test_file_path = '/data/uijih/dataset/uiji_seed.csv'
en_idiom_file_path = '/data/uijih/dataset/EN_Idiom_filtered.csv'
kr_idiom_file_path = '/data/uijih/dataset/KR_Idiom.csv'
#none_file_path = '/data/uijih/dataset/daataset_meaning.csv' # None 경우임
none_file_path = '/data/uijih/dataset/processed_texts.csv'
train_output_file_path = 'train_idioms_detection_dataset.jsonl'
test_output_file_path = 'test_idioms_detection_dataset.jsonl'

# Generate with augmentation factor of 1
train_size, test_size, stats = generate_dataset(
    test_file_path, 
    en_idiom_file_path, 
    kr_idiom_file_path,
    none_file_path,
    train_output_file_path, 
    test_output_file_path,
    augment_factor=1
)

print("\n===== 데이터셋 통계 =====")
print(f"1. 학습 데이터셋:")
print(f"   - 영어 관용구 템플릿 예문: {stats['template_en_examples']}개")
print(f"   - 한국어 관용구 템플릿 예문: {stats['template_kr_examples']}개")
print(f"   - 관용구 없는 예문: {stats['meaning_none_cases']}개")
print(f"\n2. 테스트 데이터셋:")
print(f"   - 영어 관용구 예문: {stats['text_en_with_idioms']}개")
print(f"   - 한국어 관용구 예문: {stats['test_kr_with_idioms']}개")
print(f"   - 관용구 없는 예문: {stats['test_without_idioms']}개")

print(f"\n총 데이터 크기:")
print(f"   - 학습 데이터셋: {train_size}개")
print(f"   - 테스트 데이터셋: {test_size}개")
