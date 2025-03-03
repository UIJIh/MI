import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_output_dir = "/data/uijih/8b_instruct/model_output/post_sft_idioms-NEW-2-post"
tokenizer = AutoTokenizer.from_pretrained(model_output_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_output_dir, torch_dtype=torch.bfloat16).to('cuda')

def generate_response(user_input, max_length=256):
    inputs = tokenizer(
        user_input,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    ).to('cuda')

    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=3,
        early_stopping=True,
        no_repeat_ngram_size=2
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


dataset_path_kr_to_en = "/data/uijih/dataset/final/KREN_test_141.csv"
dataset_path_en_to_kr = "/data/uijih/dataset/final/ENKR_test_187.csv"
data_kr_to_en = pd.read_csv(dataset_path_kr_to_en)
data_en_to_kr = pd.read_csv(dataset_path_en_to_kr)

shots_kr_to_en = [
    {
        "source": "회사에 불만이 있다고 해서 동료들에게 나쁜 소문을 퍼뜨리는 건 누워서 침 뱉기나 다름없어.",
        "target": "Spreading bad rumors about your colleagues just because you have complaints about the company is like cutting off your nose to spite your face."
    },
    {
        "source": "그녀는 진짜 그의 눈엣가시이다.",
        "target": "She is real thorn in his side."
    }
]

shots_en_to_kr = [
    {
        "source": "Spreading bad rumors about your colleagues just because you have complaints about the company is like cutting off your nose to spite your face.",
        "target": "회사에 불만이 있다고 해서 동료들에게 나쁜 소문을 퍼뜨리는 건 누워서 침 뱉기나 다름없어."
    },
    {
        "source": "She is real thorn in his side.",
        "target": "그녀는 진짜 그의 눈엣가시이다."
    }
]

def create_prompt(korean_sentence=None, english_sentence=None, shots=None, direction="KR_to_EN"):
    prompt = ""
    if direction == "KR_to_EN":
        prompt += "Translate the following Korean sentence to English, making sure to translate *idioms as idioms*.\n\n"
        for shot in shots:
            prompt += f"Source (KR): {shot['source']}\nTarget (EN): {shot['target']}\n\n"
        prompt += f"Source (KR): {korean_sentence}\nTarget (EN): "
    elif direction == "EN_to_KR":
        prompt += "Translate the following English sentence to Korean, making sure to translate *idioms as idioms*.\n\n"
        for shot in shots:
            prompt += f"Source (EN): {shot['source']}\nTarget (KR): {shot['target']}\n\n"
        prompt += f"Source (EN): {english_sentence}\nTarget (KR): "
    return prompt

results_kr_to_en_df = pd.DataFrame(columns=['Original_KR_Sentence', 'KR_to_EN_Translation', 'Label'])
results_en_to_kr_df = pd.DataFrame(columns=['Original_EN_Sentence', 'EN_to_KR_Translation', 'Label'])

for _, item in data_kr_to_en.iterrows():
    final_prompt_kr_to_en = create_prompt(korean_sentence=item['KR_Sentence'], shots=shots_kr_to_en, direction="KR_to_EN")
    translation_response_kr_to_en = generate_response(final_prompt_kr_to_en)
    print(f"===========================================\n{translation_response_kr_to_en}\n")  # 번역 결과 출력
    kr_to_en_translation = translation_response_kr_to_en[len(final_prompt_kr_to_en):].strip()
    new_row_kr_to_en = {
        'Original_KR_Sentence': item['KR_Sentence'],
        'KR_to_EN_Translation': kr_to_en_translation,
        'Label': item['Idiom']
    }
    results_kr_to_en_df = pd.concat([results_kr_to_en_df, pd.DataFrame([new_row_kr_to_en])], ignore_index=True)

for _, item in data_en_to_kr.iterrows():
    final_prompt_en_to_kr = create_prompt(english_sentence=item['Sentence'], shots=shots_en_to_kr, direction="EN_to_KR")
    translation_response_en_to_kr = generate_response(final_prompt_en_to_kr)
    print(f"===========================================\n{translation_response_en_to_kr}\n")  # 번역 결과 출력
    en_to_kr_translation = translation_response_en_to_kr[len(final_prompt_en_to_kr):].strip()
    new_row_en_to_kr = {
        'Original_EN_Sentence': item['Sentence'],
        'EN_to_KR_Translation': en_to_kr_translation,
        'Label': item['KR_Idiom']
    }
    results_en_to_kr_df = pd.concat([results_en_to_kr_df, pd.DataFrame([new_row_en_to_kr])], ignore_index=True)

output_csv_kr_to_en_path = 'results/NEW-2-post-kr_to_en.csv'
output_csv_en_to_kr_path = 'results/NEW-2-post-en_to_kr.csv'
results_kr_to_en_df.to_csv(output_csv_kr_to_en_path, index=False)
results_en_to_kr_df.to_csv(output_csv_en_to_kr_path, index=False)

print(f"KR to EN translation results successfully saved to {output_csv_kr_to_en_path}")
print(f"EN to KR translation results successfully saved to {output_csv_en_to_kr_path}")
