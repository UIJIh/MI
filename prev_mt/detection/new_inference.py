import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
max_memory = {
    0: "40GiB",
    1: "40GiB",
    2: "40GiB",
    3: "40GiB"
}
#model_path = "/data/uijih/detection/model_output/llama70_new-2"
model_path = "/data/uijih/8b_instruct/model_output/llama70_sft_idioms-Test"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print("Model Loading!")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    max_memory=max_memory,
    quantization_config=bnb_config,
    offload_folder="./offload",
    offload_state_dict=True,
    trust_remote_code=True,
)
print("Model Loaded!")
model.resize_token_embeddings(len(tokenizer))
print(f"Tokenizer vocab size: {len(tokenizer)}")
print(f"Model vocab size: {model.config.vocab_size}")

# 시스템 프롬프트 (Follow the examples below:)
system_prompt_en = """The task is to detect any idiom present in the given sentence and return it exactly as it appears in the sentence. If there is no idiom, respond with \"None.\" and provide no additional text or explanation.
Sentence: "She plans to travel the world before she kicks the bucket."
Output: "kicks the bucket"

Sentence: "I think I ate too much dinner, and now my stomach hurts."
Output: "None"""

system_prompt_kr = """The task is to detect any idiom present in the given sentence and return it exactly as it appears in the sentence. If there is no idiom, respond with \"None.\" and provide no additional text or explanation.
Sentence: "가족을 먹여살리고자 밤낮 없이 일하다 불의의 사고로 한 줌 재가 되어버린 그가 남긴 자산을 그리 많지 않았다."
Output: "한 줌 재가 되어버린"

Sentence: "저녁을 너무 많이 먹어서인지 배가 아프다."
Output: "None"""

def detect_idiom(sentence, language="en"):
    system_prompt = system_prompt_en if language == "en" else system_prompt_kr
    input_text = system_prompt + f'Sentence: "{sentence}"\nOutput: '
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to("cuda")
    outputs = model.generate(
        inputs["input_ids"],
        max_length=256,
        num_return_sequences=1,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    detected_idiom = output_text[len(input_text):].strip()
    return detected_idiom

def process_dataset(file_path, output_file, sentence_column, label_column=None):
    data = pd.read_csv(file_path)
    results = []

    for _, row in data.iterrows():
        english_sentence = row[sentence_column[0]]
        korean_sentence = row[sentence_column[1]]

        detected_idiom_en = detect_idiom(english_sentence, language="en")
        detected_idiom_kr = detect_idiom(korean_sentence, language="kr")

        result = {
            "English Sentence": english_sentence,
            "Detected Idiom (EN)": detected_idiom_en,
            "Korean Sentence": korean_sentence,
            "Detected Idiom (KR)": detected_idiom_kr,
        }
        if label_column:
            result["Label"] = row[label_column]
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


seed_file_path = "/data/uijih/dataset/Seed50_for_Parallel_Dataset_ENKR_idiomKB_0.8_example.csv"
none_file_path = "/data/uijih/dataset/daataset_meaning.csv"
output_file_seed = "./idiom_detection_results-2.csv"
output_file_none = "./idiom_detection_none_results-2.csv"
# 1) idiom 
process_dataset(seed_file_path, output_file_seed, ["Sentence", "KR_Sentence"])
# 2) None 
process_dataset(none_file_path, output_file_none, ["New_Example_EN", "New_Example_KR"], label_column="Label")
