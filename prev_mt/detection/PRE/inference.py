import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import random

# 1. 추론 함수 정의
def inference(model_path, tokenizer, input_data, system_prompts_en):
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    model.eval()

    # 영어 프롬프트 템플릿
    prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

{user_input}<|eot_id|>
<|start_header_id|>assistant<|end_header_id>
"""

    results = []
    for sentence in input_data:
        # 영어 프롬프트만 사용
        system_prompt = random.choice(system_prompts_en)
        user_input = f'Is there an idiom in the sentence "{sentence}"? If yes, find the idiom\'s original form.'

        # 프롬프트 템플릿에 시스템 프롬프트와 사용자 입력 삽입
        input_prompt = prompt_template.format(
            system_prompt=system_prompt,
            user_input=user_input
        )

        # 토크나이즈
        inputs = tokenizer(input_prompt, return_tensors="pt").to("cuda")

        # 모델에 입력
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=256,
                do_sample=False,
                top_k=10
            )
        
        # 토큰 디코딩
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append(decoded_output)
    
    return results

# 2. 추론 및 비교 함수
def run_inference_and_save_results(input_csv_path, output_csv_path, model_path, base_model_path, system_prompts_en):
    # 데이터 로드
    data = pd.read_csv(input_csv_path)
    sentences = data['Sentence'].tolist()  # 추론할 문장 리스트
    labels = data['Idiom'].tolist()  # 실제 레이블 (정답)

    # 3. 두 가지 모델 로드 (기본 모델과 학습된 모델)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer_base = AutoTokenizer.from_pretrained(base_model_path)

    # 4. 학습된 모델 추론
    print(f"Running inference with model: {model_path}")
    model_predictions = inference(model_path, tokenizer, sentences, system_prompts_en)

    # 5. 기본 모델 추론
    print(f"Running inference with base model: {base_model_path}")
    base_model_predictions = inference(base_model_path, tokenizer_base, sentences, system_prompts_en)

    # 6. 예측 결과와 실제 레이블 비교해서 O/X 판단
    comparison = ["O" if pred.strip() == label.strip() else "X" for pred, label in zip(model_predictions, labels)]

    # 7. 결과 데이터프레임 생성 및 저장
    result_df = pd.DataFrame({
        'Sentence': sentences,
        'Label': labels,  # 실제 레이블
        'Model_Prediction': model_predictions,  # 학습된 모델의 예측 결과
        'Base_Model_Prediction': base_model_predictions,  # 기본 모델의 예측 결과
        'Comparison': comparison  # O/X 비교 결과
    })

    result_df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")

# 8. 실행
if __name__ == "__main__":
    # 학습된 모델 경로와 기본 모델 경로
    model_path = "./saveded_instruct-full-detection-1"
    base_model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # 입력 데이터 경로와 결과 저장 경로
    input_csv_path = "/data/uijih/previous/Seed50_for_Parallel_Dataset_ENKR_idiomKB_0.8_example.csv"
    output_csv_path = "./inference_results.csv"  # 결과를 저장할 CSV 파일 경로

    # 학습 때 사용했던 영어 시스템 프롬프트만 사용
    system_prompts_en = [
        "Detect if there is an idiom in the following sentence. If there is, find the original form of the idiom. If there are no idioms, answer 'none'.",
        "Check if the following sentence contains any idioms. If so, provide the idiom's original form. If none, respond with 'none'.",
        "Find the idiom in the sentence if any. Otherwise, answer 'none'.",
        "Look for an idiom in the following sentence and return its original form if found. Otherwise, return 'none'."
    ]

    # 추론 실행 및 결과 저장
    run_inference_and_save_results(input_csv_path, output_csv_path, model_path, base_model_path, system_prompts_en)