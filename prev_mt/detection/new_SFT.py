"""
우지의 개발 일지:)
$ nohup python -u .py & 
$ tail -f nohup.out 
"""
# 우선 mismatch 디버깅 해결.. 거의 1-2주만에 완료 
## (이전 기록들 일단 삭제)
## new epoch3 1.2668 -> del
## -2 batch4/epoch4 (엄청 안정적으로 떨어져서 조금 lr 올려도 ㄱㅊ을듯하긴함) 1.5773 -> new 로 변경 후 위 삭제
## -2 epoch5  1.4335 -> del (shot은 주는게 맞아보이는데, system prompt 안한버전도, 혹은 랜덤
## (04.12~05.) -2 lr 올리고 new (이것도 여전히 안정적..) -> 1.0164

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
import torch, json, re
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"


model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
HF_token = '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'

# 70B 모델 로드
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config,
)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# QLoRA 설정
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 토크나이저 설정
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
#tokenizer.add_special_tokens({"pad_token": "<pad>", "eos_token": "<eos>", "bos_token": "<bos>"})
# model.resize_token_embeddings(len(tokenizer))
model.config.use_cache = False
# 추가 로깅
# print(f"Added tokens: {tokenizer.added_tokens_encoder}")
# print(f"Vocab before resize: {model.config.vocab_size}")
# model.resize_token_embeddings(len(tokenizer))
# print(f"Vocab after resize: {model.config.vocab_size}")


# 데이터셋 템플릿 적용
def load_dataset(jsonl_path, is_test=False, debug=True, num_debug_samples=1, tokenizer=None):
    with open(jsonl_path, 'r') as f:
        raw_data = [json.loads(line) for line in f]
    
    data = {'text': []}
    for item in raw_data:
        conversation = []
        for entry in item["conversations"]:
            conversation.append({
                "role": "system",
                "content": "You are an expert in detecting idioms. Identify only the idiom exactly as it appears in the given sentence. Do not provide additional explanations or context interpretation. If there is no idiom, respond with 'None.'"
            })
            conversation.append({'role': 'user', 'content': entry['user']})    
            conversation.append({'role': 'assistant', 'content': entry['assistant']})
        
        templated = tokenizer.apply_chat_template(conversation, tokenize=False, padding=True, max_length=400, truncation=True)
        data['text'].append(templated)

        if debug and len(data['text']) <= num_debug_samples:
            print(f"\n============= {'test' if is_test else 'train'} 템플릿 적용 결과 =============")
            print(templated)
    
    return Dataset.from_dict(data)

class AdvancedIdiomEvalCallback(TrainerCallback):
    def __init__(self, trainer, eval_dataset, tokenizer, num_examples=5):
        self.trainer = trainer
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.total_samples = 0
        self.correct_predictions = 0

    def extract_user_input(self, full_text):
        """
        Extracts the 'user' content from the text based on markers.
        """
        try:
            user_start = "<|start_header_id|>user<|end_header_id|>"
            user_end = "<|eot_id|>"
            user_content = full_text.split(user_start)[-1].split(user_end)[0].strip()
            return user_content
        except IndexError:
            return full_text.strip()

    def extract_ground_truth(self, full_text):
        """
        Extracts the 'assistant' content from the text based on markers.
        """
        try:
            assistant_start = "<|start_header_id|>assistant<|end_header_id|>"
            assistant_end = "<|eot_id|>"
            assistant_content = full_text.split(assistant_start)[-1].split(assistant_end)[0].strip()
            return assistant_content
        except IndexError:
            return "None"

    def on_evaluate(self, args, state, control, **kwargs):
        try:
            ########## 코드 수정해야됨 inference.ipynb 였나에 있을거임
            print("\n============= 테스트셋 평가 시작 =============")
            for i in range(min(self.num_examples, len(self.eval_dataset))):
                sample_data = self.eval_dataset[i]
                full_text = sample_data["text"]
                user_input = self.extract_user_input(full_text)
                ground_truth = self.extract_ground_truth(full_text)

                eval_conversation = [
                    {"role": "system", "content": "You are an expert in detecting idioms. Identify only the idiom exactly as it appears in the given sentence. Do not provide additional explanations or context interpretation. If there is no idiom, respond with 'None.'"},
                    {"role": "user", "content": user_input}
                ]

                eval_text = self.tokenizer.apply_chat_template(eval_conversation, tokenize=False)
                inputs = self.tokenizer(eval_text, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
                print("\n============= TEST EVAL =============")
                print(f"입력 문장: {user_input}")
                
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=256,
                    num_return_sequences=1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )

                # 디코딩
                decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                #print(f"Decoded Output: {decoded_output}")  # 디버깅 출력

                # "assistant" 이후 텍스트 추출
                if "assistant" in decoded_output:
                    detected_idiom = decoded_output.split("assistant")[-1].strip()
                else:
                    detected_idiom = decoded_output.strip()
                
                print(f"Ground Truth: {ground_truth}")
                print(f"모델 예측: {detected_idiom}")
                #     is_correct = (detected_idiom == ground_truth)
                #     print(f"정답 여부: {'✓' if is_correct else '✗'}")

                #     self.total_samples += 1
                #     self.correct_predictions += 1 if is_correct else 0

                # accuracy = self.correct_predictions / self.total_samples if self.total_samples > 0 else 0
                # print(f"\n============= 평가 요약 =============")
                # print(f"총 샘플 수: {self.total_samples}")
                # print(f"정확한 예측: {self.correct_predictions}")
                # print(f"정확도: {accuracy:.2%}")
        except Exception as e:
            print(f"\n[WARNING] Evaluation callback encountered an error: {e}")

train_dataset = load_dataset('train_idioms_detection_dataset.jsonl', is_test=False, tokenizer=tokenizer)
test_dataset = load_dataset('test_idioms_detection_dataset.jsonl', is_test=True, tokenizer=tokenizer)

# SFT Trainer
training_args = SFTConfig(
    output_dir="./model_output/llama70_new-2",
    dataset_text_field='text',
    remove_unused_columns=False,
    per_device_train_batch_size=8, # 원래 2, 4되고(new), 8해보기(2)
    gradient_accumulation_steps=64,
    logging_steps=1,  
    log_level='info', 
    logging_dir='./logs',  
    learning_rate=1e-4, # 원래 1e-8 -> 5e-5 -> (너무 안정적이어서) 1e-4
    packing=True,
    num_train_epochs=5, # 원래 3
    save_strategy='steps',  # 'steps'로 변경
    save_steps=30,          # 원래 10으로 해놓긴함
    # evaluation_strategy='steps', 
    # eval_steps=5,                
)
sft_trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,  
    args=training_args,
    max_seq_length=400
)

# 평가 콜백 추가
# eval_callback = AdvancedIdiomEvalCallback(
#     trainer=sft_trainer,
#     eval_dataset=test_dataset,
#     tokenizer=tokenizer,
#     num_examples=1  
# )
# sft_trainer.add_callback(eval_callback)

# 모델 훈련
#print(f"Added tokens: {tokenizer.added_tokens_encoder}")
#print(f"Vocab before resize: {model.config.vocab_size}")
model.resize_token_embeddings(len(tokenizer))
# print(f"Vocab after resize: {model.config.vocab_size}")
# print(f"Training START : {model_id}")
sft_trainer.train()

# 모델 저장
#model.resize_token_embeddings(len(tokenizer))
# print(f"Final Vocab of Tokenizer : {len(tokenizer)}")
# print(f"Final Vocab of Model : {model.config.vocab_size}")
tokenizer.save_pretrained(training_args.output_dir) 
sft_trainer.save_model(training_args.output_dir)
print("DONE!!")
