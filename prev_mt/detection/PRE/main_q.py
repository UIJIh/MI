"""
0.9838, 9e5lr, 50eps, 50steps
    - 아니.. 0.09xx까지 나와야함.. 
"""
"""
에폭 늘려 80
"""
"""
SFT 할거임 9e-5
"""
import torch, wandb
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dataset import dataset_and_tokenizer
from trl import SFTConfig, SFTTrainer

from datasets import load_from_disk

# 1. Data and Tokenizer setup (Assuming 'dataset_and_tokenizer' is defined)
model_path = "meta-llama/Meta-Llama-3.1-70B-Instruct"
data_path_1 = "/data/uijih/Seed50_for_Parallel_Dataset_ENKR_idiomKB_0.8_example.csv"
data_path_2 = "/data/uijih/detection/daataset_meaning.csv"
shuffled_tokenized_full_dataset, tokenizer = dataset_and_tokenizer(data_path_1, data_path_2, model_path)

# 2. Model loading with 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True, # 메모리 때문에 이중양자화 (성능 좀 낮아질수도)
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    quantization_config=bnb_config
)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
#model.enable_input_require_grads() # lora

# 3. Apply QLoRA
lora_config = LoraConfig(
    r=64, # 커질수록 복잡도 향상 그러나 메모리
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

for name, module in model.named_modules():
    print(name)

model.print_trainable_parameters()
# 토크나이저 설정
tokenizer.add_special_tokens({"pad_token": "<pad>", "eos_token": "<eos>", "bos_token": "<bos>"})
model.resize_token_embeddings(len(tokenizer))
model.print_trainable_parameters()
model.config.use_cache = False

# 4. Trainer setup
# trainer = Trainer(
#     model=model,
#     train_dataset=shuffled_tokenized_full_dataset,
#     #eval_dataset=eval_dataset,
#     args=TrainingArguments(
#         output_dir="./output_70b",
#         per_device_train_batch_size=4,
#         gradient_accumulation_steps=16,
#         num_train_epochs=50, #
#         learning_rate=9e-5, # 5-5 too stable -> 
#         bf16=True,
#         logging_steps=1,
#         logging_dir="./logs",
#         save_steps=75,
#         # evaluation_strategy="steps",
#         # eval_steps=1000,
#         # report_to=["wandb"],
#         # load_best_model_at_end=True,
#         optim="paged_adamw_8bit",
#     ),
#     data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
# )

training_args = SFTConfig(
    output_dir="./model_output/{}_sft_{}".format("Meta-Llama-3.1-70B", "detection"),
    dataset_text_field='text',
    remove_unused_columns=False,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=64,
    logging_steps=1,
    learning_rate=1e-5, # 0.9504 / 9e-5보다 낮춰야할듯(0.57에 20epoch)
    packing=True,
    num_train_epochs=30, # 5보다 더 가야하긴할듯
    save_strategy='no'
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=shuffled_tokenized_full_dataset,
    max_seq_length=350
)

# 5. Initialize Weights & Biases
# wandb.init(
#     project="Your_Project_Name",
#     name="Run_Name",
#     config=trainer.args,
# )

# 6. Start training
trainer.train()

# 7. Save model and tokenizer
model.save_pretrained("./saveded_instruct-70-detection-new")
tokenizer.save_pretrained("./saveded_instruct-70-detection-new")
print("Model saved successfully!")
