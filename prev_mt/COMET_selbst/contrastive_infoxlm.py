import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import XLMRobertaModel, XLMRobertaTokenizer
from datasets import Dataset
from tqdm import tqdm

# 1. 데이터 로드
en_idiom = Dataset.from_csv('EN_idiom_3990.csv')
kr_idiom = Dataset.from_csv('KR_idiom_3989.csv')
enkr_idiom = Dataset.from_csv('ENKR-hypothesis.csv')
kren_idiom = Dataset.from_csv('KREN-hypothesis.csv')

# 데이터 예시
print("English Idiom Sample:", en_idiom[0])
print("Korean Idiom Sample:", kr_idiom[0])
print("EN->KR Hypothesis Sample:", enkr_idiom[0])
print("KR->EN Hypothesis Sample:", kren_idiom[0])

# 2. 모델 및 토크나이저 로드
model_name = "microsoft/infoxlm-base"
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
model = XLMRobertaModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 3. Contrastive Loss 정의
def contrastive_loss(query, positive, negatives, temperature=0.07):
    pos_similarity = F.cosine_similarity(query, positive, dim=-1) / temperature
    neg_similarity = torch.cat([
        F.cosine_similarity(query, neg, dim=-1).unsqueeze(1) for neg in negatives
    ], dim=1) / temperature
    logits = torch.cat([pos_similarity.unsqueeze(1), neg_similarity], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long).to(query.device)
    loss = F.cross_entropy(logits, labels)
    return loss

# 4. 데이터 전처리
def tokenize_and_embed(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # [CLS] 벡터 반환

# 5. 데이터셋 준비
def prepare_batch(batch):
    query = tokenize_and_embed(batch['source'])
    positive = tokenize_and_embed(batch['target'])
    negatives = [tokenize_and_embed(neg) for neg in batch['negative']]
    return query, positive, negatives

# 6. DataLoader 설정
def collate_fn(batch):
    sources, targets, negatives = [], [], []
    for item in batch:
        sources.append(item["source"])
        targets.append(item["target"])
        negatives.append(item["negative"])
    return {"source": sources, "target": targets, "negative": negatives}

train_data = DataLoader(enkr_idiom, batch_size=16, shuffle=True, collate_fn=collate_fn)

# 7. Optimizer 설정
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# 8. 학습 루프
epochs = 3
model.train()

for epoch in range(epochs):
    total_loss = 0
    with tqdm(train_data, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
        for batch in pbar:
            query, positive, negatives = prepare_batch(batch)
            loss = contrastive_loss(query, positive, negatives)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"Loss": total_loss / (pbar.n or 1)})

    print(f"Epoch {epoch+1}: Average Loss: {total_loss / len(train_data)}")

# 9. 모델 저장
model.save_pretrained("./contrastive_infoxlm")
tokenizer.save_pretrained("./contrastive_infoxlm")

print("Training complete. Model saved to './contrastive_infoxlm'")
