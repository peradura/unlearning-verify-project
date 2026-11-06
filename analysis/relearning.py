import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
from datasets import load_dataset

# ===========================================================================
# 💡 [설정 구역] 재학습할 언러닝 모델 경로 및 저장 디렉토리 설정
# ===========================================================================
# 1. 재학습 대상 언러닝 모델 (HF Hub ID 또는 로컬 가중치 디렉토리)
# 예시:
# - GradDiff: "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha5_epoch5"
# - AltPO:    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr5e-05_beta0.1_alpha1_epoch10"
# - NPO:      "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr2e-05_beta0.5_alpha1_epoch10"
MODEL_PATH = "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha5_epoch5"

# 2. 재학습 완료 후 가중치를 저장할 로컬 디렉토리 경로
SAVE_DIR = "./checkpoints/4_graddiff_relearned_50"

# 3. 재학습 하이퍼파라미터
EPOCHS = 3
LEARNING_RATE = 1e-5
BATCH_SIZE = 4
MAX_LENGTH = 512
# ===========================================================================

print(f"=======================================================")
print(f" Start 50% Forget Re-learning Fine-Tuning")
print(f" Target Unlearned Model: {MODEL_PATH}")
print(f" Save Directory       : {SAVE_DIR}")
print(f" Epochs: {EPOCHS} | LR: {LEARNING_RATE} | Batch Size: {BATCH_SIZE}")
print(f"=======================================================")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Tokenizer 및 언러닝 모델 로드
# ---------------------------------------------------------------------------
print("\n[1/4] Loading Tokenizer and Unlearned Model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)

# ---------------------------------------------------------------------------
# 2. TOFU 데이터셋 로드 및 Forget 데이터 50% 분할
# ---------------------------------------------------------------------------
print("\n[2/4] Loading TOFU Dataset and Splitting 50% Forget Data...")
tofu_forget = list(load_dataset("locuslab/TOFU", "forget10")["train"])

# 전체 Forget 샘플 400개 중 전반부 50%(200개)만 재학습용 데이터로 추출
split_idx = len(tofu_forget) // 2
train_forget_50 = tofu_forget[:split_idx]

print(f" -> Total Forget Samples: {len(tofu_forget)}")
print(f" -> Samples used for Re-learning (50%): {len(train_forget_50)}")
print(f" -> Samples reserved for Testing (50%): {len(tofu_forget) - split_idx}")

# 데이터 콜레이터 함수
def collate_fn(batch):
    texts = [d['question'] + " " + d['answer'] for d in batch]
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    )
    inputs['labels'] = inputs['input_ids'].clone()
    # Pad 토큰 위치는 Loss 계산에서 제외 (-100)
    inputs['labels'][inputs['labels'] == tokenizer.pad_token_id] = -100
    return inputs

dataloader = DataLoader(train_forget_50, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# ---------------------------------------------------------------------------
# 3. 50% Forget 데이터 Fine-Tuning 진행
# ---------------------------------------------------------------------------
print("\n[3/4] Starting Re-learning Training Loop...")
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    for step, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f" -> Epoch {epoch + 1}/{EPOCHS} Finished. Average Loss: {avg_loss:.4f}")

# ---------------------------------------------------------------------------
# 4. 재학습된 가중치 및 토크나이저 저장 (save_pretrained)
# ---------------------------------------------------------------------------
print(f"\n[4/4] Saving Re-learned model checkpoint to: {SAVE_DIR} ...")
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print(f"\nSuccessfully completed 50% re-learning and saved model to '{SAVE_DIR}'!")