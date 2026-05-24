from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
import asyncio
import uuid
import os
import shutil
import torch
import json
import argparse
import numpy as np
import math
from safetensors.torch import load_file

from verify.KL_divergence import compute_pure_kl_divergence
from verify.mia_evaluation import run_evaluation
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logs = {}
results = {}

BEFORE_MODEL_DIR = "models/before"
AFTER_MODEL_DIR = "models/after"

@app.get("/")
async def root():
    return HTMLResponse(open("frontend/시연2.html", encoding="utf-8").read())

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/verify")
async def verify(
    background_tasks: BackgroundTasks,
    config: UploadFile = File(...),
    forget_set: UploadFile = File(...),
    retain_set: UploadFile = File(...),
    before_weights: UploadFile = File(...),
    after_weights: UploadFile = File(...),
):
    session_id = str(uuid.uuid4())
    session_dir = f"/tmp/{session_id}"
    os.makedirs(session_dir, exist_ok=True)

    file_map = {
        "config.json": config,
        "forget_set.jsonl": forget_set,
        "retain_set.jsonl": retain_set,
        "before_weights.pt": before_weights,
        "after_weights.pt": after_weights,
    }
    for filename, upload_file in file_map.items():
        path = os.path.join(session_dir, filename)
        with open(path, "wb") as f:
            shutil.copyfileobj(upload_file.file, f)

    logs[session_id] = []
    background_tasks.add_task(run_verification, session_id, session_dir)

    return {"session_id": session_id}


def compute_perplexity(model, tokenizer, texts, max_length=256):
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            labels = inputs.input_ids.clone()
            outputs = model(**inputs, labels=labels)
            total_loss += outputs.loss.item()
            count += 1
    return math.exp(total_loss / count) if count > 0 else 0


def run_verification(session_id: str, session_dir: str):
    def log(msg):
        logs[session_id].append(msg)
        print(msg)

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"디바이스: {device}")

        forget_path = os.path.join(session_dir, "forget_set.jsonl")
        retain_path = os.path.join(session_dir, "retain_set.jsonl")

        # ① KL Divergence
        log("토크나이저 로딩 중...")
        tokenizer = AutoTokenizer.from_pretrained(BEFORE_MODEL_DIR)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        log("모델 로딩 중 (before)...")
        model_before = AutoModelForCausalLM.from_pretrained(
            BEFORE_MODEL_DIR, torch_dtype=torch.float16
        )

        log("모델 로딩 중 (after)...")
        model_after = AutoModelForCausalLM.from_pretrained(
            AFTER_MODEL_DIR, torch_dtype=torch.float16
        )

        log("KL Divergence 계산 중...")
        with open(forget_path, "r") as f:
            forget_data = [json.loads(line) for line in f if line.strip()]

        kl_scores = []
        for i, sample in enumerate(forget_data[:20]):
            text = f"{sample.get('question','').strip()} {sample.get('answer','').strip()}"
            score = compute_pure_kl_divergence(model_before, model_after, tokenizer, text)
            kl_scores.append(score)
            log(f"KL 샘플 {i+1}/20: {score:.4f}")

        kl_avg = sum(kl_scores) / len(kl_scores) if kl_scores else 0
        log(f"KL Divergence 완료: 평균 {kl_avg:.4f}")

        # ② L2 Distance
        log("Layer L2 Distance 계산 중...")
        sd_before = load_file(f"{BEFORE_MODEL_DIR}/model.safetensors")
        sd_after = load_file(f"{AFTER_MODEL_DIR}/model.safetensors")

        weight_l2 = {}
        for key in sd_before.keys():
            if key not in sd_after or 'weight' not in key:
                continue
            w_orig = sd_before[key].float()
            w_unlearn = sd_after[key].float()
            l2_dist = torch.norm(w_orig - w_unlearn, p=2).item()
            norm_l2 = l2_dist / np.sqrt(w_orig.numel()) if w_orig.numel() > 0 else 0
            weight_l2[key] = norm_l2

        top_layers = sorted(weight_l2.items(), key=lambda x: x[1], reverse=True)[:5]
        max_l2 = top_layers[0][1] if top_layers else 0
        log(f"L2 Distance 완료: 최대 {max_l2:.4f} ({top_layers[0][0] if top_layers else 'N/A'})")

        # ③ MIA
        log("MIA 평가 중...")
        mia_args = argparse.Namespace(
            model_path=AFTER_MODEL_DIR,
            forget_data=forget_path,
            nonmember_data=retain_path,
            max_samples=100,
            max_length=256,
            batch_size=4,
            method="loss",
            k=0.2,
            output=os.path.join(session_dir, "mia_results.json")
        )
        run_evaluation(mia_args)

        with open(mia_args.output, "r") as f:
            mia_result = json.load(f)
        mia_auc = mia_result.get("loss", {}).get("auc", 0.5)
        mia_score = round(mia_auc * 100, 1)
        log(f"MIA 완료: AUC {mia_auc:.4f} ({mia_score}%)")

        # ④ Retain KL Divergence
        log("Retain KL Divergence 계산 중...")
        with open(retain_path, "r") as f:
            retain_data = [json.loads(line) for line in f if line.strip()]

        retain_kl_scores = []
        for i, sample in enumerate(retain_data[:20]):
            text = f"{sample.get('question','').strip()} {sample.get('answer','').strip()}"
            score = compute_pure_kl_divergence(model_before, model_after, tokenizer, text)
            retain_kl_scores.append(score)
            log(f"Retain KL 샘플 {i+1}/20: {score:.4f}")

        retain_kl_avg = sum(retain_kl_scores) / len(retain_kl_scores) if retain_kl_scores else 0
        log(f"Retain KL 완료: {retain_kl_avg:.4f}")

        # ⑤ Perplexity
        log("Perplexity 계산 중...")
        texts_retain = [
            f"{s.get('question','').strip()} {s.get('answer','').strip()}"
            for s in retain_data[:20]
        ]
        ppl_before = compute_perplexity(model_before, tokenizer, texts_retain)
        ppl_after = compute_perplexity(model_after, tokenizer, texts_retain)
        ppl_delta = round((ppl_after - ppl_before) / ppl_before * 100, 1) if ppl_before > 0 else 0
        log(f"Perplexity 완료: before={ppl_before:.2f} after={ppl_after:.2f} delta={ppl_delta:+.1f}%")

        del model_before, model_after

        # 종합 점수
        kl_pass = kl_avg >= 1.5
        mia_pass = 45 <= mia_score <= 55
        overall = min(round((kl_avg / 5.0 * 50) + (50 - abs(mia_score - 50)) + (max_l2 * 10), 1), 100)
        verdict = "PASS" if (kl_pass and mia_pass) else "FAIL"

        log(f"검증 완료! 종합 점수: {overall} — {verdict}")

        results[session_id] = {
            "kl_divergence": round(kl_avg, 4),
            "mia_score": mia_score,
            "l2_distance": round(max_l2, 4),
            "l2_layers": dict(top_layers),
            "retain_kl": round(retain_kl_avg, 4),
            "ppl_before": round(ppl_before, 2),
            "ppl_after": round(ppl_after, 2),
            "ppl_delta": ppl_delta,
            "overall_score": overall,
            "verdict": verdict
        }

    except Exception as e:
        log(f"오류 발생: {str(e)}")
        results[session_id] = {"error": str(e), "verdict": "ERROR"}


@app.get("/verify/stream/{session_id}")
async def stream(session_id: str):
    async def event_generator():
        sent = 0
        while True:
            if session_id in logs:
                current = logs[session_id]
                while sent < len(current):
                    yield f"data: {current[sent]}\n\n"
                    sent += 1
                if session_id in results:
                    yield f"data: DONE\n\n"
                    break
            await asyncio.sleep(0.3)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/report/{session_id}")
async def get_report(session_id: str):
    if session_id in results:
        return results[session_id]
    return {"error": "결과 없음"}