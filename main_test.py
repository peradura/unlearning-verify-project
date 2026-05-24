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
import torch.nn.functional as F

from verify.KL_divergence import compute_pure_kl_divergence
from verify.mia_evaluation import run_evaluation
from verify.rogue import run_log_prob_evaluation
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


def compute_neuron_activation(model_before, model_after, tokenizer, forget_data, max_samples=50):
    """
    forget set 입력 시 before/after 모델의 뉴런 활성화 패턴 코사인 유사도 계산
    낮을수록 언러닝 성공
    """
    # act_store 매번 초기화
    act_store = {
        "original": {"total_val": 0.0, "active_count": 0, "total_count": 0, "vectors": []},
        "unlearned": {"total_val": 0.0, "active_count": 0, "total_count": 0, "vectors": []}
    }

    thresh = 0.01
    hooks = []

    def make_hook(model_type):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            act_val = out.detach().float().mean(dim=(0, 1)).abs()
            act_store[model_type]["total_val"] += act_val.sum().item()
            act_store[model_type]["active_count"] += (act_val > thresh).sum().item()
            act_store[model_type]["total_count"] += act_val.numel()
            act_store[model_type]["vectors"].append(act_val)
        return hook_fn

    # 레이어 범위 설정 (50%~90% 구간)
    layers_before = model_before.model.layers
    layers_after = model_after.model.layers
    total = len(layers_before)
    start = int(total * 0.5)
    end = int(total * 0.9)

    try:
        for idx in range(start, end + 1):
            hooks.append(layers_before[idx].mlp.gate_proj.register_forward_hook(make_hook("original")))
            hooks.append(layers_after[idx].mlp.gate_proj.register_forward_hook(make_hook("unlearned")))
    except AttributeError:
        # gate_proj 없는 아키텍처면 전체 레이어에 훅
        for idx in range(start, end + 1):
            hooks.append(layers_before[idx].register_forward_hook(make_hook("original")))
            hooks.append(layers_after[idx].register_forward_hook(make_hook("unlearned")))

    model_before.eval()
    model_after.eval()

    with torch.no_grad():
        for sample in forget_data[:max_samples]:
            text = f"{sample.get('question', sample.get('prompt',''))} {sample.get('answer', sample.get('response',''))}"
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            model_before(**inputs)
            model_after(**inputs)

    for h in hooks:
        h.remove()

    if not act_store["original"]["vectors"] or not act_store["unlearned"]["vectors"]:
        return 0.0, 0.0, 0.0

    v_orig = torch.stack(act_store["original"]["vectors"]).mean(dim=0)
    v_unlearn = torch.stack(act_store["unlearned"]["vectors"]).mean(dim=0)
    cos_sim = F.cosine_similarity(v_orig.unsqueeze(0), v_unlearn.unsqueeze(0)).item()

    orig_active_pct = (act_store["original"]["active_count"] / act_store["original"]["total_count"]) * 100
    unlearn_active_pct = (act_store["unlearned"]["active_count"] / act_store["unlearned"]["total_count"]) * 100

    return round(cos_sim * 100, 2), round(orig_active_pct, 2), round(unlearn_active_pct, 2)


def run_verification(session_id: str, session_dir: str):
    def log(msg):
        logs[session_id].append(msg)
        print(msg)

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"디바이스: {device}")

        forget_path = os.path.join(session_dir, "forget_set.jsonl")
        retain_path = os.path.join(session_dir, "retain_set.jsonl")

        # forget set 로드
        with open(forget_path, "r") as f:
            first_char = f.read(1)
        with open(forget_path, "r") as f:
            if first_char == '[':
                forget_data = json.load(f)
            else:
                forget_data = [json.loads(line) for line in f if line.strip()]

        # retain set 로드
        with open(retain_path, "r") as f:
            first_char = f.read(1)
        with open(retain_path, "r") as f:
            if first_char == '[':
                retain_data = json.load(f)
            else:
                retain_data = [json.loads(line) for line in f if line.strip()]

        # ① 모델 로딩
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

        # ② KL Divergence
        log("KL Divergence 계산 중...")
        kl_scores = []
        for i, sample in enumerate(forget_data[:20]):
            text = f"{sample.get('question', sample.get('prompt',''))} {sample.get('answer', sample.get('response',''))}"
            score = compute_pure_kl_divergence(model_before, model_after, tokenizer, text)
            kl_scores.append(score)
            log(f"KL 샘플 {i+1}/20: {score:.4f}")

        kl_avg = sum(kl_scores) / len(kl_scores) if kl_scores else 0
        log(f"KL Divergence 완료: 평균 {kl_avg:.4f}")

        # ③ L2 Distance
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
        log(f"L2 Distance 완료: 최대 {max_l2:.4f}")

        # ④ MIA
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

        # ⑤ Neuron Activation
        log("뉴런 활성화 분석 중...")
        cos_sim, orig_pct, unlearn_pct = compute_neuron_activation(
            model_before, model_after, tokenizer, forget_data
        )
        log(f"뉴런 활성화 완료: 코사인 유사도 {cos_sim:.2f}% (원본 활성 {orig_pct:.2f}% → 언러닝 {unlearn_pct:.2f}%)")

        # ⑥ Rogue (Log Prob)
        log("Log Probability 평가 중...")
        # TOFU 포맷 (question/answer) → rogue 포맷 (prompt/response) 변환
        forget_path_rogue = os.path.join(session_dir, "forget_rogue.json")
        rogue_data = [
            {"prompt": s.get("question", s.get("prompt", "")),
             "response": s.get("answer", s.get("response", ""))}
            for s in forget_data[:100]
        ]
        with open(forget_path_rogue, "w") as f:
            json.dump(rogue_data, f)

        rogue_result = run_log_prob_evaluation(
            orig_path=BEFORE_MODEL_DIR,
            unlearn_path=AFTER_MODEL_DIR,
            forget_data_path=forget_path_rogue,
            max_samples=50
        )
        log_prob_gap = rogue_result.get("pure_log_prob_gap", 0)
        log(f"Log Prob 완료: gap={log_prob_gap:.4f}")

        # ⑦ Retain KL Divergence
        log("Retain KL Divergence 계산 중...")
        retain_kl_scores = []
        for i, sample in enumerate(retain_data[:20]):
            text = f"{sample.get('question', sample.get('prompt',''))} {sample.get('answer', sample.get('response',''))}"
            score = compute_pure_kl_divergence(model_before, model_after, tokenizer, text)
            retain_kl_scores.append(score)
            log(f"Retain KL 샘플 {i+1}/20: {score:.4f}")

        retain_kl_avg = sum(retain_kl_scores) / len(retain_kl_scores) if retain_kl_scores else 0
        log(f"Retain KL 완료: {retain_kl_avg:.4f}")

        # ⑧ Perplexity
        log("Perplexity 계산 중...")
        texts_retain = [
            f"{s.get('question', s.get('prompt',''))} {s.get('answer', s.get('response',''))}"
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
        overall = min(round(
            (kl_avg / 5.0 * 50) +
            (50 - abs(mia_score - 50)) +
            (max_l2 * 10), 1), 100)
        verdict = "PASS" if (kl_pass and mia_pass) else "FAIL"

        log(f"검증 완료! 종합 점수: {overall} — {verdict}")

        results[session_id] = {
            "kl_divergence": round(kl_avg, 4),
            "mia_score": mia_score,
            "l2_distance": round(max_l2, 4),
            "l2_layers": dict(top_layers),
            "neuron_cos_sim": cos_sim,
            "neuron_orig_pct": orig_pct,
            "neuron_unlearn_pct": unlearn_pct,
            "log_prob_gap": round(log_prob_gap, 4),
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