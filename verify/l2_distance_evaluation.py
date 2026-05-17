import torch
import numpy as np
from transformers import AutoConfig, AutoModelForCausalLM

def _load_weights(path, device):
    ckpt = torch.load(path, map_location=device)
    return ckpt.get('model_state_dict', ckpt)

def calculate_metrics(config_data, orig_weights, unlearn_weights, unlearn_dataset, tokenizer, device="cuda", num_samples: int=4):
    """
    1) 레이어별 가중치 L2 Distance 계산
    2) 데이터셋 기반 순전파 Activation Distance 계산
    """
    config = AutoConfig.from_dict(config_data) if isinstance(config_data, dict) else AutoConfig.from_pretrained(config_data)
    
    model_orig = AutoModelForCausalLM.from_config(config).to(device).eval()
    model_unlearn = AutoModelForCausalLM.from_config(config).to(device).eval()
    
    sd_orig = _load_weights(orig_weights, device) if isinstance(orig_weights, str) else orig_weights
    sd_unlearn = _load_weights(unlearn_weights, device) if isinstance(unlearn_weights, str) else unlearn_weights
    
    model_orig.load_state_dict(sd_orig, strict=False)
    model_unlearn.load_state_dict(sd_unlearn, strict=False)
    
    # ----------------------------------------------------
    # [검증 1] 가중치 L2 Distance 계산
    # ----------------------------------------------------
    weight_l2_dict = {}
    for key in sd_orig.keys():
        if key not in sd_unlearn or 'weight' not in key:
            continue
        w_orig = sd_orig[key].float()
        w_unlearn = sd_unlearn[key].float()
        
        l2_dist = torch.norm(w_orig - w_unlearn, p=2).item()
        norm_l2 = l2_dist / np.sqrt(w_orig.numel()) if w_orig.numel() > 0 else 0
        weight_l2_dict[key] = norm_l2

    # # ----------------------------------------------------
    # # [검증 2] 데이터셋 순전파 Activation 차이 계산
    # # ----------------------------------------------------
    # samples = [batch['text'] for batch in list(unlearn_dataset)[:num_samples]]
    # inputs = tokenizer(samples, return_tensors="pt", padding=True, truncation=True).to(device)
    
    # with torch.no_grad():
    #     outputs_orig = model_orig(**inputs, output_hidden_states=True)
    #     outputs_unlearn = model_unlearn(**inputs, output_hidden_states=True)
        
    #     activation_distances = []
    #     for h_orig, h_unlearn in zip(outputs_orig.hidden_states, outputs_unlearn.hidden_states):
    #         dist = torch.norm(h_orig - h_unlearn, p=2, dim=-1).mean().item()
    #         activation_distances.append(dist)
            
    del model_orig, model_unlearn
    torch.cuda.empty_cache()
    
    # return weight_l2_dict, activation_distances
    return weight_l2_dict
