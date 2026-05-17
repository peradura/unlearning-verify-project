import matplotlib.pyplot as plt

def plot_unified_verification(weight_metrics, activation_metrics, target_layer_keyword="layers.12", save_path=None):
    """ 두 검증 결과 데이터를 받아 하나의 피겨에 세로로 배치하여 시각화 """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # 1. 상단 차트: 가중치 변화
    w_keys = list(weight_metrics.keys())
    w_values = list(weight_metrics.values())
    w_short_keys = [k.replace("model.layers.", "L").replace(".weight", "") for k in w_keys]
    
    bars1 = ax1.bar(range(len(w_short_keys)), w_values, color='lightgray', edgecolor='k')
    for idx, key in enumerate(w_keys):
        if target_layer_keyword in key:
            bars1[idx].set_color('crimson') # 타겟 레이어 강조
            
    ax1.set_xticks(range(len(w_short_keys)))
    ax1.set_xticklabels(w_short_keys, rotation=90, fontsize=7)
    ax1.set_ylabel('Normalized Weight $L_2$ Distance')
    ax1.set_title(f'1. Layer-wise Weight Variations (Target: {target_layer_keyword})')
    ax1.grid(axis='y', linestyle=':', alpha=0.6)

    # 2. 하단 차트: Activation 변화
    act_labels = [f"L{i}" for i in range(len(activation_metrics))]
    ax2.bar(range(len(activation_metrics)), activation_metrics, color='skyblue', edgecolor='k')
    ax2.set_xticks(range(len(activation_metrics)))
    ax2.set_xticklabels(act_labels, rotation=45)
    ax2.set_ylabel('Activation Mean Distance')
    ax2.set_title('2. Forward Pass Activation Distance (Using Unlearning Dataset)')
    ax2.grid(axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()