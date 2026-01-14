import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# 定义模型和对应的文件路径
FILES = {
    "LLaMA-3.1-8B (K=3)": "result_llama3.1_8b_K=3_tem=1.json", 
}

# 需要指定模型
CURRENT_MODEL_KEY = "LLaMA-3.1-8B (K=3)"  


def get_accuracy_at_percentiles(y_true, scores):
    """
    计算不同百分位下的准确率
    """
    # 传入的 scores 越大越自信
    sorted_indices = np.argsort(scores)[::-1] 
    y_sorted = y_true[sorted_indices]
    total = len(y_sorted)
    
    # Retention Rate: 5%, 10%, ..., 100%
    percentiles = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 
                   55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    
    results = {}
    
    for p in percentiles:
        k = int(total * (p / 100))
        if k < 1: k = 1 
        acc = np.mean(y_sorted[:k])
        results[p] = acc
        
    return results

def plot_heatmap(df, title, filename, cmap, center=None, annot_fmt=".3f"):
    
    # 通用热力图绘制函数

    plt.figure(figsize=(6, 10)) 
    
    sns.heatmap(df, annot=True, fmt=annot_fmt, cmap=cmap, center=center,
                cbar_kws={'label': 'Accuracy' if center is None else 'Accuracy Gain (Delta)'})
    
    plt.title(title)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f" 图表已保存: {filename}")
    plt.close() 

# 主程序

def main():
    
    filepath = FILES.get(CURRENT_MODEL_KEY)
    if not filepath or not os.path.exists(filepath):
        print(f" 错误: 找不到文件 {filepath}，请检查路径或 FILES 配置。")
        return

    print(f" 正在处理: {CURRENT_MODEL_KEY} | 文件: {filepath}")
    
    output_base = Path(filepath).stem

    with open(filepath, 'r') as f:
        data = json.load(f)
        
    y_true = np.array([1 if d['is_correct'] else 0 for d in data])
    
    # 写入 scores
    score_prob = np.array([d['score_prob'] for d in data])
    score_ent = np.array([d['score_entropy'] for d in data]) # 假设json里已经是负熵或越大越好了
    
    # LogTokU score
    if 'score_product' in data[0]:
        score_ours = np.array([d['score_product'] for d in data])
    else:
        score_ours = np.array([- (d.get('score_eu', 0) * d.get('score_au', 0)) for d in data])
    
    # EU 和 AU 
    score_eu = np.array([d['score_eu'] for d in data])
    score_au = np.array([d['score_au'] for d in data])

    # 计算准确率
    dict_prob = get_accuracy_at_percentiles(y_true, score_prob)
    dict_ent  = get_accuracy_at_percentiles(y_true, score_ent)
    dict_ours = get_accuracy_at_percentiles(y_true, score_ours)
    dict_eu = get_accuracy_at_percentiles(y_true, score_eu)
    dict_au = get_accuracy_at_percentiles(y_true, score_au)

    # 图 1: 绝对准确率热力图
    df_acc = pd.DataFrame({
        'Prob (Baseline)': dict_prob,
        'Entropy': dict_ent,
        'LogTokU': dict_ours,
        'EU': dict_eu,
        'AU': dict_au
    })
    df_acc.index.name = 'Retention Percentile'
    
    plot_heatmap(
        df_acc, 
        title=f"Accuracy vs. Retention Rate\n({CURRENT_MODEL_KEY})",
        filename=f"{output_base}_acc_heatmap.png",
        cmap="viridis_r" 
    )

    # 图 2: Delta Heatmap
    
    df_delta = pd.DataFrame()
    
    # 计算公式：LogTokU - Prob 和 LogTokU - Entropy
    percentiles = df_acc.index

    df_delta = pd.DataFrame({
        'LogTokU vs Prob': df_acc['LogTokU'] - df_acc['Prob (Baseline)'],
        'LogTokU vs Entropy': df_acc['LogTokU'] - df_acc['Entropy']
    })
    
    df_delta.index.name = 'Retention Percentile'

    plot_heatmap(
        df_delta,
        title=f"Performance Gain over Baseline \n({CURRENT_MODEL_KEY})",
        filename=f"{output_base}_delta_heatmap.png",
        cmap="RdBu_r", 
        center=0       
    )

if __name__ == "__main__":
    main()