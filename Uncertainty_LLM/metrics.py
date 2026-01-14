from cmath import exp
import numpy as np
from scipy.special import softmax, digamma

def topk(arr, k):
    indices = np.argpartition(arr, -k)[-k:]
    values = arr[indices]
    return values, indices

def softmax_stable(logits):
    """数值稳定的 Softmax"""
    # Revise: x - max(x) 保证了指数部分最大是 0 (exp(0)=1)，防止 exp 爆炸。这不会改变 softmax 的结果。
    logits_shifted = logits - np.max(logits)
    exps = np.exp(logits_shifted)
    return exps / np.sum(exps)

def get_eu(mode="prob", k=None):
    if mode == "eu":
        if k is None:
            raise ValueError("k must be provided for 'eu' mode.")

        def eu(logits):
            top_k = k
            if len(logits) < top_k:
                raise ValueError("Logits array length is less than top_k.")
            top_values = np.partition(logits, -top_k)[-top_k:]
            temperature = 5.0 # 可以尝试 2.0 到 10.0 之间的值
            alpha = np.exp(top_values / temperature)
            mean_scores = top_k / (np.sum(alpha) + top_k)
            return mean_scores

        return eu

    elif mode == "prob":
        def eu(logits):
            logits = softmax_stable(logits)
            top_k = 1
            if len(logits) < top_k:
                raise ValueError("Logits array length is less than top_k.")
            top_values, _ = topk(logits, top_k)
            return top_values[0]

        return eu

    elif mode == "entropy":
        def eu(logits):
            probs = softmax_stable(logits)
            log_probs = np.zeros_like(probs)
            mask = probs > 0
            log_probs[mask] = np.log(probs[mask])
            entropy = -np.sum(probs * log_probs)
            return entropy

        return eu

    elif mode == "au":
        def cal_au(logits):
            top_k = k # 确保 k 被传入
            if len(logits) < top_k:
                raise ValueError("...")
            
            top_values = np.partition(logits, -top_k)[-top_k:]
            
            # 应用 exp，确保 Alpha 非负
            alpha = np.exp(top_values)
            
            alpha_0 = alpha.sum(keepdims=True)
            
            if alpha_0 == 0:
                return 0.0 # 或者其他默认值，表示没有证据
                
            psi_alpha_k_plus_1 = digamma(alpha + 1)
            psi_alpha_0_plus_1 = digamma(alpha_0 + 1)
            
            # 加上负号
            result = - (alpha / alpha_0) * (psi_alpha_k_plus_1 - psi_alpha_0_plus_1)
            
            return result.sum()

        return cal_au

    elif mode == "eu_2":
        def eu(logits):
            top_k = 2
            if len(logits) < top_k:
                raise ValueError("Logits array length is less than top_k.")
            top_values, _ = topk(logits, top_k)
            mean_scores = top_k / (np.sum(np.maximum(0, top_values)) + top_k)
            return mean_scores

        return eu

    elif mode == "au_2":
        def cal_au(logits):
            top_k = 2
            if len(logits) < top_k:
                raise ValueError("Logits array length is less than top_k.")
            top_values = np.partition(logits, -top_k)[-top_k:]
            alpha = np.maximum(0, top_values)
            alpha = np.array([alpha])
            alpha_0 = alpha.sum(axis=1, keepdims=True)
            psi_alpha_k_plus_1 = digamma(alpha + 1)
            psi_alpha_0_plus_1 = digamma(alpha_0 + 1)
            result = - (alpha / alpha_0) * (psi_alpha_k_plus_1 - psi_alpha_0_plus_1)
            return result.sum(axis=1)[0]

        return cal_au

    else:
        raise ValueError(f"Unsupported mode: {mode}")