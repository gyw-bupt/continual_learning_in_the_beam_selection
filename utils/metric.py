import time
import torch
import numpy as np
from tqdm import tqdm
from numpy import log2
import torch.nn.functional as F
def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()  # 转置为 (maxk, batch_size)


        correct = pred.eq(target.view(1, -1).expand_as(pred))  # shape (maxk, batch_size)

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum()  # 使用 reshape 替代 view
            res.append(correct_k.item() * 100.0 / batch_size)  # 计算准确率并转换为百分比

        if len(res) == 1:
            return res[0]
        else:
            return res


def throughput_ratio(preds, targets, ):
    """Compute throughput ratio for Top-k predictions."""
    k_values = [1, 2, 5, 10]
    throughputs = []


    for k in k_values:
        up = []
        down = []
        for exp in range(len(targets)):
            preds_exp_np = preds[exp].detach().cpu().numpy()
            true_1 = targets[exp].detach().cpu().numpy()
            true_log = 1  # 真实标签的对数 (假设得分为 true_1)

            top_preds = np.argsort(preds_exp_np)[-k:]
            pred_log = log2(1 + preds_exp_np[top_preds[0]])  # 取前 k 个中得分最高的预测

            up.append(pred_log)
            down.append(true_log)

        throughputs.append(np.sum(up) / np.sum(down))

    return throughputs


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum) / self.count


class Timer(object):
    """
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.interval = 0
        self.time = time.time()

    def value(self):
        return time.time() - self.time

    def tic(self):
        self.time = time.time()

    def toc(self):
        self.interval = time.time() - self.time
        self.time = time.time()
        return self.interval

def load_results_from_npz(filename):
    with np.load(filename) as data:
        results_dict = {key: data[key] for key in data.files}
    return results_dict

def save_results_to_npz(results_dict, filename):
    np.savez_compressed(filename, **results_dict)
