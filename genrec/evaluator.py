from collections import OrderedDict

import torch


class Evaluator:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.eos_token = tokenizer.eos_token
        self.maxk = max(config["topk"])

    def calculate_pos_index(self, preds, labels):
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()
        if preds.shape[1] != self.maxk:
            raise ValueError(f"Expected {self.maxk} predictions, got {preds.shape[1]}.")

        pos_index = torch.zeros((preds.shape[0], self.maxk), dtype=torch.bool)
        for row in range(preds.shape[0]):
            label = labels[row].tolist()
            if self.eos_token in label:
                label = label[:label.index(self.eos_token)]
            for rank in range(self.maxk):
                if preds[row, rank].tolist() == label:
                    pos_index[row, rank] = True
                    break
        return pos_index

    @staticmethod
    def recall_at_k(pos_index, k):
        return pos_index[:, :k].sum(dim=1).float()

    @staticmethod
    def ndcg_at_k(pos_index, k):
        ranks = torch.arange(1, pos_index.shape[-1] + 1, dtype=torch.float32)
        gains = 1.0 / torch.log2(ranks + 1.0)
        dcg = torch.where(pos_index, gains, torch.zeros_like(gains))
        return dcg[:, :k].sum(dim=1).float()

    def calculate_metrics(self, preds, labels):
        pos_index = self.calculate_pos_index(preds, labels)
        results = OrderedDict()
        for metric in self.config["metrics"]:
            for k in self.config["topk"]:
                key = f"{metric}@{k}"
                if metric == "recall":
                    results[key] = self.recall_at_k(pos_index, k)
                elif metric == "ndcg":
                    results[key] = self.ndcg_at_k(pos_index, k)
                else:
                    raise ValueError(f"Unsupported metric: {metric}")
        return results
