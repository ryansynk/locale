import torch


class BaseIndexer:
    def build(self, features, ids):
        raise NotImplementedError

    def search(self, query_features, topk):
        raise NotImplementedError


class DenseIndexer(BaseIndexer):
    def __init__(self):
        pass

    def build(self, features, ids):
        self.index = features
        self.ids = ids  # Keep track of mapping if needed

    @torch.no_grad()
    def search(self, query_features, topk):
        logits = torch.matmul(query_features, self.index.T)  # (num_queries, num_keys)
        _, indices = logits.topk(k=topk, dim=1)  # (num_queries, k)
        return indices.cpu()


class SourMashIndexer(BaseIndexer):
    def __init__(self, threshold=0.5, num_perm=128):
        raise NotImplementedError

    def build(self, features, ids):
        raise NotImplementedError

    def search(self, query_features, topk):
        raise NotImplementedError
