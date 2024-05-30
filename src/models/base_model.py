import torch
from torch import nn
from abc import ABC, abstractmethod

class NLVLBaseNet(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    def forward(self, video_frames, query_tokens, label_ids=None):        
        video_embedding = self.embed_video(video_frames)
        query_embedding = self.embed_query(query_tokens)

        # dim=1 to concat over batch dimension
        features = torch.cat([video_embedding, query_embedding], dim=1)
        features = self.backbone(features)

        preds = self.regression_head(features)
        
        return preds

    @abstractmethod
    def embed_video(self, video_frames):
        pass

    @abstractmethod
    def embed_query(self, query_tensor):
        pass

    @abstractmethod
    def backbone(self, features):
        pass

    @abstractmethod
    def regression_head(self, features):
        pass