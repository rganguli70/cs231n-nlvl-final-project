import torch
from torch import nn
from abc import ABC, abstractmethod

class NLVLBaseNet(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    def forward(self, **kwargs):
        video_frames = kwargs["video_frames"]
        query_tensor = kwargs["query_tensor"]

        video_embedding = self.embed_video(video_frames)
        query_embedding = self.embed_query(query_tensor)

        features = torch.cat([video_embedding, query_embedding])
        features = self.backbone(features)

        pred_start_s, pred_end_s = self.regression_head(features)

        return pred_start_s, pred_end_s

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