import torch
from torch import nn
import torch.nn.functional as F
from models.base_model import NLVLBaseNet
from transformers import ViTConfig, ViTModel, PhiConfig, PhiModel
import os

class NLVLTransformerNet(NLVLBaseNet):
    def __init__(self):
        super().__init__()
        self.vit_config = ViTConfig(image_size=480, num_hidden_layers=2, hidden_size=12)
        self.vit = ViTModel(self.vit_config)
        self.embed_video_norm = nn.BatchNorm2d(1)
        self.embed_video_conv = nn.Conv2d(1, 1, 4, stride=2)
        self.embed_video_pool = nn.MaxPool2d(4)
        self.embed_video_dim = 112 # calculated from above layers

        self.phi_config = PhiConfig(num_hidden_layers=2, hidden_size=32)
        self.phi = PhiModel(self.phi_config)
        self.embed_query_dim = self.embed_video_dim
        self.embed_query_fc = nn.Linear(self.phi_config.hidden_size, self.embed_query_dim)

        self.backbone_dim = self.embed_video_dim + self.embed_query_dim
        self.backbone_fc = nn.Linear(self.backbone_dim, 100)

        self.regression_fc1 = nn.Linear(100, 50)
        self.regression_fc2 = nn.Linear(50, 25)
        self.regression_fc3 = nn.Linear(25, 2)

    def embed_video(self, video_frames):
        batch_size, num_frames, height, width, channels = video_frames.shape
        embedding = torch.zeros((batch_size, self.embed_video_dim), device=video_frames.device)
        for f in range(num_frames):
            frame = video_frames[:, f, :, :, :]
            frame = torch.permute(frame, (0, 3, 1, 2)) # N, H, W, C -> N, C, H, W

            output = self.vit(frame)

            # dimensionality reduction
            x = output.last_hidden_state.unsqueeze(1)
            x = self.embed_video_conv(x)
            x = F.relu(self.embed_video_norm(x))
            x = self.embed_video_pool(x)
            x = x.reshape(batch_size, -1)

            embedding = embedding + x
            
        return embedding

    def embed_query(self, query_tokens):
        output = self.phi(**query_tokens)

        # average along sequence length dimension
        embedding = torch.mean(output.last_hidden_state, dim=1)

        # dimensionality reduction
        embedding = F.relu(self.embed_query_fc(embedding))

        return embedding

    def backbone(self, features):
        x = F.relu(self.backbone_fc(features))
        return x
    
    def regression_head(self, features):
        x = F.relu(self.regression_fc1(features))
        x = F.relu(self.regression_fc2(x))
        x = self.regression_fc3(x)

        return x