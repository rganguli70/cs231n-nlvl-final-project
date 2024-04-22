import torch
from torch import nn
import torch.nn.functional as F
from base_model import NLVLBaseNet

class NLVLLinearNet(NLVLBaseNet):
    def __init__(self):
        super().__init__()

        self.embed_video_conv1 = nn.Conv2d(3, 16, 10, stride=5)
        self.embed_video_pool1 = nn.MaxPool2d(5)
        self.embed_video_conv2 = nn.Conv2d(16, 32, 2, stride=2)
        self.embed_video_pool2 = nn.MaxPool2d(2)
        self.embed_video_fc = nn.Linear(384, 100)

        vocabulary = open("data/vocabulary.txt", 'r').readlines()
        self.embed_query_fc1 = nn.Linear(len(vocabulary)*10, 1000)
        self.embed_query_fc2 = nn.Linear(1000, 500)
        self.embed_query_fc3 = nn.Linear(500, 100)

        self.backbone_fc = nn.Linear(200, 200)

        self.regression_fc1 = nn.Linear(200, 100)
        self.regression_fc2 = nn.Linear(100, 50)
        self.regression_fc3 = nn.Linear(50, 2)
    
    def embed_video(self, video_frames):
        embedding = torch.zeros(100, device=video_frames.device)
        for frame in video_frames:
            x = torch.permute(frame, (2, 0, 1)).float()

            x = self.embed_video_conv1(x)
            x = self.embed_video_pool1(x)
            x = self.embed_video_conv2(x)
            x = self.embed_video_pool2(x)

            x = torch.flatten(x)
            embedding += self.embed_video_fc(x)
        embedding /= len(video_frames)

        return embedding
    
    def embed_query(self, query_tensor):
        x = torch.flatten(query_tensor)
        x = F.relu(self.embed_query_fc1(x))
        x = F.relu(self.embed_query_fc2(x))
        x = self.embed_query_fc3(x)
        return x
    
    def backbone(self, features):
        x = F.relu(self.backbone_fc(features))
        return x
    
    def regression_head(self, features):
        x = F.relu(self.regression_fc1(features))
        x = F.relu(self.regression_fc2(x))
        x = self.regression_fc3(x)

        return x