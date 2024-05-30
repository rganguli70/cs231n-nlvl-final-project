import os
import pandas as pd
from torchvision.io import read_video
from torch.utils.data import Dataset
import random
import torch
from transformers import AutoTokenizer

class CharadesDataset(Dataset):
    def __init__(self, annotations_file, classes_file, video_dir):
        self.annotations = pd.read_csv(annotations_file)
        self.max_frames = 45
        self.annotations = self.annotations[self.annotations["length"] < self.max_frames]
        self.annotations = self.annotations[self.annotations["actions"].notnull()]

        self.classes = {}
        for item in open(classes_file, 'r').readlines():
            self.classes[item.split(' ', 1)[0]] = item.split(' ', 1)[1].strip()

        self.video_dir = video_dir

        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
        self.max_words = 10
    
    def __pad_video(self, video_frames, max_height=480, max_width=480):
        return torch.nn.functional.pad(video_frames, 
                                      (0, 0, 
                                       0, max_width - video_frames.shape[2], 
                                       0, max_height - video_frames.shape[1], 
                                       0, self.max_frames - video_frames.shape[0])
                                       )
    
    def __pad_query_tokens(self, query_tokens):
        query_tokens["input_ids"] = torch.nn.functional.pad(query_tokens["input_ids"],
                                                            (0, self.max_words - query_tokens["input_ids"].shape[1])
                                                            )
        query_tokens["attention_mask"] = torch.nn.functional.pad(query_tokens["attention_mask"], 
                                                                 (0, self.max_words - query_tokens["attention_mask"].shape[1])
                                                                )
            
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        data = self.annotations.iloc[idx]
        # print(data)
        video_fname = data["id"] + ".mp4"
        video_path = os.path.join(self.video_dir, video_fname)
        video_frames, _, metadata = read_video(video_path, pts_unit='sec')
        fps = int(metadata["video_fps"])
        # use slicing with the step size of fps to 
        # subsample the video frames to 1 frame per second
        video_frames = video_frames[::fps]
        video_frames = self.__pad_video(video_frames)
        # video_frames.to("cpu")

        actions = data["actions"]
        actions = actions.split(";")

        label = random.choice(actions)
        label = label.split(" ")

        query = self.classes[label[0]]
        query_tokens = self.tokenizer(query, return_tensors="pt")
        self.__pad_query_tokens(query_tokens)

        start_s = float(label[1])
        end_s = float(label[2])

        return {
            "video_frames": video_frames.unsqueeze(0), 
            "query_tokens": query_tokens,
            "video_positions": torch.arange(0, self.max_frames).unsqueeze(1).repeat(1, 1),  # Example video positions
            "text_positions": torch.arange(0, self.max_words).unsqueeze(1).repeat(1, 1),  # Example text positions
            "label_ids": torch.tensor([start_s, end_s])
            }
