import os
import pandas as pd
from torchvision.io import read_video
from torch.utils.data import Dataset
import random
import torch

class CharadesDataset(Dataset):
    def __init__(self, annotations_file, classes_file, video_dir):
        self.annotations = pd.read_csv(annotations_file)
        self.annotations = self.annotations[self.annotations["length"] < 75.0]
        self.annotations = self.annotations[self.annotations["actions"].notnull()]

        self.classes = {}
        for item in open(classes_file, 'r').readlines():
            self.classes[item.split(' ', 1)[0]] = item.split(' ', 1)[1].strip()

        self.video_dir = video_dir

        self.vocabulary = open("data/vocabulary.txt", 'r').readlines()
    
    def __query_to_tensor(self, query, max_words=10):
        embedding = []
        
        words = query.lower().split()
        
        for word in words:
            if word in self.vocabulary:
                # create a one-hot encoded vector for the word
                word_vector = [0] * len(self.vocabulary)
                word_index = self.vocabulary.index(word)
                word_vector[word_index] = 1
                embedding.append(word_vector)
            else:
                # if the word is not in the vocabulary, use a placeholder vector
                embedding.append([0] * len(self.vocabulary))
        
        # convert the embedding list to a PyTorch tensor and pad to max word length
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
        embedding_tensor = torch.nn.functional.pad(embedding_tensor, (0, 0, 0, max_words - embedding_tensor.shape[0]))
        
        return embedding_tensor
    
    def __pad_video(self, video_frames, max_frames=75, max_height=360, max_width=480):
        return torch.nn.functional.pad(video_frames, 
            (0, 0, 0, max_width - video_frames.shape[2], 0, max_height - video_frames.shape[1], 0, max_frames - video_frames.shape[0]))
            
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        data = self.annotations.iloc[idx]
        # print(data)
        video_fname = data["id"] + ".mp4"
        video_path = os.path.join(self.video_dir, video_fname)
        video_frames, _, metadata = read_video(video_path, pts_unit='sec')

        fps = int(float(metadata["video_fps"])) # str -> float -> int
        # use slicing with the step size of fps to 
        # subsample the video frames to 1 frame per second
        video_frames = video_frames[::fps]
        video_frames = self.__pad_video(video_frames)

        actions = data["actions"]
        actions = actions.split(";")

        label = random.choice(actions)
        label = label.split(" ")

        query = self.classes[label[0]]
        query_tensor = self.__query_to_tensor(query)

        start_s = float(label[1])
        end_s = float(label[2])

        return {
            "video_frames": video_frames, 
            "query_tensor": query_tensor,
            "start_s": start_s, 
            "end_s": end_s
            }