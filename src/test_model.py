from utils import CharadesDataset
from models.nlvl_detr_v2 import NLVL_DETR
import torch
from tqdm import tqdm 

device = None
if torch.cuda.is_available():
    device = "cuda"  
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print("Using device:", device)

dataset = CharadesDataset("../data/Charades_v1_test.csv", 
                          "../data/Charades_v1_classes.txt", 
                          "../data/videos/")

model = NLVL_DETR()
checkpoint = torch.load("trained_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])

pbar = tqdm(dataset)
for i, data in enumerate(pbar):
    video_frames = data["video_frames"]
    query_tensor = data["query_tensor"]
    start_s, end_s = data["start_s"], data["end_s"]

    span_preds = model(video_frames=video_frames, query_tensor=query_tensor)

    print(f"ground truth start_s={start_s}, end_s={end_s}")
    print(f"prediction start_s={span_preds[0].item()}, end_s=start_s={span_preds[1].item()}")

    if i == 10:
        break # DEBUG