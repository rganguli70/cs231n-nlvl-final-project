from dataloader import CharadesDataset
from linear_model import NLVLLinearNet
import torch
from tqdm import tqdm 

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

dataset = CharadesDataset("data/Charades_v1_test.csv", 
                          "data/Charades_v1_classes.txt", 
                          "data/videos/")

model = NLVLLinearNet()
checkpoint = torch.load("trained_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])

criterion = torch.nn.MSELoss()

cum_loss = 0
pbar = tqdm(dataset)
for i, data in enumerate(pbar):
    video_frames = data["video_frames"]
    query_tensor = data["query_tensor"]
    start_s, end_s = data["start_s"], data["end_s"]
    labels = torch.tensor([start_s, end_s], dtype=torch.float32, requires_grad=True, device=device)

    pred_start_s, pred_end_s = model(video_frames=video_frames, query_tensor=query_tensor)
    preds = torch.tensor([pred_start_s, pred_end_s], dtype=torch.float32, requires_grad=True, device=device)

    loss = criterion(preds, labels)
    cum_loss += loss

    pbar.set_description(f"Testing loss: {loss}")

    # if i == 10:
    #     break

print(f"Average test set loss: {cum_loss / i}")