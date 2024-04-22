from dataloader import CharadesDataset
from linear_model import NLVLLinearNet
from torch import optim
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

writer = SummaryWriter()

dataset = CharadesDataset("data/Charades_v1_train.csv", 
                          "data/Charades_v1_classes.txt", 
                          "data/videos/")
trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

model = NLVLLinearNet()
model.to(device)

criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters())

num_epochs = 1

for epoch in range(num_epochs):
    cum_loss = 0
    pbar = tqdm(trainloader)
    for i, data in enumerate(pbar):

        video_frames = data["video_frames"].to(device)
        video_frames = video_frames.squeeze(0)

        query_tensor = data["query_tensor"].to(device)
        query_tensor = query_tensor.squeeze(0)

        start_s, end_s = data["start_s"], data["end_s"]
        labels = torch.tensor([start_s, end_s], dtype=torch.float32, requires_grad=True, device=device)
        
        optimizer.zero_grad()
        pred_start_s, pred_end_s = model(video_frames=video_frames, query_tensor=query_tensor)
        preds = torch.tensor([pred_start_s, pred_end_s], dtype=torch.float32, requires_grad=True, device=device)

        loss = criterion(preds, labels)
        writer.add_scalar("Loss/train", loss, i)
        loss.backward()
        optimizer.step()

        cum_loss += loss
        pbar.set_description(f"Training loss: {loss}")

        # if i == 10:
        #     break

    print(f"Average train set loss: {cum_loss / i}")

    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': (cum_loss / i),
            }, "trained_model.pth")

writer.flush()
writer.close()