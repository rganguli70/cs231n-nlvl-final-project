import torch
import torch.nn as nn
from transformers import ViTModel, ViTImageProcessor, AutoModel
import os
import time

VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"

# class KMeansLayer(nn.Module):
#     def __init__(self, n_clusters, max_iter=100):
#         super(KMeansLayer, self).__init__()
#         self.n_clusters = n_clusters
#         self.max_iter = max_iter

#     def forward(self, x):
#         if VERBOSE: print("Applying kmeans clustering...")
#         num_frames, d_model = x.shape
#         x = x.view(-1, d_model)  # Flatten the input to (num_samples, num_features)

#         # Randomly initialize cluster centers
#         initial_indices = torch.randperm(x.size(0))[:self.n_clusters]
#         cluster_centers = x[initial_indices]

#         for _ in range(self.max_iter):
#             # Compute distances to cluster centers
#             distances = torch.cdist(x, cluster_centers)
#             # Assign samples to the nearest cluster center
#             cluster_assignments = torch.argmin(distances, dim=1)

#             # Compute new cluster centers
#             new_cluster_centers = torch.stack([x[cluster_assignments == k].mean(dim=0) for k in range(self.n_clusters)])

#             # Check for convergence (if cluster centers do not change)
#             if torch.all(cluster_centers == new_cluster_centers):
#                 break
#             cluster_centers = new_cluster_centers

#         return cluster_centers

class NLVL_DETR(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=5, num_decoder_layers=5, dim_feedforward=2048, dropout=0.1):
        super(NLVL_DETR, self).__init__()
        
        self.d_model = d_model
        self.nhead = nhead

        # Feature extractors
        self.image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.embed_video_fc = nn.Linear(self.vit.config.hidden_size, self.d_model)

        self.max_frames = 60
        self.position_encoder_video = nn.Embedding(self.max_frames, d_model)
        # self.kmeans_layer = KMeansLayer(n_clusters=20)
        
        self.phi = AutoModel.from_pretrained("microsoft/phi-2") # tokenization handled by dataloader
        self.embed_query_fc = nn.Linear(self.phi.config.hidden_size, self.d_model)
        
        # Transformer encoder and decoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        decoder_layers = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_decoder_layers)
        
        # Output layers
        self.span_predictor = nn.Linear(d_model, 4)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # video_frames.shape   [1, 60, 480, 480, 3]
    def embed_video(self, video_frames):
        if VERBOSE: print("Embedding video...", end=" "); start_time = time.time()
        batch_size, num_frames, height, width, channels = video_frames.shape
        embedding = torch.zeros((num_frames, self.d_model), device=video_frames.device)
        for f in range(num_frames):
            frame = video_frames[:, f, :, :, :]
            frame = torch.permute(frame, (0, 3, 1, 2)) # N, H, W, C -> N, C, H, W
            
            with torch.no_grad(): # FREEZE image embedding
                inputs = self.image_processor(frame)
                inputs["pixel_values"] = torch.as_tensor(inputs["pixel_values"][0]).unsqueeze(0) # numpy to tensor
                inputs = inputs.to(self.vit.device)
                output = self.vit(**inputs)
            
            embedding[f] = self.embed_video_fc(output.pooler_output.squeeze(0))
        
        if VERBOSE: print(f"time: {round(time.time() - start_time, 4)} sec")
        return embedding

    # query_tokens[{"input_ids", "attention_mask"}].shape   [1, 15]
    def embed_query(self, query_tokens): 
        if VERBOSE: print("Embedding query...", end=" "); start_time = time.time()
        del query_tokens['attention_mask'] # no masking required for text input
    
        with torch.no_grad(): # FREEZE text embedding
            embedding = self.phi(**query_tokens)

        embedding = self.embed_query_fc(torch.mean(embedding.last_hidden_state, 1))

        if VERBOSE: print(f"time: {round(time.time() - start_time, 4)} sec")
        return embedding

    # video_frames.shape                                    [1, 60, 480, 480, 3]
    # query_tokens[{"input_ids", "attention_mask"}].shape   [1, 15]
    def forward(self, video_frames, query_tokens, label_ids=None):
        # Extract Features
        video_features = self.embed_video(video_frames) # [60, 512]
        # video_features = self.kmeans_layer(video_features) # [10, 512]
        text_features = self.embed_query(query_tokens) # [1, 512]

        # Positional encoding
        if VERBOSE: print("Positional encoding for video...", end=" "); start_time = time.time()
        video_positions = self.position_encoder_video(torch.arange(0, video_frames.shape[1], device=video_frames.device))
        video_features = video_features + video_positions # [60, 512]
        if VERBOSE: print(f"time: {round(time.time() - start_time, 4)} sec")

        # Transformer encoder
        if VERBOSE: print("Encoding video...", end=" "); start_time = time.time()
        memory = self.transformer_encoder(video_features) # [60, 512]
        if VERBOSE: print(f"time: {round(time.time() - start_time, 4)} sec")
        
        # Transformer decoder
        if VERBOSE: print("Decoding query with video context...", end=" "); start_time = time.time()
        output = self.transformer_decoder(text_features, memory) # [1, 512]
        if VERBOSE: print(f"time: {round(time.time() - start_time, 4)} sec")

        """
        The following code is a computational trick help utilize 2d IoU loss. 2d IoU loss
        expects input as (x1, y1, x2, y2), so the model is adjusted to predict 4 values 
        but only the 1st and 3rd values hold semantic meaning. 2nd and 4th are always set
        to 0 and 1 respectively.
        """
        # Output predictions
        if VERBOSE: print("Predicting span...", end=" "); start_time = time.time()
        span_logits = self.span_predictor(output.squeeze(0)) \
            * torch.as_tensor([1, 0, 1, 0]).to(video_frames.device) \
            + torch.as_tensor([0, 0, 0, 1]).to(video_frames.device) # [1, 4]

        if VERBOSE: print(f"time: {round(time.time() - start_time, 4)} sec")
        return span_logits