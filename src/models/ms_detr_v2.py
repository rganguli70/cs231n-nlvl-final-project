import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
# from transformers import ViTConfig, ViTModel, PhiConfig, PhiModel
from transformers import ViTModel, ViTImageProcessor, AutoModel, AutoTokenizer

class MS_DETR(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=5, num_decoder_layers=5, dim_feedforward=2048, dropout=0.1):
        super(MS_DETR, self).__init__()
        
        self.d_model = d_model
        self.nhead = nhead

        # Feature extractors
        self.image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.embed_video_norm = nn.BatchNorm2d(1)
        self.embed_video_conv = nn.Conv2d(1, 1, 2, stride=2)
        self.embed_video_pool = nn.MaxPool2d(2)
        self.embed_video_dim = 9408 # calculated from above layers
        self.embed_video_fc = nn.Linear(self.embed_video_dim, self.d_model)

        self.phi = AutoModel.from_pretrained("microsoft/phi-2") # tokenization done in dataloader
        self.embed_query_fc = nn.Linear(self.phi.config.hidden_size, self.d_model)

        # Positional encodings
        self.position_encoder = nn.Embedding(5000, d_model)
        self.position_decoder = nn.Embedding(5000, d_model)
        
        # Transformer encoder and decoder
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        
        decoder_layers = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_decoder_layers)
        
        # Output layers
        self.span_predictor = nn.Linear(d_model, 2)
        self.mask_predictor = nn.Linear(d_model, 2)
        
        # Auxiliary losses
        self.span_loss_fn = nn.BCEWithLogitsLoss()
        self.mask_loss_fn = nn.CrossEntropyLoss()

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

    def embed_video(self, video_frames):
        batch_size, num_frames, height, width, channels = video_frames.shape
        embedding = torch.zeros((num_frames, batch_size, self.d_model), device=video_frames.device)
        for f in range(num_frames):
            frame = video_frames[:, f, :, :, :]
            frame = torch.permute(frame, (0, 3, 1, 2)) # N, H, W, C -> N, C, H, W
            
            with torch.no_grad(): # FREEZE image embedding
                inputs = self.image_processor(frame)
                inputs["pixel_values"] = torch.as_tensor(inputs["pixel_values"][0]).unsqueeze(0) # WORKAROUND since bsz=1
                inputs = inputs.to(self.vit.device)
                output = self.vit(**inputs)

            # dimensionality reduction
            x = output.last_hidden_state.unsqueeze(1)
            x = self.embed_video_conv(x)
            x = F.relu(self.embed_video_norm(x))
            x = self.embed_video_pool(x)
            x = x.reshape(batch_size, -1)
            x = self.embed_video_fc(x)

            embedding[f] = x
            
        return embedding

    def embed_query(self, query_tokens):
        with torch.no_grad(): # FREEZE text embedding
            embedding = self.phi(**query_tokens)

        embedding = self.embed_query_fc(embedding.last_hidden_state)

        # N, L, C -> L, N, C
        return torch.permute(embedding, (1, 0, 2))

    def forward(self, video_frames, query_tokens, video_positions, text_positions, label_ids=None):
        # Extract Features
        video_features = self.embed_video(video_frames)
        text_features = self.embed_query(query_tokens)

        # Positional encoding
        video_features = video_features + self.position_encoder(video_positions)
        text_features = text_features + self.position_decoder(text_positions)
        
        # Add batch dimension if necessary
        if video_features.dim() == 2:
            video_features = video_features.unsqueeze(1)
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(1)
        
        # Transformer encoder
        memory = self.transformer_encoder(video_features)
        
        # Transformer decoder
        output = self.transformer_decoder(text_features, memory)

        # Output predictions
        span_logits = self.span_predictor(output)
        mask_logits = self.mask_predictor(output)
        
        return span_logits, mask_logits
    
    def compute_loss(self, span_logits, mask_logits, span_labels, mask_labels):
        span_loss = self.span_loss_fn(span_logits, span_labels)
        mask_loss = self.mask_loss_fn(mask_logits.view(-1, mask_logits.size(-1)), mask_labels.view(-1))
        
        return span_loss + mask_loss