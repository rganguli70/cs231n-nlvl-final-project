import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
# from transformers import ViTConfig, ViTModel, PhiConfig, PhiModel
from transformers import ViTModel, ViTImageProcessor, AutoModel, AutoTokenizer

class NLVL_DETR(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=5, num_decoder_layers=5, dim_feedforward=2048, dropout=0.1):
        super(NLVL_DETR, self).__init__()
        
        self.d_model = d_model
        self.nhead = nhead

        # Feature extractors
        self.max_frames = 60
        self.image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.embed_video_fc = nn.Linear(self.vit.config.hidden_size, self.d_model)
        
        self.max_words = 10
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
        self.phi = AutoModel.from_pretrained("microsoft/phi-2")
        self.embed_query_fc = nn.Linear(self.phi.config.hidden_size, self.d_model)

        # Positional encodings
        self.position_encoder_video = nn.Embedding(self.max_frames, d_model)
        self.position_encoder_text = nn.Embedding(self.max_words, d_model)
        
        # Transformer encoder and decoder
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        
        decoder_layers = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_decoder_layers)
        
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
            
            embedding[f] = self.embed_video_fc(output.pooler_output)
            
        return embedding

    def embed_query(self, query_tokens):        
        with torch.no_grad(): # FREEZE text embedding
            embedding = self.phi(**query_tokens)

        embedding = self.embed_query_fc(embedding.last_hidden_state)

        return torch.permute(embedding, (1, 0, 2)) # N, L, C -> L, N, C

    def forward(self, video_frames, query_tokens, label_ids=None):
        # Extract Features
        video_features = self.embed_video(video_frames)
        text_features = self.embed_query(query_tokens)

        # Positional encoding
        video_positions = torch.arange(0, video_frames.shape[1]).unsqueeze(1).repeat(1, 1).to(video_frames.device)
        text_positions = torch.arange(0, self.max_words).unsqueeze(1).repeat(1, 1).to(video_frames.device)
        video_features = video_features + self.position_encoder_video(video_positions)
        text_features = text_features + self.position_encoder_text(text_positions)
        
        # Transformer encoder
        memory = self.transformer_encoder(video_features)
        
        # Transformer decoder
        output = self.transformer_decoder(text_features, memory)

        # Output predictions
        span_logits = self.span_predictor(output) \
            * torch.as_tensor([1, 0, 1, 0]).to(video_frames.device) \
            + torch.as_tensor([0, 0, 0, 1]).to(video_frames.device)
        
        return span_logits