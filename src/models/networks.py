import torch
import torch.nn as nn

from .modules import build_audio_encoder, build_text_encoder
from transformers import AutoModel

# Create Multi-modal model - layer norm
class MMSERA(nn.Module):
    def __init__(
        self,
        num_classes=4,
        num_attention_head=8,
        dropout=0.5,
        text_encoder_type="bert",
        text_encoder_dim=768,
        text_unfreeze=False,
        audio_encoder_type="vggish",
        audio_encoder_dim=128,
        audio_unfreeze=True,
        audio_norm_type="layer_norm",
        fusion_head_output_type="cls",
        device="cpu",
        video_encoder_dim = 768
    ):
        """

        Args: MMSERA model extends from MMSER model in the paper
            num_classes (int, optional): The number of classes. Defaults to 4.
            num_attention_head (int, optional): The number of self-attention heads. Defaults to 8.
            dropout (float, optional): Whether to use dropout. Defaults to 0.5.
            device (str, optional): The device to use. Defaults to "cpu".
        """
        super(MMSERA, self).__init__()

        # Text module
        self.text_encoder = build_text_encoder(text_encoder_type)
        self.text_encoder.to(device)
        for param in self.text_encoder.parameters():
            param.requires_grad = text_unfreeze

        # Audio module
        self.audio_norm_type = audio_norm_type
        self.audio_encoder = build_audio_encoder(audio_encoder_type)
        self.audio_encoder.to(device)
        if audio_norm_type == "layer_norm":
            self.audio_encoder_layer_norm = nn.LayerNorm(audio_encoder_dim)
        for param in self.audio_encoder.parameters():
            param.requires_grad = audio_unfreeze

        # Audio self-attention module
        self.audio_self_attention = nn.MultiheadAttention(
            embed_dim=audio_encoder_dim, num_heads=num_attention_head, dropout=dropout, batch_first=True
        )

        # Text attention module
        self.text_attention = nn.MultiheadAttention(
            embed_dim=text_encoder_dim, num_heads=num_attention_head, dropout=dropout, batch_first=True
        )
        self.text_linear = nn.Linear(text_encoder_dim, audio_encoder_dim)
        self.text_layer_norm = nn.LayerNorm(audio_encoder_dim)

        # Cross-attention modules
        self.audio_text_cross_attention = nn.MultiheadAttention(
            embed_dim=audio_encoder_dim, num_heads=num_attention_head, dropout=dropout, batch_first=True
        )
        self.text_audio_cross_attention = nn.MultiheadAttention(
            embed_dim=audio_encoder_dim, num_heads=num_attention_head, dropout=dropout, batch_first=True
        )

        # Fusion module
        self.fusion_linear = nn.Linear(audio_encoder_dim, 128)  # Adjusted for concatenated dimensions
        self.fusion_layer_norm = nn.LayerNorm(128)

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(128, 64)
        self.classifier = nn.Linear(64, num_classes)

        self.fusion_head_output_type = fusion_head_output_type

    def forward(self, input_ids, audio, output_attentions=False, attention_mask=None, video_embedding=None):
        # Text processing
        text_embeddings = self.text_encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        text_attention, _ = self.text_attention(text_embeddings, text_embeddings, text_embeddings)
        text_linear = self.text_linear(text_attention)
        text_norm = self.text_layer_norm(text_linear)

        # Audio processing
        audio_embeddings = self.audio_encoder(audio)
        audio_self_attn, _ = self.audio_self_attention(audio_embeddings, audio_embeddings, audio_embeddings)
        if self.audio_norm_type == "layer_norm":
            audio_embeddings = self.audio_encoder_layer_norm(audio_self_attn)

        # Cross-attention: Audio as query, Normalized Text (text_norm) as key and value
        audio_text_attn, _ = self.audio_text_cross_attention(
            audio_embeddings, text_norm, text_norm
        )

        # Cross-attention: Normalized Text (text_norm) as query, Audio as key and value
        text_audio_attn, _ = self.text_audio_cross_attention(
            text_norm, audio_embeddings, audio_embeddings
        )

        #print(text_audio_attn.size(), audio_text_attn.size())
        # Fusion Embeddings: Concatenate the outputs of cross-attention modules
        fusion_embeddings = torch.cat((audio_text_attn, text_audio_attn), dim=1)
        #print(fusion_embeddings)
        # Fusion module processing
        fusion_linear = self.fusion_linear(fusion_embeddings)
        fusion_norm = self.fusion_layer_norm(fusion_linear)

        # Get classification output
        if self.fusion_head_output_type == "cls":
            cls_token_final_fusion_norm = fusion_norm[:, 0, :]
        elif self.fusion_head_output_type == "mean":
            cls_token_final_fusion_norm = fusion_norm.mean(dim=1)
        elif self.fusion_head_output_type == "max":
            cls_token_final_fusion_norm = fusion_norm.max(dim=1)
        else:
            raise ValueError("Invalid fusion head output type")

        # Classification head
        x = self.dropout(cls_token_final_fusion_norm)
        x = self.linear(x)
        x = nn.functional.leaky_relu(x)
        out = self.classifier(x)

        return out, cls_token_final_fusion_norm, text_norm.mean(dim=1), audio_embeddings.mean(dim=1)


class SERVER(nn.Module):
    def __init__(
        self,
        num_classes=4,
        num_attention_head=8,
        dropout=0.5,
        text_encoder_type="bert",
        text_encoder_dim=768,
        text_unfreeze=False,
        audio_encoder_type="vggish",
        audio_encoder_dim=128,
        audio_unfreeze=True,
        audio_norm_type="layer_norm",
        fusion_head_output_type="cls",
        device="cpu",
    ):
        """

        Args: MMSERA model extends from MMSER model in the paper
            num_classes (int, optional): The number of classes. Defaults to 4.
            num_attention_head (int, optional): The number of self-attention heads. Defaults to 8.
            dropout (float, optional): Whether to use dropout. Defaults to 0.5.
            device (str, optional): The device to use. Defaults to "cpu".
        """
        super(SERVER, self).__init__()
        # Text module
        self.text_encoder = build_text_encoder(text_encoder_type)
        self.text_encoder.to(device)
        # Freeze/Unfreeze the text module
        for param in self.text_encoder.parameters():
            param.requires_grad = text_unfreeze

        # Audio module
        self.audio_encoder = build_audio_encoder(audio_encoder_type)
        self.audio_encoder.to(device)

        # Freeze/Unfreeze the audio module
        for param in self.audio_encoder.parameters():
            param.requires_grad = audio_unfreeze

        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(text_encoder_dim, 128)
        self.linear2 = nn.Linear(256, 64)
        self.classifer = nn.Linear(64, num_classes)

    def forward(self, input_ids, audio, output_attentions=False):
        # Text processing
        text_embeddings = self.text_encoder(input_ids).pooler_output
        text_embeddings = self.linear1(text_embeddings)
        # Audio processing
        audio_embeddings = self.audio_encoder(audio)
        # Get classification token from the audio module
        audio_embeddings = audio_embeddings.sum(dim=1)

        # Concatenate the text and audio embeddings
        fusion_embeddings = torch.cat((text_embeddings, audio_embeddings), 1)

        # Classification head
        x = self.dropout(fusion_embeddings)
        x = self.linear2(x)
        out = self.classifer(x)

        return out, fusion_embeddings, text_embeddings, audio_embeddings



