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
        
        #self.text_encoder = SimilarityClassifier("SamLowe/roberta-base-go_emotions", 768, 0.2,False,True)
        #self.text_encoder.to(device)
        #self.text_encoder.load_state_dict(torch.load("D:/bert-based-triplet/ckpt/best_model_v6_triplet_epoch_10",map_location=device))

        # self.text_encoder = AutoModel.from_pretrained("SamLowe/roberta-base-go_emotions")
        # self.text_encoder.to(device)

        # Freeze/Unfreeze the text module
        for param in self.text_encoder.parameters():
            param.requires_grad = text_unfreeze

        # Audio module
        self.audio_norm_type = audio_norm_type
        self.audio_encoder = build_audio_encoder(audio_encoder_type)
        self.audio_encoder.to(device)
        if audio_norm_type == "layer_norm":
            self.audio_encoder_layer_norm = nn.LayerNorm(audio_encoder_dim)

        # Freeze/Unfreeze the audio module
        for param in self.audio_encoder.parameters():
            param.requires_grad = audio_unfreeze

        # Fusion module
        self.text_attention = nn.MultiheadAttention(
            embed_dim=text_encoder_dim, num_heads=num_attention_head, dropout=dropout, batch_first=True
        )

        self.text_linear = nn.Linear(text_encoder_dim, audio_encoder_dim)
        self.text_layer_norm = nn.LayerNorm(audio_encoder_dim)

        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=audio_encoder_dim, num_heads=num_attention_head, dropout=dropout, batch_first=True
        )
        self.fusion_linear = nn.Linear(audio_encoder_dim, 128)
        self.fusion_layer_norm = nn.LayerNorm(128)

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(128, 64)
        self.classifer = nn.Linear(64, num_classes)

        self.fusion_head_output_type = fusion_head_output_type

    def forward(self, input_ids, audio, output_attentions=False, attention_mask= None, video_embedding = None):
        # Text processing
        text_embeddings = self.text_encoder(input_ids,attention_mask = attention_mask).last_hidden_state
        
        #text_embeddings = self.text_encoder(input_ids,attention_mask)
        #text_embeddings = text_embeddings.unsqueeze(1)
        #text_embeddings = text_embeddings.unsqueeze(1).expand(text_embeddings.shape[0], 2, -1) #Reshape it to (batch_size, 2, 768) by duplicating the first dimension

        #text_embeddings = self.text_encoder(input_ids,attention_mask).last_hidden_state

        # Audio processing
        audio_embeddings = self.audio_encoder(audio)
        if self.audio_norm_type == "layer_norm":
            audio_embeddings = self.audio_encoder_layer_norm(audio_embeddings)
        elif self.audio_norm_type == "min_max":
            # Min-max normalization
            audio_embeddings = (audio_embeddings - audio_embeddings.min()) / (audio_embeddings.max() - audio_embeddings.min())

        ## Fusion Module
        # Self-attention to reduce the dimensionality of the text embeddings
        text_attention, text_attn_output_weights = self.text_attention(
            text_embeddings, text_embeddings, text_embeddings, average_attn_weights=False
        )
        text_linear = self.text_linear(text_attention)
        text_norm = self.text_layer_norm(text_linear)

        # Concatenate the text and audio embeddings
        fusion_embeddings = torch.cat((text_norm, audio_embeddings), 1)

        # Selt-attention module
        fusion_attention, fusion_attn_output_weights = self.fusion_attention(
            fusion_embeddings, fusion_embeddings, fusion_embeddings, average_attn_weights=False
        )
        fusion_linear = self.fusion_linear(fusion_attention)
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
        out = self.classifer(x)

        if output_attentions:
            return [out, cls_token_final_fusion_norm], [text_attn_output_weights, fusion_attn_output_weights]

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

    def forward(self, input_ids, audio, output_attentions=False, attention_mask= None, video_embedding= None):
        # Text processing
        text_embeddings = self.text_encoder(input_ids,attention_mask= attention_mask).pooler_output
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


class JointSpace(nn.Module):
    def __init__(self, hidden_size, n_dim):
        super(JointSpace, self).__init__()
        self.hidden_size = hidden_size
        self.out1 = nn.Linear(hidden_size, hidden_size // 2)
        self.out2 = nn.Linear(hidden_size // 2, n_dim)

    def forward(self, x):
        out = self.out1(x)
        out = self.out2(out)
        return out


class SimilarityClassifier(nn.Module):
    def __init__(self, PRE_TRAINED_MODEL_NAME, embed_dim, dropout_p, freeze=False, space_joiner = True):
        super(SimilarityClassifier, self).__init__()
        # self.bert = DistilBertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.bert = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME,)
        self.drop = nn.Dropout(p=0.3)
        self.sigmoid = nn.Sigmoid()
        self.space_joiner = space_joiner
        if space_joiner:
            self.space_joiner = JointSpace(self.bert.config.hidden_size, embed_dim)
        else:
            assert embed_dim == 768
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )[0]
        bert_output = torch.mean(bert_output, dim=1)
        out = self.drop(bert_output)
        if self.space_joiner:
            out = self.space_joiner(out)

        return out

def get_model(config):
    model = SimilarityClassifier(config.PRE_TRAINED_MODEL_NAME, config.embed_dim, config.dropout,config.freeze,config.space_joiner)
    return model


class MMSERA_TAV(nn.Module):
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
        
        #self.text_encoder = SimilarityClassifier("SamLowe/roberta-base-go_emotions", 768, 0.2,False,True)
        #self.text_encoder.to(device)
        #self.text_encoder.load_state_dict(torch.load("D:/bert-based-triplet/ckpt/best_model_v6_triplet_epoch_10",map_location=device))

        # self.text_encoder = AutoModel.from_pretrained("SamLowe/roberta-base-go_emotions")
        # self.text_encoder.to(device)

        # Freeze/Unfreeze the text module
        for param in self.text_encoder.parameters():
            param.requires_grad = text_unfreeze

        # Audio module
        self.audio_norm_type = audio_norm_type
        self.audio_encoder = build_audio_encoder(audio_encoder_type)
        self.audio_encoder.to(device)
        if audio_norm_type == "layer_norm":
            self.audio_encoder_layer_norm = nn.LayerNorm(audio_encoder_dim)

        # Freeze/Unfreeze the audio module
        for param in self.audio_encoder.parameters():
            param.requires_grad = audio_unfreeze

        # Fusion module
        self.text_attention = nn.MultiheadAttention(
            embed_dim=text_encoder_dim, num_heads=num_attention_head, dropout=dropout, batch_first=True
        )

        self.video_attention = nn.MultiheadAttention(
            embed_dim=video_encoder_dim, num_heads=num_attention_head, dropout=dropout, batch_first=True
        )

        self.text_linear = nn.Linear(text_encoder_dim, audio_encoder_dim)
        self.text_layer_norm = nn.LayerNorm(audio_encoder_dim)

        self.video_linear = nn.Linear(video_encoder_dim, audio_encoder_dim)
        self.video_layer_norm = nn.LayerNorm(audio_encoder_dim)

        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=audio_encoder_dim, num_heads=num_attention_head, dropout=dropout, batch_first=True
        )
        self.fusion_linear = nn.Linear(audio_encoder_dim, 128)
        self.fusion_layer_norm = nn.LayerNorm(128)

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(128, 64)
        self.classifer = nn.Linear(64, num_classes)

        self.fusion_head_output_type = fusion_head_output_type

    def forward(self, input_ids, audio, output_attentions=False, attention_mask= None, video_embedding = None):
        # Text processing
        text_embeddings = self.text_encoder(input_ids,attention_mask = attention_mask).last_hidden_state
        
        #text_embeddings = self.text_encoder(input_ids,attention_mask)
        #text_embeddings = text_embeddings.unsqueeze(1)
        #text_embeddings = text_embeddings.unsqueeze(1).expand(text_embeddings.shape[0], 2, -1) #Reshape it to (batch_size, 2, 768) by duplicating the first dimension

        #text_embeddings = self.text_encoder(input_ids,attention_mask).last_hidden_state


        # Audio processing
        audio_embeddings = self.audio_encoder(audio)
        if self.audio_norm_type == "layer_norm":
            audio_embeddings = self.audio_encoder_layer_norm(audio_embeddings)
        elif self.audio_norm_type == "min_max":
            # Min-max normalization
            audio_embeddings = (audio_embeddings - audio_embeddings.min()) / (audio_embeddings.max() - audio_embeddings.min())

        ## Fusion Module
        # Self-attention to reduce the dimensionality of the text embeddings
        text_attention, text_attn_output_weights = self.text_attention(
            text_embeddings, text_embeddings, text_embeddings, average_attn_weights=False
        )
        text_linear = self.text_linear(text_attention)
        text_norm = self.text_layer_norm(text_linear)

        video_attention, video_attn_output_weights = self.video_attention(
            video_embedding, video_embedding, video_embedding, average_attn_weights=False
        )
        video_linear = self.video_linear(video_attention)
        video_norm = self.video_layer_norm(video_linear)


        # Concatenate the text and audio embeddings
        fusion_embeddings = torch.cat((text_norm, audio_embeddings,video_norm), 1)

        # Selt-attention module
        fusion_attention, fusion_attn_output_weights = self.fusion_attention(
            fusion_embeddings, fusion_embeddings, fusion_embeddings, average_attn_weights=False
        )
        fusion_linear = self.fusion_linear(fusion_attention)
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
        out = self.classifer(x)

        if output_attentions:
            return [out, cls_token_final_fusion_norm], [text_attn_output_weights, fusion_attn_output_weights]

        return out, cls_token_final_fusion_norm, text_norm.mean(dim=1), audio_embeddings.mean(dim=1)