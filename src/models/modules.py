import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvggish import vggish
from transformers import BertConfig, BertModel


def build_bert_encoder() -> nn.Module:
    """A function to build bert encoder"""
    config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
    bert = BertModel.from_pretrained("bert-base-uncased", config=config)
    return bert


class VGGish(nn.Module):
    def __init__(self):
        super(VGGish, self).__init__()
        self.vggish = vggish()

    def forward(self, x):
        out = []
        for i in range(x.size(0)):
            out.append(self.vggish(x[i]))
        x = torch.stack(out, axis=0)
        if len(x.size()) == 2:
            x = x.unsqueeze(1)
        return x


def build_vggish_encoder() -> nn.Module:
    """A function to build vggish encoder"""
    return VGGish()


def build_audio_encoder(type: str = "vggish") -> nn.Module:
    """A function to build audio encoder

    Args:
        type (str, optional): Type of audio encoder. Defaults to "vggish".

    Returns:
        nn.Module: Audio encoder
    """
    encoders = {
        "vggish": build_vggish_encoder,
    }
    assert type in encoders.keys(), f"Invalid audio encoder type: {type}"
    return encoders[type]()


def build_text_encoder(type: str = "bert") -> nn.Module:
    """A function to build text encoder

    Args:
        type (str, optional): Type of text encoder. Defaults to "bert".

    Returns:
        torch.nn.Module: Text encoder
    """
    encoders = {
        "bert": build_bert_encoder,
    }
    assert type in encoders.keys(), f"Invalid text encoder type: {type}"
    return encoders[type]()
