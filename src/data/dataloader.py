import os
import pickle
from typing import Dict, List, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torchvggish.vggish_input import waveform_to_examples
from transformers import BertTokenizer, RobertaTokenizer,AutoTokenizer


class IEMOCAPDataset(Dataset):
    def __init__(
        self,
        path: str = "path/to/data.pkl",
        tokenizer: Union[BertTokenizer, RobertaTokenizer, AutoTokenizer] = BertTokenizer.from_pretrained("bert-base-uncased"),
        audio_max_length: int = 546220,
        text_max_length: int = 100,
        audio_encoder_type: str = "vggish",
        video_dir : str = None,
    ):
        """Dataset for IEMOCAP

        Args:
            path (str, optional): Path to data.pkl. Defaults to "path/to/data.pkl".
            tokenizer (BertTokenizer, optional): Tokenizer for text. Defaults to BertTokenizer.from_pretrained("bert-base-uncased").
            audio_max_length (int, optional): The maximum length of audio. Defaults to 546220. None for no padding and truncation.
            text_max_length (int, optional): The maximum length of text. Defaults to 100. None for no padding and truncation.
        """
        super(IEMOCAPDataset, self).__init__()
        with open(path, "rb") as train_file:
            self.data_list = pickle.load(train_file)
        self.audio_max_length = audio_max_length
        self.text_max_length = text_max_length
        self.tokenizer = tokenizer
        self.audio_encoder_type = audio_encoder_type
        self.video_dir = video_dir

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fileName, text, spectrogram, label = self.data_list[index].values()

        # Tokenize text
        text_input = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.text_max_length,
            truncation=True,
            padding='max_length',
            return_tensors="np"
        )
        input_ids = torch.tensor(text_input['input_ids'].squeeze(), dtype=torch.long)
        attention_mask = torch.tensor(text_input['attention_mask'].squeeze(), dtype=torch.long)

        # Load video embedding
        file_path = os.path.join(self.video_dir, f"{fileName}_embeddings.pkl")
        with open(file_path, 'rb') as f:
            video_embedding = pickle.load(f)
        video_embedding = torch.tensor(np.array(video_embedding)).squeeze(1)
        video_embedding = torch.mean(video_embedding, dim=1)

        if isinstance(spectrogram, torch.Tensor):
            spectrogram = spectrogram.clone().detach()
        else:  # assuming it's a numpy array
            spectrogram = torch.tensor(spectrogram)

        label = torch.tensor(label, dtype=torch.long)  # assuming label is an integer

        return input_ids, spectrogram, label, attention_mask, video_embedding

    def __len__(self):
        return len(self.data_list)


def build_train_test_dataset(
    root: str = "data/",
    batch_size: int = 64,
    tokenizer: Union[BertTokenizer, RobertaTokenizer, AutoTokenizer]  = BertTokenizer.from_pretrained("bert-base-uncased"),
    audio_max_length: int = 546220,
    text_max_length: int = 100,
    audio_encoder_type: str = "vggish",
) -> Tuple[DataLoader, DataLoader]:
    """Read train and test data from pickle files

    Args:
        root (str, optional): Path to data directory. Defaults to "data/".
        Your data directory should contain train_data.pkl and test_data.pkl

    Returns:
        Tuple[List, List]: Tuple of train and test data
    """

    if batch_size == 1:
        audio_max_length = None
        text_max_length = None
    training_data = IEMOCAPDataset(
        os.path.join(root, "train_data.pkl"),
        tokenizer,
        audio_max_length,
        text_max_length,
        audio_encoder_type,
        video_dir= 'D:/MELD/video_embeddings/MELD_train_embeddings'
        )
    test_data = IEMOCAPDataset(
        os.path.join(root, "test_data.pkl"), 
        tokenizer, 
        None, 
        None, 
        audio_encoder_type,
        video_dir= 'D:/MELD/video_embeddings/MELD_dev_embeddings'
        )

    train_dataloader = DataLoader(
        training_data, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn)
    test_dataloader = DataLoader(
        test_data,
        batch_size=1, 
        shuffle=False,
        collate_fn=collate_fn)
    return (train_dataloader, test_dataloader)

def collate_fn(batch):
    # Unzip the batch
    input_ids, spectrograms, labels, attention_masks, video_embeddings = zip(*batch)
    
    # Padding for spectrograms
    max_length_spec = max([spec.shape[0] for spec in spectrograms])
    padded_spectrograms = []
    for spec in spectrograms:
        target_shape = (max_length_spec, spec.shape[1], spec.shape[2], spec.shape[3])
        padded_spec = torch.zeros(target_shape, dtype=spec.dtype)
        padded_spec[:spec.shape[0]] = spec
        padded_spectrograms.append(padded_spec)
    
    # Padding for video_embeddings
    max_length_video = max([ve.shape[0] for ve in video_embeddings])
    padded_video_embeddings = []
    for ve in video_embeddings:
        target_shape = (max_length_video,) + ve.shape[1:]
        padded_ve = torch.zeros(target_shape, dtype=ve.dtype)
        padded_ve[:ve.shape[0]] = ve
        padded_video_embeddings.append(padded_ve)

    return (torch.stack(input_ids),
            torch.stack(padded_spectrograms),
            torch.stack(labels),
            torch.stack(attention_masks),
            torch.stack(padded_video_embeddings))