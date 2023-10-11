from typing import Dict, Union

import torch
from torch import Tensor

from models.networks import MMSERA, SERVER
from utils.torch.trainer import TorchTrainer


class Trainer(TorchTrainer):
    def __init__(self, network: Union[MMSERA, SERVER], criterion: torch.nn.CrossEntropyLoss = None, **kwargs):
        super().__init__(**kwargs)
        self.network = network
        self.criterion = criterion
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        self.network.train()
        self.optimizer.zero_grad()

        # Prepare batch
        input_ids, audio, label, attention_mask, video_embedding = batch

        # Move inputs to cpu or gpu
        audio = audio.to(self.device)
        label = label.to(self.device)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        if video_embedding is not None:
            video_embedding = video_embedding.to(self.device)
        
        # Forward pass
        output = self.network(input_ids, audio, attention_mask=attention_mask, video_embedding=video_embedding)
        loss = self.criterion(output, label)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        # Calculate accuracy
        _, preds = torch.max(output[0], 1)
        accuracy = torch.mean((preds == label).float())
        return {"loss": loss.detach().cpu().item(), "acc": accuracy.detach().cpu().item()}

    def test_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        self.network.eval()
        
        # Prepare batch
        input_ids, audio, label, attention_mask, video_embedding = batch

        # Move inputs to cpu or gpu
        audio = audio.to(self.device)
        label = label.to(self.device)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        if video_embedding is not None:
            video_embedding = video_embedding.to(self.device)
        
        with torch.no_grad():
            # Forward pass
            output = self.network(input_ids, audio, attention_mask=attention_mask, video_embedding=video_embedding)
            loss = self.criterion(output, label)
            # Calculate accuracy
            _, preds = torch.max(output[0], 1)
            accuracy = torch.mean((preds == label).float())
        return {"loss": loss.detach().cpu().item(), "acc": accuracy.detach().cpu().item()}
