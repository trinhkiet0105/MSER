from configs.base import Config as BaseConfig


class Config(BaseConfig):
    # Base
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.name = "bert_vggish"
        self.add_args()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_args(self, **kwargs):
        self.batch_size = 8
        self.num_epochs = 29

        self.loss_type = "CrossEntropyLoss"

        self.model_type = "SERVER"  # [MMSERA, SERVER]
        self.name = self.name + "_" + self.model_type
        self.text_encoder_type = "bert"
        self.text_encoder_dim = 768
        self.text_unfreeze = False
        self.audio_encoder_type = "vggish"
        self.audio_encoder_dim = 128
        self.audio_norm_type = "layer_norm"  # [layer_norm, min_max, None]
        self.audio_unfreeze = False

        self.fusion_head_output_type = "mean"  # [cls, mean, max]

        for key, value in kwargs.items():
            setattr(self, key, value)
