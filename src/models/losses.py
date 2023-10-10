from torch.nn import CrossEntropyLoss as CELoss


class CrossEntropyLoss(CELoss):
    def __init__(self, feat_dim, num_classes, lambda_c=1.0, **kwargs):
        super(CrossEntropyLoss, self).__init__(**kwargs)

    def forward(self, input, target):
        out = input[0]
        return super().forward(out, target)
