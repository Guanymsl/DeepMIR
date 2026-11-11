import torch.nn as nn
from hear21passt.base import get_basic_model
from av_bench.vggish.vggish import VGGish


class ExtractionModels(nn.Module):
    def __init__(self):
        super().__init__()
        self.vggish = VGGish(postprocess=False).eval()
        self.passt_model = get_basic_model(mode="all").eval()
