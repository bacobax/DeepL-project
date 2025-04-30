import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from model.prompt_learner import PromptLearner
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from easydict import EasyDict

_tokenizer = _Tokenizer()


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner(image_features)
        
        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)
        
        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)
        
        return logits


if __name__ == '__main__':
    clip_model = clip.load("ViT-B/32", jit=False)[0]

    # Define sample classnames for testing
    classnames = [
        "dog", "cat", "bird", "fish", "horse",
        "elephant", "bear", "zebra", "giraffe", "lion"
    ]

    cfg = EasyDict()

    # Training configuration
    cfg.TRAINER = EasyDict()
    cfg.TRAINER.COCOOP = EasyDict()
    cfg.TRAINER.COCOOP.N_CTX = 16  # Number of context tokens
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # Leave empty for random initialization
    cfg.TRAINER.COCOOP.PREC = "fp16"  # Precision for meta network

    # Input configuration
    cfg.INPUT = EasyDict()

    resolution = clip_model.visual.input_resolution

    cfg.INPUT.SIZE = [resolution, resolution]  # Must match CLIP model's input resolution
    # Initialize CustomCLIP with all required parameters
    custom_clip = CustomCLIP(cfg, classnames, clip_model)