import os.path as osp
from collections import OrderedDict
import math
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from model.cocoop.prompt_learner import PromptLearner
from model.cocoop.mlp_adversary import GradientReversalLayer, AdversarialMLP
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
        print(f"self.dtype={self.dtype}")

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
        self.clip_model = clip_model
        self.cfg = cfg



    @contextmanager
    def temporary_classnames(self, new_classnames):
        # --- Save original state ---
        original_classnames = self.prompt_learner.n_cls
        original_tokenized_prompts = self.tokenized_prompts
        original_token_prefix = self.prompt_learner.token_prefix
        original_token_suffix = self.prompt_learner.token_suffix

        # --- Apply temporary state ---
        temp_prompt_learner = PromptLearner(
            cfg=self.cfg,
            classnames=new_classnames,
            clip_model=self.clip_model
        )

        self.tokenized_prompts = temp_prompt_learner.tokenized_prompts
        self.prompt_learner.tokenized_prompts = temp_prompt_learner.tokenized_prompts
        self.prompt_learner.token_prefix = temp_prompt_learner.token_prefix
        self.prompt_learner.token_suffix = temp_prompt_learner.token_suffix
        self.prompt_learner.n_cls = len(new_classnames)

        try:
            yield
        finally:
            # --- Restore original state ---
            self.tokenized_prompts = original_tokenized_prompts
            self.prompt_learner.tokenized_prompts = original_tokenized_prompts
            self.prompt_learner.token_prefix = original_token_prefix
            self.prompt_learner.token_suffix = original_token_suffix
            self.prompt_learner.n_cls = original_classnames


    def forward(self, image, label=None, get_image_features=False):
        # tokenized_prompts: [num_classes, context_length] (e.g., [10, 77])
        tokenized_prompts = self.tokenized_prompts

        # logit_scale: scalar (e.g., initialized as a learnable parameter like torch.tensor(1/0.07).log())
        logit_scale = self.logit_scale.exp()

        # image: [B, 3, H, W]
        # image_features: [B, D] where D = transformer width (e.g., 512 for ViT-B/32)
        #print(f"image device: {image.device} | image encoder device: {next(self.image_encoder.parameters()).device}")
        image_features = self.image_encoder(image.type(self.dtype))
        if image_features.isnan().any():
            raise ValueError("NaN detected in image_features.")
        # Normalize image features: each row to unit length
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # prompts: List of [num_classes, context_length, D] (one per image feature)
        # Each element is generated conditioned on an image feature
        prompts, ctx, bias = self.prompt_learner(image_features) # [B , n_cls, n_ctx, D]
        if prompts.isnan().any():
            raise ValueError("NaN detected in prompts.")
        # prompts: [B, n_cls, n_ctx, D] -> [B * n_cls, n_ctx, D]
        logits = []
        all_text_features = []
        # Iterate over batch
        for pts_i, imf_i in zip(prompts, image_features):
            # pts_i: [num_classes, context_length, D]
            # tokenized_prompts: [num_classes, context_length]
            # text_features: [num_classes, D]
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            if text_features.isnan().any():
                raise ValueError("NaN detected in text ft.")
            all_text_features.append(text_features)
            # Normalize text features
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # imf_i: [D], text_features.T: [D, num_classes]
            # l_i: [num_classes], similarity scores between image and all class prompts
            l_i = logit_scale * imf_i @ text_features.t()
            # Append l_i (1D tensor) to logits list
            logits.append(l_i)

        all_text_features = torch.stack(all_text_features)  # [B, num_classes, D]
        #avarage over num_classes
        avg_text_features = all_text_features.mean(dim=1)

        # logits: list of B tensors each of shape [num_classes]
        # stacked into a tensor of shape [B, num_classes]
        logits = torch.stack(logits)
        if logits.isnan().any():
            raise ValueError("NaN detected in logits")

        # If in training mode, compute and return cross-entropy loss
        if self.prompt_learner.training:
            # logits: [B, num_classes], label: [B]
            if get_image_features:
                # If get_image_features is True, return logits and image features
                return logits, F.cross_entropy(logits, label), image_features, ctx, bias, avg_text_features
            else:
                return logits, F.cross_entropy(logits, label)

        # Otherwise, return logits for evaluation: [B, num_classes]
        return logits
