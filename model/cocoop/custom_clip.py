import os.path as osp
from collections import OrderedDict
import math
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from model.cocoop.prompt_learner import PromptLearner
from model.cocoop.mlp_adversary import CLSBiasAdderMLP, CLSDiscriminatorMLP
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
        self.meta_net_2 = CLSBiasAdderMLP(self.prompt_learner.cls_dim, dropout=0.3)


    def change_classnames(self, new_classnames):
        """
        Change the class names and update the prompt learner's tokenized prompts accordingly.

        Args:
            new_classnames (list): List of new class names.
        """
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


    def forward(self, image, label=None, get_image_features=False, meta_net_2=True):
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

        image_features = self.meta_net_2(image_features, apply_bias=meta_net_2)
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
        selected_text_features = []    
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
            best_idx = l_i.argmax()
            best_text_feat = text_features[best_idx]  # [D]
            selected_text_features.append(best_text_feat)

        all_text_features = torch.stack(all_text_features)  # [B, num_classes, D]
        #avarage over num_classes
        avg_text_features = all_text_features.mean(dim=1)

        # Shape: [B, D]
        selected_text_features = torch.stack(selected_text_features)
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
                return logits, F.cross_entropy(logits, label), image_features, ctx, bias, avg_text_features, selected_text_features
            else:
                return logits, F.cross_entropy(logits, label)

        # Otherwise, return logits for evaluation: [B, num_classes]
        return logits

    def print_all_dtypes(self):
        print(f"CustomCLIP dtype: {self.dtype}")
        print(f"  logit_scale dtype: {getattr(self.logit_scale, 'dtype', type(self.logit_scale))}")
        print(f"  tokenized_prompts dtype: {getattr(self.tokenized_prompts, 'dtype', type(self.tokenized_prompts))}")
        print(f"  image_encoder: {type(self.image_encoder)}")
        for name, param in self.image_encoder.named_parameters():
            print(f"    image_encoder param {name}: {param.dtype}")
        for name, buf in self.image_encoder.named_buffers():
            print(f"    image_encoder buffer {name}: {buf.dtype}")
        print(f"  text_encoder: {type(self.text_encoder)}")
        for name, param in self.text_encoder.named_parameters():
            print(f"    text_encoder param {name}: {param.dtype}")
        for name, buf in self.text_encoder.named_buffers():
            print(f"    text_encoder buffer {name}: {buf.dtype}")
        print(f"  prompt_learner: {type(self.prompt_learner)}")
        for name, param in self.prompt_learner.named_parameters():
            print(f"    prompt_learner param {name}: {param.dtype}")
        for name, buf in self.prompt_learner.named_buffers():
            print(f"    prompt_learner buffer {name}: {buf.dtype}")
        # Also print dtype for ctx, token_prefix, token_suffix if present
        if hasattr(self.prompt_learner, 'ctx'):
            print(f"    prompt_learner.ctx dtype: {self.prompt_learner.ctx.dtype}")
        if hasattr(self.prompt_learner, 'token_prefix'):
            print(f"    prompt_learner.token_prefix dtype: {self.prompt_learner.token_prefix.dtype}")
        if hasattr(self.prompt_learner, 'token_suffix'):
            print(f"    prompt_learner.token_suffix dtype: {self.prompt_learner.token_suffix.dtype}")
        print(f"  clip_model: {type(self.clip_model)}")
        for name, param in self.clip_model.named_parameters():
            print(f"    clip_model param {name}: {param.dtype}")
        for name, buf in self.clip_model.named_buffers():
            print(f"    clip_model buffer {name}: {buf.dtype}")
    @staticmethod
    def load_from_checkpoint(classnames, checkpoint_path, device, n_ctx, clip_model, ctx_8_coop, ctx_4_coop, ctx_init=""):
        """
        Load a CustomCLIP model from a saved checkpoint.

        Args:
            classnames (list): List of class names for the model.
            checkpoint_path (str): Path to the saved checkpoint file.
            device (str): Device to load the model onto (e.g., 'cuda', 'cpu').

        Returns:
            CustomCLIP: An instance of CustomCLIP loaded with the checkpoint.
        """
        ctx_load = (
            ctx_4_coop
            if n_ctx == 4
            else ctx_8_coop
        )
        resolution = clip_model.visual.input_resolution
        cfg = EasyDict(
            {
                "TRAINER": {
                    "COCOOP": {
                        "CTX_LOAD": ctx_load,
                        "N_CTX": n_ctx,
                        "CTX_INIT": ctx_init,
                        "PREC": "fp16",
                    }
                },
                "INPUT": {"SIZE": [resolution, resolution]},
            }
        )
        state_dict = torch.load(checkpoint_path, map_location=device)
        n_cls = state_dict["prompt_learner.token_prefix"].shape[0]

        clip_model, _ = clip.load("ViT-B/16", device=device)

        model = CustomCLIP(cfg, ["X"] * n_cls, clip_model)
        model.load_state_dict(state_dict)
        model.change_classnames(classnames)
        return model
