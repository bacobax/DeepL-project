from collections import OrderedDict
import os
import torch
import torch.nn as nn
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COCOOP.N_CTX
        ctx_init = cfg.TRAINER.COCOOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)


        self.ctx = nn.Parameter(ctx_vectors)

        # Optional: Load pre-trained ctx from a file
        if hasattr(cfg.TRAINER.COCOOP, "CTX_LOAD") and cfg.TRAINER.COCOOP.CTX_LOAD:
            ctx_path = cfg.TRAINER.COCOOP.CTX_LOAD
            if os.path.isfile(ctx_path):
                #print(f"🔁 Loading ctx from: {ctx_path}")
                state_dict = torch.load(ctx_path, map_location="cpu")
                if "ctx" in state_dict:
                    with torch.no_grad():
                        self.ctx.copy_(state_dict["ctx"])
                else:
                    raise KeyError(f"'ctx' not found in {ctx_path}")
            else:
                raise FileNotFoundError(f"CTX_LOAD path not found: {ctx_path}")
            #print("PROMPT LEARNER LOADED FROM A COOP PRETRAINED ONE")

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))

        if cfg.TRAINER.COCOOP.PREC == "fp16" and not torch.backends.mps.is_available():
            self.meta_net.half()
        else:
            print("⚠️ Using float32 for meta_net due to MPS")

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            device = clip_model.token_embedding.weight.device

            # Ensure tokenized_prompts is on the right device BEFORE embedding
            tokenized_prompts = tokenized_prompts.to(device)

            embedding = clip_model.token_embedding(tokenized_prompts)

            # Do not convert to fp16 on MPS (Apple doesn't support it fully)
            if device.type == "mps" and dtype == torch.float16:
                print("⚠️ fp16 not fully supported on MPS; using float32 instead")
                dtype = torch.float32

            embedding = embedding.to(dtype)
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx                     # (n_ctx, ctx_dim)
        if im_features.isnan().any():
            raise ValueError("NaN in im_features before meta_net")
        
        #print("im_features stats", im_features.min().item(), im_features.max().item(), im_features.norm(dim=1).mean().item())

        meta_net_dtype = next(self.meta_net.parameters()).dtype
        bias = self.meta_net(im_features.to(meta_net_dtype))  # (batch, ctx_dim)
        if bias.isnan().any():
            raise ValueError("NaN detected in bias")
        bias = bias.unsqueeze(1)           # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)             # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias           # (batch, n_ctx, ctx_dim)
        if ctx_shifted.isnan().any():
            raise ValueError("NaN detected in ctx_shifted")
        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1) # (n_cls, n_ctx, ctx_dim)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            if pts_i.isnan().any():
                raise ValueError("NaN detected in pts_i")
            prompts.append(pts_i)
        prompts = torch.stack(prompts)

        return prompts, ctx, bias
