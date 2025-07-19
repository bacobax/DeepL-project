import clip
import torch
from torch import nn
from torch.nn import functional as F
from contextlib import contextmanager


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding
        x = x.permute(1, 0, 2)  # [batch_size, n_ctx, transformer.width] -> [n_ctx, batch_size, transformer.width]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [n_ctx, batch_size, transformer.width] -> [batch_size, n_ctx, transformer.width]
        x = self.ln_final(x)
        if torch.isnan(x).any():
            print("DSFSADSDFA")

        # Take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        

        return x

    

class PromptLearnerCoOp(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal clip_imsize ({clip_imsize})"

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        #print(f'Initial context: "{prompt_prefix}"')
        #print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts.to(clip_model.token_embedding.weight.device))
            embedding = embedding.type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])

        self.tokenized_prompts = tokenized_prompts
        self.n_cls = n_cls
        self.n_ctx = n_ctx

    def construct_prompts(self, ctx, prefix, suffix):
        return torch.cat([prefix, ctx, suffix], dim=1)

    def forward(self):
        ctx = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        return self.construct_prompts(ctx, self.token_prefix, self.token_suffix)


class CustomCLIPCoOp(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearnerCoOp(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.cfg = cfg
        self.dtype = clip_model.dtype

    def forward(self, image, label=None):
        logit_scale = self.logit_scale.exp()

        #print("Raw logit_scale:", self.logit_scale.item())
        #print("Exp logit_scale:", logit_scale)

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts.type(self.dtype), self.tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if torch.isnan(image_features).any():
            print("⚠️ NaNs in image_features!")

        if torch.isnan(text_features).any():
            print("⚠️ NaNs in text_features!")

        if torch.isinf(image_features).any():
            print("⚠️ Infs in image_features!")

        if torch.isinf(text_features).any():
            print("⚠️ Infs in text_features!")

        #print("Image feature norm:", image_features.norm(dim=-1).mean().item())
        #print("Text feature norm:", text_features.norm(dim=-1).mean().item())

        logits = logit_scale * image_features @ text_features.t()

        #print(f"is training: {self.prompt_learner.training}, label: {label}")
        
        if label is not None:
            
            return F.cross_entropy(logits, label), logits

        return logits
    @contextmanager
    def temporary_classnames(self, new_classnames):
        # --- Save original state ---
        original_classnames = self.prompt_learner.n_cls
        original_tokenized_prompts = self.tokenized_prompts
        original_token_prefix = self.prompt_learner.token_prefix
        original_token_suffix = self.prompt_learner.token_suffix

        # --- Apply temporary state ---
        temp_prompt_learner = PromptLearnerCoOp(
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
