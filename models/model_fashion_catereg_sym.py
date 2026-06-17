"""
model_fashion_catereg_sym.py

Downstream Category Recognition (CR) + Subcategory Recognition (SCR) model
adapted to load from a FashionSAP-Sym pre-trained checkpoint.

Key differences vs the original model_fashion_catereg.py:
  1. No special symbol tokens ([tops_sign] etc.) added to the tokenizer.
     The new pre-training does not use them.

  2. DecoupledTextAttention is copied from the pre-training model.
     It produces two 768-dim vectors from the BERT output:
       S  — symbol stream, focused on category-level words
       D  — description stream, focused on attribute/detail words

  3. Option C classification heads:
       concat [S ; D]  →  1536-dim vector
       cr_head  : Linear(1536) → LayerNorm → GELU → Dropout → Linear(48)
       scr_head : Linear(1536) → LayerNorm → GELU → Dropout → Linear(121)
     Both heads share the same forward pass.

  4. Dropout(0.1) added after GELU in both heads to prevent overfitting.
     Without dropout the SCR train loss collapses to ~0 while test
     accuracy barely moves — classic memorisation of the training set.

  5. Checkpoint loading handles the new key names gracefully with
     strict=False — only the two new classification heads are
     uninitialized (missing_keys); everything else transfers cleanly
     from the pre-training checkpoint.
"""

from functools import partial
import torch
from torch import nn
import torch.nn.functional as F

from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel


# ── DecoupledTextAttention (identical to pre-training model) ──────────────────
class DecoupledTextAttention(nn.Module):
    """
    Sits on top of the BERT text encoder output.

    Input : text_embeds  (B, L, 768)
    Output: S (B, 768)  — symbol stream, category focus
            D (B, 768)  — description stream, attribute focus

    Both streams use the CLS token as query and attend over all tokens
    with completely separate Q/K/V weights, forcing specialisation.
    Residual connection + LayerNorm keeps output on the BERT hidden scale.
    """

    def __init__(self, hidden_size: int = 768, num_heads: int = 8):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = hidden_size // num_heads
        self.scale     = self.head_dim ** -0.5

        # Stream A — symbol (category focus)
        self.sym_q    = nn.Linear(hidden_size, hidden_size)
        self.sym_k    = nn.Linear(hidden_size, hidden_size)
        self.sym_v    = nn.Linear(hidden_size, hidden_size)
        self.sym_out  = nn.Linear(hidden_size, hidden_size)
        self.sym_norm = nn.LayerNorm(hidden_size)

        # Stream B — description (attribute focus)
        self.desc_q    = nn.Linear(hidden_size, hidden_size)
        self.desc_k    = nn.Linear(hidden_size, hidden_size)
        self.desc_v    = nn.Linear(hidden_size, hidden_size)
        self.desc_out  = nn.Linear(hidden_size, hidden_size)
        self.desc_norm = nn.LayerNorm(hidden_size)

    def _attn(self, q_proj, k_proj, v_proj, out_proj, norm, cls, tokens):
        B, L, D = tokens.shape
        H, hd   = self.num_heads, self.head_dim
        Q    = q_proj(cls.unsqueeze(1)).view(B, 1, H, hd).transpose(1, 2)
        K    = k_proj(tokens).view(B, L, H, hd).transpose(1, 2)
        V    = v_proj(tokens).view(B, L, H, hd).transpose(1, 2)
        attn = F.softmax(
            torch.matmul(Q, K.transpose(-1, -2)) * self.scale, dim=-1)
        out  = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, 1, D)
        return norm(out_proj(out).squeeze(1) + cls)

    def forward(self, text_embeds):
        cls = text_embeds[:, 0, :]
        S   = self._attn(self.sym_q,  self.sym_k,  self.sym_v,
                         self.sym_out,  self.sym_norm,  cls, text_embeds)
        D   = self._attn(self.desc_q, self.desc_k, self.desc_v,
                         self.desc_out, self.desc_norm, cls, text_embeds)
        return S, D     # each (B, 768)


# ── Main downstream model ─────────────────────────────────────────────────────
class FashionSAP(nn.Module):
    """
    Fine-tuning model for CR and SCR using Option-C classification heads.
    Forward returns (cr_logit, scr_logit).
    """

    def __init__(self, config=None, args=None):
        super().__init__()
        self.config = config
        self.args   = args

        # ── image encoder ─────────────────────────────────────────────────
        self.visual_encoder = VisionTransformer(
            img_size   = config['image_res'],
            patch_size = 16,
            embed_dim  = 768,
            depth      = 12,
            num_heads  = 12,
            mlp_ratio  = 4,
            qkv_bias   = True,
            norm_layer = partial(nn.LayerNorm, eps=1e-6),
        )

        # ── text encoder ──────────────────────────────────────────────────
        self.bert_config  = BertConfig.from_json_file(config['bert_config'])
        self.text_encoder = BertModel(
            config=self.bert_config, add_pooling_layer=False)
        text_width = self.text_encoder.config.hidden_size   # 768

        # ── DecoupledTextAttention (weights loaded from pre-train ckpt) ───
        self.decoupled_text_attn = DecoupledTextAttention(hidden_size=text_width)

        # ── Option C: concat [S ; D]  →  1536-dim ─────────────────────────
        sd_dim = text_width * 2   # 1536

        # FIX: Dropout(0.1) added after GELU in both heads.
        # Without it, SCR train loss collapses to ~0 while test accuracy
        # barely improves — the model memorises training captions instead
        # of generalising.  0.1 dropout is gentle enough not to hurt CR
        # (which is already near-perfect) while regularising SCR.
        self.cr_head = nn.Sequential(
            nn.Linear(sd_dim, sd_dim),
            nn.LayerNorm(sd_dim, eps=self.bert_config.layer_norm_eps),
            nn.GELU(),
            nn.Dropout(0.1),                        # ← regularisation
            nn.Linear(sd_dim, args.cr_class_num),
        )

        self.scr_head = nn.Sequential(
            nn.Linear(sd_dim, sd_dim),
            nn.LayerNorm(sd_dim, eps=self.bert_config.layer_norm_eps),
            nn.GELU(),
            nn.Dropout(0.1),                        # ← regularisation
            nn.Linear(sd_dim, args.scr_class_num),
        )

        # itm_head kept for checkpoint compatibility (weights load but
        # are not used in forward — safe to ignore)
        self.itm_head = nn.Linear(text_width, 2)

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, image, text_input_ids, text_attention_mask):
        """
        Args:
            image               : (B, 3, H, W)
            text_input_ids      : (B, L)
            text_attention_mask : (B, L)

        Returns:
            cr_logit  : (B, cr_class_num)
            scr_logit : (B, scr_class_num)
        """
        # image encoding
        image_embeds = self.visual_encoder(image)
        image_atts   = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        # text encoding
        text_output = self.text_encoder(
            text_input_ids,
            attention_mask = text_attention_mask,
            return_dict    = True,
            mode           = 'text',
        )
        text_embeds = text_output.last_hidden_state   # (B, L, 768)

        # S and D streams
        S, D = self.decoupled_text_attn(text_embeds)  # each (B, 768)

        # cross-modal fusion — output not used for classification but keeps
        # the fusion encoder weights warm; remove if GPU memory is tight
        _ = self.text_encoder(
            encoder_embeds         = text_embeds,
            attention_mask         = text_attention_mask,
            encoder_hidden_states  = image_embeds,
            encoder_attention_mask = image_atts,
            return_dict            = True,
            mode                   = 'fusion',
        )

        # Option C: concat [S ; D] → two heads
        sd        = torch.cat([S, D], dim=-1)   # (B, 1536)
        cr_logit  = self.cr_head(sd)             # (B, 48)
        scr_logit = self.scr_head(sd)            # (B, 121)

        return cr_logit, scr_logit