"""
model_fashion_pretrain_sym_V5.py

What changed vs original model_fashion_pretrain.py:
  1. DecoupledTextAttention (Stage 1):
       - reads text_embeds (B, L, 768)
       - two separate Q,K,V projections on same text tokens
       - Stream A → S (B, 768): learns to focus on category words
       - Stream B → D (B, 768): learns to focus on attribute/detail words

  2. loss_moco (NEW, replaces symbol_simloss / FSIS):
       - S (symbol from text) vs image_feat_m (momentum image)
       - self-supervised: no category labels needed
       - uses a dedicated symbol queue

  3. ITA loss now uses Description stream D instead of raw [CLS] token:
       - online  branch: text_feat  = proj(D)   instead of proj(text_embeds[:, 0, :])
       - momentum branch: text_feat_m = proj(D_m) instead of proj(last_hidden_state[:, 0, :])
       - rationale: D attends to attribute/detail words, giving the ITA loss
         a richer fine-grained anchor than the generic [CLS] pooling.
       - no new parameters needed — reuses the existing text_proj /
         combine_text_proj (and their _m counterparts).

  Everything else unchanged.
"""

from functools import partial
from models.vit import VisionTransformer
from models.resnet import resnet50
from models.xbert import BertConfig, BertModel, BertOnlyMLMHead, ACT2FN

import torch
from torch import nn
import torch.nn.functional as F


# ── unchanged from original ───────────────────────────────────────────────────
class Replace_Predictor(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        self.dense_layer   = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.act_fun       = ACT2FN[config.hidden_act]
        self.norm_layer    = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.predict_layer = nn.Linear(config.hidden_size, args.replace_kind_num)

    def forward(self, hidden_embedding):
        hidden_feat = self.dense_layer(hidden_embedding)
        hidden_feat = self.act_fun(hidden_feat)
        hidden_feat = self.norm_layer(hidden_feat)
        hidden_feat = self.predict_layer(hidden_feat)
        return hidden_feat


# ── CHANGE 1: new class — DecoupledTextAttention ─────────────────────────────
class DecoupledTextAttention(nn.Module):
    """
    Input : text_embeds (B, L, 768)
    Output: S (B, 768) — symbol stream,      attends to category words
            D (B, 768) — description stream, attends to attribute words

    Both streams:
      - use the same CLS token as query
      - use ALL text tokens as keys and values
      - have completely separate K and V projection weights
    This separation forces each stream to specialise differently.
    """
    def __init__(self, hidden_size=768, num_heads=8):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = hidden_size // num_heads
        self.scale     = self.head_dim ** -0.5

        # Stream A: symbol — separate weights
        self.sym_q    = nn.Linear(hidden_size, hidden_size)
        self.sym_k    = nn.Linear(hidden_size, hidden_size)
        self.sym_v    = nn.Linear(hidden_size, hidden_size)
        self.sym_out  = nn.Linear(hidden_size, hidden_size)
        self.sym_norm = nn.LayerNorm(hidden_size)

        # Stream B: description — separate weights
        self.desc_q    = nn.Linear(hidden_size, hidden_size)
        self.desc_k    = nn.Linear(hidden_size, hidden_size)
        self.desc_v    = nn.Linear(hidden_size, hidden_size)
        self.desc_out  = nn.Linear(hidden_size, hidden_size)
        self.desc_norm = nn.LayerNorm(hidden_size)

    def _attn(self, q_proj, k_proj, v_proj, out_proj, norm, cls, tokens):
        B, L, D = tokens.shape
        H, hd   = self.num_heads, self.head_dim
        Q = q_proj(cls.unsqueeze(1)).view(B, 1, H, hd).transpose(1, 2)
        K = k_proj(tokens).view(B, L, H, hd).transpose(1, 2)
        V = v_proj(tokens).view(B, L, H, hd).transpose(1, 2)
        attn = F.softmax(torch.matmul(Q, K.transpose(-1, -2)) * self.scale, dim=-1)
        out  = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, 1, D)
        return norm(out_proj(out).squeeze(1) + cls)

    def forward(self, text_embeds):
        cls = text_embeds[:, 0, :]
        S   = self._attn(self.sym_q,  self.sym_k,  self.sym_v,
                         self.sym_out,  self.sym_norm,  cls, text_embeds)
        D   = self._attn(self.desc_q, self.desc_k, self.desc_v,
                         self.desc_out, self.desc_norm, cls, text_embeds)
        return S, D   # each (B, 768)


# ── main model ────────────────────────────────────────────────────────────────
class FashionSAP(nn.Module):
    def __init__(self, config=None, args=None):
        super().__init__()

        self.config = config
        self.args   = args
        embed_dim    = config['embed_dim']
        vision_width = config['vision_width']

        # ── unchanged ────────────────────────────────────────────────────
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768,
            depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))

        self.bert_config  = BertConfig.from_json_file(config['bert_config'])
        self.text_encoder = BertModel(config=self.bert_config, add_pooling_layer=False)
        text_width        = self.text_encoder.config.hidden_size

        self.vision_proj         = nn.Linear(vision_width, embed_dim)
        self.text_proj           = nn.Linear(text_width,   embed_dim)
        self.combine_vision_proj = nn.Linear(embed_dim,    embed_dim)
        self.combine_text_proj   = nn.Linear(embed_dim,    embed_dim)

        self.temp       = nn.Parameter(torch.ones([]) * config['temp'])
        self.queue_size = config['queue_size']
        self.momentum   = config['momentum']
        self.itm_head   = nn.Linear(text_width, 2)

        # ── CHANGE 1: add DecoupledTextAttention (online + momentum) ─────
        self.decoupled_text_attn   = DecoupledTextAttention(hidden_size=text_width)
        self.decoupled_text_attn_m = DecoupledTextAttention(hidden_size=text_width)

        # symbol projection layers (S → embed space for MoCo loss)
        self.symbol_proj           = nn.Linear(text_width, embed_dim)
        self.combine_symbol_proj   = nn.Linear(embed_dim,  embed_dim)
        self.symbol_proj_m         = nn.Linear(text_width, embed_dim)
        self.combine_symbol_proj_m = nn.Linear(embed_dim,  embed_dim)

        # ── unchanged: momentum encoders ─────────────────────────────────
        self.visual_encoder_m = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768,
            depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.vision_proj_m           = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m          = BertModel(config=self.bert_config, add_pooling_layer=False)
        self.text_proj_m             = nn.Linear(text_width,   embed_dim)
        self.combine_vision_proj_m   = nn.Linear(embed_dim,    embed_dim)
        self.combine_text_proj_m     = nn.Linear(embed_dim,    embed_dim)

        # ── unchanged: decode and replace layers ─────────────────────────
        self.decoder_layer         = BertOnlyMLMHead(config=self.bert_config)
        self.replace_predict_layer = Replace_Predictor(config=self.bert_config, args=args)

        # ── EMA pairs — new layers added, rest unchanged ─────────────────
        self.model_pairs = [
            [self.visual_encoder,        self.visual_encoder_m],
            [self.vision_proj,           self.vision_proj_m],
            [self.text_encoder,          self.text_encoder_m],
            [self.text_proj,             self.text_proj_m],
            [self.combine_text_proj,     self.combine_text_proj_m],
            [self.combine_vision_proj,   self.combine_vision_proj_m],
            # NEW pairs for decoupled attention
            [self.decoupled_text_attn,   self.decoupled_text_attn_m],
            [self.symbol_proj,           self.symbol_proj_m],
            [self.combine_symbol_proj,   self.combine_symbol_proj_m],
        ]
        self.copy_params()

        # ── unchanged: ITA queue ──────────────────────────────────────────
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue",  torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr",   torch.zeros(1, dtype=torch.long))
        self.register_buffer("idx_queue",   torch.full((1, self.queue_size), -100))

        # ── CHANGE 2: new symbol queue for MoCo loss ─────────────────────
        self.register_buffer("sym_image_queue",
                             torch.randn(embed_dim, self.queue_size))
        self.register_buffer("sym_queue_ptr",
                             torch.zeros(1, dtype=torch.long))

        self.image_queue     = F.normalize(self.image_queue,     dim=0)
        self.text_queue      = F.normalize(self.text_queue,      dim=0)
        self.sym_image_queue = F.normalize(self.sym_image_queue, dim=0)

    # ─────────────────────────────────────────────────────────────────────
    def forward(self, image, text_input_ids, text_attention_mask,
                alpha, idx, mask_labels=None, replace_labels=None,use_d_for_ita: bool = True):

        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        # ── unchanged: image encoding ─────────────────────────────────────
        image_embeds = self.visual_encoder(image)
        image_atts   = torch.ones(image_embeds.size()[:-1],
                                  dtype=torch.long).to(image.device)
        image_feat   = F.normalize(
            self.combine_vision_proj(self.vision_proj(image_embeds[:, 0, :])), dim=-1)

        # ── unchanged: text encoding ──────────────────────────────────────
        text_output  = self.text_encoder(
            text_input_ids, attention_mask=text_attention_mask,
            return_dict=True, mode='text')
        text_embeds  = text_output.last_hidden_state

        # ── CHANGE 1: get S and D from decoupled text attention ───────────
        S, D = self.decoupled_text_attn(text_embeds)  # each (B, 768)

        # ── CHANGE 3: use Description stream D for ITA instead of raw [CLS]
        _ita_anchor   = D if use_d_for_ita else text_embeds[:, 0, :]
        text_feat = F.normalize(
            self.combine_text_proj(self.text_proj(_ita_anchor)), dim=-1)

        # S is the symbol — projected for MoCo loss
        S_feat = F.normalize(
            self.combine_symbol_proj(self.symbol_proj(S)), dim=-1)

        # ── unchanged: similarity targets ────────────────────────────────
        idx     = idx.view(-1, 1)
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)
        pos_idx = torch.eq(idx, idx_all).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

        # ── unchanged: momentum forward ───────────────────────────────────
        with torch.no_grad():
            self._momentum_update()

            image_embeds_m = self.visual_encoder_m(image)
            image_feat_m   = F.normalize(
                self.combine_vision_proj_m(
                    self.vision_proj_m(image_embeds_m[:, 0, :])), dim=-1)
            image_feat_all = torch.cat(
                [image_feat_m.t(), self.image_queue.clone().detach()], dim=1)

            text_output_m  = self.text_encoder_m(
                text_input_ids, attention_mask=text_attention_mask,
                return_dict=True, mode='text')

            # CHANGE 3 (momentum): keep D_m so it can drive text_feat_m
            S_m, D_m = self.decoupled_text_attn_m(text_output_m.last_hidden_state)

            # CHANGE 3: use D_m (description stream) instead of momentum [CLS]
            _ita_anchor_m = D_m if use_d_for_ita else text_output_m.last_hidden_state[:, 0, :]
            text_feat_m = F.normalize(
                self.combine_text_proj_m(self.text_proj_m(_ita_anchor_m)), dim=-1)
            text_feat_all = torch.cat(
                [text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

            # symbol queue for MoCo
            sym_image_feat_all = torch.cat(
                [image_feat_m.t(), self.sym_image_queue.clone().detach()], dim=1)

            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m  @ image_feat_all / self.temp

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

        # ── unchanged: ITA loss ───────────────────────────────────────────
        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat  @ image_feat_all / self.temp

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
        loss_ita = (loss_i2t + loss_t2i) / 2
        self._dequeue_and_enqueue(image_feat_m, text_feat_m, idx=idx)

        # ── CHANGE 2: loss_moco replaces symbol_simloss ───────────────────
        # S came only from text. MoCo forces it to align with the image.
        # Positive = same item's momentum image. Negatives = symbol queue.
        # No category label needed — self-supervised.
        bs = image.size(0)
        sim_sym2img = S_feat @ sym_image_feat_all / self.temp   # (B, Q+B)
        sym_targets = torch.zeros(sim_sym2img.size()).to(image.device)
        sym_targets[:, :bs].fill_diagonal_(1.0)
        loss_moco = -torch.sum(
            F.log_softmax(sim_sym2img, dim=1) * sym_targets, dim=1).mean()
        self._sym_dequeue_and_enqueue(image_feat_m)

        # ── unchanged: ITM ────────────────────────────────────────────────
        output_pos = self.text_encoder(
            encoder_embeds=text_embeds,
            attention_mask=text_attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True, mode='fusion')

        with torch.no_grad():
            weights_i2t = sim_i2t[:, :bs].clone().detach()
            weights_t2i = sim_t2i[:, :bs].clone().detach()
            weights_i2t.fill_diagonal_(-1000)
            weights_t2i.fill_diagonal_(-1000)
            weights_i2t = F.softmax(weights_i2t, dim=1)
            weights_t2i = F.softmax(weights_t2i, dim=1)
            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        text_embeds_neg, text_atts_neg = [], []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text_attention_mask[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg   = torch.stack(text_atts_neg,   dim=0)

        text_embeds_all  = torch.cat([text_embeds,         text_embeds_neg], dim=0)
        text_atts_all    = torch.cat([text_attention_mask, text_atts_neg],   dim=0)
        image_embeds_all = torch.cat([image_embeds_neg,    image_embeds],    dim=0)
        image_atts_all   = torch.cat([image_atts,          image_atts],      dim=0)

        output_neg = self.text_encoder(
            encoder_embeds=text_embeds_all,
            attention_mask=text_atts_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True, mode='fusion')

        vl_embeddings = torch.cat([
            output_pos.last_hidden_state[:, 0, :],
            output_neg.last_hidden_state[:, 0, :]], dim=0)
        vl_output  = self.itm_head(vl_embeddings)
        itm_labels = torch.cat([
            torch.ones(bs,    dtype=torch.long),
            torch.zeros(2*bs, dtype=torch.long)], dim=0).to(image.device)
        loss_itm   = F.cross_entropy(vl_output, itm_labels)

        # ── unchanged: MLM + Replace ──────────────────────────────────────
        crossentropy_func = nn.CrossEntropyLoss()
        fusion_embeds     = output_pos.last_hidden_state
        mask_logit        = self.decoder_layer(fusion_embeds)
        replace_logit     = self.replace_predict_layer(fusion_embeds)
        mask_loss         = crossentropy_func(
            mask_logit.view(-1, self.bert_config.vocab_size),
            mask_labels.view(-1))
        replace_loss      = crossentropy_func(
            replace_logit.view(-1, self.args.replace_kind_num),
            replace_labels.view(-1))

        # same return signature as original — loss_moco replaces symbol_simloss
        return loss_ita, loss_itm, loss_moco, mask_loss, replace_loss

    # ── unchanged ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(),
                                      model_pair[1].parameters()):
                param_m.data.copy_(param.data)
                param_m.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(),
                                      model_pair[1].parameters()):
                param_m.data = (param_m.data * self.momentum
                                + param.data  * (1. - self.momentum))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, idx=None):
        image_feats = concat_all_gather(image_feat)
        text_feats  = concat_all_gather(text_feat)
        idxs        = concat_all_gather(idx)
        batch_size  = image_feats.shape[0]
        ptr         = int(self.queue_ptr)

        # FIX: the original hard assert crashes one process when the last
        # batch is smaller than expected (drop_last=False with uneven dataset).
        # Instead, only enqueue as many samples as fit without wrapping; the
        # small remainder is simply discarded for this step.
        space = self.queue_size - ptr
        n     = min(batch_size, space)   # samples we can write without wrapping
        self.image_queue[:, ptr:ptr + n] = image_feats[:n].T
        self.text_queue[:,  ptr:ptr + n] = text_feats[:n].T
        self.idx_queue[:,   ptr:ptr + n] = idxs[:n].T
        ptr = (ptr + n) % self.queue_size
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _sym_dequeue_and_enqueue(self, image_feat):
        """Dedicated queue for MoCo symbol loss."""
        image_feats = concat_all_gather(image_feat)
        batch_size  = image_feats.shape[0]
        ptr         = int(self.sym_queue_ptr)
        # FIX: same safe enqueue — no hard assert
        space = self.queue_size - ptr
        n     = min(batch_size, space)
        self.sym_image_queue[:, ptr:ptr + n] = image_feats[:n].T
        ptr = (ptr + n) % self.queue_size
        self.sym_queue_ptr[0] = ptr


# ── unchanged ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    return torch.cat(tensors_gather, dim=0)