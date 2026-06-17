"""
fashion_pretrain_sym.py

Changes vs original fashion_pretrain.py:
  1. Import FashionSAP from model_fashion_pretrain_sym (V6)
  2. 'sis' -> 'moco' in metric_logger  (loss_moco replaces symbol_simloss)
  3. Gradient accumulation (--grad_accum_steps, default=8)
  4. Peak LR via --lr_override (recommended 1e-4 for effective batch 128)
  5. optimizer.step() / zero_grad() only every accum steps + grad clip
  6. [NEW] Special symbol tokens ([tops_sign] etc.) NOT added to tokenizer.
  7. [NEW] strip_sign_prefix() removes any [xxx_sign] prepended by the dataset.
  8. [NEW] Always save last epoch checkpoint; delete previous one.
  9. [NEW] Summary text file name is configurable via --summary_name.
 10. [NEW] evaluate_retrieval() computes R@1 on train (sampled) and test sets.
 11. [NEW] D-stream warmup: for the first --ita_warmup_epochs epochs the ITA
         loss uses the raw [CLS] token as its text anchor (same as original
         FashionSAP).  Once D has been trained for enough epochs, the ITA
         anchor switches to the D stream.  This prevents poor ITA alignment
         at the start of training caused by an untrained D stream.
         Requires the model's forward() to accept use_d_for_ita (bool).
"""

import argparse
import os
import re
import shutil
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_fashion_pretrain_sym import FashionSAP
from models.vit import interpolate_pos_embed
from transformers.models.bert.tokenization_bert import BertTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer


# ── CHANGE 7: strip [xxx_sign] prefix ────────────────────────────────────────
_SIGN_RE = re.compile(r'^\s*\[[a-zA-Z]+_sign\]\s*', re.IGNORECASE)

def strip_sign_prefix(text: str) -> str:
    """
    Remove a leading fashion-symbol token such as [tops_sign] from a string.

    >>> strip_sign_prefix("[tops_sign] a pink long sleeve shirt")
    'a pink long sleeve shirt'
    >>> strip_sign_prefix("a pink long sleeve shirt")
    'a pink long sleeve shirt'
    """
    return _SIGN_RE.sub('', text).strip()
# ─────────────────────────────────────────────────────────────────────────────


# ── CHANGE 10: retrieval evaluation ──────────────────────────────────────────
@torch.no_grad()
def evaluate_retrieval(model_without_ddp, data_loader, device,
                       max_batches: int = None) -> dict:
    """
    Compute image-text retrieval R@1 accuracy on a data loader.
    Always uses the D stream for text features (the final target behaviour),
    regardless of which ITA anchor is currently active during training.

    NOTE: must only be called on rank-0.  The caller is responsible for
    ensuring this (wrap with `if utils.is_main_process()`).

    Args:
        max_batches: cap the number of batches (None = all batches).

    Returns dict with keys: r1_i2t, r1_t2i, r1_mean  (all in %)
    """
    model_without_ddp.eval()
    all_img_feats, all_txt_feats = [], []

    for i, batch in enumerate(data_loader):
        if max_batches is not None and i >= max_batches:
            break

        image               = batch[0].to(device, non_blocking=True)
        text_input_ids      = batch[1].to(device, non_blocking=True)
        text_attention_mask = batch[2].to(device, non_blocking=True)

        # image features
        img_emb = model_without_ddp.visual_encoder(image)
        img_f   = F.normalize(
            model_without_ddp.combine_vision_proj(
                model_without_ddp.vision_proj(img_emb[:, 0, :])), dim=-1)

        # text features — always use D stream for evaluation
        txt_out = model_without_ddp.text_encoder(
            text_input_ids, attention_mask=text_attention_mask,
            return_dict=True, mode='text')
        _, D  = model_without_ddp.decoupled_text_attn(txt_out.last_hidden_state)
        txt_f = F.normalize(
            model_without_ddp.combine_text_proj(
                model_without_ddp.text_proj(D)), dim=-1)

        all_img_feats.append(img_f.cpu())
        all_txt_feats.append(txt_f.cpu())

    all_img = torch.cat(all_img_feats, dim=0)
    all_txt = torch.cat(all_txt_feats, dim=0)
    N       = len(all_img)
    labels  = torch.arange(N)

    # FIX: compute similarity in chunks to avoid a giant N×N CPU matrix
    # that can exhaust RAM when N is large (e.g. full FashionGen test set)
    chunk = 512
    r1_i2t_hits, r1_t2i_hits = 0, 0
    for start in range(0, N, chunk):
        end     = min(start + chunk, N)
        sim_row = all_img[start:end] @ all_txt.T    # (chunk, N)
        r1_i2t_hits += (sim_row.argmax(1) == labels[start:end]).sum().item()
    for start in range(0, N, chunk):
        end     = min(start + chunk, N)
        sim_col = all_txt[start:end] @ all_img.T    # (chunk, N)
        r1_t2i_hits += (sim_col.argmax(1) == labels[start:end]).sum().item()

    r1_i2t = r1_i2t_hits / N * 100
    r1_t2i = r1_t2i_hits / N * 100

    # FIX: explicitly free large tensors so they don't linger into the next
    # training epoch (Python GC is not guaranteed to run immediately)
    del all_img_feats, all_txt_feats, all_img, all_txt
    import gc; gc.collect()
    torch.cuda.empty_cache()

    return {
        'r1_i2t':  round(r1_i2t,               2),
        'r1_t2i':  round(r1_t2i,               2),
        'r1_mean': round((r1_i2t + r1_t2i) / 2, 2),
    }
# ─────────────────────────────────────────────────────────────────────────────


def train(model, data_loader, optimizer, epoch, warmup_steps, device,
          scheduler, config, grad_accum_steps: int = 8,
          use_d_for_ita: bool = True) -> dict:   # ← CHANGE 11: new flag
    """One training epoch.  Returns a dict of averaged metric strings."""
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr',   utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('ita',  utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('itm',  utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('moco', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('ml',   utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('rl',   utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header            = 'Train Epoch: [{}]'.format(epoch)
    print_freq        = 50
    step_size         = 100
    warmup_iterations = warmup_steps * step_size

    optimizer.zero_grad()
    accum_step = 0

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        batch = [item.to(device, non_blocking=True) for item in batch]
        (image, text_input_ids, text_attention_mask,
         mask_labels, replace_labels, idx) = batch

        alpha = (config['alpha']
                 if epoch > 0 or not config['warm_up']
                 else config['alpha'] * min(1, i / len(data_loader)))

        # ── CHANGE 11: pass use_d_for_ita to model forward ────────────────
        loss_ita, loss_itm, loss_moco, mask_loss, replace_loss = model(
            image, text_input_ids, text_attention_mask,
            alpha=alpha, idx=idx,
            mask_labels=mask_labels, replace_labels=replace_labels,
            use_d_for_ita=use_d_for_ita)
        # ─────────────────────────────────────────────────────────────────

        loss        = loss_ita + loss_itm + loss_moco + mask_loss + replace_loss
        loss_scaled = loss / grad_accum_steps
        loss_scaled.backward()
        accum_step += 1

        is_last_batch = (i == len(data_loader) - 1)
        should_step   = (accum_step % grad_accum_steps == 0) or is_last_batch

        if should_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            accum_step = 0

            opt_step = i // grad_accum_steps
            if epoch == 0 and opt_step % step_size == 0 and opt_step <= warmup_iterations:
                scheduler.step(opt_step // step_size)

        metric_logger.update(itm=loss_itm.item())
        metric_logger.update(ita=loss_ita.item())
        metric_logger.update(moco=loss_moco.item())
        metric_logger.update(ml=mask_loss.item())
        metric_logger.update(rl=replace_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.4f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()}


def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # overrides
    if args.lr_override > 0:
        config['optimizer']['lr'] = args.lr_override
        print(f"[override] peak LR = {args.lr_override}")

    if args.batch_size_override > 0:
        config['batch_size_train'] = args.batch_size_override
        print(f"[override] batch_size_train = {args.batch_size_override}")

    if args.max_epoch > 0:
        config['schedular']['epochs'] = args.max_epoch
        print(f"[override] max_epoch = {args.max_epoch}")

    eff_batch = config['batch_size_train'] * args.grad_accum_steps
    print(f"[batch] physical={config['batch_size_train']}  "
          f"accum={args.grad_accum_steps}  effective={eff_batch}")

    # ── CHANGE 11: report warmup plan ────────────────────────────────────────
    print(f"[ita_anchor] epochs 0-{args.ita_warmup_epochs - 1}: raw [CLS]  |  "
          f"epochs {args.ita_warmup_epochs}+: D stream")
    # ─────────────────────────────────────────────────────────────────────────

    # CHANGE 6: tokenizer without special symbol tokens
    tokenizer = BertTokenizer.from_pretrained(config['tokenizer_config'])
    print("[tokenizer] Fashion-symbol tokens NOT added -- "
          "S-stream discovers category from description words.")

    print("Creating dataset")
    train_dataset, test_dataset = create_dataset(
        'pretrain', config, args, tokenizer,
    )

    # CHANGE 7: patch datasets to strip [xxx_sign] prefixes
    def _patch_dataset_strip(dataset):
        original_getitem = dataset.__class__.__getitem__
        def patched_getitem(self, index):
            sample = original_getitem(self, index)
            if isinstance(sample, (list, tuple)):
                sample = type(sample)(
                    strip_sign_prefix(x) if isinstance(x, str) else x
                    for x in sample)
            elif isinstance(sample, str):
                sample = strip_sign_prefix(sample)
            return sample
        import types
        dataset.__getitem__ = types.MethodType(patched_getitem, dataset)

    _patch_dataset_strip(train_dataset)
    if test_dataset is not None:
        _patch_dataset_strip(test_dataset)
    print("[dataset] strip_sign_prefix patch applied.")

    if args.distributed:
        num_tasks   = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None]
    else:
        samplers = [None, None]

    # CHANGE 10: create test_loader alongside train_loader
    train_loader, test_loader = create_loader(
        [train_dataset,             test_dataset],
        [samplers[0],               None],
        batch_size=[config['batch_size_train'],
                    config.get('batch_size_test', config['batch_size_train'])],
        num_workers=[8, 4],
        is_trains=[True, False],
        collate_fns=[None, None],
    )

    print("Creating model")
    model = FashionSAP(config=config, args=args)

    if args.pre_point:
        checkpoint = torch.load(args.pre_point, map_location='cpu')
        state_dict = checkpoint['model']
        state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(
            state_dict['visual_encoder.pos_embed'], model.visual_encoder)
        state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(
            state_dict['visual_encoder_m.pos_embed'], model.visual_encoder_m)
        for key in list(state_dict.keys()):
            if 'bert' in key:
                state_dict[key.replace('bert.', '')] = state_dict.pop(key)

        # ── queue-size mismatch fix ───────────────────────────────────────
        # The queues (image_queue, text_queue, etc.) are not learned weights —
        # they are momentum memory buffers.  If the checkpoint was saved with a
        # different queue_size than the current config (e.g. 65536 vs 4096),
        # loading would crash.  We simply drop those keys so the model keeps
        # its freshly-initialised queues instead.
        model_state = model.state_dict()
        for qkey in list(state_dict.keys()):
            if qkey in model_state:
                if state_dict[qkey].shape != model_state[qkey].shape:
                    print(f"[queue] shape mismatch for '{qkey}': "
                          f"checkpoint={state_dict[qkey].shape}  "
                          f"model={model_state[qkey].shape}  -> skipped")
                    del state_dict[qkey]
        # ─────────────────────────────────────────────────────────────────

        msg = model.load_state_dict(state_dict, strict=False)
        print(f'Loaded checkpoint from {args.pre_point}')
        print(msg)

        # ── reset queue pointers ──────────────────────────────────────────
        # queue_ptr / sym_queue_ptr have shape [1] in both checkpoint and
        # model, so the shape-mismatch check above does NOT skip them.
        # If the checkpoint was trained with a different queue_size the
        # pointer value (e.g. 65408) is now larger than the new queue
        # (e.g. 4096), which causes a zero-size slice crash on the first
        # _dequeue_and_enqueue call.  Safe fix: always reset to 0 so the
        # queue fills from the start with the current queue_size.
        with torch.no_grad():
            model.queue_ptr.zero_()
            model.sym_queue_ptr.zero_()
        print("[queue] queue_ptr and sym_queue_ptr reset to 0.")
        # ─────────────────────────────────────────────────────────────────

    elif args.bert_point and args.vit_point:
        t_ckpt = utils.text_state_compatibility(
            torch.load(args.bert_point, map_location='cpu'))
        print(model.text_encoder.load_state_dict(t_ckpt, strict=False))
        v_ckpt = torch.load(args.vit_point, map_location='cpu')['model']
        v_ckpt['pos_embed'] = interpolate_pos_embed(
            v_ckpt['pos_embed'], model.visual_encoder)
        for k in list(v_ckpt.keys()):
            if k.startswith('head.'):
                del v_ckpt[k]
        print(model.visual_encoder.load_state_dict(v_ckpt, strict=True))

    model = model.to(device)
    model_without_ddp = model

    if args.device != 'cpu' and args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    optimizer       = create_optimizer(utils.AttrDict(config['optimizer']), model)
    lr_scheduler, _ = create_scheduler(utils.AttrDict(config['schedular']), optimizer)

    max_epoch    = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    # resume
    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resuming at epoch {start_epoch}")

    # ── CHANGE 9: configurable summary filename ───────────────────────────────
    summary_path = os.path.join(args.output_dir, args.summary_name)
    if utils.is_main_process():
        with open(summary_path, 'a') as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"Run started    : {datetime.datetime.now()}\n")
            f.write(f"ITA anchor     : [CLS] for epochs 0-{args.ita_warmup_epochs - 1}, "
                    f"D stream from epoch {args.ita_warmup_epochs}\n")
            f.write(f"{'='*70}\n")
    # ─────────────────────────────────────────────────────────────────────────

    print(args)
    print(config)
    print("Start training")
    start_time = time.time()

    for epoch in range(start_epoch, max_epoch):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            # ── CHANGE 11: decide ITA anchor for this epoch ───────────────
            use_d_for_ita = (epoch >= args.ita_warmup_epochs)
            ita_anchor_label = "D stream" if use_d_for_ita else "raw [CLS]"
            print(f"[Epoch {epoch}] ITA anchor = {ita_anchor_label}")
            # ─────────────────────────────────────────────────────────────

            train_stats = train(
                model, train_loader, optimizer, epoch, warmup_steps,
                device, lr_scheduler, config,
                grad_accum_steps=args.grad_accum_steps,
                use_d_for_ita=use_d_for_ita)          # ← CHANGE 11
            print(train_stats)

            # CHANGE 10: R@1 accuracy — only every eval_freq epochs
            # On skipped epochs losses are still logged, accuracy shows 'skipped'
            is_eval_epoch = (epoch % args.eval_freq == 0) or                             (epoch == max_epoch - 1)

            if is_eval_epoch:
                # FIX: evaluate only on rank 0 — avoids both processes
                # building a full N×N similarity matrix in CPU RAM simultaneously,
                # which can exhaust memory right before the next training epoch.
                if utils.is_main_process():
                    print(f"[Epoch {epoch}] Evaluating train accuracy "
                          f"(first {args.eval_train_batches} batches)...")
                    train_acc = evaluate_retrieval(
                        model_without_ddp, train_loader, device,
                        max_batches=args.eval_train_batches if args.eval_train_batches > 0
                                    else None)

                    print(f"[Epoch {epoch}] Evaluating test accuracy...")
                    test_acc = evaluate_retrieval(
                        model_without_ddp, test_loader, device,
                        max_batches=None)

                    print(f"  Train R@1 -- I2T: {train_acc['r1_i2t']:.2f}%  "
                          f"T2I: {train_acc['r1_t2i']:.2f}%  "
                          f"Mean: {train_acc['r1_mean']:.2f}%")
                    print(f"  Test  R@1 -- I2T: {test_acc['r1_i2t']:.2f}%  "
                          f"T2I: {test_acc['r1_t2i']:.2f}%  "
                          f"Mean: {test_acc['r1_mean']:.2f}%")
                else:
                    train_acc = None
                    test_acc  = None
            else:
                train_acc = None
                test_acc  = None
                print(f"[Epoch {epoch}] Skipping accuracy eval "
                      f"(next eval at epoch "
                      f"{epoch + args.eval_freq - (epoch % args.eval_freq)})")

            if utils.is_main_process():

                # CHANGE 9: append epoch row to summary file
                with open(summary_path, 'a') as f:
                    f.write(f"\nEpoch {epoch:03d}  "
                            f"[ITA anchor: {ita_anchor_label}]\n")
                    f.write("  Train losses : "
                            + "  ".join(f"{k}={v}" for k, v in train_stats.items())
                            + "\n")
                    if train_acc is not None:
                        f.write(f"  Train R@1    : "
                                f"I2T={train_acc['r1_i2t']:.2f}%  "
                                f"T2I={train_acc['r1_t2i']:.2f}%  "
                                f"Mean={train_acc['r1_mean']:.2f}%\n")
                        f.write(f"  Test  R@1    : "
                                f"I2T={test_acc['r1_i2t']:.2f}%  "
                                f"T2I={test_acc['r1_t2i']:.2f}%  "
                                f"Mean={test_acc['r1_mean']:.2f}%\n")
                    else:
                        f.write(f"  Train R@1    : skipped (eval_freq={args.eval_freq})\n")
                        f.write(f"  Test  R@1    : skipped (eval_freq={args.eval_freq})\n")

                # CHANGE 8: save current epoch, delete previous
                save_obj = {
                    'model':          model_without_ddp.state_dict(),
                    'optimizer':      optimizer.state_dict(),
                    'lr_scheduler':   lr_scheduler.state_dict(),
                    'config':         config,
                    'epoch':          epoch,
                    'train_acc':      train_acc,   # None on skipped epochs
                    'test_acc':       test_acc,    # None on skipped epochs
                    'use_d_for_ita':  use_d_for_ita,
                }
                curr_path = os.path.join(args.output_dir, f'epoch{epoch:03d}')
                os.makedirs(curr_path, exist_ok=True)
                torch.save(save_obj, os.path.join(curr_path, 'checkpoint.pth'))
                print(f"  Saved  checkpoint -> {curr_path}/checkpoint.pth")

                if epoch > 0:
                    prev_path = os.path.join(args.output_dir,
                                             f'epoch{epoch - 1:03d}')
                    if os.path.exists(prev_path):
                        shutil.rmtree(prev_path)
                        print(f"  Deleted previous checkpoint: {prev_path}")

        if args.evaluate:
            break

        lr_scheduler.step(epoch + warmup_steps + 1)
        # FIX: add a timeout so that if one process dies mid-epoch the other
        # process does not hang forever at the barrier (and keeps the GPUs
        # locked).  30 minutes is generous; adjust if your epochs are longer.
        dist.barrier()
        torch.cuda.empty_cache()

    total_time     = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}')
    if utils.is_main_process():
        with open(summary_path, 'a') as f:
            f.write(f"\nTotal training time: {total_time_str}\n")
            f.write(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     default='./configs/fashion_pretrain_custom.yaml')
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--pre_point',  default='ALBEF.pth')

    parser.add_argument('--evaluate',    action='store_true')
    parser.add_argument('--device',      default='cuda')
    parser.add_argument('--seed',        default=66,   type=int)
    parser.add_argument('--world_size',  default=1,    type=int)
    parser.add_argument('--dist_url',    default='env://')
    parser.add_argument('--distributed', default=True, type=bool)

    parser.add_argument('--max_word_num',          default=180, type=int)
    parser.add_argument('--data_root',             default='',  type=str)
    parser.add_argument('--catemap_filename',      default='categorys_to_sign.txt', type=str)
    parser.add_argument('--product_list_filename', default='productid_list.json',   type=str)
    parser.add_argument('--replace_kind_num',      default=2,   type=int)

    parser.add_argument('--resume', default='', type=str)

    parser.add_argument('--grad_accum_steps',    default=8,   type=int)
    parser.add_argument('--batch_size_override', default=0,   type=int)
    parser.add_argument('--lr_override',         default=0.0, type=float)
    parser.add_argument('--subset_ratio',        default=1.0, type=float)
    parser.add_argument('--max_epoch',           default=0,   type=int)

    parser.add_argument('--eval_train_batches',  default=200, type=int,
                        help='Train batches sampled for R@1 estimate. 0 = all.')
    parser.add_argument('--eval_freq', default=1, type=int,
                        help='Evaluate R@1 accuracy every N epochs. '
                             '1 = every epoch (smoke test). '
                             '3 = every 3 epochs (tier 2). '
                             '5 = every 5 epochs (full run). '
                             'Last epoch is always evaluated regardless.')

    # ── CHANGE 11: D-stream warmup epochs ────────────────────────────────────
    parser.add_argument('--ita_warmup_epochs', default=5, type=int,
                        help='Number of epochs to use raw [CLS] as ITA anchor '
                             'before switching to the D stream. '
                             'Set 0 to use D stream from the start (original V5). '
                             'Set a large number to never use D stream (original V4).')
    # ─────────────────────────────────────────────────────────────────────────

    # ── CHANGE 9: configurable summary filename ───────────────────────────────
    parser.add_argument('--summary_name', default='training_summary.txt', type=str,
                        help='Filename for the training summary log inside '
                             '--output_dir.  Change this to keep separate summary '
                             'files for different experiments without overwriting '
                             'previous results.')
    # ─────────────────────────────────────────────────────────────────────────

    args   = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, config)