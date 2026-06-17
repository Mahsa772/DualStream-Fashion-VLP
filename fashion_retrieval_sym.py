"""
fashion_retrieval_sym.py

Changes vs previous version:
  [CHANGE 1] evaluation() now uses SD = (S + D) / 2 for text features,
             consistent with the model forward() change in
             model_fashion_retrieval_sym.py. Train and eval use same features.

  [CHANGE 2] Checkpoint saved based on best TINY Mean R@1
             = (tiny_I2T_R@1 + tiny_T2I_R@1) / 2
             instead of global Mean R@1.
             Reason: Table 2 (tiny) is the main comparison metric.
             The checkpoint saved will be the one that best matches
             what we report in the paper comparison table.

  [CHANGE 3] Summary log now writes ALL metrics every epoch:
             Global: I2T R@1/R@5/R@10, T2I R@1/R@5/R@10, global Mean R@1
             Tiny:   I2T R@1/R@5/R@10, T2I R@1/R@5/R@10, tiny Mean R@1
             So you have everything needed for both Table 2 and Table 3.

  [CHANGE 4] dist.barrier() called without timeout argument
             (fixes TypeError on older PyTorch versions).

  [CHANGE 5] LR printed with 6 decimal places in summary
             so small values like 0.000010 show correctly
             instead of rounding to 0.0000.

  [CHANGE 6] Tiny (Table 2) metric is now averaged over --tiny_eval_sets
             (default 5) randomly sampled 1k-query / 101-candidate test sets,
             matching the paper ("average of 5 randomly chosen retrieval test
             sets"). Reports mean +/- std per metric and uses the averaged
             tiny Mean R@1 as the best-tiny checkpoint criterion. The original
             code (and the official repo) evaluated only ONE random set per
             epoch; the 5-set averaging was never in the released code.
"""

import argparse
import os
import re
import datetime
import time
import random
from pathlib import Path

import ruamel.yaml as yaml
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_fashion_retrieval_sym import FashionSAP
from models.vit import interpolate_pos_embed
from transformers.models.bert.tokenization_bert import BertTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer


# ── strip [xxx_sign] prefix ───────────────────────────────────────────────────
_SIGN_RE = re.compile(r'^\s*\[[a-zA-Z]+_sign\]\s*', re.IGNORECASE)

def strip_sign_prefix(text: str) -> str:
    return _SIGN_RE.sub('', text).strip()

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


# ── training loop ─────────────────────────────────────────────────────────────
def train(model, data_loader, optimizer, tokenizer, epoch,
          warmup_steps, device, scheduler, config,
          grad_accum_steps: int = 4):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr',  utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('ita', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header            = f'Train Epoch: [{epoch}]'
    print_freq        = 50
    step_size         = 100
    warmup_iterations = warmup_steps * step_size

    optimizer.zero_grad()
    accum_step = 0

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        batch = [t.to(device, non_blocking=True) for t in batch]
        image, text_input_ids, text_attention_mask, idx = batch

        alpha = (config['alpha']
                 if epoch > 0 or not config['warm_up']
                 else config['alpha'] * min(1, i / len(data_loader)))

        loss_ita, loss_itm = model(
            image, text_input_ids, text_attention_mask,
            alpha=alpha, idx=idx)

        loss        = loss_ita + loss_itm
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

        metric_logger.update(ita=loss_ita.item())
        metric_logger.update(itm=loss_itm.item())
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    # [CHANGE 5] use 6 decimal places so small LR values show correctly
    return {k: f"{meter.global_avg:.6f}"
            for k, meter in metric_logger.meters.items()}


# ── evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config, args):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header        = 'Evaluation:'
    start_time    = time.time()

    # encode all texts
    text_feats = []
    texts      = data_loader.dataset.texts
    text_num   = len(texts)
    tbs        = 256
    print(f'[eval] encoding {text_num} texts ...')
    for i in range(0, text_num, tbs):
        text = texts[i: min(text_num, i + tbs)]
        text_input = tokenizer(
            text, padding='max_length', truncation=True,
            max_length=args.max_word_num, return_tensors='pt').to(device)
        text_out = model.text_encoder(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            return_dict=True, mode='text')
        # [CHANGE 1] SD = (S + D) / 2 — same as forward() in model
        S, D = model.decoupled_text_attn(text_out.last_hidden_state)
        SD   = (S + D) / 2
        feat = model.combine_text_proj(model.text_proj(SD))
        text_feats.append(feat)

    # encode all images
    img_feats = []
    print('[eval] encoding images ...')
    for i, batch in enumerate(metric_logger.log_every(data_loader, 50, header)):
        img = batch[0].to(device)
        img_emb  = model.visual_encoder(img)
        img_feat = model.combine_vision_proj(
            model.vision_proj(img_emb[:, 0, :]))
        img_feats.append(img_feat)

    text_feats   = torch.cat(text_feats, dim=0)
    img_feats    = torch.cat(img_feats,  dim=0)
    text_feats_n = F.normalize(text_feats, dim=-1)
    img_feats_n  = F.normalize(img_feats,  dim=-1)
    sim_t2i      = text_feats_n @ img_feats_n.t()

    elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f'Evaluation time {elapsed}')
    return sim_t2i.T.cpu().numpy(), sim_t2i.cpu().numpy()


# ── metric computation ────────────────────────────────────────────────────────
@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i,
             img2txt=None, txt2img=None,
             tiny_i2t=None, tiny_t2i=None):

    if img2txt is None:
        img2txt = {i: i for i in range(scores_i2t.shape[0])}
        txt2img = {k: [v] for k, v in img2txt.items()}

    # Global I2T
    print('calculating i2t ranks')
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == img2txt[index])[0][0]
    tr1  = 100.0 * len(np.where(ranks < 1)[0])  / len(ranks)
    tr5  = 100.0 * len(np.where(ranks < 5)[0])  / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Tiny I2T
    tr1_tiny = tr5_tiny = tr10_tiny = 0
    if tiny_i2t is not None:
        img_indexes, txt_indexes = tiny_i2t
        tiny_score_i2t = np.zeros(txt_indexes.shape)
        tiny_score     = scores_i2t[img_indexes]
        for i, t_score in enumerate(tiny_score):
            tiny_score_i2t[i] = t_score[txt_indexes[i]]
        ranks_t = np.zeros(tiny_score_i2t.shape[0])
        for index, score in enumerate(tiny_score_i2t):
            inds = np.argsort(score)[::-1]
            ranks_t[index] = np.where(inds == 100)[0][0]
        tr1_tiny  = 100.0 * len(np.where(ranks_t < 1)[0])  / len(ranks_t)
        tr5_tiny  = 100.0 * len(np.where(ranks_t < 5)[0])  / len(ranks_t)
        tr10_tiny = 100.0 * len(np.where(ranks_t < 10)[0]) / len(ranks_t)

    # Global T2I
    print('calculating t2i ranks')
    ranks = np.zeros(scores_t2i.shape[0])
    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        rank = 100000
        for i in txt2img[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
    ir1  = 100.0 * len(np.where(ranks < 1)[0])  / len(ranks)
    ir5  = 100.0 * len(np.where(ranks < 5)[0])  / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Tiny T2I
    ir1_tiny = ir5_tiny = ir10_tiny = 0
    if tiny_t2i is not None:
        txt_indexes, img_indexes = tiny_t2i
        tiny_score_t2i = np.zeros(img_indexes.shape)
        tiny_score     = scores_t2i[txt_indexes]
        for i, t_score in enumerate(tiny_score):
            tiny_score_t2i[i] = t_score[img_indexes[i]]
        ranks_t = np.zeros(tiny_score_t2i.shape[0])
        for index, score in enumerate(tiny_score_t2i):
            inds = np.argsort(score)[::-1]
            ranks_t[index] = np.where(inds == 100)[0][0]
        ir1_tiny  = 100.0 * len(np.where(ranks_t < 1)[0])  / len(ranks_t)
        ir5_tiny  = 100.0 * len(np.where(ranks_t < 5)[0])  / len(ranks_t)
        ir10_tiny = 100.0 * len(np.where(ranks_t < 10)[0]) / len(ranks_t)

    return {
        # Global (Table 3)
        'txt_r1':       tr1,   'txt_r5':       tr5,   'txt_r10':       tr10,
        'img_r1':       ir1,   'img_r5':       ir5,   'img_r10':       ir10,
        'r_mean':       (tr1 + ir1) / 2,
        # Tiny (Table 2)
        'tiny_txt_r1':  tr1_tiny,  'tiny_txt_r5':  tr5_tiny,  'tiny_txt_r10':  tr10_tiny,
        'tiny_img_r1':  ir1_tiny,  'tiny_img_r5':  ir5_tiny,  'tiny_img_r10':  ir10_tiny,
        'tiny_r_mean':  (tr1_tiny + ir1_tiny) / 2,
    }


# ── [CHANGE 6] 5-fold tiny evaluation (paper protocol) ────────────────────────
# The FashionSAP paper reports "the average result of 5 randomly chosen
# retrieval test sets ... each contains 1k queries ... 100 candidates from the
# same subcategory".  The dataset already re-samples a fresh 1k-query /
# 101-candidate set on every get_i2t_test()/get_t2i_test() call, so we simply
# call itm_eval n_sets times and report mean +/- std on the tiny metrics.
#
# The expensive feature encoding in evaluation() runs ONCE; only the cheap
# candidate-scoring inside itm_eval re-runs per set.  The global (Table 3)
# metrics are deterministic, so we keep the first set's global result as-is.
def tiny_eval_5fold(scores_i2t, scores_t2i, test_dataset,
                    img2text, text2img, n_sets=5, base_seed=None):
    """
    Returns:
        val_result : full itm_eval dict from the first set (use for GLOBAL
                     metrics + global checkpoint criterion; global is the
                     same across sets).
        tiny_stats : {<metric>_mean, <metric>_std} for the 7 tiny metrics.
        tiny_runs  : {<metric>: [per-set values]} for logging the raw draws.
    """
    import random as _random
    keys = ['tiny_txt_r1', 'tiny_txt_r5', 'tiny_txt_r10',
            'tiny_img_r1', 'tiny_img_r5', 'tiny_img_r10', 'tiny_r_mean']
    runs = {k: [] for k in keys}
    val_result = None

    # save/restore the global RNG so seeding the tiny draws does NOT perturb
    # the training data stream in later epochs (the dataset uses `random`).
    saved_state = _random.getstate() if base_seed is not None else None

    for s in range(n_sets):
        if base_seed is not None:
            _random.seed(base_seed + s)       # reproducible per-set sampling
        ti_img, ti_txt = test_dataset.get_i2t_test()
        tt_txt, tt_img = test_dataset.get_t2i_test()
        r = itm_eval(scores_i2t, scores_t2i, img2text, text2img,
                     tiny_i2t=(ti_img, ti_txt), tiny_t2i=(tt_txt, tt_img))
        if val_result is None:
            val_result = r                    # global metrics (deterministic)
        for k in keys:
            runs[k].append(r[k])
        print(f"  [tiny set {s}] "
              f"I2T R@1={r['tiny_txt_r1']:.2f}  "
              f"T2I R@1={r['tiny_img_r1']:.2f}  "
              f"Mean R@1={r['tiny_r_mean']:.2f}")

    if saved_state is not None:
        _random.setstate(saved_state)

    tiny_stats = {}
    for k in keys:
        v = np.asarray(runs[k], dtype=np.float64)
        tiny_stats[k + '_mean'] = float(v.mean())
        # ddof=1 -> unbiased sample std, the right estimator for n_sets=5
        tiny_stats[k + '_std']  = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    return val_result, tiny_stats, runs
# ──────────────────────────────────────────────────────────────────────────────


# ── main ──────────────────────────────────────────────────────────────────────
def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    n_gpu     = utils.get_world_size()
    eff_batch = config['batch_size_train'] * args.grad_accum_steps * n_gpu
    print(f'[batch] physical={config["batch_size_train"]}  '
          f'accum={args.grad_accum_steps}  GPUs={n_gpu}  effective={eff_batch}')

    # tokenizer — NO [xxx_sign] tokens
    tokenizer = BertTokenizer.from_pretrained(config['tokenizer_config'])
    print('[tokenizer] Fashion-symbol tokens NOT added (sym-pretrain mode).')

    # datasets
    print('Creating dataset')
    train_dataset, test_dataset = create_dataset('retrieval', config, args, tokenizer)
    _patch_dataset_strip(train_dataset)
    if test_dataset is not None:
        _patch_dataset_strip(test_dataset)
    print('[dataset] strip_sign_prefix patch applied.')

    # subset the train dataset for smoke testing
    # test eval always uses the FULL test set regardless of subset_ratio
    if args.subset_ratio < 1.0:
        import math
        n_full   = len(train_dataset)
        n_subset = max(1, math.floor(n_full * args.subset_ratio))
        indices  = list(range(n_subset))
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        print(f'[subset] using {n_subset}/{n_full} train samples ({args.subset_ratio*100:.0f}%)')

    if args.distributed:
        num_tasks   = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = (create_sampler([train_dataset], [True], num_tasks, global_rank)
                    + [None])
    else:
        samplers = [None, None]

    train_loader, val_loader = create_loader(
        [train_dataset, test_dataset], samplers,
        batch_size=[config['batch_size_train'], config['batch_size_test']],
        num_workers=[4, 4],
        is_trains=[True, False],
        collate_fns=[None, None])

    # model
    print('Creating model')
    model = FashionSAP(config=config, args=args)

    # load pretrain checkpoint
    if args.pre_point:
        print(f'Loading pretrain checkpoint from {args.pre_point}')
        checkpoint = torch.load(args.pre_point, map_location='cpu')
        state_dict = checkpoint['model']

        state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(
            state_dict['visual_encoder.pos_embed'], model.visual_encoder)
        state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(
            state_dict['visual_encoder_m.pos_embed'], model.visual_encoder_m)

        model_sd = model.state_dict()
        for k in list(state_dict.keys()):
            if k in model_sd and state_dict[k].shape != model_sd[k].shape:
                print(f'  [shape mismatch] dropping "{k}"')
                del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
        with torch.no_grad():
            model.queue_ptr.zero_()
        print('[queue] queue_ptr reset to 0.')

    elif args.checkpoint:
        print(f'Resuming from fine-tune checkpoint {args.checkpoint}')
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        print(msg)

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
    best               = 0.0   # tracks best tiny Mean R@1
    best_epoch         = 0
    best_global        = 0.0   # tracks best global Mean R@1
    best_global_epoch  = 0

    summary_path = os.path.join(args.output_dir, 'retrieval_summary.txt')
    if utils.is_main_process():
        with open(summary_path, 'a') as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"Run started      : {datetime.datetime.now()}\n")
            f.write(f"pre_point        : {args.pre_point}\n")
            f.write(f"checkpoint       : {args.checkpoint}\n")
            f.write(f"effective batch  : {eff_batch} "
                    f"(physical={config['batch_size_train']} "
                    f"x accum={args.grad_accum_steps} "
                    f"x GPUs={n_gpu})\n")
            f.write(f"text feature     : SD = (S + D) / 2\n")
            f.write(f"save criterion   : best tiny Mean R@1\n")
            f.write(f"{'='*70}\n")

    print('Start training')
    print(args)
    print(config)
    start_time = time.time()

    for epoch in range(0, max_epoch):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            train_stats = train(
                model, train_loader, optimizer, tokenizer,
                epoch, warmup_steps, device, lr_scheduler, config,
                grad_accum_steps=args.grad_accum_steps)
            print(train_stats)

        # evaluation
        score_i2t, score_t2i = evaluation(
            model_without_ddp, val_loader, tokenizer, device, config, args)

        if utils.is_main_process():
            img2text, text2img = test_dataset.get_test_labels()

            # [CHANGE 6] average the tiny (Table 2) metric over
            # args.tiny_eval_sets random 1k-query / 101-candidate sets,
            # exactly as the paper describes. Global metrics come from the
            # first set (they are deterministic).
            val_result, tiny_stats, tiny_runs = tiny_eval_5fold(
                score_i2t, score_t2i, test_dataset,
                img2text, text2img,
                n_sets=args.tiny_eval_sets, base_seed=args.seed)
            print(val_result)

            # [CHANGE 3] write ALL metrics — global and tiny — every epoch
            with open(summary_path, 'a') as f:
                f.write(f"\nEpoch {epoch:03d}\n")
                f.write(f"  Train         : "
                        + "  ".join(f"{k}={v}" for k, v in train_stats.items())
                        + "\n")
                # Global (Table 3)
                f.write(f"  [Global] I2T  R@1={val_result['txt_r1']:.2f}  "
                        f"R@5={val_result['txt_r5']:.2f}  "
                        f"R@10={val_result['txt_r10']:.2f}\n")
                f.write(f"  [Global] T2I  R@1={val_result['img_r1']:.2f}  "
                        f"R@5={val_result['img_r5']:.2f}  "
                        f"R@10={val_result['img_r10']:.2f}\n")
                f.write(f"  [Global] Mean R@1 = {val_result['r_mean']:.2f}\n")
                # Tiny (Table 2) — averaged over args.tiny_eval_sets sets
                f.write(f"  [Tiny x{args.tiny_eval_sets}] I2T  "
                        f"R@1={tiny_stats['tiny_txt_r1_mean']:.2f}+/-{tiny_stats['tiny_txt_r1_std']:.2f}  "
                        f"R@5={tiny_stats['tiny_txt_r5_mean']:.2f}+/-{tiny_stats['tiny_txt_r5_std']:.2f}  "
                        f"R@10={tiny_stats['tiny_txt_r10_mean']:.2f}+/-{tiny_stats['tiny_txt_r10_std']:.2f}\n")
                f.write(f"  [Tiny x{args.tiny_eval_sets}] T2I  "
                        f"R@1={tiny_stats['tiny_img_r1_mean']:.2f}+/-{tiny_stats['tiny_img_r1_std']:.2f}  "
                        f"R@5={tiny_stats['tiny_img_r5_mean']:.2f}+/-{tiny_stats['tiny_img_r5_std']:.2f}  "
                        f"R@10={tiny_stats['tiny_img_r10_mean']:.2f}+/-{tiny_stats['tiny_img_r10_std']:.2f}\n")
                f.write(f"  [Tiny x{args.tiny_eval_sets}] Mean R@1 = "
                        f"{tiny_stats['tiny_r_mean_mean']:.2f}+/-{tiny_stats['tiny_r_mean_std']:.2f}\n")
                f.write(f"  [Tiny] per-set Mean R@1: "
                        + ", ".join(f"{x:.2f}" for x in tiny_runs['tiny_r_mean'])
                        + "\n")

            # build checkpoint object once — reused for both saves below
            save_obj = {
                'model':        model_without_ddp.state_dict(),
                'optimizer':    optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config':       config,
                'epoch':        epoch,
                'val_result':   val_result,
                'tiny_stats':   tiny_stats,   # [CHANGE 6] mean/std over sets
            }

            # save best TINY checkpoint (Table 2 criterion)
            # [CHANGE 6] criterion is now the MEAN tiny Mean R@1 across the
            # args.tiny_eval_sets random sets — more stable than one draw.
            tiny_mean = tiny_stats['tiny_r_mean_mean']
            if tiny_mean > best:
                best       = tiny_mean
                best_epoch = epoch
                torch.save(save_obj,
                           os.path.join(args.output_dir, 'checkpoint_best_tiny.pth'))
                print(f'  [best-tiny]   epoch={epoch}  '
                      f'tiny Mean R@1={best:.2f}'
                      f'+/-{tiny_stats["tiny_r_mean_std"]:.2f}  '
                      f'tiny I2T={tiny_stats["tiny_txt_r1_mean"]:.2f}  '
                      f'tiny T2I={tiny_stats["tiny_img_r1_mean"]:.2f}  '
                      f'-> checkpoint_best_tiny.pth')

            # save best GLOBAL checkpoint (Table 3 criterion)
            # best_global = highest (I2T_R@1 + T2I_R@1) / 2
            global_mean = val_result['r_mean']
            if global_mean > best_global:
                best_global       = global_mean
                best_global_epoch = epoch
                torch.save(save_obj,
                           os.path.join(args.output_dir, 'checkpoint_best_global.pth'))
                print(f'  [best-global] epoch={epoch}  '
                      f'global Mean R@1={best_global:.2f}  '
                      f'I2T={val_result["txt_r1"]:.2f}  '
                      f'T2I={val_result["img_r1"]:.2f}  '
                      f'-> checkpoint_best_global.pth')

        if args.evaluate:
            break

        lr_scheduler.step(epoch + warmup_steps + 1)
        dist.barrier()   # [CHANGE 4] no timeout argument
        torch.cuda.empty_cache()

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f'Training time {total_time}')
    if utils.is_main_process():
        with open(summary_path, 'a') as f:
            f.write(f"\nBest tiny   epoch={best_epoch}  tiny Mean R@1={best:.2f}\n")
            f.write(f"Best global epoch={best_global_epoch}  global Mean R@1={best_global:.2f}\n")
            f.write(f"Total time : {total_time}\n")
            f.write(f"{'='*70}\n")
        with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
            f.write(f"best epoch: {best_epoch}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',      default='./configs/fashion_retrieval.yaml')
    parser.add_argument('--output_dir',  default='./output/finetune_retrieval5set_sym')
    parser.add_argument('--pre_point',   default='',
                        help='Path to sym-pretrain checkpoint')
    parser.add_argument('--checkpoint',  default='',
                        help='Path to a fine-tune checkpoint to resume from')
    parser.add_argument('--evaluate',    action='store_true')
    parser.add_argument('--device',      default='cuda')
    parser.add_argument('--seed',        default=66,  type=int)
    parser.add_argument('--world_size',  default=1,   type=int)
    parser.add_argument('--dist_url',    default='env://')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--sub_dataset', default=False, type=bool)

    parser.add_argument('--max_word_num',          default=180, type=int)
    parser.add_argument('--data_root',             default='',  type=str)
    parser.add_argument('--catemap_filename',      default='categorys_to_sign.txt', type=str)
    parser.add_argument('--product_list_filename', default='productid_list.json',   type=str)
    parser.add_argument('--replace_kind_num',      default=2,   type=int)

    parser.add_argument('--grad_accum_steps', default=4, type=int,
                        help='2 GPUs x batch 16 x accum 4 = effective 128.')

    parser.add_argument('--subset_ratio', default=1.0, type=float,
                        help='Fraction of train data to use. '
                             '0.05 = 5%% smoke test. 1.0 = full training.')

    # [CHANGE 6] number of random 1k-query/101-candidate sets to average for
    # the tiny (Table 2) metric, matching the FashionSAP paper protocol.
    parser.add_argument('--tiny_eval_sets', default=5, type=int,
                        help='Random tiny test sets to average (paper = 5). '
                             'Each is a fresh 1k-query / 101-candidate draw '
                             'with 100 same-subcategory negatives. '
                             'Set 1 to reproduce the old single-set behaviour.')

    args   = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, config)