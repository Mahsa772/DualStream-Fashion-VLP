"""
fashion_catereg_sym.py

Fine-tuning script for Category Recognition (CR) and
Subcategory Recognition (SCR) using a FashionSAP-Sym pre-trained checkpoint.

Changes vs original fashion_catereg.py:
  1. Imports FashionSAP from model_fashion_catereg_sym (Option-C model).
  2. No special symbol tokens added to the tokenizer.
  3. strip_sign_prefix() patches the dataset to strip any [xxx_sign] token.
  4. Model forward() returns (cr_logit, scr_logit).
     --task {cr | scr | both} controls which loss(es) are back-propagated.
  5. evaluate() returns per-task metrics: acc, macro_f, micro_f.
  6. Best checkpoint saved separately for CR (by acc) and SCR (by macro_F).
  7. Summary .txt log written after every epoch.
  8. --subset_ratio allows smoke / tier-2 / full runs.
  9. Gradient accumulation via --grad_accum_steps.
 10. [NEW] Early stopping on SCR macro_F with configurable patience.
     Training stops when macro_F does not improve for --scr_patience epochs.
     The saved checkpoint (best_scr_macroF/) always holds the best macro_F
     seen so far — it is overwritten only when a strictly better value is found.
"""

import argparse
import os
import re
import json
import datetime
import time
import random
from pathlib import Path

import numpy as np
import ruamel.yaml as yaml
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from sklearn.metrics import f1_score

from models.model_fashion_catereg_sym import FashionSAP
from models.vit import interpolate_pos_embed
from transformers.models.bert.tokenization_bert import BertTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer


# ── strip residual [xxx_sign] prefixes ───────────────────────────────────────
_SIGN_RE = re.compile(r'^\s*\[[a-zA-Z]+_sign\]\s*', re.IGNORECASE)

def strip_sign_prefix(text: str) -> str:
    return _SIGN_RE.sub('', text).strip()

def _patch_dataset_strip(dataset):
    import types
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
    dataset.__getitem__ = types.MethodType(patched_getitem, dataset)


# ── training ──────────────────────────────────────────────────────────────────
def train(model, data_loader, optimizer, epoch,
          warmup_steps, device, scheduler, config, args):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if args.task in ('cr', 'both'):
        metric_logger.add_meter(
            'cr_loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if args.task in ('scr', 'both'):
        metric_logger.add_meter(
            'scr_loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header            = f'Train Epoch: [{epoch}]'
    print_freq        = 50
    step_size         = 100
    warmup_iterations = warmup_steps * step_size
    loss_fn           = nn.CrossEntropyLoss()

    optimizer.zero_grad()

    for i, batch in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        batch     = [t.to(device, non_blocking=True) for t in batch]
        image, text_ids, text_mask, cr_label, scr_label = batch
        cr_label  = cr_label.squeeze()
        scr_label = scr_label.squeeze()

        cr_logit, scr_logit = model(image, text_ids, text_mask)

        loss = torch.tensor(0.0, device=device)
        if args.task in ('cr', 'both'):
            cr_loss = loss_fn(cr_logit, cr_label)
            loss    = loss + cr_loss
            metric_logger.update(cr_loss=cr_loss.item())
        if args.task in ('scr', 'both'):
            scr_loss = loss_fn(scr_logit, scr_label)
            loss     = loss + scr_loss
            metric_logger.update(scr_loss=scr_loss.item())

        (loss / args.grad_accum_steps).backward()

        if (i + 1) % args.grad_accum_steps == 0 or (i + 1) == len(data_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        metric_logger.update(lr=optimizer.param_groups[0]['lr'])

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: '{:.4f}'.format(meter.global_avg)
            for k, meter in metric_logger.meters.items()}


# ── evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, data_loader, device, args):
    model.eval()

    all_cr_logits,  all_cr_labels  = [], []
    all_scr_logits, all_scr_labels = [], []
    metric_logger = utils.MetricLogger(delimiter="  ")

    for batch in metric_logger.log_every(data_loader, 50, 'Evaluation:'):
        batch = [t.to(device) for t in batch]
        image, text_ids, text_mask, cr_label, scr_label = batch
        cr_logit, scr_logit = model(image, text_ids, text_mask)
        all_cr_logits.append(cr_logit.cpu())
        all_cr_labels.append(cr_label.squeeze().cpu())
        all_scr_logits.append(scr_logit.cpu())
        all_scr_labels.append(scr_label.squeeze().cpu())

    cr_logits  = torch.cat(all_cr_logits,  dim=0).numpy()
    cr_labels  = torch.cat(all_cr_labels,  dim=0).numpy()
    scr_logits = torch.cat(all_scr_logits, dim=0).numpy()
    scr_labels = torch.cat(all_scr_labels, dim=0).numpy()

    results = {'cr': {}, 'scr': {}}
    if args.task in ('cr', 'both'):
        results['cr']  = _calc_metrics(cr_logits,  cr_labels)
    if args.task in ('scr', 'both'):
        results['scr'] = _calc_metrics(scr_logits, scr_labels)
    return results


def _calc_metrics(logits, labels):
    pred    = np.argmax(logits, axis=-1)
    acc     = np.equal(pred, labels).astype(np.float64).mean()
    macro_f = f1_score(y_true=labels, y_pred=pred, average='macro', zero_division=0)
    micro_f = f1_score(y_true=labels, y_pred=pred, average='micro', zero_division=0)
    return {'acc':     round(float(acc),     4),
            'macro_f': round(float(macro_f), 4),
            'micro_f': round(float(micro_f), 4)}


# ── checkpoint loading ────────────────────────────────────────────────────────
def load_pretrain_checkpoint(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint)

    if 'visual_encoder.pos_embed' in state_dict:
        state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(
            state_dict['visual_encoder.pos_embed'], model.visual_encoder)

    keys_to_drop = [k for k in state_dict
                    if any(tag in k for tag in (
                        '_m.', 'image_queue', 'text_queue', 'sym_image_queue',
                        'queue_ptr', 'sym_queue_ptr', 'idx_queue',
                        'symbol_proj', 'combine_symbol_proj',
                        'decoder_layer', 'replace_predict_layer', 'itm_head',
                    ))]
    for k in keys_to_drop:
        state_dict.pop(k, None)

    msg = model.load_state_dict(state_dict, strict=False)
    print(f'[checkpoint] Loaded from {ckpt_path}')
    print(f'[checkpoint] missing_keys   : {msg.missing_keys}')
    print(f'[checkpoint] unexpected_keys: {msg.unexpected_keys}')

    unexpected_missing = [k for k in msg.missing_keys
                          if not k.startswith(('cr_head.', 'scr_head.'))]
    if unexpected_missing:
        print(f'[WARNING] Unexpected missing keys: {unexpected_missing}')
    return msg


# ── main ──────────────────────────────────────────────────────────────────────
def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    args.cr_class_num  = 48
    args.scr_class_num = 121

    # tokenizer — plain BERT, no special symbol tokens
    tokenizer = BertTokenizer.from_pretrained(config['tokenizer_config'])
    print('[tokenizer] Plain BERT vocabulary — no [xxx_sign] tokens added.')

    # datasets
    print('Creating dataset')
    train_dataset, test_dataset = create_dataset('catereg', config, args, tokenizer)
    _patch_dataset_strip(train_dataset)
    if test_dataset is not None:
        _patch_dataset_strip(test_dataset)
    print('[dataset] strip_sign_prefix patch applied.')

    if args.distributed:
        num_tasks   = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers    = (create_sampler([train_dataset], [True],
                                      num_tasks, global_rank) + [None])
    else:
        samplers = [None, None]

    train_loader, val_loader = create_loader(
        [train_dataset, test_dataset], samplers,
        batch_size  = [config['batch_size_train'], config['batch_size_test']],
        num_workers = [4, 4],
        is_trains   = [True, False],
        collate_fns = [None, None],
    )

    # model
    print('Creating model')
    model = FashionSAP(config=config, args=args)

    if args.pre_point:
        load_pretrain_checkpoint(model, args.pre_point)
    elif args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        msg  = model.load_state_dict(ckpt['model'], strict=True)
        print(f'[checkpoint] Resumed from {args.checkpoint}')
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

    # summary log
    summary_path = os.path.join(args.output_dir, args.summary_name)
    if utils.is_main_process():
        with open(summary_path, 'a') as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"Run started  : {datetime.datetime.now()}\n")
            f.write(f"Task         : {args.task}\n")
            f.write(f"CR classes   : {args.cr_class_num}\n")
            f.write(f"SCR classes  : {args.scr_class_num}\n")
            f.write(f"Pre-trained  : {args.pre_point}\n")
            f.write(f"SCR patience : {args.scr_patience} epochs\n")
            f.write(f"{'='*70}\n")

    print(args)
    print(config)
    print('Start training')
    start_time = time.time()

    # ── tracking variables ────────────────────────────────────────────────
    best_cr_acc       = 0.0
    best_cr_epoch     = 0
    best_scr_acc      = 0.0
    best_scr_epoch    = 0
    # NEW: early stopping tracked on macro_F (more honest than accuracy)
    best_scr_macro_f  = 0.0   # best macro_F seen so far
    scr_patience_count = 0    # epochs since last macro_F improvement
    stop_training      = False

    for epoch in range(0, max_epoch):

        # ── train ─────────────────────────────────────────────────────────
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, epoch,
                                warmup_steps, device, lr_scheduler, config, args)
            print(train_stats)

        # ── evaluate ──────────────────────────────────────────────────────
        eval_results = evaluate(model_without_ddp, val_loader, device, args)

        if utils.is_main_process():
            cr_res  = eval_results.get('cr',  {})
            scr_res = eval_results.get('scr', {})

            # console print
            if cr_res:
                print(f'[Epoch {epoch}] CR  — '
                      f'acc={cr_res["acc"]:.4f}  '
                      f'macro_F={cr_res["macro_f"]:.4f}  '
                      f'micro_F={cr_res["micro_f"]:.4f}')
            if scr_res:
                print(f'[Epoch {epoch}] SCR — '
                      f'acc={scr_res["acc"]:.4f}  '
                      f'macro_F={scr_res["macro_f"]:.4f}  '
                      f'micro_F={scr_res["micro_f"]:.4f}')

            # summary log
            with open(summary_path, 'a') as f:
                f.write(f'\nEpoch {epoch:03d}\n')
                if not args.evaluate:
                    f.write('  Train : '
                            + '  '.join(f'{k}={v}' for k, v in train_stats.items())
                            + '\n')
                if cr_res:
                    f.write(f'  CR    : acc={cr_res["acc"]:.4f}  '
                            f'macro_F={cr_res["macro_f"]:.4f}  '
                            f'micro_F={cr_res["micro_f"]:.4f}\n')
                if scr_res:
                    f.write(f'  SCR   : acc={scr_res["acc"]:.4f}  '
                            f'macro_F={scr_res["macro_f"]:.4f}  '
                            f'micro_F={scr_res["micro_f"]:.4f}\n')

            # json log
            log_entry = {'epoch': epoch}
            if not args.evaluate:
                log_entry.update({f'train_{k}': v for k, v in train_stats.items()})
            if cr_res:
                log_entry.update({f'cr_{k}': v for k, v in cr_res.items()})
            if scr_res:
                log_entry.update({f'scr_{k}': v for k, v in scr_res.items()})
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

            # ── checkpoint saving ─────────────────────────────────────────
            if not args.evaluate:
                def _save_ckpt(tag):
                    save_obj = {
                        'model':        model_without_ddp.state_dict(),
                        'optimizer':    optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'config':       config,
                        'epoch':        epoch,
                        'cr_result':    cr_res,
                        'scr_result':   scr_res,
                    }
                    out_dir = os.path.join(args.output_dir, tag)
                    os.makedirs(out_dir, exist_ok=True)
                    path = os.path.join(out_dir, 'checkpoint_best.pth')
                    torch.save(save_obj, path)
                    print(f'  Saved best {tag} checkpoint → {path}')

                # save best CR by accuracy
                if cr_res and cr_res['acc'] > best_cr_acc:
                    best_cr_acc   = cr_res['acc']
                    best_cr_epoch = epoch
                    if args.task in ('cr', 'both'):
                        _save_ckpt('best_cr')

                # save best SCR by accuracy (keeps existing behaviour)
                if scr_res and scr_res['acc'] > best_scr_acc:
                    best_scr_acc   = scr_res['acc']
                    best_scr_epoch = epoch
                    if args.task in ('scr', 'both'):
                        _save_ckpt('best_scr')

                # ── NEW: save best SCR by macro_F + early stopping ────────
                # macro_F is the honest metric — it cannot be inflated by
                # ignoring rare subcategories the way accuracy can.
                # The checkpoint in best_scr_macroF/ is overwritten ONLY
                # when a strictly better macro_F is found, so when training
                # ends (normally or via early stopping) that file always
                # holds the single best checkpoint seen during this run.
                if scr_res:
                    current_macro_f = scr_res['macro_f']
                    if current_macro_f > best_scr_macro_f:
                        # improvement found — save and reset patience
                        best_scr_macro_f   = current_macro_f
                        scr_patience_count = 0
                        if args.task in ('scr', 'both'):
                            _save_ckpt('best_scr_macroF')
                        print(f'  [early-stop] New best SCR macro_F = '
                              f'{best_scr_macro_f:.4f} @ epoch {epoch}')
                        with open(summary_path, 'a') as f:
                            f.write(f'  [best macro_F so far: {best_scr_macro_f:.4f}]\n')
                    else:
                        # no improvement — increment patience counter
                        scr_patience_count += 1
                        print(f'  [early-stop] No improvement '
                              f'({scr_patience_count}/{args.scr_patience})  '
                              f'best so far = {best_scr_macro_f:.4f}')
                        if scr_patience_count >= args.scr_patience:
                            print(f'  [early-stop] Patience exhausted. '
                                  f'Stopping at epoch {epoch}. '
                                  f'Best SCR macro_F = {best_scr_macro_f:.4f}')
                            with open(summary_path, 'a') as f:
                                f.write(f'  [early-stop] Stopped at epoch {epoch}. '
                                        f'Best macro_F = {best_scr_macro_f:.4f}\n')
                            stop_training = True

        if args.evaluate:
            break

        # ── broadcast early-stop signal to all ranks ──────────────────────
        # stop_training is only set on rank-0 (inside is_main_process).
        # We must broadcast it so every rank breaks together; otherwise
        # rank-1 will hang at the next dist.barrier() waiting for rank-0.
        if args.distributed:
            stop_tensor = torch.tensor(
                int(stop_training), dtype=torch.int32, device=device)
            dist.broadcast(stop_tensor, src=0)
            stop_training = bool(stop_tensor.item())

        if stop_training:
            break

        lr_scheduler.step(epoch + warmup_steps + 1)
        if args.distributed:
            dist.barrier()
        torch.cuda.empty_cache()

    # ── final summary ─────────────────────────────────────────────────────
    total_time     = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}')

    if utils.is_main_process():
        with open(summary_path, 'a') as f:
            f.write(f'\nTotal training time : {total_time_str}\n')
            if args.task in ('cr', 'both'):
                f.write(f'Best CR  acc     = {best_cr_acc:.4f}  @ epoch {best_cr_epoch}\n')
            if args.task in ('scr', 'both'):
                f.write(f'Best SCR acc     = {best_scr_acc:.4f}  @ epoch {best_scr_epoch}\n')
                f.write(f'Best SCR macro_F = {best_scr_macro_f:.4f}\n')
            f.write(f"{'='*70}\n")

        with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
            f.write(json.dumps({
                'best_cr_acc':      best_cr_acc,
                'best_cr_epoch':    best_cr_epoch,
                'best_scr_acc':     best_scr_acc,
                'best_scr_epoch':   best_scr_epoch,
                'best_scr_macro_f': best_scr_macro_f,
            }) + '\n')


# ── argument parser ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # paths
    parser.add_argument('--config',     default='./configs/fashion_catereg.yaml')
    parser.add_argument('--output_dir', default='output/catereg')
    parser.add_argument('--pre_point',  default='')
    parser.add_argument('--checkpoint', default='')

    # task
    parser.add_argument('--task', default='both',
                        choices=['cr', 'scr', 'both'])

    # distributed / device
    parser.add_argument('--evaluate',    action='store_true')
    parser.add_argument('--device',      default='cuda')
    parser.add_argument('--seed',        default=66,   type=int)
    parser.add_argument('--world_size',  default=1,    type=int)
    parser.add_argument('--dist_url',    default='env://')
    parser.add_argument('--distributed', default=True, type=bool)

    # data
    parser.add_argument('--max_word_num',     default=75,   type=int)
    parser.add_argument('--data_root',        default='',   type=str)
    parser.add_argument('--prompt',           default='',   type=str)
    parser.add_argument('--cate_kind',        default='cate', type=str)
    parser.add_argument('--class_num',        default=48,   type=int)
    parser.add_argument('--subset_ratio',     default=1.0,  type=float)
    parser.add_argument('--replace_kind_num', default=2,    type=int)

    # training
    parser.add_argument('--grad_accum_steps', default=1, type=int)

    # early stopping
    parser.add_argument('--scr_patience', default=5, type=int,
                        help='Stop training when SCR macro_F does not improve '
                             'for this many consecutive epochs. '
                             'Set 0 to disable early stopping.')

    # logging
    parser.add_argument('--summary_name', default='training_summary_catereg.txt')

    args   = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, config)