"""
check_dataset_overlap.py

Checks whether the train and test splits of the catereg dataset share any
items — which would explain suspiciously perfect accuracy.

Argument names and defaults are taken EXACTLY from fashion_pretrain_sym.py
so create_dataset() receives the same args object it expects.

No GPU, no distributed setup, no model needed.

Run from the FashionSAP project root:

    python check_dataset_overlap.py \
        --config   configs/fashion_catereg.yaml \
        --data_root /aul/homes/hsale014/Project1/Fashion/FashionSAP/data/data-fashion
"""

import argparse
import sys
import ruamel.yaml as yaml
from pathlib import Path
from transformers.models.bert.tokenization_bert import BertTokenizer


# ── annotation-list probe ─────────────────────────────────────────────────────
def _get_ann_list(dataset):
    """
    Try every attribute name the FashionGen dataset classes use for their
    internal annotation list.  Returns the list or None.
    """
    for attr in ('raw_data', 'info_data', 'ann', 'annotation', 'data',
                 'samples', 'items', 'img_list', 'train_data', 'test_data'):
        val = getattr(dataset, attr, None)
        if isinstance(val, (list, tuple)) and len(val) > 0:
            return list(val)
    return None


# ── identifier extraction ─────────────────────────────────────────────────────
def _extract_ids(dataset, split_name):
    """
    Pull three types of identifier out of a dataset split.
    Returns dict: {product_ids, img_paths, captions}  — each a plain list.
    """
    result = {'product_ids': [], 'img_paths': [], 'captions': []}

    anns = _get_ann_list(dataset)

    if anns is None:
        print(f"  [{split_name}] WARNING: no annotation list attribute found.")
        print(f"  [{split_name}] Trying common attribute names on dataset object:")
        print(f"  [{split_name}]   {[a for a in dir(dataset) if not a.startswith('_')]}")
        return result

    print(f"  [{split_name}] {len(anns)} annotations found.")
    print(f"  [{split_name}] Keys in first annotation: {list(anns[0].keys())}")

    for ann in anns:
        # ── product / item ID ────────────────────────────────────────────
        pid = ann.get('input_productID')
        result['product_ids'].append(str(pid) if pid is not None else None)

        # ── image file path ──────────────────────────────────────────────
        img = ann.get('input_pose')
        result['img_paths'].append(str(img) if img is not None else None)

        # ── caption / description text ───────────────────────────────────
        cap = ann.get('input_concat_description') or ann.get('input_description')
        result['captions'].append(str(cap).strip() if cap is not None else None)

    return result


# ── overlap stats ─────────────────────────────────────────────────────────────
def _check_overlap(train_vals, test_vals, id_type):
    train_known = {v for v in train_vals if v is not None}
    test_known  = {v for v in test_vals  if v is not None}

    if not train_known or not test_known:
        return {'status': 'UNKNOWN', 'overlap_count': 0,
                'overlap_pct': 0.0, 'examples': [],
                'train_known': len(train_known), 'test_known': len(test_known)}

    overlap = train_known & test_known
    pct     = len(overlap) / len(test_known) * 100

    return {
        'status':        'LEAK' if overlap else 'CLEAN',
        'train_known':   len(train_known),
        'test_known':    len(test_known),
        'overlap_count': len(overlap),
        'overlap_pct':   round(pct, 2),
        'examples':      list(overlap)[:5],
    }


# ── label distribution check ──────────────────────────────────────────────────
def _check_labels(train_anns, test_anns):
    if train_anns is None or test_anns is None:
        print("  (skipped — annotation list not accessible)")
        return

    # try every label key name the dataset might use
    for key in ('input_category', 'input_subcategory'):
        train_labels = [a[key] for a in train_anns if key in a]
        test_labels  = [a[key] for a in test_anns  if key in a]
        if not train_labels:
            continue

        train_set   = set(train_labels)
        test_set    = set(test_labels)
        unseen      = test_set - train_set   # classes in test but not train

        print(f"\n  Key '{key}':")
        print(f"    Train unique classes : {len(train_set)}")
        print(f"    Test  unique classes : {len(test_set)}")
        if unseen:
            print(f"    !! {len(unseen)} class(es) in test but NOT in train "
                  f"— model never saw these during fine-tuning:")
            print(f"       {list(unseen)[:10]}")
        else:
            print(f"    All test classes also appear in train (expected).")


# ── main ──────────────────────────────────────────────────────────────────────
def main(args, config):

    # ── tokenizer — plain BERT, no special symbol tokens ─────────────────
    # Matches fashion_pretrain_sym.py exactly
    tokenizer = BertTokenizer.from_pretrained(config['tokenizer_config'])
    print("[tokenizer] Plain BERT vocabulary — no [xxx_sign] tokens added.")

    # ── import dataset utilities ──────────────────────────────────────────
    try:
        from dataset import create_dataset
    except ImportError:
        print("ERROR: cannot import 'dataset'. "
              "Make sure you are running from the FashionSAP project root.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  FashionSAP — Dataset Overlap Checker  (catereg)")
    print("=" * 60)

# mirrors the logic in the original fashion_catereg.py main()
    args.class_num = 48 if args.cate_kind == 'cate' else 121
    # ── load both splits ──────────────────────────────────────────────────
    print("\n[1] Loading train + test splits ...")
    try:
        train_dataset, test_dataset = create_dataset(
            'catereg', config, args, tokenizer
        )
    except Exception as exc:
        print(f"\nERROR while loading dataset:\n  {exc}")
        print("\nIf the error mentions a missing attribute on args, "
              "add the missing --flag to this script's argument parser below.")
        sys.exit(1)

    print(f"  Train samples : {len(train_dataset)}")
    print(f"  Test  samples : {len(test_dataset)}")

    # ── extract identifiers ───────────────────────────────────────────────
    print("\n[2] Reading train annotations ...")
    train_ids = _extract_ids(train_dataset, 'TRAIN')

    print("\n[3] Reading test annotations ...")
    test_ids  = _extract_ids(test_dataset,  'TEST')

    # ── three overlap checks ──────────────────────────────────────────────
    print("\n[4] Overlap checks")
    print("-" * 60)

    any_leak = False
    checks = [
        ('Product ID',   train_ids['product_ids'], test_ids['product_ids']),
        ('Image path',   train_ids['img_paths'],   test_ids['img_paths']),
        ('Caption text', train_ids['captions'],    test_ids['captions']),
    ]

    for label, tr, te in checks:
        r = _check_overlap(tr, te, label)

        icon = {'CLEAN': '✓ CLEAN', 'LEAK': '✗ LEAK', 'UNKNOWN': '? UNKNOWN'}[r['status']]
        print(f"\n  {label}: {icon}")
        print(f"    Train identifiers known : {r['train_known']}")
        print(f"    Test  identifiers known : {r['test_known']}")
        print(f"    Shared items            : {r['overlap_count']}")
        print(f"    Overlap %               : {r['overlap_pct']:.2f}%")
        if r.get('examples'):
            print(f"    Example shared IDs      : {r['examples']}")
        if r['status'] == 'LEAK':
            any_leak = True

    # ── label distribution check ──────────────────────────────────────────
    print("\n[5] Label distribution")
    print("-" * 60)
    _check_labels(_get_ann_list(train_dataset), _get_ann_list(test_dataset))

    # ── verdict ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if any_leak:
        print("  VERDICT: DATA LEAKAGE DETECTED")
        print("  Train and test share items — 100% CR accuracy is NOT real.")
        print("  Fix the split (group by product_id) before reporting results.")
    else:
        print("  VERDICT: NO OVERLAP FOUND")
        print("  Splits are clean.")
        print("  100% CR accuracy reflects genuine pre-trained S stream quality.")
    print("=" * 60 + "\n")


# ── argument parser — mirrors fashion_pretrain_sym.py exactly ─────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ── same as fashion_pretrain_sym.py ───────────────────────────────────
    parser.add_argument('--config',     default='./configs/fashion_catereg.yaml')
    parser.add_argument('--data_root',  default='', type=str)

    parser.add_argument('--device',      default='cuda')
    parser.add_argument('--seed',        default=66,  type=int)
    parser.add_argument('--world_size',  default=1,   type=int)
    parser.add_argument('--dist_url',    default='env://')
    parser.add_argument('--distributed', default=False, type=bool)  # OFF — no GPU needed

    parser.add_argument('--max_word_num',          default=180, type=int)
    parser.add_argument('--catemap_filename',      default='categorys_to_sign.txt', type=str)
    parser.add_argument('--product_list_filename', default='productid_list.json',   type=str)
    parser.add_argument('--replace_kind_num',      default=2,   type=int)
    parser.add_argument('--subset_ratio',          default=1.0, type=float)

    # ── extra args create_dataset('catereg') needs ────────────────────────
    # These come from the original fashion_catereg.py
    parser.add_argument('--cate_kind', default='cate', type=str,
                        choices=['cate', 'subcate'],
                        help='Which label to use: cate=48 classes, subcate=121.')
    parser.add_argument('--prompt',    default=False, type=bool)

    args   = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    main(args, config)