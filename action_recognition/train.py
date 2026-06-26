import os
import sys
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, os.path.dirname(__file__))
from dataset import AnimalKingdomDataset
from model import ActionTransformer
from utils import compute_map

# ── Config ────────────────────────────────────────────────────────────────────
CFG = {
    'gt_path':      'dataset/gt.json',
    'feature_root': 'dataset/clip_features',
    'ckpt_dir':     'checkpoints',
    'max_len':      256,
    'batch_size':   64,     # drop to 16 + grad_accum=2 if OOM on 6GB
    'num_workers':  4,
    'lr':           3e-4,
    'weight_decay': 1e-4,
    'epochs':       50,
    'val_ratio':    0.15,
    'seed':         42, # 42
    # model
    'd_model':      512,
    'nhead':        4,
    'num_layers':   2,
    'dropout':      0.1,
    # training tricks
    'grad_accum':   1,      # effective batch = batch_size * grad_accum
    'threshold':    0.3,
    'device':       'cuda' if torch.cuda.is_available() else 'cpu',
}
# ──────────────────────────────────────────────────────────────────────────────


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, grad_accum):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    for step, (feats, masks, labels) in enumerate(loader):
        print(f'Step {step}: feats {feats.shape}')
        feats  = feats.to(device)
        masks  = masks.to(device)
        labels = labels.to(device)
        with autocast():
            logits = model(feats, masks)
            loss   = criterion(logits, labels) / grad_accum
        scaler.scale(loss).backward()
        if (step + 1) % grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        total_loss += loss.item() * grad_accum
        print(f'Total loss {total_loss}, progress {step/len(loader)*100.0}%')
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_scores, all_labels = [], []
    for feats, masks, labels in loader:
        feats  = feats.to(device)
        masks  = masks.to(device)
        labels = labels.to(device)
        with autocast():
            logits = model(feats, masks)
            loss   = criterion(logits, labels)
        total_loss += loss.item()
        all_scores.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    scores  = np.concatenate(all_scores)
    targets = np.concatenate(all_labels)
    return total_loss / len(loader), compute_map(scores, targets)


def main():
    os.makedirs(CFG['ckpt_dir'], exist_ok=True)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join(CFG['ckpt_dir'], 'tb_logs', run_name))
    device = torch.device(CFG['device'])
    print(f"[Train] device={device}")

    train_ds = AnimalKingdomDataset(
        CFG['gt_path'], CFG['feature_root'], split='train',
        max_len=CFG['max_len'], val_ratio=CFG['val_ratio'], seed=CFG['seed'])
    val_ds = AnimalKingdomDataset(
        CFG['gt_path'], CFG['feature_root'], split='val',
        max_len=CFG['max_len'], val_ratio=CFG['val_ratio'], seed=CFG['seed'])

    pos_weights = train_ds.compute_pos_weights().to(device)
    criterion   = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True,
                              num_workers=CFG['num_workers'], pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG['batch_size'], shuffle=False,
                              num_workers=CFG['num_workers'], pin_memory=True)

    model = ActionTransformer(
        d_model=CFG['d_model'], nhead=CFG['nhead'],
        num_layers=CFG['num_layers'], dropout=CFG['dropout'],
        max_len=CFG['max_len'],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['epochs'])
    scaler    = GradScaler()

    best_map = 0.0
    for epoch in range(1, CFG['epochs'] + 1):
        train_loss              = train_one_epoch(model, train_loader, optimizer, criterion,
                                                  scaler, device, CFG['grad_accum'])
        val_loss, val_map       = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val',   val_loss,   epoch)
        writer.add_scalar('mAP/val',    val_map,    epoch)
        writer.add_scalar('LR',         lr,         epoch)

        print(f"Epoch {epoch:03d} | train={train_loss:.4f} | val={val_loss:.4f} | mAP={val_map:.4f} | lr={lr:.2e}")

        if val_map > best_map:
            best_map = val_map
            torch.save(
                {'epoch': epoch, 'model': model.state_dict(), 'cfg': CFG, 'best_map': best_map},
                os.path.join(CFG['ckpt_dir'], 'best.pt')
            )
            print(f"  → best checkpoint saved  (mAP={best_map:.4f})")

    writer.close()
    print(f"\nDone. Best val mAP: {best_map:.4f}")


if __name__ == '__main__':
    main()