import os, sys, time, random, math, torch, torchvision
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime
from collections import Counter
import torch.nn.functional as F

IM = InterpolationMode.BILINEAR

device = ("mps" if torch.backends.mps.is_available()
         else "cuda" if torch.cuda.is_available()
         else "cpu")
print(f"Using device: {device}")

random.seed(1337)
torch.manual_seed(1337)

data_dir     = "data"
batch_size   = 64
epochs       = 40
lr           = 1e-3
num_workers  = 0
weight_decay = 1e-4
use_focal    = True      
gamma_focal  = 2.0
triangle_boost = 1.35    
mixup_prob   = 0.50
mixup_alpha  = 0.20
cutmix_prob  = 0.10
cutmix_alpha = 0.8

def _format_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m:d}m {s:02d}s"
    return f"{s:d}s"

def _progress_bar(current: int, total: int, start_time: float, bar_len: int = 28, prefix: str = ""):
    """Print a single-line progress bar with ETA for the current epoch."""
    current = min(current, total)
    elapsed = time.time() - start_time
    pct = current / total if total else 1.0
    filled = int(bar_len * pct)
    rate = current / elapsed if elapsed > 0 else 0.0
    remain = (total - current) / rate if rate > 0 else 0.0
    bar = "#" * filled + "." * (bar_len - filled)
    sys.stdout.write(
        f"\r{prefix}[{bar}] {current}/{total} • {pct*100:5.1f}% • ETA {_format_time(remain)}"
    )
    sys.stdout.flush()

train_tf = transforms.Compose([
    transforms.Resize((288, 288), interpolation=IM),
    transforms.RandomResizedCrop((256, 256), scale=(0.65, 1.0), ratio=(0.9, 1.1), interpolation=IM),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.08),
    transforms.RandomRotation(degrees=18, interpolation=IM),
    transforms.RandomAffine(degrees=0, translate=(0.07,0.07), shear=7, interpolation=IM),
    transforms.RandomPerspective(distortion_scale=0.22, p=0.12, interpolation=IM),
    transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05, hue=0.015),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
    transforms.RandomErasing(p=0.20, scale=(0.02, 0.12), ratio=(0.3, 3.3), inplace=True),
])

val_tf = transforms.Compose([
    transforms.Resize((288,288), interpolation=IM),
    transforms.CenterCrop((256,256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

train_ds = torchvision.datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
val_ds   = torchvision.datasets.ImageFolder(os.path.join(data_dir, "val"),   transform=val_tf)
print("Class mapping:", train_ds.class_to_idx)

expected = {"not_triangle", "triangle"}
assert set(train_ds.class_to_idx.keys()) == expected, f"Expected classes {expected}, got {set(train_ds.class_to_idx.keys())}"

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

classes = train_ds.classes
num_classes = len(classes)
idx_tri = classes.index("triangle")
idx_not = classes.index("not_triangle")

def count_targets(ds):
    return Counter(ds.targets) if hasattr(ds, "targets") else Counter([y for _,y in ds.samples])

print("Train counts:", {classes[i]: c for i,c in count_targets(train_ds).items()})
print("Val counts:",   {classes[i]: c for i,c in count_targets(val_ds).items()})

class TinyCNNv2(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(3, 48, 3, padding=1),   nn.BatchNorm2d(48),   nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1),  nn.BatchNorm2d(96),   nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(96, 192, 3, padding=1), nn.BatchNorm2d(192),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(192, 256, 3, padding=1),nn.BatchNorm2d(256),  nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Dropout(0.30),
        )
        self.fc = nn.Linear(256, k)
    def forward(self, x):
        return self.fc(self.f(x).view(x.size(0), -1))

model = TinyCNNv2(num_classes).to(device)

with torch.no_grad():
    tr_counts = count_targets(train_ds)
    w = torch.ones(num_classes, dtype=torch.float32)
    w[idx_tri] *= triangle_boost
class_weights = w.to(device)

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction="none", weight=self.weight, label_smoothing=self.label_smoothing)
        pt = torch.softmax(logits, dim=1).gather(1, target.view(-1,1)).squeeze(1).clamp_(1e-6, 1.0)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()

criterion = FocalLoss(weight=class_weights, gamma=gamma_focal, label_smoothing=0.05) if use_focal \
            else nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

def mixup(x, y, alpha):
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def cutmix(x, y, alpha):
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(x.size(0), device=x.device)
    b, c, h, w = x.size()
    cx = random.randint(0, w-1); cy = random.randint(0, h-1)
    rw = int(w * math.sqrt(1 - lam))
    rh = int(h * math.sqrt(1 - lam))
    x1 = max(0, cx - rw // 2); y1 = max(0, cy - rh // 2)
    x2 = min(w, cx + rw // 2);  y2 = min(h, cy + rh // 2)
    x_aug = x.clone()
    x_aug[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam_adj = 1 - ((x2-x1)*(y2-y1) / (w*h))
    return x_aug, y, y[idx], lam_adj

def mix_criterion(pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def tri_metrics_from_cm(cm):
    tn, fp = cm[idx_not, idx_not].item(), cm[idx_not, idx_tri].item()
    fn, tp = cm[idx_tri, idx_not].item(), cm[idx_tri, idx_tri].item()
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2*prec*rec / (prec + rec + 1e-9)
    acc  = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    return prec, rec, f1, acc

best_f1 = 0.0
os.makedirs("models", exist_ok=True)

for epoch in range(1, epochs+1):
    model.train()
    total = correct = 0
    total_batches = len(train_loader)
    epoch_start = time.time()

    for batch_idx, (x, y) in enumerate(train_loader, 1):
        x, y = x.to(device), y.to(device)

        r = random.random()
        if r < cutmix_prob:
            x_aug, y_a, y_b, lam = cutmix(x, y, cutmix_alpha)
            use_mix = True
        elif r < cutmix_prob + mixup_prob:
            x_aug, y_a, y_b, lam = mixup(x, y, mixup_alpha)
            use_mix = True
        else:
            x_aug = x; use_mix = False

        optimizer.zero_grad()
        out = model(x_aug)
        loss = mix_criterion(out, y_a, y_b, lam) if use_mix else criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total   += y.size(0)

        _progress_bar(batch_idx, total_batches, epoch_start, prefix=f"epoch: {epoch:02d}/{epochs:02d} ")

    sys.stdout.write("\n")

    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(1)
            for t, p in zip(y, pred):
                cm[t, p] += 1

    tri_prec, tri_rec, tri_f1, val_acc = tri_metrics_from_cm(cm)
    scheduler.step()

    print(f"epoch: {epoch:02d}/{epochs:02d} — Training accuracy: {val_acc*100:5.2f}%")

    if tri_f1 > best_f1:
        best_f1 = tri_f1
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        torch.save({
            "model": model.state_dict(),
            "classes": classes,
            "norm_mean": [0.5,0.5,0.5],
            "norm_std":  [0.5,0.5,0.5],
            "arch": "tinycnn_v2_triangle_f1",
            "best_triangle_f1": float(best_f1),
        }, f"models/tri_vs_not_{ts}.pt")

print(f"=> Training complete. Best training accuracy: {best_f1*100:5.2f}")
