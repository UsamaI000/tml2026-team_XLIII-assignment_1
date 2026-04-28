import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from collections import defaultdict
from pathlib import Path
from membership_dataset import MembershipDataset, ShadowDataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def make_stratified_shadow_splits(dataset, n_shadow=4, seed=42):
    """
    For each shadow model, create a stratified 50/50 split of the dataset
    into shadow_train (member) and shadow_out (non-member).

    Returns:
        List of (shadow_train_indices, shadow_out_indices) tuples, one per shadow model.
    """
    rng = np.random.default_rng(seed)

    # Group indices by class label
    label_to_indices = defaultdict(list)
    for idx in range(len(dataset)):
        _, _, label, _ = dataset[idx]          # MembershipDataset returns (id, img, label, membership)
        label_to_indices[int(label)].append(idx)

    print(label_to_indices[0])

    # Shuffle within each class
    for label in label_to_indices:
        rng.shuffle(label_to_indices[label])

    print(label_to_indices[0])

    # this list will store the 'in' and 'out' ids for each shadow model
    splits = []
    for shadow_idx in range(n_shadow):
        train_indices = []
        out_indices   = []

        for label, indices in label_to_indices.items():
            mid = len(indices) // 2
            # Rotate which half is "train" per shadow model to maximise coverage
            offset = (shadow_idx * mid) % len(indices)
            rotated = indices[offset:] + indices[:offset]
            train_indices.extend(rotated[:mid])
            out_indices.extend(rotated[mid:])

        splits.append((train_indices, out_indices))

    print(len(splits))

    return splits


def make_shadow_dataset(dataset, indices, member_label):
    """
    Wrap a Subset as a ShadowDataset that returns (id, img, label, is_member).
    member_label: 1 for shadow_train, 0 for shadow_out.
    """
    return ShadowDataset(dataset, indices, member_label)


# TODO: do we need this as a separate function???
def make_shadow_model(num_classes=9):
    """
    ResNet-18 matching the target model architecture.
    Replaces the final FC layer for num_classes outputs.
    """
    model = models.resnet18(weights=None)          # no pretrained weights
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# TODO: this is running with some default settings - we need to try different hyperparams
# train loop for ONE shadow model
def train_shadow_model(train_loader, num_classes=9, epochs=50, lr=0.1, 
                       momentum=0.9, weight_decay=5e-4, device=None):
    """
    Trains a single shadow model on the given train_loader.
    Uses SGD + cosine LR schedule, standard for ResNet on small datasets.

    Returns:
        model (eval mode, on device)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = make_shadow_model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            _, imgs, labels, _ = batch          # unpack (id, img, label, membership)
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += imgs.size(0)

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch {epoch+1}/{epochs} "
                f"| loss: {running_loss/total:.4f} "
                f"| acc: {correct/total:.4f} "
                f"| lr: {scheduler.get_last_lr()[0]:.5f}"
            )

    model.eval()
    return model


# training loop for ALL shadow models
def train_all_shadow_models(shadow_loaders, num_classes=9, epochs=50,
                            lr=0.1, save_dir=None):
    """
    Trains one shadow model per (train_loader, out_loader) pair.
    Optionally saves each model checkpoint to save_dir.

    Returns:
        List of trained models (eval mode)
    """
    import os
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}\n")

    trained_models = []

    for i, (train_loader, out_loader) in enumerate(shadow_loaders):
        print(f"----- Shadow model {i} -----")
        model = train_shadow_model(
            train_loader,
            num_classes=num_classes,
            epochs=epochs,
            lr=lr,
            device=device
        )
        trained_models.append(model)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"shadow_{i}.pt")
            torch.save(model.state_dict(), path)
            print(f"  Saved - {path}")

        print()

    return trained_models


# extract outputs from one model over one loader
def collect_outputs(model, loader, device):
    """
    Runs model over all batches in loader and collects per-sample signals.

    Returns dict with:
        ids         : list of sample ids
        labels      : (N,)  true class labels
        memberships : (N,)  1 = member, 0 = non-member
        logits      : (N, C) raw model outputs
        probs       : (N, C) softmax probabilities
        loss        : (N,)  per-sample cross-entropy loss
        confidence  : (N,)  max softmax probability
        entropy     : (N,)  entropy of softmax distribution
        margin      : (N,)  top1 - top2 softmax probability
        correct     : (N,)  1 if prediction == label, else 0
    """
    model.eval()

    all_ids         = []
    all_labels      = []
    all_memberships = []
    all_logits      = []

    with torch.no_grad():
        for batch in loader:
            ids, imgs, labels, memberships = batch
            imgs, labels = imgs.to(device), labels.to(device)

            logits = model(imgs)                        # (B, C)

            all_ids.extend(ids)
            all_labels.append(labels.cpu())
            all_memberships.append(memberships)
            all_logits.append(logits.cpu())

    # Concatenate everything
    labels      = torch.cat(all_labels)                 # (N,)
    memberships = torch.cat(all_memberships)            # (N,)
    logits      = torch.cat(all_logits)                 # (N, C)

    # Derive signals from logits
    probs       = F.softmax(logits, dim=1)              # (N, C)

    # Per-sample cross-entropy loss
    loss = F.cross_entropy(logits, labels, reduction='none')   # (N,)

    # Confidence: probability assigned to the true class
    true_class_probs = probs[torch.arange(len(labels)), labels]   # (N,)

    # Max softmax confidence (may differ from true class prob if prediction is wrong)
    confidence  = probs.max(dim=1).values                         # (N,)

    # Entropy of the output distribution
    entropy     = -(probs * (probs + 1e-9).log()).sum(dim=1)      # (N,)

    # Margin: gap between top-1 and top-2 probabilities
    top2        = probs.topk(2, dim=1).values                     # (N, 2)
    margin      = top2[:, 0] - top2[:, 1]                         # (N,)

    # Correct prediction flag
    correct     = (logits.argmax(dim=1) == labels).float()        # (N,)

    return {
        "ids"         : all_ids,
        "labels"      : labels.numpy(),
        "memberships" : memberships.numpy(),
        "logits"      : logits.numpy(),
        "probs"       : probs.numpy(),
        "loss"        : loss.numpy(),
        "true_prob"   : true_class_probs.numpy(),
        "confidence"  : confidence.numpy(),
        "entropy"     : entropy.numpy(),
        "margin"      : margin.numpy(),
        "correct"     : correct.numpy(),
    }


# collect outputs from all shadow models
def collect_all_shadow_outputs(shadow_models, shadow_loaders, device):
    """
    For each shadow model, collects outputs from both its train loader
    (members) and out loader (non-members).

    Returns:
        List of dicts, one per shadow model, each containing
        'train' and 'out' output dicts.
    """
    all_outputs = []

    for i, (model, (train_loader, out_loader)) in enumerate(
        zip(shadow_models, shadow_loaders)
    ):
        print(f"Collecting outputs for shadow model {i}...")

        train_outputs = collect_outputs(model, train_loader, device)
        out_outputs   = collect_outputs(model, out_loader,   device)

        print(f"  train samples : {len(train_outputs['labels'])}"
              f"  |  out samples : {len(out_outputs['labels'])}")

        all_outputs.append({"train": train_outputs, "out": out_outputs})

    return all_outputs


# pool all shadow outputs into one attack training dataset
def build_attack_dataset(all_shadow_outputs):
    """
    Pools outputs from all shadow models into a single flat dataset
    for training the attack model.

    Returns dict with same keys as collect_outputs, plus 'memberships'
    as the attack label (1 = member, 0 = non-member).
    """
    pooled = {k: [] for k in [
        "labels", "memberships", "logits", "probs",
        "loss", "true_prob", "confidence", "entropy", "margin", "correct"
    ]}

    for shadow_out in all_shadow_outputs:
        for split in ("train", "out"):
            d = shadow_out[split]
            for k in pooled:
                pooled[k].append(d[k])

    return {k: np.concatenate(v) for k, v in pooled.items()}


if __name__ == "__main__":
    # Quick local test (small subset)

    N_SHADOW      = 4      # use fewer locally, scale up on GPU machine
    LOCAL_TEST    = True   # flip to False on full run
    LOCAL_SUBSET  = 1000   # samples to use for local smoke test

    EPOCHS     = 10 if LOCAL_TEST else 50     # fewer epochs locally
    SAVE_DIR   = "shadow_checkpoints"

    # config
    BASE = Path(__file__).parent
    PUB_PATH = BASE / "pub.pt"
    PRIV_PATH = BASE / "priv.pt"
    MODEL_PATH = BASE / "model.pt"
    OUTPUT_CSV = BASE / "submission.csv"

    print("Loading datasets...")
    pub_ds = torch.load(PUB_PATH, weights_only=False)
    priv_ds = torch.load(PRIV_PATH, weights_only=False)

    # normalization (same as training)
    MEAN = [0.7406, 0.5331, 0.7059]
    STD = [0.1491, 0.1864, 0.1301]

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    pub_ds.transform = transform
    priv_ds.transform = transform


    # ------------------------------------------------------------------------------

    # JUST FOR POC TESTING - REMOVE THIS LATER
    if LOCAL_TEST:
        # Take a small stratified slice of pub_ds for fast local iteration
        local_indices = []
        
        # print(local_indices)

        # to create key value pairs for all 9 classes (labels) and their corresponding ids
        label_to_indices = defaultdict(list)

        # print(label_to_indices)

        # getting list of dataset ids against each label
        for idx in range(len(pub_ds)):
            _, _, label, _ = pub_ds[idx]
            label_to_indices[int(label)].append(idx)

        # print(label_to_indices[2])

        per_class = LOCAL_SUBSET // 9

        # print(label_to_indices.values())

        # print(local_indices)

        for indices in label_to_indices.values():
            local_indices.extend(indices[:per_class])

        # print(len(local_indices))

        from torch.utils.data import Subset
        test_ds = Subset(pub_ds, local_indices)

        # Wrap Subset so make_stratified_shadow_splits can call dataset[idx]
        class SubsetWrapper(Dataset):
            def __init__(self, subset): self.subset = subset
            def __len__(self): return len(self.subset)
            def __getitem__(self, idx): return self.subset[idx]

        working_ds = SubsetWrapper(test_ds)
        # print(working_ds)
    else:
        working_ds = pub_ds


    # ------------------------------------------------------------------------------

    # MAIN CODE FOR SPLITTING DATASET FOR SHADOW MODELS

    # Build splits
    splits = make_stratified_shadow_splits(working_ds, n_shadow=N_SHADOW, seed=42)

    # get training and 'out' data for each shadow model
    # final list has len = n_shadow_models and a tuple for (in, out) at each index
    shadow_loaders = []
    for i, (train_idx, out_idx) in enumerate(splits):
        train_ds = make_shadow_dataset(working_ds, train_idx, member_label=1)
        out_ds   = make_shadow_dataset(working_ds, out_idx,   member_label=0)

        # TODO: change batch_size while testing
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        out_loader   = DataLoader(out_ds,   batch_size=64, shuffle=False)

        shadow_loaders.append((train_loader, out_loader))
        print(f"Shadow {i}: train={len(train_ds)}  out={len(out_ds)}")

    # ------------------------------------------------------------------------------
    # MAIN CODE FOR TRAINING SHADOW MODELS

    shadow_models = train_all_shadow_models(
        shadow_loaders,
        num_classes=9,
        epochs=EPOCHS,
        lr=0.1,
        save_dir=SAVE_DIR
    )

    # ------------------------------------------------------------------------------
    # GETTING OUTPUT FROM SHADOW MODELS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_shadow_outputs = collect_all_shadow_outputs(
        shadow_models, shadow_loaders, device
    )

    attack_dataset = build_attack_dataset(all_shadow_outputs)

    print(attack_dataset)

    print(f"\nAttack dataset size : {len(attack_dataset['labels'])}")
    print(f"Members             : {attack_dataset['memberships'].sum()}")
    print(f"Non-members         : {(1 - attack_dataset['memberships']).sum()}")
    print(f"Mean loss  members  : {attack_dataset['loss'][attack_dataset['memberships'] == 1].mean():.4f}")
    print(f"Mean loss  non-mem  : {attack_dataset['loss'][attack_dataset['memberships'] == 0].mean():.4f}")
