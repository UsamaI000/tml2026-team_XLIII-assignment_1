import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.models import resnet18

from collections import defaultdict
from pathlib import Path
from membership_dataset import MembershipDataset, ShadowDataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")
import random


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
    
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
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
def collect_outputs(model, loader, device, has_membership=True):
    """
    Runs model over all batches in loader and collects per-sample signals.
    has_membership: set to False for private dataset where membership is None

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
            if has_membership:
                ids, imgs, labels, memberships = batch
                all_memberships.append(memberships)
            else:
                ids, imgs, labels = batch[:3]   # ignore the None membership field

            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)

            all_ids.extend(ids)
            all_labels.append(labels.cpu())
            all_logits.append(logits.cpu())

    labels  = torch.cat(all_labels)
    logits  = torch.cat(all_logits)
    probs   = F.softmax(logits, dim=1)

    loss        = F.cross_entropy(logits, labels, reduction='none')
    true_prob   = probs[torch.arange(len(labels)), labels]
    confidence  = probs.max(dim=1).values
    entropy     = -(probs * (probs + 1e-9).log()).sum(dim=1)
    top2        = probs.topk(2, dim=1).values
    margin      = top2[:, 0] - top2[:, 1]
    correct     = (logits.argmax(dim=1) == labels).float()

    result = {
        "ids"        : all_ids,
        "labels"     : labels.numpy(),
        "logits"     : logits.numpy(),
        "probs"      : probs.numpy(),
        "loss"       : loss.numpy(),
        "true_prob"  : true_prob.numpy(),
        "confidence" : confidence.numpy(),
        "entropy"    : entropy.numpy(),
        "margin"     : margin.numpy(),
        "correct"    : correct.numpy(),
    }

    if has_membership:
        result["memberships"] = torch.cat(all_memberships).numpy()
    else:
        result["memberships"] = np.zeros(len(labels), dtype=np.float32)  # dummy

    return result


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


# for attack model
def build_features(outputs, mode="full"):
    """
    Constructs feature matrix from collected model outputs.

    Modes:
        "simple"  : just loss + confidence + entropy + margin (fast baseline)
        "full"    : all scalar features
        "with_probs" : full + raw softmax vector (9 extra dims, per-class signal)

    Always call with the same mode for shadow outputs and target outputs.
    """
    loss       = outputs["loss"].reshape(-1, 1)
    true_prob  = outputs["true_prob"].reshape(-1, 1)
    confidence = outputs["confidence"].reshape(-1, 1)
    entropy    = outputs["entropy"].reshape(-1, 1)
    margin     = outputs["margin"].reshape(-1, 1)
    correct    = outputs["correct"].reshape(-1, 1)

    if mode == "simple":
        return np.hstack([loss, confidence, entropy, margin])

    if mode == "full":
        return np.hstack([loss, true_prob, confidence, entropy, margin, correct])

    if mode == "with_probs":
        probs = outputs["probs"]                        # (N, 9)
        return np.hstack([loss, true_prob, confidence,
                          entropy, margin, correct, probs])

    raise ValueError(f"Unknown mode: {mode}")


# classifier for each class (9 labels)
# def train_attack_classifiers(attack_dataset, feature_mode="full", use_mlp=False):
#     """
#     Trains one attack classifier per class label (9 total).

#     For each class:
#         - filters attack_dataset to samples of that class
#         - scales features
#         - trains a logistic regression (or MLP) binary classifier
#         - evaluates TPR @ 5% FPR on the same data (training signal check)

#     Returns:
#         classifiers : dict {label -> fitted classifier}
#         scalers     : dict {label -> fitted scaler}
#     """
#     labels      = attack_dataset["labels"]
#     memberships = attack_dataset["memberships"]
#     features    = build_features(attack_dataset, mode=feature_mode)

#     classifiers = {}
#     scalers     = {}

#     for cls in range(9):
#         mask = labels == cls
#         X    = features[mask]
#         y    = memberships[mask]

#         if len(np.unique(y)) < 2:
#             print(f"  Class {cls}: skipping — only one membership class present")
#             continue

#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X)

#         if use_mlp:
#             clf = MLPClassifier(
#                 hidden_layer_sizes=(64, 32),
#                 activation="relu",
#                 max_iter=500,
#                 random_state=RANDOM_SEED
#             )
#         else:
#             clf = LogisticRegression(
#                 C=1.0,
#                 max_iter=1000,
#                 random_state=RANDOM_SEED
#             )

#         clf.fit(X_scaled, y)

#         # Quick training signal: TPR @ 5% FPR
#         probs      = clf.predict_proba(X_scaled)[:, 1]
#         tpr_at_fpr = compute_tpr_at_fpr(y, probs, fpr_threshold=0.05)

#         print(f"  Class {cls} | n={mask.sum():5d} "
#               f"| members={y.sum():4d} "
#               f"| TPR@5%FPR (train): {tpr_at_fpr:.4f}")

#         classifiers[cls] = clf
#         scalers[cls]     = scaler

#     return classifiers, scalers
def train_attack_classifiers(attack_dataset, feature_mode="full", use_mlp=False):
    labels      = attack_dataset["labels"]
    memberships = attack_dataset["memberships"]
    features    = build_features(attack_dataset, mode=feature_mode)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    if use_mlp:
        clf = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=500,
            random_state=RANDOM_SEED
        )
    else:
        clf = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=RANDOM_SEED
        )

    clf.fit(X_scaled, memberships)

    probs      = clf.predict_proba(X_scaled)[:, 1]
    tpr_at_fpr = compute_tpr_at_fpr(memberships, probs, fpr_threshold=0.05)
    print(f"  Global classifier | n={len(memberships)} "
          f"| members={int(memberships.sum())} "
          f"| TPR@5%FPR (train): {tpr_at_fpr:.4f}")

    return clf, scaler

# cross-validated evaluation on the public dataset
# def evaluate_on_public(pub_ds_outputs, classifiers, scalers,
#                        feature_mode="full", n_folds=5):
#     """
#     Evaluates attack classifiers on the public dataset using
#     stratified k-fold cross-validation per class.

#     pub_ds_outputs: output dict from collect_outputs() run on pub_ds
#                     with the TARGET model (not shadow models)
#     """
#     labels      = pub_ds_outputs["labels"]
#     memberships = pub_ds_outputs["memberships"]
#     features    = build_features(pub_ds_outputs, mode=feature_mode)

#     all_scores  = np.zeros(len(labels))

#     for cls in range(9):
#         mask = labels == cls
#         X    = features[mask]
#         y    = memberships[mask]
#         idx  = np.where(mask)[0]

#         if cls not in classifiers:
#             # Fallback: use raw loss (negated, higher = more likely member)
#             all_scores[idx] = -X[:, 0]
#             continue

#         scaler = scalers[cls]
#         clf    = classifiers[cls]

#         X_scaled         = scaler.transform(X)
#         scores           = clf.predict_proba(X_scaled)[:, 1]
#         all_scores[idx]  = scores

#     overall_tpr = compute_tpr_at_fpr(memberships, all_scores, fpr_threshold=0.05)
#     print(f"\nOverall TPR@5%FPR on public dataset: {overall_tpr:.4f}")

#     # Per-class breakdown
#     for cls in range(9):
#         mask = labels == cls
#         if mask.sum() == 0:
#             continue
#         t = compute_tpr_at_fpr(memberships[mask], all_scores[mask], fpr_threshold=0.05)
#         print(f"  Class {cls}: TPR@5%FPR = {t:.4f}  (n={mask.sum()})")

#     return all_scores, overall_tpr
def evaluate_on_public(pub_ds_outputs, classifier, scaler, feature_mode="full"):
    memberships = pub_ds_outputs["memberships"]
    features    = build_features(pub_ds_outputs, mode=feature_mode)

    X_scaled    = scaler.transform(features)
    all_scores  = classifier.predict_proba(X_scaled)[:, 1]

    overall_tpr = compute_tpr_at_fpr(memberships, all_scores, fpr_threshold=0.05)
    print(f"\nOverall TPR@5%FPR on public dataset: {overall_tpr:.4f}")

    # Per-class breakdown (just for diagnostics, still one global model)
    labels = pub_ds_outputs["labels"]
    for cls in range(9):
        mask = labels == cls
        if mask.sum() == 0:
            continue
        t = compute_tpr_at_fpr(memberships[mask], all_scores[mask], fpr_threshold=0.05)
        print(f"  Class {cls}: TPR@5%FPR = {t:.4f}  (n={mask.sum()})")

    return all_scores, overall_tpr


def compute_tpr_at_fpr(y_true, y_scores, fpr_threshold=0.05):
    """
    Computes TPR at a given FPR threshold using the ROC curve.
    This is the competition metric.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    # Find the TPR at the largest FPR that is still <= threshold
    valid = fpr <= fpr_threshold
    if not valid.any():
        return 0.0
    return tpr[valid].max()


# inference on prvate dataset
# def predict_private(priv_outputs, classifiers, scalers, feature_mode="full"):
#     """
#     Produces membership scores for the private dataset.
#     Returns array of scores in [0, 1], one per sample, aligned with priv_outputs.
#     """
#     labels   = priv_outputs["labels"]
#     features = build_features(priv_outputs, mode=feature_mode)
#     scores   = np.zeros(len(labels))

#     for cls in range(9):
#         mask = labels == cls
#         if not mask.any():
#             continue

#         X = features[mask]

#         if cls not in classifiers:
#             scores[mask] = -X[:, 0]          # fallback: negated loss
#             continue

#         X_scaled      = scalers[cls].transform(X)
#         scores[mask]  = classifiers[cls].predict_proba(X_scaled)[:, 1]

#     return scores
def predict_private(priv_outputs, classifier, scaler, feature_mode="full"):
    features = build_features(priv_outputs, mode=feature_mode)
    X_scaled = scaler.transform(features)
    return classifier.predict_proba(X_scaled)[:, 1]


if __name__ == "__main__":
    N_SHADOW     = 50     # use fewer locally, scale up on GPU machine
    EPOCHS       = 30
    SAVE_DIR     = "shadow_checkpoints"
    FEATURE_MODE = "full"      # try "with_probs" if results are weak
    USE_MLP      = False       # flip to True to try a small neural attack model

    RANDOM_SEED  = 12

    # config
    BASE = Path(__file__).parent
    PUB_PATH = BASE / "pub.pt"
    PRIV_PATH = BASE / "priv.pt"
    MODEL_PATH = BASE / "model.pt"
    OUTPUT_CSV = BASE / "single_attack_50m_30e.csv"
    # OUTPUT_CSV = BASE / "submission.csv"

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

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    # ------------------------------------------------------------------------------

    working_ds = pub_ds

    # ------------------------------------------------------------------------------

    # MAIN CODE FOR SPLITTING DATASET FOR SHADOW MODELS

    # Build splits
    splits = make_stratified_shadow_splits(working_ds, n_shadow=N_SHADOW, seed=RANDOM_SEED)

    # get training and 'out' data for each shadow model
    # final list has len = n_shadow_models and a tuple for (in, out) at each index
    shadow_loaders = []
    for i, (train_idx, out_idx) in enumerate(splits):
        train_ds = make_shadow_dataset(working_ds, train_idx, member_label=1)
        out_ds   = make_shadow_dataset(working_ds, out_idx,   member_label=0)

        # TODO: change batch_size while testing
        g = torch.Generator()
        g.manual_seed(RANDOM_SEED)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, generator=g)
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

    # ------------------------------------------------------------------------------
    # TRAINING ATTACK MODELS
    print("Training attack classifiers...\n")
    classifiers, scalers = train_attack_classifiers(
        attack_dataset,
        feature_mode=FEATURE_MODE,
        use_mlp=USE_MLP
    )

    # -- Collect target model outputs on public dataset for validation
    print("\nCollecting target model outputs on public dataset...")
    print("Loading model...")
    target_model = resnet18(weights=None)
    target_model.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    target_model.maxpool = torch.nn.Identity()
    target_model.fc = torch.nn.Linear(512, 9)
    target_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    target_model.to(device)
    target_model.eval()

    pub_target_outputs  = collect_outputs(target_model, DataLoader(pub_ds,  batch_size=64), device, has_membership=True)

    print("\nEvaluating on public dataset...")
    pub_scores, pub_tpr = evaluate_on_public(
        pub_target_outputs, classifiers, scalers, feature_mode=FEATURE_MODE
    )

    # -- Collect target model outputs on private dataset
    print("\nCollecting target model outputs on private dataset...")
    
    def collate_fn(batch):
        ids, imgs, labels, _ = zip(*batch)
        return list(ids), torch.stack(list(imgs)), torch.tensor(labels)

    priv_loader = DataLoader(priv_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    priv_target_outputs = collect_outputs(target_model, priv_loader, device, has_membership=False)

    priv_scores = predict_private(priv_target_outputs, classifiers, scalers, feature_mode=FEATURE_MODE)

    # -- Build submission
    import pandas as pd
    submission = pd.DataFrame({
        "id"    : priv_target_outputs["ids"],
        "score" : priv_scores
    })
    submission.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSubmission saved — {len(submission)} rows")
    print(OUTPUT_CSV)
    print(submission.head())
