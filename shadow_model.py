import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from collections import defaultdict
from pathlib import Path
from membership_dataset import MembershipDataset


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


class ShadowDataset(Dataset):
    """
    Wraps a MembershipDataset + index list.
    Overrides the membership field with the shadow membership label.
    """
    def __init__(self, base_dataset, indices, member_label):
        self.base    = base_dataset
        self.indices = indices
        self.member_label = member_label

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        id_, img, label, _ = self.base[real_idx]   # ignore original membership
        return id_, img, label, self.member_label


if __name__ == "__main__":
    # Quick local test (small subset)

    N_SHADOW      = 4      # use fewer locally, scale up on GPU machine
    LOCAL_TEST    = True   # flip to False on full run
    LOCAL_SUBSET  = 1000   # samples to use for local smoke test

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

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        out_loader   = DataLoader(out_ds,   batch_size=64, shuffle=False)

        shadow_loaders.append((train_loader, out_loader))
        print(f"Shadow {i}: train={len(train_ds)}  out={len(out_ds)}")

    print(shadow_loaders)

    # Sanity check: verify class balance in first shadow split
    from collections import Counter
    train_labels = [int(working_ds[idx][2]) for idx in splits[0][0]]
    out_labels   = [int(working_ds[idx][2]) for idx in splits[0][1]]
    print("\nShadow 0 train class dist:", dict(sorted(Counter(train_labels).items())))
    print("Shadow 0 out   class dist:", dict(sorted(Counter(out_labels).items())))