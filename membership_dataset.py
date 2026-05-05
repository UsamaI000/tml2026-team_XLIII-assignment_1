from torch.utils.data import Dataset


class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index):
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)


class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index):
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]


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
