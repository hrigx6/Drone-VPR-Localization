import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_transform(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

class FlatImageDataset(Dataset):
    """
    Loads all images from a root directory recursively.
    Each item: (image_tensor, building_id, image_path)
    building_id is the parent folder name e.g. '0000', '0001'
    Used for: feature extraction over query_drone or gallery_satellite
    """
    def __init__(self, root_dir, transform=None):
        self.root = Path(root_dir)
        self.transform = transform or get_transform()
        self.samples = []

        for img_path in sorted(self.root.rglob("*.jp*g")):
            building_id = img_path.parent.name
            self.samples.append((img_path, building_id))

        if len(self.samples) == 0:
            raise ValueError(f"No images found in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, building_id = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, building_id, str(img_path)


class DroneDataset(FlatImageDataset):
    """Query drone images — test/query_drone/"""
    pass


class SatelliteDataset(FlatImageDataset):
    """Gallery satellite images — test/gallery_satellite/ or train/satellite/"""
    pass


def get_dataloader(root_dir, batch_size=64, num_workers=4, shuffle=False):
    dataset = FlatImageDataset(root_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


if __name__ == "__main__":
    import sys
    base = Path("data/university1652")

    print("--- Query drone ---")
    drone_ds = DroneDataset(base / "test/query_drone")
    print(f"  Total images : {len(drone_ds)}")
    img, bid, path = drone_ds[0]
    print(f"  Sample shape : {img.shape}")
    print(f"  Building ID  : {bid}")
    print(f"  Path         : {path}")

    print("\n--- Gallery satellite ---")
    sat_ds = SatelliteDataset(base / "test/gallery_satellite")
    print(f"  Total images : {len(sat_ds)}")
    img, bid, path = sat_ds[0]
    print(f"  Sample shape : {img.shape}")
    print(f"  Building ID  : {bid}")
    print(f"  Path         : {path}")

    print("\n--- Dataloader batch test ---")
    loader = get_dataloader(base / "test/gallery_satellite", batch_size=8)
    imgs, bids, paths = next(iter(loader))
    print(f"  Batch shape  : {imgs.shape}")
    print(f"  Building IDs : {list(bids)}")
