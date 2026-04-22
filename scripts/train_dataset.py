import random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from dataloader import IMAGENET_MEAN, IMAGENET_STD


def get_drone_transform(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_sat_transform(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomRotation(180),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class TripletDroneDataset(Dataset):
    """
    Returns triplets: (drone_img, sat_positive, sat_negative, building_id)
    Drone gets ColorJitter only; satellite gets rotation + jitter + hflip.
    HNM is handled in the training loop (in-batch).
    """
    def __init__(self, root_dir):
        self.root          = Path(root_dir)
        self.drone_tfm     = get_drone_transform()
        self.sat_tfm       = get_sat_transform()

        drone_root = self.root / "drone"
        sat_root   = self.root / "satellite"

        self.drone_samples = []
        for img_path in sorted(drone_root.rglob("*.jp*g")):
            building_id = img_path.parent.name
            self.drone_samples.append((img_path, building_id))

        self.sat_lookup = {}
        for img_path in sorted(sat_root.rglob("*.jp*g")):
            building_id = img_path.parent.name
            self.sat_lookup[building_id] = img_path

        self.building_ids = list(self.sat_lookup.keys())
        print(f"  Drone images : {len(self.drone_samples)}")
        print(f"  Satellite    : {len(self.sat_lookup)} buildings")

    def __len__(self):
        return len(self.drone_samples)

    def __getitem__(self, idx):
        drone_path, building_id = self.drone_samples[idx]

        anchor   = self._load(drone_path,                      self.drone_tfm)
        positive = self._load(self.sat_lookup[building_id],    self.sat_tfm)

        neg_id = building_id
        while neg_id == building_id:
            neg_id = random.choice(self.building_ids)
        negative = self._load(self.sat_lookup[neg_id], self.sat_tfm)

        return anchor, positive, negative, building_id

    def _load(self, path, tfm):
        img = Image.open(path).convert("RGB")
        return tfm(img)


class PairDroneDataset(TripletDroneDataset):
    """
    Returns pairs instead of triplets: (anchor, positive, building_id).
    Used with InfoNCE loss where in-batch negatives are implicit.
    """
    def __getitem__(self, idx):
        drone_path, building_id = self.drone_samples[idx]
        anchor   = self._load(drone_path,                   self.drone_tfm)
        positive = self._load(self.sat_lookup[building_id], self.sat_tfm)
        return anchor, positive, building_id


if __name__ == "__main__":
    ds = TripletDroneDataset("data/university1652/train")
    anchor, pos, neg, bid = ds[0]
    print(f"\nSample triplet:")
    print(f"  building ID : {bid}")
    print(f"  anchor shape: {anchor.shape}")
    print(f"  pos shape   : {pos.shape}")
    print(f"  neg shape   : {neg.shape}")
