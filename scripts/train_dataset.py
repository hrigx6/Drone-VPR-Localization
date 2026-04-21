import random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from dataloader import get_transform

class TripletDroneDataset(Dataset):
    """
    Returns triplets: (drone_img, sat_positive, sat_negative)
    
    For each sample:
      anchor   = one drone image of building ID X
      positive = satellite image of building ID X
      negative = satellite image of randomly sampled building ID Y (Y != X)

    ML concept — why triplets:
      We want embeddings where same-place images cluster together
      and different-place images are spread apart.
      Triplet loss directly optimizes this geometric property.
    """
    def __init__(self, root_dir, transform=None):
        self.root      = Path(root_dir)
        self.transform = transform or get_transform()

        drone_root = self.root / "drone"
        sat_root   = self.root / "satellite"

        # build list of all drone images with their building ID
        self.drone_samples = []
        for img_path in sorted(drone_root.rglob("*.jp*g")):
            building_id = img_path.parent.name
            self.drone_samples.append((img_path, building_id))

        # build satellite lookup: building_id → satellite image path
        self.sat_lookup = {}
        for img_path in sorted(sat_root.rglob("*.jp*g")):
            building_id = img_path.parent.name
            self.sat_lookup[building_id] = img_path

        #self.drone_samples = self.drone_samples[:2000]    ##testing
        self.building_ids = list(self.sat_lookup.keys())
        print(f"  Drone images : {len(self.drone_samples)}")
        print(f"  Satellite    : {len(self.sat_lookup)} buildings")

    def __len__(self):
        return len(self.drone_samples)

    def __getitem__(self, idx):
        drone_path, building_id = self.drone_samples[idx]

        # anchor — drone image
        anchor = self._load(drone_path)

        # positive — satellite of same building
        positive = self._load(self.sat_lookup[building_id])

        # negative — satellite of randomly different building
        neg_id = building_id
        while neg_id == building_id:
            neg_id = random.choice(self.building_ids)
        negative = self._load(self.sat_lookup[neg_id])

        return anchor, positive, negative, building_id

    def _load(self, path):
        img = Image.open(path).convert("RGB")
        return self.transform(img)


if __name__ == "__main__":
    ds = TripletDroneDataset("data/university1652/train")
    anchor, pos, neg, bid = ds[0]
    print(f"\nSample triplet:")
    print(f"  building ID : {bid}")
    print(f"  anchor shape: {anchor.shape}")
    print(f"  pos shape   : {pos.shape}")
    print(f"  neg shape   : {neg.shape}")
