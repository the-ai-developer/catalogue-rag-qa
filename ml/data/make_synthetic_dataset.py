"""Seeded synthetic catalogue corpus: spec → description pairs + product images.

Generates ml/data/generated/:
  train.jsonl / val.jsonl   {"spec": {...}, "description": "..."}  (T5 fine-tune)
  manifest.csv              item_id,image_path,description          (projection training)
  images/*.png              deterministic placeholder product images (PIL)
  qrels.jsonl               question, item_id (retrieval eval)

Fully deterministic: same seed -> same corpus.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random

CATEGORIES = {
    "drinkware": {
        "material": ["18/8 stainless steel", "double-wall ceramic", "tritan copolyester"],
        "features": ["double-wall vacuum insulation", "leak-proof lid", "BPA-free",
                     "dishwasher safe", "non-slip base", "carry loop"],
        "dims": {"height_cm": (14.0, 30.0), "diameter_cm": (6.0, 9.0)},
        "names": ["Insulated Bottle", "Travel Mug", "Tumbler", "Water Flask"],
    },
    "bags": {
        "material": ["recycled nylon", "full-grain leather", "waxed canvas"],
        "features": ["laptop sleeve", "rain cover", "padded straps",
                     "luggage pass-through", "hidden pocket", "YKK zips"],
        "dims": {"height_cm": (30.0, 55.0), "width_cm": (25.0, 45.0)},
        "names": ["Travel Pack", "Daypack", "Duffel", "Messenger Bag"],
    },
    "apparel": {
        "material": ["organic cotton", "merino wool", "recycled polyester"],
        "features": ["moisture-wicking", "flatlock seams", "UPF 50+",
                     "four-way stretch", "anti-odour finish"],
        "dims": {"chest_cm": (90.0, 110.0), "length_cm": (65.0, 78.0)},
        "names": ["Training Tee", "Base Layer", "Zip Hoodie", "Trail Shirt"],
    },
    "electronics": {
        "material": ["anodised aluminium", "ABS+PC blend", "recycled polycarbonate"],
        "features": ["USB-C fast charge", "Bluetooth 5.3", "ANC",
                     "40-hour battery", "IPX5 rated", "multipoint pairing"],
        "dims": {"width_cm": (6.0, 22.0), "depth_cm": (2.0, 12.0)},
        "names": ["Headphones", "Power Bank", "Smart Speaker", "Earbuds Case"],
    },
    "home": {
        "material": ["solid oak", "stoneware", "borosilicate glass"],
        "features": ["stackable", "oven safe", "non-toxic glaze",
                     "hand-finished edges", "easy-clean coating"],
        "dims": {"height_cm": (8.0, 40.0), "width_cm": (15.0, 60.0)},
        "names": ["Serving Bowl", "Storage Jar", "Chopping Board", "Table Lamp"],
    },
    "outdoor": {
        "material": ["ripstop nylon", "aluminium alloy", "EVA foam"],
        "features": ["UV resistant", "quick-dry", "packable",
                     "windproof", "reinforced stress points"],
        "dims": {"length_cm": (20.0, 220.0), "width_cm": (10.0, 120.0)},
        "names": ["Camp Chair", "Dry Bag", "Tarp", "Sleeping Pad"],
    },
    "beauty": {
        "material": ["bamboo fibre", "silicone", "glass"],
        "features": ["dermatologically tested", "fragrance-free",
                     "refillable", "travel size"],
        "dims": {"height_cm": (5.0, 20.0), "width_cm": (3.0, 10.0)},
        "names": ["Cleansing Brush", "Serum Bottle", "Travel Case", "Hair Brush"],
    },
    "toys": {
        "material": ["FSC beechwood", "food-grade silicone", "recycled ABS"],
        "features": ["BPA-free", "develops fine motor skills", "wipe-clean",
                     "chunky grips", "tested to EN71"],
        "dims": {"height_cm": (8.0, 30.0), "width_cm": (8.0, 30.0)},
        "names": ["Stacking Rings", "Shape Sorter", "Building Blocks", "Push Toy"],
    },
}

OPENERS = ["Meet the {name}", "The {name} is built for everyday use",
           "Designed for daily life, the {name}", "A dependable essential, the {name}"]
CLOSERS = ["A solid pick for everyday use.", "Built to last, season after season.",
           "An easy addition to your routine.", "Thoughtful details throughout."]


def _dim_value(rng: random.Random, lo: float, hi: float) -> float:
    return round(rng.uniform(lo, hi), 1)


def make_item(rng: random.Random, item_no: int) -> dict:
    cat = rng.choice(sorted(CATEGORIES))
    spec_cat = CATEGORIES[cat]
    title = f"{rng.choice(['Kestrel', 'Ridge', 'Harbor', 'Summit', 'Drift', 'Atlas'])} " \
            f"{rng.choice(['12 oz', '40 L', 'M', '20000 mAh', '35 cm', 'Set of 2'])} " \
            f"{rng.choice(spec_cat['names'])}"
    dims = {k: _dim_value(rng, *bounds) for k, bounds in spec_cat["dims"].items()}
    feats = rng.sample(spec_cat["features"], k=min(3, len(spec_cat["features"])))
    material = rng.choice(spec_cat["material"])
    spec = {"title": title, "category": cat, "material": material,
            "dimensions": dims, "features": feats,
            "extra": {"colour": rng.choice(["slate", "sand", "olive", "ink", "clay"])}}

    dim_text = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in sorted(dims.items()))
    description = (
        f"{rng.choice(OPENERS).format(name=title.lower())}. "
        f"Made from {material}, it features {', '.join(feats[:-1])} and {feats[-1]}. "
        f"Measures {dim_text}. "
        f"Finished in {spec['extra']['colour']}. {rng.choice(CLOSERS)}")
    return {"item_id": f"synth-{item_no:05d}", "spec": spec,
            "description": " ".join(description.split())}


def write_image(path: str, label: str, seed: int) -> None:
    """Deterministic placeholder product photo (coloured panel + name)."""
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    img = Image.new("RGB", (256, 256),
                    (rng.randint(120, 220), rng.randint(120, 220), rng.randint(120, 220)))
    d = ImageDraw.Draw(img)
    d.rectangle([48, 48, 208, 208], fill=(rng.randint(30, 90),) * 3)
    d.text((12, 12), label[:28], fill=(255, 255, 255))
    img.save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "generated"))
    ap.add_argument("--n", type=int, default=3200)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-images", dest="images", action="store_false",
                    help="write manifest.csv without rendering the placeholder PNGs "
                         "(the manifest still lists their paths, so train_projection "
                         "needs them unless you also pass --no-images there)")
    ap.set_defaults(images=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)

    items = [make_item(rng, i) for i in range(args.n)]
    n_val = max(1, int(args.n * args.val_fraction))

    for name, split in (("train", items[n_val:]), ("val", items[:n_val])):
        with open(os.path.join(args.out, f"{name}.jsonl"), "w") as fh:
            for it in split:
                fh.write(json.dumps(it) + "\n")

    with open(os.path.join(args.out, "manifest.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "image_path", "description"])
        for idx, it in enumerate(items):
            img_rel = f"images/{it['item_id']}.png"
            if args.images:
                write_image(os.path.join(args.out, img_rel),
                            it["spec"]["title"], args.seed + idx)
            w.writerow([it["item_id"], img_rel, it["description"]])

    with open(os.path.join(args.out, "qrels.jsonl"), "w") as fh:
        for it in items[: max(20, n_val)]:
            feats = it["spec"]["features"][0]
            fh.write(json.dumps({
                "question": f"Which item features {feats}?",
                "item_id": it["item_id"]}) + "\n")

    print(json.dumps({"out": args.out, "items": args.n,
                      "val": n_val, "seed": args.seed}))


if __name__ == "__main__":
    main()
