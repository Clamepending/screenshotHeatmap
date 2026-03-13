# Webpage Interactive Elements Dataset

Dataset for training ML models to detect and label interactive UI elements (buttons, links, inputs, etc.) in webpage screenshots.

## Structure

```
dataset/
├── sites.txt           # URLs to crawl (one per line)
├── images/             # Screenshots: {id:05d}.png
├── annotations.json    # COCO-format annotations
├── crawl_dataset.py    # Crawler script
└── README.md
```

## Annotation Format (COCO)

The `annotations.json` follows the [COCO Object Detection format](https://cocoodataset.org/#format-data), widely supported by PyTorch, TensorFlow, Detectron2, YOLO, etc.

```json
{
  "info": { "description": "Webpage interactive elements" },
  "images": [
    { "id": 0, "file_name": "00000.png", "width": 1280, "height": 720, "url": "https://..." }
  ],
  "annotations": [
    { "id": 0, "image_id": 0, "category_id": 5, "bbox": [x, y, w, h], "area": w*h }
  ],
  "categories": [
    { "id": 0, "name": "Search" },
    { "id": 1, "name": "Sign in" }
  ]
}
```

- **bbox**: `[x, y, width, height]` in pixels (top-left origin)
- **category_id**: Maps to `categories` for the interactive element label
- **label**: Raw string (convenience; same as `categories[category_id].name`)

## Load for ML

```python
import json
from PIL import Image

with open("dataset/annotations.json") as f:
    data = json.load(f)

# Map image_id -> annotations
from collections import defaultdict
ann_by_image = defaultdict(list)
for ann in data["annotations"]:
    ann_by_image[ann["image_id"]].append(ann)

# Iterate
for img in data["images"]:
    pil_img = Image.open(f"dataset/images/{img['file_name']}")
    boxes = [(a["bbox"], a["label"]) for a in ann_by_image[img["id"]]]
    # boxes = [([x,y,w,h], "Search"), ...]
```

## Training

- **Object detection**: Use with Detectron2, MMDetection, Ultralytics YOLO, etc.
- **Grounding/Open-vocabulary**: Use `categories[].name` as text prompts for models like Grounding DINO, OWL-ViT
- **Custom**: Load with `json.load()` and iterate `images` + `annotations`

## Crawling

```bash
uv run python dataset/crawl_dataset.py
uv run python dataset/crawl_dataset.py --limit 5    # Crawl first 5 sites only
uv run python dataset/crawl_dataset.py --headed    # Visible browser (for sites that block headless)
```
