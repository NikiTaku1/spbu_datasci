"""
Utility functions: dataset loader for inference, crop-and-aggregate, angle helpers, visualization.
"""
from PIL import Image
import torch
import torchvision.transforms as T
from typing import List
import random

ROTATIONS = [0, 90, 180, 270]


def load_model_state(model, path, device="cpu"):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def sample_crops(img: Image.Image, crop_size=(224, 224), grid=3) -> List[Image.Image]:
    """Return list of crops sampled in a grid (center + corners + edges)"""
    w, h = img.size
    cw, ch = crop_size
    crops = []

    # center crop
    cx = max((w - cw) // 2, 0)
    cy = max((h - ch) // 2, 0)
    crops.append(img.crop((cx, cy, cx+cw, cy+ch)))

    # grid positions
    xs = [0, (w-cw)//2, max(w-cw,0)] if w > cw else [0]
    ys = [0, (h-ch)//2, max(h-ch,0)] if h > ch else [0]
    for x in xs:
        for y in ys:
            c = img.crop((x, y, min(x+cw,w), min(y+ch,h)))
            if c.size != (cw, ch):
                c = c.resize((cw, ch))
            crops.append(c)

    # random crops
    for _ in range(4):
        if w > cw and h > ch:
            rx = random.randint(0, w-cw)
            ry = random.randint(0, h-ch)
            c = img.crop((rx, ry, rx+cw, ry+ch))
            crops.append(c)
    # deduplicate by size checks
    return crops


def predict_orientation_aggregate(img: Image.Image, model, device="cpu", transform=None):
    """Perform crop-and-aggregate inference over 4 rotations.
    For each rotation: sample crops, predict logits, average logits across crops.
    Return best rotation angle (one of ROTATIONS).
    """
    if transform is None:
        transform = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

    model.eval()
    agg_scores = []
    for angle in ROTATIONS:
        rimg = img.rotate(angle, expand=True, fillcolor="#fafafa")
        crops = sample_crops(rimg, crop_size=(224,224))
        logits_sum = None
        with torch.no_grad():
            for c in crops:
                x = transform(c).unsqueeze(0).to(device)
                out = model(x)  # logits
                if logits_sum is None:
                    logits_sum = out.squeeze(0).cpu()
                else:
                    logits_sum += out.squeeze(0).cpu()
        # average
        logits_mean = logits_sum / len(crops)
        # probability assigned to class '0' (meaning 'upright')
        prob0 = torch.softmax(logits_mean, dim=0)[0].item()
        agg_scores.append(prob0)
    best_idx = int(torch.tensor(agg_scores).argmax().item())
    return ROTATIONS[best_idx]


def correct_orientation_img(img_path, model, device="cpu"):
    img = Image.open(img_path).convert("RGB")
    angle = predict_orientation_aggregate(img, model, device=device)
    corrected = img.rotate(-angle, expand=True, fillcolor="#fafafa")
    return angle, corrected