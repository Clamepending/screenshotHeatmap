import gradio as gr
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

matplotlib.use("Agg")

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
# GroundingDINO may not support MPS; use CPU for detection models if needed
DET_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
print(f"Device: {DEVICE}, Detection: {DET_DEVICE}", flush=True)

print("Loading SigLIP...", flush=True)
from transformers import SiglipModel, SiglipProcessor
siglip_model = SiglipModel.from_pretrained("google/siglip-base-patch16-384").to(DEVICE)
siglip_model.eval()
siglip_processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-384")

print("Loading CLIPSeg...", flush=True)
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(DEVICE)
clipseg_model.eval()

print("Loading GEM (CLIP ViT-B/16)...", flush=True)
import gem
gem_preprocess = gem.get_gem_img_transform()
gem_model = gem.create_gem_model(model_name="ViT-B/16", pretrained="openai", device=DEVICE)

print("Loading GroundingDINO...", flush=True)
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
grounding_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
    "IDEA-Research/grounding-dino-tiny"
).to(DET_DEVICE)
grounding_model.eval()

print("Loading OWL-ViT...", flush=True)
from transformers import pipeline
owlvit_detector = pipeline(
    model="google/owlvit-base-patch32",
    task="zero-shot-object-detection",
    device=0 if torch.cuda.is_available() else -1,
)

print("All models loaded.", flush=True)

# ---------------------------------------------------------------------------
# Heatmap helpers
# ---------------------------------------------------------------------------
SIGLIP_GRID = 384 // 16  # 24


def _overlay(image_rgb: Image.Image, heatmap_2d: np.ndarray, title: str,
             blur_sigma: float = 2.0) -> Image.Image:
    """Normalize, blur, and overlay a heatmap on the original image."""
    h = gaussian_filter(heatmap_2d.astype(np.float32), sigma=blur_sigma)
    h = (h - h.min()) / (h.max() - h.min() + 1e-8)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(image_rgb)
    resized = np.array(
        Image.fromarray((h * 255).astype(np.uint8)).resize(
            image_rgb.size, resample=Image.BICUBIC
        )
    ) / 255.0
    ax.imshow(resized, cmap="jet", alpha=0.5,
              extent=[0, image_rgb.width, image_rgb.height, 0])
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    result = Image.frombuffer("RGBA", fig.canvas.get_width_height(), buf).convert("RGB")
    plt.close(fig)
    return result


# ---------------------------------------------------------------------------
# Method 1: SigLIP GradCAM
# ---------------------------------------------------------------------------
def siglip_gradcam(image_rgb: Image.Image, text_query: str, blur_sigma: float) -> Image.Image:
    inputs = siglip_processor(text=[text_query], images=[image_rgb],
                              return_tensors="pt", padding="max_length")
    pixel_values = inputs["pixel_values"].to(DEVICE)
    input_ids = inputs["input_ids"].to(DEVICE)

    activations, gradients = {}, {}

    def fwd_hook(m, i, o):
        activations["v"] = o

    def bwd_hook(m, gi, go):
        gradients["v"] = go[0]

    h1 = siglip_model.vision_model.post_layernorm.register_forward_hook(fwd_hook)
    h2 = siglip_model.vision_model.post_layernorm.register_full_backward_hook(bwd_hook)

    for p in siglip_model.parameters():
        p.requires_grad_(True)
    siglip_model.zero_grad()

    out = siglip_model(pixel_values=pixel_values, input_ids=input_ids)
    out.logits_per_image[0, 0].backward()

    h1.remove()
    h2.remove()
    for p in siglip_model.parameters():
        p.requires_grad_(False)

    act = activations["v"][0].detach().cpu()
    grad = gradients["v"][0].detach().cpu()
    weights = grad.mean(dim=0)
    cam = torch.relu((act * weights).sum(dim=-1)).numpy()
    cam_grid = cam.reshape(SIGLIP_GRID, SIGLIP_GRID)

    return _overlay(image_rgb, cam_grid, "SigLIP GradCAM", blur_sigma)


# ---------------------------------------------------------------------------
# Method 2: CLIPSeg
# ---------------------------------------------------------------------------
def clipseg_heatmap(image_rgb: Image.Image, text_query: str, blur_sigma: float) -> Image.Image:
    inputs = clipseg_processor(text=[text_query], images=[image_rgb],
                               return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = clipseg_model(**inputs)

    heatmap = torch.sigmoid(outputs.logits[0]).cpu().numpy()
    return _overlay(image_rgb, heatmap, "CLIPSeg", blur_sigma)


# ---------------------------------------------------------------------------
# Bounding box overlay helper
# ---------------------------------------------------------------------------
def _draw_boxes(image: Image.Image, detections: list, title: str, color: str = "lime") -> Image.Image:
    """Draw bounding boxes and labels on image. detections: list of {box, score, label}."""
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except OSError:
        font = ImageFont.load_default()

    color_map = {"lime": (0, 255, 0), "red": (255, 0, 0), "cyan": (0, 255, 255)}
    rgb = color_map.get(color, (0, 255, 0))

    for d in detections:
        box = d.get("box") or d.get("boxes")
        if box is None:
            continue
        if isinstance(box, dict):
            xmin = box.get("xmin", box.get("x_min", 0))
            ymin = box.get("ymin", box.get("y_min", 0))
            xmax = box.get("xmax", box.get("x_max", 0))
            ymax = box.get("ymax", box.get("y_max", 0))
        else:
            if hasattr(box, "tolist"):
                box = box.tolist()
            xmin, ymin, xmax, ymax = float(box[0]), float(box[1]), float(box[2]), float(box[3])

        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        score = d.get("score", d.get("scores", 0))
        if hasattr(score, "item"):
            score = score.item()
        label = d.get("label", d.get("text_labels", "?"))
        if isinstance(label, (list, tuple)):
            label = label[0] if label else "?"
        text = f"{label} {score:.2f}"

        draw.rectangle([xmin, ymin, xmax, ymax], outline=rgb, width=5)
        draw.text((xmin, max(0, ymin - 20)), text, fill=rgb, font=font)

    draw.text((10, 10), f"{title} ({len(detections)} detections)", fill=rgb, font=font)
    return img


# ---------------------------------------------------------------------------
# Method 3: GEM
# ---------------------------------------------------------------------------
def gem_heatmap(image_rgb: Image.Image, text_query: str, blur_sigma: float) -> Image.Image:
    img_tensor = gem_preprocess(image_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = gem_model(img_tensor, [text_query])  # [1, 1, H, W]

    heatmap = logits[0, 0].cpu().numpy()
    return _overlay(image_rgb, heatmap, "GEM (ViT-B/16)", blur_sigma)


# ---------------------------------------------------------------------------
# Detection: GroundingDINO
# ---------------------------------------------------------------------------
def grounding_dino_detect(image_rgb: Image.Image, text_query: str) -> Image.Image:
    text_labels = [[text_query.strip()]]
    inputs = grounding_processor(images=image_rgb, text=text_labels, return_tensors="pt")
    inputs = {k: v.to(DET_DEVICE) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    target_sizes = [(image_rgb.height, image_rgb.width)]
    results = grounding_processor.post_process_grounded_object_detection(
        outputs, inputs["input_ids"], threshold=0.3, text_threshold=0.25, target_sizes=target_sizes
    )
    result = results[0]
    detections = []
    for box, score, label in zip(result["boxes"], result["scores"], result["text_labels"]):
        detections.append({"box": box.cpu().tolist(), "score": score.item(), "label": label})
    return _draw_boxes(image_rgb, detections, "GroundingDINO", color="lime")


# ---------------------------------------------------------------------------
# Detection: OWL-ViT
# ---------------------------------------------------------------------------
def owlvit_detect(image_rgb: Image.Image, text_query: str) -> Image.Image:
    candidate_labels = [text_query.strip()]
    raw = owlvit_detector(image_rgb, candidate_labels=candidate_labels, threshold=0.02)
    detections = [{"box": r["box"], "score": r["score"], "label": r["label"]} for r in raw]
    return _draw_boxes(image_rgb, detections, "OWL-ViT", color="cyan")


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------
def generate_all(image: Image.Image, text_query: str, blur_sigma: float):
    if image is None or not text_query.strip():
        return None, None, None, None, None

    image_rgb = image.convert("RGB")
    return (
        siglip_gradcam(image_rgb, text_query, blur_sigma),
        clipseg_heatmap(image_rgb, text_query, blur_sigma),
        gem_heatmap(image_rgb, text_query, blur_sigma),
        grounding_dino_detect(image_rgb, text_query),
        owlvit_detect(image_rgb, text_query),
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Text→Image Heatmap & Detection Comparison") as demo:
    gr.Markdown(
        "# Text → Image Heatmap & Detection Comparison\n"
        "Upload an image and enter a text query. Compare **heatmaps** (SigLIP GradCAM, CLIPSeg, GEM) "
        "and **detection** (GroundingDINO, OWL-ViT) side-by-side."
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Image")
            text_input = gr.Textbox(label="Text Query",
                                    placeholder="e.g. search bar, button, logo...")
            blur_slider = gr.Slider(minimum=0, maximum=10, value=2,
                                    step=0.5, label="Blur Sigma (heatmaps only)")
            run_btn = gr.Button("Generate All", variant="primary")

    gr.Markdown("### Heatmaps")
    with gr.Row():
        out_siglip = gr.Image(type="pil", label="SigLIP GradCAM")
        out_clipseg = gr.Image(type="pil", label="CLIPSeg")
        out_gem = gr.Image(type="pil", label="GEM (ViT-B/16)")

    gr.Markdown("### Detection (bounding boxes)")
    with gr.Row():
        out_grounding = gr.Image(type="pil", label="GroundingDINO")
        out_owlvit = gr.Image(type="pil", label="OWL-ViT")

    run_btn.click(fn=generate_all,
                  inputs=[image_input, text_input, blur_slider],
                  outputs=[out_siglip, out_clipseg, out_gem, out_grounding, out_owlvit])

demo.launch(share=False, show_error=True)
