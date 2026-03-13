#!/usr/bin/env python3
"""
Visualize interactive elements from a webpage using Playwright.

Takes a URL, extracts bounding boxes of interactive elements (links, buttons,
inputs, etc.), draws numbered boxes on a screenshot, and prints a numbered list.

Usage:
    python visualize_interactives.py <url>
    python visualize_interactives.py https://example.com
    python visualize_interactives.py --headed https://www.amazon.com  # visible browser (helps with sites that block headless)
"""

import io
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright
from PIL import Image, ImageDraw, ImageFont

from interactive_extractor import extract_interactives




def draw_boxes_on_image(image: Image.Image, elements: list[dict]) -> Image.Image:
    """Draw numbered bounding boxes on the screenshot."""
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except OSError:
        font = ImageFont.load_default()
        small_font = font

    color = (255, 50, 50)  # Red
    for idx, el in enumerate(elements, start=1):
        b = el["box"]
        x, y = int(b["x"]), int(b["y"])
        w, h = int(b["width"]), int(b["height"])

        # Only draw if box is in viewport (screenshot bounds)
        if x + w < 0 or y + h < 0 or x > img.width or y > img.height:
            continue

        # Clamp to image
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img.width, x + w)
        y2 = min(img.height, y + h)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw number badge
        label = str(idx)
        draw.rectangle([x1, y1 - 24, x1 + 28, y1], fill=color)
        draw.text((x1 + 4, y1 - 22), label, fill="white", font=small_font)

    return img


def main():
    args = sys.argv[1:]
    headed = "--headed" in args
    if headed:
        args = [a for a in args if a != "--headed"]

    limit = 80
    if "--limit" in args:
        idx = args.index("--limit")
        if idx + 1 < len(args) and args[idx + 1].isdigit():
            limit = int(args[idx + 1])
            args = [a for i, a in enumerate(args) if i not in (idx, idx + 1)]
        else:
            args = [a for a in args if a != "--limit"]

    if not args:
        print("Usage: python visualize_interactives.py [--headed] [--limit N] <url>")
        print("Example: python visualize_interactives.py https://example.com")
        print("  --headed  Show browser window (helps with sites that block headless)")
        print("  --limit N  Max elements to show (default 80)")
        sys.exit(1)

    url = args[0]
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    viewport = {"width": 1280, "height": 720}
    output_path = Path("screenshot_with_boxes.png")

    print(f"Loading {url}...")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=not headed,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(
            viewport=viewport,
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page = context.new_page()
        page.goto(url, wait_until="load", timeout=30000)

        # Let JS-heavy pages render
        page.wait_for_timeout(3000)

        # Take screenshot
        screenshot_bytes = page.screenshot(type="png")
        image = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")

        # Extract interactive elements (viewport only, sorted top-to-bottom)
        elements = extract_interactives(page, viewport["width"], viewport["height"], limit=None)

        # Limit for readable output (keep first N by position)
        total = len(elements)
        if total > limit:
            elements = elements[:limit]
            print(f"\nFound {total} interactive elements (showing top {limit}).\n")
        else:
            print(f"\nFound {total} interactive elements.\n")

        # Draw boxes
        annotated = draw_boxes_on_image(image, elements)

        # Save
        annotated.save(output_path)
        print(f"Saved: {output_path}\n")

        # Print numbered list
        print("--- Numbered list ---")
        for idx, el in enumerate(elements, start=1):
            print(f"  {idx}. {el['name']}")

        browser.close()

    print(f"\nDone. Open {output_path} to view the screenshot with bounding boxes.")


if __name__ == "__main__":
    main()
