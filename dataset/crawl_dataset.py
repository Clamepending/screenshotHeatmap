#!/usr/bin/env python3
"""
BFS crawler that builds a dataset of webpage screenshots + interactive element annotations.

Starts with seed URLs from sites.txt, then follows links in BFS order up to a configurable depth.
Uses content hashing to skip duplicate pages (same visual content).

Creates COCO-format annotations suitable for ML training.

Usage:
    uv run python dataset/crawl_dataset.py
    uv run python dataset/crawl_dataset.py --depth 1
    uv run python dataset/crawl_dataset.py --depth 3 --limit 50
    uv run python dataset/crawl_dataset.py --headed
"""

import hashlib
import io
import json
import sys
from collections import deque
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from playwright.sync_api import sync_playwright
from PIL import Image

from interactive_extractor import extract_interactives

DATASET_DIR = Path(__file__).resolve().parent
SITES_FILE = DATASET_DIR / "sites.txt"
IMAGES_DIR = DATASET_DIR / "images"
ANNOTATIONS_FILE = DATASET_DIR / "annotations.json"

VIEWPORT = {"width": 1280, "height": 720}
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Tracking params to strip for URL dedup
TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "fbclid", "gclid", "ref"}


def normalize_url(url: str, base: str | None = None) -> str:
    """Normalize URL for deduplication: resolve, strip fragment and tracking params."""
    if base:
        url = urljoin(base, url)
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return ""
    # Strip fragment
    # Filter tracking params from query
    if parsed.query:
        params = parse_qs(parsed.query, keep_blank_values=True)
        filtered = {k: v for k, v in params.items() if k.lower() not in TRACKING_PARAMS and not k.lower().startswith("utm_")}
        query = urlencode(filtered, doseq=True)
    else:
        query = ""
    path = parsed.path.rstrip("/") or "/"
    return urlunparse((parsed.scheme, parsed.netloc.lower(), path, "", query, ""))


def extract_links(page, current_url: str, same_domain: bool) -> list[str]:
    """Extract crawlable links from the page."""
    current_domain = urlparse(current_url).netloc.lower()
    links = []

    try:
        locator = page.locator("a[href]")
        count = locator.count()
    except Exception:
        return links

    for i in range(count):
        try:
            el = locator.nth(i)
            href = el.get_attribute("href")
            if not href or href.startswith("#") or href.startswith("javascript:") or href.startswith("mailto:"):
                continue
            abs_url = urljoin(current_url, href)
            norm = normalize_url(abs_url)
            if not norm:
                continue
            if same_domain and urlparse(norm).netloc.lower() != current_domain:
                continue
            links.append(norm)
        except Exception:
            continue

    return list(dict.fromkeys(links))  # dedupe preserving order


def content_hash(screenshot_bytes: bytes) -> str:
    """Hash screenshot bytes to detect duplicate page content."""
    return hashlib.sha256(screenshot_bytes).hexdigest()


def load_sites() -> list[str]:
    """Load seed URLs from sites.txt."""
    if not SITES_FILE.exists():
        raise FileNotFoundError(f"Create {SITES_FILE} with one URL per line")
    urls = []
    for line in SITES_FILE.read_text().strip().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            if not line.startswith(("http://", "https://")):
                line = "https://" + line
            urls.append(normalize_url(line))
    return [u for u in urls if u]


def crawl(depth: int, limit: int | None, headed: bool, same_domain: bool) -> None:
    """BFS crawl URLs and build dataset."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    coco = {
        "info": {"description": "Webpage interactive elements dataset (BFS crawl)", "version": "1.0"},
        "images": [],
        "annotations": [],
        "categories": [],
    }

    label_to_id: dict[str, int] = {}
    image_id = 0
    ann_id = 0

    seen_urls: set[str] = set()
    seen_content_hashes: set[str] = set()

    # BFS: (url, depth)
    seed_urls = load_sites()
    queue: deque[tuple[str, int]] = deque((u, 0) for u in seed_urls)
    seen_urls.update(seed_urls)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=not headed,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(viewport=VIEWPORT, user_agent=USER_AGENT)
        page = context.new_page()

        while queue and (limit is None or image_id < limit):
            url, d = queue.popleft()

            if d > depth:
                continue

            print(f"[depth={d}] {url[:80]}{'...' if len(url) > 80 else ''}")

            try:
                page.goto(url, wait_until="load", timeout=30000)
                page.wait_for_timeout(2000)

                # Resolve redirects
                final_url = page.url
                norm_final = normalize_url(final_url)
                if norm_final != url and norm_final in seen_urls:
                    print(f"  -> Redirect to already-seen URL, skipping")
                    continue

                screenshot_bytes = page.screenshot(type="png")
                ch = content_hash(screenshot_bytes)
                if ch in seen_content_hashes:
                    print(f"  -> Duplicate content (hash match), skipping")
                    # Still extract links for BFS
                    if d < depth:
                        for link in extract_links(page, final_url, same_domain):
                            if link not in seen_urls:
                                seen_urls.add(link)
                                queue.append((link, d + 1))
                    continue

                seen_content_hashes.add(ch)

                image = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")
                w, h = image.size

                elements = extract_interactives(
                    page, VIEWPORT["width"], VIEWPORT["height"], limit=None
                )

                if not elements:
                    print(f"  -> No elements found, skipping")
                    if d < depth:
                        for link in extract_links(page, final_url, same_domain):
                            if link not in seen_urls:
                                seen_urls.add(link)
                                queue.append((link, d + 1))
                    continue

                # Save image
                img_filename = f"{image_id:05d}.png"
                image.save(IMAGES_DIR / img_filename)

                coco["images"].append({
                    "id": image_id,
                    "file_name": img_filename,
                    "width": w,
                    "height": h,
                    "url": final_url,
                    "depth": d,
                })

                for el in elements:
                    b = el["box"]
                    x, y = float(b["x"]), float(b["y"])
                    bw, bh = float(b["width"]), float(b["height"])
                    label = el["name"]

                    if label not in label_to_id:
                        label_to_id[label] = len(label_to_id)
                        coco["categories"].append({"id": label_to_id[label], "name": label})

                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": label_to_id[label],
                        "bbox": [round(x, 1), round(y, 1), round(bw, 1), round(bh, 1)],
                        "area": round(bw * bh, 1),
                        "label": label,
                    })
                    ann_id += 1

                print(f"  -> {len(elements)} elements, saved {img_filename}")
                image_id += 1

                # Add outlinks for BFS
                if d < depth:
                    for link in extract_links(page, final_url, same_domain):
                        if link not in seen_urls:
                            seen_urls.add(link)
                            queue.append((link, d + 1))

            except Exception as e:
                print(f"  -> Error: {e}")

        browser.close()

    coco["categories"] = [{"id": v, "name": k} for k, v in sorted(label_to_id.items(), key=lambda x: x[1])]
    ANNOTATIONS_FILE.write_text(json.dumps(coco, indent=2))
    print(f"\nSaved {len(coco['images'])} images, {len(coco['annotations'])} annotations")
    print(f"Output: {IMAGES_DIR}/, {ANNOTATIONS_FILE}")


def main():
    args = sys.argv[1:]
    headed = "--headed" in args
    args = [a for a in args if a != "--headed"]

    cross_domain = "--cross-domain" in args
    args = [a for a in args if a != "--cross-domain"]

    depth = 2
    if "--depth" in args:
        idx = args.index("--depth")
        if idx + 1 < len(args) and args[idx + 1].isdigit():
            depth = int(args[idx + 1])
            args = [a for i, a in enumerate(args) if i not in (idx, idx + 1)]
        else:
            args = [a for a in args if a != "--depth"]

    limit = None
    if "--limit" in args:
        idx = args.index("--limit")
        if idx + 1 < len(args) and args[idx + 1].isdigit():
            limit = int(args[idx + 1])
            args = [a for i, a in enumerate(args) if i not in (idx, idx + 1)]
        else:
            args = [a for a in args if a != "--limit"]

    print(f"BFS crawl: depth={depth}, same_domain={not cross_domain}, limit={limit}")

    crawl(depth=depth, limit=limit, headed=headed, same_domain=not cross_domain)


if __name__ == "__main__":
    main()
