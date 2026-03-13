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

import base64
import hashlib
import io
import json
import random
import sys
import time
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
STATE_FILE = DATASET_DIR / "crawl_state.json"

VIEWPORT = {"width": 1280, "height": 720}
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Tracking params to strip for URL dedup
TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "fbclid", "gclid", "ref"}

# URL patterns to skip (login/sign-in pages - usually not useful for UI element dataset)
SKIP_URL_PATTERNS = (
    "/login", "/signin", "/sign-in", "/sign_up", "/signup", "/register",
    "/auth", "/oauth", "/account/login", "/accounts/login",
    "login.live.com", "login.microsoftonline.com", "accounts.google.com",
    "facebook.com/login", "twitter.com/i/flow/login", "github.com/login",
    "outlook.live.com", "outlook.com/", "onedrive.live.com",
    "microsoft365.com", "office.com/", "portal.azure.com",
    "live.com", "signup.live.com",
)


def is_login_url(url: str) -> bool:
    """Return True if URL is a login/sign-in page."""
    url_lower = url.lower()
    return any(p in url_lower for p in SKIP_URL_PATTERNS)


def resolve_redirect_url(url: str) -> str:
    """Resolve Bing/Google-style redirect URLs to the actual destination."""
    parsed = urlparse(url)
    if "bing.com" in parsed.netloc and "/ck/a" in parsed.path:
        params = parse_qs(parsed.query)
        if "u" in params:
            encoded = params["u"][0]
            try:
                # Bing uses "a1" prefix before base64-encoded URL
                if encoded.startswith("a1") and len(encoded) > 2:
                    encoded = encoded[2:]
                decoded = base64.b64decode(encoded + "==").decode("utf-8", errors="ignore")
                if decoded.startswith(("http://", "https://")):
                    return decoded
            except Exception:
                pass
    if "google.com" in parsed.netloc and "/url" in parsed.path:
        params = parse_qs(parsed.query)
        if "q" in params:
            return params["q"][0]
        if "url" in params:
            return params["url"][0]
    return url


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
    """Extract crawlable links from the page. Skips links that don't change the page."""
    current_domain = urlparse(current_url).netloc.lower()
    current_norm = normalize_url(current_url)
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
            # Resolve search engine redirect URLs (Bing /ck/a, Google /url) to actual destination
            abs_url = resolve_redirect_url(abs_url)
            norm = normalize_url(abs_url)
            if not norm:
                continue
            # Skip same-page links (would not change the page)
            if norm == current_norm:
                continue
            # Skip login/sign-in pages
            if is_login_url(norm):
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


def save_state(
    queue: deque,
    seen_urls: set,
    seen_content_hashes: set,
    coco: dict,
    image_id: int,
    ann_id: int,
    label_to_id: dict,
    depth: int,
    same_domain: bool,
) -> None:
    """Persist crawler state for resume."""
    state = {
        "queue": list(queue),
        "seen_urls": list(seen_urls),
        "seen_content_hashes": list(seen_content_hashes),
        "coco": coco,
        "image_id": image_id,
        "ann_id": ann_id,
        "label_to_id": label_to_id,
        "depth": depth,
        "same_domain": same_domain,
    }
    STATE_FILE.write_text(json.dumps(state, indent=2))
    ANNOTATIONS_FILE.write_text(json.dumps(coco, indent=2))
    print(f"\nState saved to {STATE_FILE} ({image_id} images)")


def load_state() -> dict | None:
    """Load crawler state if it exists."""
    if not STATE_FILE.exists():
        return None
    return json.loads(STATE_FILE.read_text())


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


def crawl(
    depth: int,
    max_websites: int | None,
    headed: bool,
    same_domain: bool,
    resume: bool,
) -> None:
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
    queue: deque[tuple[str, int]] = deque()

    if resume and (state := load_state()):
        queue = deque(tuple(x) for x in state["queue"])
        seen_urls = set(state["seen_urls"])
        seen_content_hashes = set(state["seen_content_hashes"])
        coco = state["coco"]
        image_id = state["image_id"]
        ann_id = state["ann_id"]
        label_to_id = state["label_to_id"]
        depth = state["depth"]
        same_domain = state["same_domain"]
        print(f"Resumed: {image_id} images, {len(queue)} URLs in queue")
    else:
        seed_urls = load_sites()
        queue = deque((u, 0) for u in seed_urls)
        seen_urls.update(seed_urls)

    with sync_playwright() as p:
        launch_args = [
            "--disable-blink-features=AutomationControlled",
            "--disable-dev-shm-usage",
            "--no-sandbox",
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ]
        browser = p.chromium.launch(
            headless=not headed,
            args=launch_args,
        )
        context = browser.new_context(
            viewport=VIEWPORT,
            user_agent=USER_AGENT,
            locale="en-US",
            timezone_id="America/Los_Angeles",
        )
        # Reduce automation detection
        context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        """)
        page = context.new_page()

        while queue and (max_websites is None or image_id < max_websites):
            url, d = queue.popleft()

            if d > depth:
                continue

            # Skip login pages (may have snuck in from previous runs)
            if is_login_url(url):
                print(f"[depth={d}] {url[:60]}... -> Skipping (login page)")
                continue

            print(f"[depth={d}] {url[:80]}{'...' if len(url) > 80 else ''}")

            # Random delay between requests (reduces rate limiting / captcha triggers)
            if image_id > 0:
                delay = random.uniform(2, 5)
                time.sleep(delay)

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

                # Check max_websites - save state and exit
                if max_websites is not None and image_id >= max_websites:
                    save_state(
                        queue, seen_urls, seen_content_hashes,
                        coco, image_id, ann_id, label_to_id, depth, same_domain,
                    )
                    print(f"\nReached max_websites={max_websites}. Run with --resume to continue.")
                    return

            except Exception as e:
                print(f"  -> Error: {e}")

        browser.close()

    coco["categories"] = [{"id": v, "name": k} for k, v in sorted(label_to_id.items(), key=lambda x: x[1])]
    if STATE_FILE.exists():
        STATE_FILE.unlink()  # Clear state on normal completion
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

    max_websites = None
    for flag in ("--max-websites", "--limit"):
        if flag in args:
            idx = args.index(flag)
            if idx + 1 < len(args) and args[idx + 1].isdigit():
                max_websites = int(args[idx + 1])
                args = [a for i, a in enumerate(args) if i not in (idx, idx + 1)]
            else:
                args = [a for a in args if a != flag]
            break

    resume = "--resume" in args
    args = [a for a in args if a != "--resume"]

    print(f"BFS crawl: depth={depth}, same_domain={not cross_domain}, max_websites={max_websites}, resume={resume}")

    crawl(depth=depth, max_websites=max_websites, headed=headed, same_domain=not cross_domain, resume=resume)


if __name__ == "__main__":
    main()
