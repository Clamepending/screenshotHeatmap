"""
Shared logic for extracting interactive elements from webpages.
Used by visualize_interactives.py and dataset/crawl_dataset.py.
"""

import re
from urllib.parse import urlparse

# Selectors for interactive elements
INTERACTIVE_SELECTORS = [
    "a[href]",
    "button",
    "input:not([type='hidden'])",
    "select",
    "textarea",
    "[role='button']",
    "[role='link']",
    "[role='menuitem']",
    "[role='tab']",
]

NAME_MAX_LEN = 50
MIN_WIDTH = 20
MIN_HEIGHT = 20


def _clean_aria_label(label: str) -> str:
    """Strip keyboard shortcuts from aria-labels."""
    if not label:
        return label
    cleaned = re.sub(r",\s*(shift|option|ctrl|control|alt|cmd|command|meta)\s*,.*$", "", label, flags=re.I)
    cleaned = re.sub(r",\s*[a-z]\s*$", "", cleaned, flags=re.I)
    return cleaned.strip()


def get_element_name(element) -> str:
    """Extract a human-readable name for an element."""
    attrs = element.get_attribute
    text = element.inner_text().strip() or ""

    if label := attrs("aria-label"):
        return _clean_aria_label(label)[:NAME_MAX_LEN]

    tag = element.evaluate("el => el.tagName.toLowerCase()")
    if tag == "input":
        if placeholder := attrs("placeholder"):
            return placeholder.strip()[:NAME_MAX_LEN]
        if name := attrs("name"):
            return name.strip()[:NAME_MAX_LEN]
        if input_type := attrs("type"):
            return f"input ({input_type})"
        return "input"

    if title := attrs("title"):
        return title.strip()[:NAME_MAX_LEN]
    if text:
        return text.replace("\n", " ").strip()[:NAME_MAX_LEN]

    if tag == "a":
        img_alt = element.evaluate(
            "el => el.querySelector('img[alt]')?.getAttribute('alt')?.trim() || null"
        )
        if img_alt:
            return img_alt[:NAME_MAX_LEN]
        href = attrs("href")
        if href and not href.startswith("#") and not href.startswith("javascript:"):
            try:
                parsed = urlparse(href)
                path = parsed.path.strip("/")
                parts = [p for p in path.split("/") if p]
                if not parts or (len(parts) == 1 and parts[0] in ("", "index", "home")):
                    return "home"
                if len(parts) >= 2 and parts[0] in ("itm", "usr", "sch"):
                    slug = parts[1] if len(parts) > 1 else parts[0]
                    if parts[0] == "itm":
                        return "product" if re.match(r"^\d+$", slug) else slug[:NAME_MAX_LEN]
                    if parts[0] == "usr":
                        return f"user {slug}"[:NAME_MAX_LEN]
                last = parts[-1] if parts else ""
                if last and last not in ("ref", "s", "gp", "dp", "b") and not re.match(r"^[A-Z0-9]{10,}$", last):
                    return last.replace("-", " ").replace("_", " ")[:NAME_MAX_LEN]
            except Exception:
                pass
        return "link"

    role = attrs("role") or ""
    if role:
        return f"{tag} ({role})"[:NAME_MAX_LEN]
    return tag


def extract_interactives(page, viewport_width: int, viewport_height: int, limit: int | None = None) -> list[dict]:
    """Extract interactive elements with bounding boxes and names."""
    seen_boxes = set()
    results = []

    for selector in INTERACTIVE_SELECTORS:
        try:
            locator = page.locator(selector)
            count = locator.count()
        except Exception:
            continue

        for i in range(count):
            try:
                el = locator.nth(i)
                if not el.is_visible():
                    continue

                box = el.bounding_box()
                if not box:
                    continue

                w, h = box["width"], box["height"]
                if w < MIN_WIDTH or h < MIN_HEIGHT:
                    continue

                x, y = box["x"], box["y"]
                in_view = (
                    x + w > 0 and y + h > 0
                    and x < viewport_width and y < viewport_height
                )
                if not in_view:
                    continue

                key = (round(x), round(y), round(w), round(h))
                if key in seen_boxes:
                    continue
                seen_boxes.add(key)

                name = get_element_name(el)
                results.append({
                    "box": box,
                    "name": name or "(unnamed)",
                })
            except Exception:
                continue

    results.sort(key=lambda r: (round(r["box"]["y"] / 50) * 50, r["box"]["x"]))

    if limit is not None:
        results = results[:limit]

    return results
