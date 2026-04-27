#!/usr/bin/env python3
"""
Image Collector for Sri Lanka Tourism Locations.

Downloads images for all 80 locations in locations_metadata.csv using
Google Places API (New) as the primary source, with Wikimedia Commons
as a fallback, and generates the metadata.json required by the image
knowledge base (Phase 2).

Data Sources (in priority order):
    1. Google Places API (New) — uses GOOGLE_MAPS_API_KEY from .env
       - Text Search to find the place
       - Place Photos to download high-quality images
    2. Wikimedia Commons API — free fallback, no key needed

Usage:
    # Full collection (all 80 locations, 5 images each):
    python3 scripts/collect_location_images.py

    # Fewer images per location (faster):
    python3 scripts/collect_location_images.py --per-location 3

    # Test with specific locations:
    python3 scripts/collect_location_images.py --locations "Sigiriya" "Galle Fort"

    # Resume after interruption (Ctrl+C safe):
    python3 scripts/collect_location_images.py --resume

    # Dry run (show what would be downloaded):
    python3 scripts/collect_location_images.py --dry-run
"""

import argparse
import csv
import json
import os
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# SSL Context (handle macOS certificate issues)
# ---------------------------------------------------------------------------
try:
    _ssl_context = ssl.create_default_context()
    urllib.request.urlopen(
        "https://places.googleapis.com", timeout=5, context=_ssl_context
    )
except Exception:
    _ssl_context = ssl._create_unverified_context()
    print("[SSL] Using unverified SSL context (macOS certificate issue)")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
CSV_PATH = PROJECT_ROOT / "data" / "locations_metadata.csv"
IMAGE_DIR = PROJECT_ROOT / "data" / "image_knowledge" / "images"
METADATA_PATH = PROJECT_ROOT / "data" / "image_knowledge" / "metadata.json"
PROGRESS_PATH = PROJECT_ROOT / "data" / "image_knowledge" / ".progress.json"
ENV_PATH = PROJECT_ROOT / ".env"


# ---------------------------------------------------------------------------
# Load API key from .env
# ---------------------------------------------------------------------------
def load_google_api_key() -> Optional[str]:
    """Load GOOGLE_MAPS_API_KEY from .env file or environment."""
    # Check environment first
    key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if key:
        return key

    # Try python-dotenv
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_PATH)
        return os.environ.get("GOOGLE_MAPS_API_KEY")
    except ImportError:
        pass

    # Manual .env parsing as fallback
    if ENV_PATH.exists():
        with open(ENV_PATH) as f:
            for line in f:
                line = line.strip()
                if line.startswith("GOOGLE_MAPS_API_KEY="):
                    return line.split("=", 1)[1].strip().strip("'\"")
    return None


# ============================================================================
# SOURCE 1: Google Places API (New) — Primary
# ============================================================================

PLACES_API_URL = "https://places.googleapis.com/v1"


def google_find_place(
    location_name: str,
    lat: float,
    lng: float,
    api_key: str,
) -> Optional[Dict]:
    """
    Find a place using Google Places Text Search (New API).

    Uses the location name + "Sri Lanka" as query, with a location bias
    around the known coordinates for accurate matching.

    Args:
        location_name: Name of the tourism location.
        lat: Known latitude.
        lng: Known longitude.
        api_key: Google Maps API key.

    Returns:
        Place dict with id, displayName, photos, or None if not found.
    """
    url = f"{PLACES_API_URL}/places:searchText"

    body = {
        "textQuery": f"{location_name} Sri Lanka",
        "languageCode": "en",
        "maxResultCount": 1,
        "locationBias": {
            "circle": {
                "center": {"latitude": lat, "longitude": lng},
                "radius": 5000.0,  # 5 km radius around known coordinates
            }
        },
    }

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": (
                "places.id,places.displayName,places.photos,"
                "places.formattedAddress,places.editorialSummary"
            ),
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=15, context=_ssl_context) as resp:
            result = json.loads(resp.read().decode())
    except Exception as e:
        print(f"    [Google] Place search failed: {e}")
        return None

    places = result.get("places", [])
    if not places:
        print(f"    [Google] No place found for '{location_name}'")
        return None

    return places[0]


def google_get_photo_url(
    photo_name: str,
    api_key: str,
    max_width: int = 1280,
) -> Optional[str]:
    """
    Get the actual image URL for a Google Places photo reference.

    Uses skipHttpRedirect=true to get the URL without downloading.

    Args:
        photo_name: The photo resource name from Places API.
        api_key: Google Maps API key.
        max_width: Maximum image width in pixels.

    Returns:
        Direct image URL string, or None on failure.
    """
    url = (
        f"{PLACES_API_URL}/{photo_name}/media"
        f"?maxWidthPx={max_width}&key={api_key}&skipHttpRedirect=true"
    )

    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=15, context=_ssl_context) as resp:
            data = json.loads(resp.read().decode())
            return data.get("photoUri")
    except Exception as e:
        print(f"    [Google] Photo URL fetch failed: {e}")
        return None


def search_google_places(
    location_name: str,
    lat: float,
    lng: float,
    api_key: str,
    limit: int = 5,
) -> List[Dict]:
    """
    Search Google Places for photos of a location.

    Pipeline:
        1. Text Search -> find the place by name + coordinates
        2. Extract photo references from the place result
        3. Resolve each photo reference to a direct image URL

    Args:
        location_name: Name of the location.
        lat: Known latitude from CSV.
        lng: Known longitude from CSV.
        api_key: Google Maps API key.
        limit: Max number of photos to collect.

    Returns:
        List of result dicts with url, description, source, etc.
    """
    # Step 1: Find the place
    place = google_find_place(location_name, lat, lng, api_key)
    if not place:
        return []

    photos = place.get("photos", [])
    if not photos:
        print(f"    [Google] Place found but no photos available")
        return []

    display_name = place.get("displayName", {}).get("text", location_name)
    editorial = place.get("editorialSummary", {}).get("text", "")
    address = place.get("formattedAddress", "")

    # Step 2: Resolve photo URLs (cap at limit)
    results = []
    for i, photo in enumerate(photos[:limit]):
        photo_name = photo.get("name", "")
        if not photo_name:
            continue

        # Get the direct image URL
        photo_url = google_get_photo_url(photo_name, api_key, max_width=1280)
        if not photo_url:
            continue

        # Build description from available metadata
        author = ""
        attributions = photo.get("authorAttributions", [])
        if attributions:
            author = attributions[0].get("displayName", "")

        description = editorial or f"{display_name} - {address}" if address else display_name
        if len(description) > 300:
            description = description[:297] + "..."

        results.append({
            "url": photo_url,
            "original_url": photo_url,
            "title": f"{display_name} ({i + 1})",
            "description": description,
            "source": "google_places",
            "width": photo.get("widthPx", 0),
            "height": photo.get("heightPx", 0),
            "photographer": author,
            "place_id": place.get("id", ""),
        })

        # Small delay between photo URL resolutions
        time.sleep(0.1)

    return results


# ============================================================================
# SOURCE 2: Wikimedia Commons API (Free fallback)
# ============================================================================

WIKI_API_URL = "https://commons.wikimedia.org/w/api.php"


def search_wikimedia(query: str, limit: int = 5) -> List[Dict]:
    """
    Search Wikimedia Commons for images. Used as fallback when Google
    Places doesn't return enough photos.
    """
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrnamespace": "6",
        "gsrsearch": query,
        "gsrlimit": str(min(limit * 2, 20)),
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": "1280",
    }

    url = f"{WIKI_API_URL}?{urllib.parse.urlencode(params)}"

    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "TravionImageCollector/1.0"}
        )
        with urllib.request.urlopen(req, timeout=15, context=_ssl_context) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"    [Wikimedia] Search failed: {e}")
        return []

    results = []
    pages = data.get("query", {}).get("pages", {})

    for _page_id, page in pages.items():
        imageinfo_list = page.get("imageinfo", [])
        if not imageinfo_list:
            continue

        info = imageinfo_list[0]
        mime = info.get("mime", "")
        if mime not in ("image/jpeg", "image/png"):
            continue

        width = info.get("width", 0)
        height = info.get("height", 0)
        if width < 400 or height < 300:
            continue

        thumb_url = info.get("thumburl", info.get("url", ""))
        original_url = info.get("url", "")

        ext_meta = info.get("extmetadata", {})
        description = ext_meta.get("ImageDescription", {}).get(
            "value", ""
        ) or page.get("title", "").replace("File:", "")
        description = re.sub(r"<[^>]+>", "", description).strip()
        if len(description) > 300:
            description = description[:297] + "..."

        results.append(
            {
                "url": thumb_url or original_url,
                "original_url": original_url,
                "title": page.get("title", "").replace("File:", ""),
                "description": description,
                "source": "wikimedia_commons",
                "width": width,
                "height": height,
                "license": ext_meta.get("LicenseShortName", {}).get("value", "CC"),
            }
        )

        if len(results) >= limit:
            break

    return results


# ============================================================================
# Image Downloader
# ============================================================================


def download_image(url: str, save_path: Path, timeout: int = 30) -> bool:
    """Download an image from a URL and save it to disk."""
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "TravionImageCollector/1.0 (Sri Lanka Tourism Research)",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context) as resp:
            data = resp.read()

            # Verify minimum file size (skip broken/tiny images)
            if len(data) < 5000:
                print(f"    [Download] Skipped tiny file ({len(data)} bytes)")
                return False

            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(data)

            size_kb = len(data) / 1024
            print(f"              ({size_kb:.0f} KB)")
            return True

    except Exception as e:
        print(f"    [Download] Failed: {e}")
        return False


def sanitize_filename(name: str) -> str:
    """Convert a location name to a safe filename slug."""
    slug = name.lower().strip()
    slug = slug.replace("'", "").replace("\u2019", "")
    slug = "".join(c if c.isalnum() else "_" for c in slug)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


# ============================================================================
# Progress Tracking
# ============================================================================


def load_progress() -> Dict:
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH) as f:
            return json.load(f)
    return {"completed_locations": [], "failed_locations": []}


def save_progress(progress: Dict) -> None:
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f, indent=2)


# ============================================================================
# Tag Generation
# ============================================================================


def load_locations(csv_path: Path) -> List[Dict]:
    """Load locations from the CSV file."""
    locations = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Location_Name", "").strip()
            if not name:
                continue
            locations.append(
                {
                    "name": name,
                    "lat": float(row.get("l_lat", 0)),
                    "lng": float(row.get("l_lng", 0)),
                    "scores": {
                        "history": float(row.get("l_hist", 0)),
                        "adventure": float(row.get("l_adv", 0)),
                        "nature": float(row.get("l_nat", 0)),
                        "relaxation": float(row.get("l_rel", 0)),
                    },
                    "outdoor": bool(int(float(row.get("l_outdoor", 1)))),
                }
            )
    return locations


def determine_tags(location: Dict) -> List[str]:
    """Generate tags based on location scores and name."""
    tags = []
    scores = location["scores"]

    if scores["history"] >= 0.7:
        tags.append("heritage")
    if scores["adventure"] >= 0.7:
        tags.append("adventure")
    if scores["nature"] >= 0.7:
        tags.append("nature")
    if scores["relaxation"] >= 0.7:
        tags.append("relaxation")
    if location["outdoor"]:
        tags.append("outdoor")

    name_lower = location["name"].lower()
    tag_keywords = {
        "beach": "beach",
        "temple": "temple",
        "fort": "fort",
        "falls": "waterfall",
        "lake": "lake",
        "park": "national_park",
        "garden": "garden",
        "rock": "rock_formation",
        "mountain": "mountain",
        "bridge": "bridge",
        "museum": "museum",
        "kovil": "hindu_temple",
        "vihara": "buddhist_temple",
        "stupa": "stupa",
        "reef": "marine",
        "island": "island",
        "forest": "forest",
        "tea": "tea_plantation",
        "surf": "surfing",
        "whale": "whale_watching",
        "safari": "safari",
        "hike": "hiking",
        "tower": "landmark",
    }
    for keyword, tag in tag_keywords.items():
        if keyword in name_lower:
            tags.append(tag)

    return list(set(tags)) or ["tourism"]


# ============================================================================
# Main Collection Pipeline
# ============================================================================


def collect_images_for_location(
    location: Dict,
    per_location: int = 5,
    google_api_key: Optional[str] = None,
    dry_run: bool = False,
) -> List[Dict]:
    """
    Collect images for a single location.

    Primary: Google Places API (high coverage for Sri Lanka)
    Fallback: Wikimedia Commons (if Google returns fewer than needed)
    """
    name = location["name"]
    slug = sanitize_filename(name)
    tags = determine_tags(location)

    all_results = []

    # Source 1: Google Places API (primary)
    if google_api_key:
        print(f"  Searching Google Places...")
        google_results = search_google_places(
            location_name=name,
            lat=location["lat"],
            lng=location["lng"],
            api_key=google_api_key,
            limit=per_location,
        )
        print(f"    [Google] Found {len(google_results)} photos")
        all_results.extend(google_results)

    # Source 2: Wikimedia fallback (if we need more)
    if len(all_results) < per_location:
        remaining = per_location - len(all_results)
        query = f"{name} Sri Lanka"
        print(f"  Searching Wikimedia Commons (need {remaining} more)...")
        wiki_results = search_wikimedia(query, limit=remaining)
        print(f"    [Wikimedia] Found {len(wiki_results)} images")
        all_results.extend(wiki_results)

    # Cap at target
    all_results = all_results[:per_location]

    if not all_results:
        print(f"    No images found for {name}")
        return []

    if dry_run:
        for r in all_results:
            src = r.get("source", "unknown")
            print(f"    [DRY RUN] [{src}] {r.get('title', '')[:60]}")
        return []

    # Download images and build metadata
    metadata_entries = []
    for idx, result in enumerate(all_results, 1):
        # Determine extension from URL
        ext = ".jpg"
        url_lower = result["url"].lower()
        if ".png" in url_lower:
            ext = ".png"
        elif ".webp" in url_lower:
            ext = ".webp"

        filename = f"{slug}_{idx:02d}{ext}"
        save_path = IMAGE_DIR / filename

        if save_path.exists() and save_path.stat().st_size > 5000:
            print(f"    [{idx}/{len(all_results)}] Already exists: {filename}")
        else:
            print(f"    [{idx}/{len(all_results)}] Downloading: {filename}")
            success = download_image(result["url"], save_path)
            if not success:
                print(f"    [{idx}/{len(all_results)}] FAILED - skipping")
                continue
            time.sleep(0.3)

        # Build metadata entry
        image_id = f"{slug}_{idx:02d}"
        description = result.get("description", "")
        if not description or len(description) < 10:
            description = f"{name} - tourism destination in Sri Lanka"

        entry = {
            "image_id": image_id,
            "file_path": f"data/image_knowledge/images/{filename}",
            "location_name": name,
            "description": description,
            "tags": tags,
            "coordinates": {
                "lat": location["lat"],
                "lng": location["lng"],
            },
            "image_url": result.get("original_url", result["url"]),
            "source": result.get("source", "unknown"),
        }

        if result.get("photographer"):
            entry["photographer"] = result["photographer"]
        if result.get("place_id"):
            entry["google_place_id"] = result["place_id"]
        if result.get("license"):
            entry["license"] = result["license"]

        metadata_entries.append(entry)

    return metadata_entries


def main():
    parser = argparse.ArgumentParser(
        description="Download tourism images for Sri Lanka locations"
    )
    parser.add_argument(
        "--per-location",
        type=int,
        default=5,
        help="Number of images per location (default: 5)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last progress (skip completed locations)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Search but don't download",
    )
    parser.add_argument(
        "--locations",
        nargs="*",
        help="Only process specific locations (by name substring)",
    )
    args = parser.parse_args()

    # Load Google API key
    google_api_key = load_google_api_key()

    print("=" * 65)
    print("  Travion Image Collector - Sri Lanka Tourism Locations")
    print("  Primary Source: Google Places API (New)")
    print("=" * 65)

    if not google_api_key:
        print(
            "\nWARNING: GOOGLE_MAPS_API_KEY not found in .env"
            "\n         Falling back to Wikimedia Commons only."
            "\n         For best results, set GOOGLE_MAPS_API_KEY in .env"
        )

    # Load locations
    if not CSV_PATH.exists():
        print(f"\nERROR: CSV not found at {CSV_PATH}")
        sys.exit(1)

    locations = load_locations(CSV_PATH)
    print(f"\nLoaded {len(locations)} locations from {CSV_PATH.name}")

    # Filter if specified
    if args.locations:
        filtered = []
        for loc in locations:
            for pattern in args.locations:
                if pattern.lower() in loc["name"].lower():
                    filtered.append(loc)
                    break
        locations = filtered
        print(f"Filtered to {len(locations)} locations matching: {args.locations}")

    # Load progress for resume
    progress = load_progress() if args.resume else {
        "completed_locations": [],
        "failed_locations": [],
    }
    if args.resume:
        skip_count = len(progress["completed_locations"])
        locations = [
            loc
            for loc in locations
            if loc["name"] not in progress["completed_locations"]
        ]
        print(f"Resuming: skipping {skip_count} already completed locations")

    # Show config
    sources = "Google Places"
    if not google_api_key:
        sources = "Wikimedia Commons (no Google key)"
    sources += " + Wikimedia (fallback)"

    print(f"\nConfiguration:")
    print(f"  Images per location : {args.per_location}")
    print(f"  Target directory    : {IMAGE_DIR}")
    print(f"  Metadata output     : {METADATA_PATH}")
    print(f"  Sources             : {sources}")
    print(f"  Dry run             : {args.dry_run}")
    print(f"  Locations to process: {len(locations)}")
    print(f"  Estimated images    : ~{len(locations) * args.per_location}")

    # Load existing metadata (append, not overwrite)
    existing_metadata = []
    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            existing_metadata = json.load(f)
        print(f"  Existing metadata   : {len(existing_metadata)} entries")

    existing_ids = {entry["image_id"] for entry in existing_metadata}
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    # Process each location
    all_metadata = list(existing_metadata)
    total_downloaded = 0
    total_failed = 0

    for i, location in enumerate(locations, 1):
        name = location["name"]
        print(f"\n{'─' * 55}")
        print(f"[{i}/{len(locations)}] {name}")
        print(f"  Coordinates: ({location['lat']}, {location['lng']})")

        try:
            entries = collect_images_for_location(
                location,
                per_location=args.per_location,
                google_api_key=google_api_key,
                dry_run=args.dry_run,
            )

            new_count = 0
            for entry in entries:
                if entry["image_id"] not in existing_ids:
                    all_metadata.append(entry)
                    existing_ids.add(entry["image_id"])
                    new_count += 1

            total_downloaded += new_count
            print(f"  Result: {new_count} new images added")

            progress["completed_locations"].append(name)

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Saving progress...")
            break
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            progress["failed_locations"].append(name)
            total_failed += 1

        # Save checkpoint every 5 locations
        if i % 5 == 0 and not args.dry_run:
            save_progress(progress)
            with open(METADATA_PATH, "w") as f:
                json.dump(all_metadata, f, indent=2, ensure_ascii=False)
            print(f"  [Checkpoint] Saved {len(all_metadata)} metadata entries")

        # Rate limiting between locations
        if not args.dry_run:
            time.sleep(0.5)

    # Final save
    if not args.dry_run:
        with open(METADATA_PATH, "w") as f:
            json.dump(all_metadata, f, indent=2, ensure_ascii=False)
        save_progress(progress)

    # Summary
    print(f"\n{'=' * 65}")
    print("  Collection Complete!")
    print(f"{'=' * 65}")
    print(f"  Total locations processed : {len(progress['completed_locations'])}")
    print(f"  Total images collected    : {total_downloaded}")
    print(f"  Total metadata entries    : {len(all_metadata)}")
    print(f"  Failed locations          : {total_failed}")
    if total_failed > 0:
        print(f"  Failed names              : {progress['failed_locations']}")
    print(f"\n  Images saved to  : {IMAGE_DIR}")
    print(f"  Metadata saved to: {METADATA_PATH}")

    if args.dry_run:
        print("\n  [DRY RUN] No files were actually downloaded.")

    print(f"\nNext step: Run the Phase 2 build script:")
    print(f"  python3 scripts/build_image_knowledge.py")


if __name__ == "__main__":
    main()
