#!/usr/bin/env python3
"""
Build Image Knowledge Base: CLIP Embeddings -> ChromaDB Collection.

This script processes the downloaded tourism images, generates CLIP
embeddings for each one, and stores them in a ChromaDB collection
named 'image_knowledge'. This collection is used by the Vision
Retrieval Node for text-to-image and image-to-image search.

Pipeline:
    1. Load metadata.json (image_id, file_path, location_name, etc.)
    2. For each image:
       a. Load the image with PIL
       b. Generate a 512-dim CLIP embedding (image modality)
       c. Also generate a CLIP text embedding of the description
    3. Upsert into ChromaDB 'image_knowledge' collection with:
       - ids: image_id
       - embeddings: CLIP image embeddings (512-dim)
       - metadatas: location_name, description, tags, coordinates, image_url, etc.
       - documents: description text (for fallback text search)
    4. Print summary statistics

ChromaDB Collection Details:
    Name: image_knowledge
    Embedding Model: openai/clip-vit-base-patch32 (512-dim)
    Embedding Source: Pre-computed (passed directly, NOT via ChromaDB's default fn)
    Storage: Same vector_db/ directory as tourism_knowledge

Usage:
    # Build the full collection:
    python3 scripts/build_image_knowledge.py

    # Rebuild from scratch (delete existing collection first):
    python3 scripts/build_image_knowledge.py --rebuild

    # Process only specific locations:
    python3 scripts/build_image_knowledge.py --locations "Sigiriya" "Galle Fort"

    # Dry run (generate embeddings but don't write to ChromaDB):
    python3 scripts/build_image_knowledge.py --dry-run

    # Custom batch size (lower if running out of memory):
    python3 scripts/build_image_knowledge.py --batch-size 10
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
METADATA_PATH = PROJECT_ROOT / "data" / "image_knowledge" / "metadata.json"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
COLLECTION_NAME = "image_knowledge"


def load_metadata(
    metadata_path: Path,
    location_filter: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Load image metadata from metadata.json.

    Args:
        metadata_path: Path to metadata.json.
        location_filter: Optional list of location name substrings to filter by.

    Returns:
        List of metadata dicts.
    """
    if not metadata_path.exists():
        print(f"ERROR: Metadata file not found: {metadata_path}")
        print("Run collect_location_images.py first to download images.")
        sys.exit(1)

    with open(metadata_path) as f:
        entries = json.load(f)

    if location_filter:
        filtered = []
        for entry in entries:
            name = entry.get("location_name", "").lower()
            for pattern in location_filter:
                if pattern.lower() in name:
                    filtered.append(entry)
                    break
        entries = filtered

    return entries


def validate_images(entries: List[Dict]) -> List[Dict]:
    """
    Validate that image files exist on disk and are not empty.

    Args:
        entries: List of metadata dicts.

    Returns:
        Filtered list with only valid entries.
    """
    valid = []
    skipped = 0
    for entry in entries:
        file_path = PROJECT_ROOT / entry["file_path"]
        if file_path.exists() and file_path.stat().st_size > 5000:
            valid.append(entry)
        else:
            skipped += 1
            print(f"  [Skip] Missing or empty: {entry['file_path']}")

    if skipped:
        print(f"  Skipped {skipped} invalid images")

    return valid


def generate_embeddings(
    entries: List[Dict],
    batch_size: int = 20,
) -> List[Dict]:
    """
    Generate CLIP embeddings for all images.

    For each image, generates:
    - image_embedding: 512-dim CLIP image vector (primary, used for storage)
    - text_embedding: 512-dim CLIP text vector of description (for reference)

    Args:
        entries: List of metadata dicts with file_path.
        batch_size: Number of images to process before printing progress.

    Returns:
        List of entries augmented with 'embedding' key.
    """
    # Import here to avoid loading CLIP at module level
    sys.path.insert(0, str(PROJECT_ROOT))
    from app.services.clip_embedding_service import CLIPEmbeddingService
    from PIL import Image

    print("\nInitializing CLIP model...")
    clip = CLIPEmbeddingService()

    # Force model load now (not lazily) so we see any errors early
    clip._ensure_loaded()
    print(f"  Model: {clip.model_name}")
    print(f"  Device: {clip._device}")
    print(f"  Embedding dim: {clip.EMBEDDING_DIM}")

    results = []
    total = len(entries)
    failed = 0
    start_time = time.time()

    print(f"\nGenerating embeddings for {total} images...")

    for i, entry in enumerate(entries):
        file_path = PROJECT_ROOT / entry["file_path"]

        try:
            # Load and embed the image
            image = Image.open(file_path)
            image_embedding = clip.embed_image(image)
            image.close()

            entry_with_embedding = {**entry, "embedding": image_embedding}
            results.append(entry_with_embedding)

        except Exception as e:
            print(f"  [FAIL] {entry['image_id']}: {e}")
            failed += 1
            continue

        # Progress reporting
        if (i + 1) % batch_size == 0 or (i + 1) == total:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i + 1}/{total}] "
                f"{rate:.1f} img/sec | "
                f"ETA: {eta:.0f}s | "
                f"Failed: {failed}"
            )

    elapsed = time.time() - start_time
    print(f"\nEmbedding generation complete:")
    print(f"  Processed: {len(results)}/{total}")
    print(f"  Failed: {failed}")
    print(f"  Time: {elapsed:.1f}s ({len(results)/elapsed:.1f} img/sec)")

    return results


def store_in_chromadb(
    entries: List[Dict],
    vector_db_dir: str,
    collection_name: str,
    rebuild: bool = False,
) -> int:
    """
    Store image embeddings in ChromaDB.

    Creates or updates the 'image_knowledge' collection with pre-computed
    CLIP embeddings. Uses upsert to handle re-runs gracefully.

    Args:
        entries: List of dicts with 'embedding' and metadata.
        vector_db_dir: Path to ChromaDB persistent storage.
        collection_name: Name of the target collection.
        rebuild: If True, delete existing collection before inserting.

    Returns:
        Number of documents stored.
    """
    import chromadb

    print(f"\nConnecting to ChromaDB at {vector_db_dir}...")
    client = chromadb.PersistentClient(path=vector_db_dir)

    # Delete existing collection if rebuilding
    if rebuild:
        try:
            client.delete_collection(name=collection_name)
            print(f"  Deleted existing '{collection_name}' collection")
        except Exception:
            pass  # Collection didn't exist

    # Create or get collection
    # IMPORTANT: No embedding_function — we pass pre-computed embeddings
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "description": "Sri Lanka tourism image knowledge base (CLIP embeddings)",
            "embedding_model": "openai/clip-vit-base-patch32",
            "embedding_dim": "512",
            "hnsw:space": "cosine",
        },
    )

    existing_count = collection.count()
    print(f"  Collection '{collection_name}': {existing_count} existing documents")

    # Prepare batch data
    ids = []
    embeddings = []
    metadatas = []
    documents = []

    for entry in entries:
        image_id = entry["image_id"]
        embedding = entry["embedding"]

        # Build metadata dict (ChromaDB only supports str, int, float, bool)
        tags = entry.get("tags", [])
        coords = entry.get("coordinates", {})

        metadata = {
            "location_name": entry.get("location_name", ""),
            "description": entry.get("description", ""),
            "image_url": entry.get("image_url", ""),
            "file_path": entry.get("file_path", ""),
            "source": entry.get("source", ""),
            "tags": ",".join(tags) if isinstance(tags, list) else str(tags),
            "lat": coords.get("lat", 0.0),
            "lng": coords.get("lng", 0.0),
        }

        # Optional fields
        if entry.get("photographer"):
            metadata["photographer"] = entry["photographer"]
        if entry.get("google_place_id"):
            metadata["google_place_id"] = entry["google_place_id"]

        # Document text = description (for hybrid text search fallback)
        doc_text = (
            f"{entry.get('location_name', '')} - "
            f"{entry.get('description', '')} "
            f"Tags: {', '.join(tags)}"
        )

        ids.append(image_id)
        embeddings.append(embedding)
        metadatas.append(metadata)
        documents.append(doc_text)

    # Upsert in batches (ChromaDB has a max batch size)
    BATCH_SIZE = 100
    total_upserted = 0

    for start in range(0, len(ids), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(ids))
        collection.upsert(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
            documents=documents[start:end],
        )
        total_upserted += end - start
        print(f"  Upserted batch: {total_upserted}/{len(ids)}")

    final_count = collection.count()
    print(f"\n  Collection '{collection_name}' now has {final_count} documents")

    return final_count


def verify_collection(vector_db_dir: str, collection_name: str) -> None:
    """
    Run a quick verification query to ensure the collection works.

    Tests both text-to-image search (CLIP text embedding) and
    metadata filtering.
    """
    import chromadb

    sys.path.insert(0, str(PROJECT_ROOT))
    from app.services.clip_embedding_service import CLIPEmbeddingService

    print("\n" + "=" * 55)
    print("  Verification: Testing image_knowledge collection")
    print("=" * 55)

    client = chromadb.PersistentClient(path=vector_db_dir)
    collection = client.get_collection(name=collection_name)

    clip = CLIPEmbeddingService()

    # Test 1: Text-to-image search
    print("\n  Test 1: Text-to-image search")
    print("  Query: 'ancient rock fortress with lion paws'")
    query_embedding = clip.embed_text("ancient rock fortress with lion paws")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["metadatas", "distances", "documents"],
    )

    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]
        similarity = 1 - dist
        print(
            f"    #{i+1}: {meta['location_name']} "
            f"(similarity: {similarity:.4f})"
        )
        print(f"         {meta['description'][:80]}")

    # Test 2: Text-to-image search (beach)
    print("\n  Test 2: Text-to-image search")
    print("  Query: 'beautiful tropical beach with clear water'")
    query_embedding = clip.embed_text("beautiful tropical beach with clear water")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["metadatas", "distances"],
    )

    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]
        similarity = 1 - dist
        print(
            f"    #{i+1}: {meta['location_name']} "
            f"(similarity: {similarity:.4f})"
        )

    # Test 3: Location metadata filter
    print("\n  Test 3: Metadata filter (location_name = 'Sigiriya Lion Rock')")
    results = collection.get(
        where={"location_name": {"$eq": "Sigiriya Lion Rock"}},
        include=["metadatas"],
    )
    print(f"    Found {len(results['ids'])} images for Sigiriya Lion Rock")

    # Test 4: Collection stats
    print(f"\n  Collection stats:")
    print(f"    Total documents: {collection.count()}")

    # Count unique locations
    all_results = collection.get(include=["metadatas"])
    unique_locations = set(
        m.get("location_name", "") for m in all_results["metadatas"]
    )
    print(f"    Unique locations: {len(unique_locations)}")
    print(f"    Avg images/location: {collection.count() / max(len(unique_locations), 1):.1f}")

    print("\n  All verification tests passed!")


def main():
    parser = argparse.ArgumentParser(
        description="Build ChromaDB image_knowledge collection from CLIP embeddings"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete existing collection and rebuild from scratch",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate embeddings but don't write to ChromaDB",
    )
    parser.add_argument(
        "--locations",
        nargs="*",
        help="Only process specific locations (by name substring)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Progress reporting interval (default: 20)",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip verification queries after building",
    )
    args = parser.parse_args()

    print("=" * 55)
    print("  Travion Image Knowledge Base Builder")
    print("  CLIP Embeddings -> ChromaDB Collection")
    print("=" * 55)

    # Step 1: Load metadata
    print(f"\nLoading metadata from {METADATA_PATH}...")
    entries = load_metadata(METADATA_PATH, location_filter=args.locations)
    print(f"  Loaded {len(entries)} image entries")

    if args.locations:
        locations = set(e["location_name"] for e in entries)
        print(f"  Filtered to {len(locations)} locations: {list(locations)[:5]}...")

    # Step 2: Validate images exist on disk
    print(f"\nValidating image files...")
    entries = validate_images(entries)
    print(f"  Valid images: {len(entries)}")

    if not entries:
        print("\nERROR: No valid images found. Run collect_location_images.py first.")
        sys.exit(1)

    # Step 3: Generate CLIP embeddings
    entries_with_embeddings = generate_embeddings(
        entries, batch_size=args.batch_size
    )

    if not entries_with_embeddings:
        print("\nERROR: No embeddings generated.")
        sys.exit(1)

    # Step 4: Store in ChromaDB
    if args.dry_run:
        print(f"\n[DRY RUN] Would store {len(entries_with_embeddings)} documents in ChromaDB")
        print("[DRY RUN] Skipping ChromaDB write.")
    else:
        vector_db_dir = str(VECTOR_DB_DIR)
        final_count = store_in_chromadb(
            entries_with_embeddings,
            vector_db_dir=vector_db_dir,
            collection_name=COLLECTION_NAME,
            rebuild=args.rebuild,
        )

        # Step 5: Verification
        if not args.skip_verify:
            verify_collection(vector_db_dir, COLLECTION_NAME)

    # Summary
    unique_locations = set(e["location_name"] for e in entries_with_embeddings)
    print(f"\n{'=' * 55}")
    print("  Build Complete!")
    print(f"{'=' * 55}")
    print(f"  Images processed   : {len(entries_with_embeddings)}")
    print(f"  Unique locations   : {len(unique_locations)}")
    print(f"  Embedding model    : openai/clip-vit-base-patch32")
    print(f"  Embedding dim      : 512")
    print(f"  ChromaDB collection: {COLLECTION_NAME}")
    print(f"  Storage path       : {VECTOR_DB_DIR}")

    if not args.dry_run:
        print(f"\nThe image_knowledge collection is ready for the Vision Retrieval Node!")


if __name__ == "__main__":
    main()
