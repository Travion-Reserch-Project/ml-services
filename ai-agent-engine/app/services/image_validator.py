"""
Image Validator: Tourism Destination Image Validation.

This module validates that uploaded images are tourism-related Sri Lankan
destinations using CLIP's zero-shot classification capability. Non-tourism
images (selfies, screenshots, documents, memes, etc.) are rejected with
a descriptive message.

Validation Strategy:
    1. File-level checks: format (JPEG/PNG/WebP), size (<= 10 MB)
    2. Content-level checks: CLIP zero-shot classification against
       positive (tourism) and negative (non-tourism) label sets
    3. If the highest positive score exceeds the highest negative score
       AND exceeds a configurable threshold, the image is accepted.

Research Note:
    CLIP's zero-shot classification works by encoding candidate text labels
    and the input image into the same embedding space, then selecting the
    label with the highest cosine similarity. This avoids the need for a
    fine-tuned classification model.
"""

import io
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Accepted image MIME types and their file signatures
ACCEPTED_FORMATS = {
    "image/jpeg": [b"\xff\xd8\xff"],
    "image/png": [b"\x89PNG"],
    "image/webp": [b"RIFF"],
}

ACCEPTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# ---------------------------------------------------------------------------
# CLIP Zero-Shot Classification Labels
# ---------------------------------------------------------------------------

# Tourism-related labels (positive set)
TOURISM_POSITIVE_LABELS = [
    "a photo of a tourist destination in Sri Lanka",
    "a scenic landscape with mountains or hills",
    "a tropical beach with ocean waves",
    "a historical monument or ancient ruins",
    "a Buddhist temple or stupa",
    "a Hindu kovil or temple",
    "a colonial era building or fort",
    "a waterfall in a tropical forest",
    "a wildlife sanctuary or national park",
    "a scenic tea plantation",
    "a lake or reservoir surrounded by nature",
    "a scenic viewpoint or lookout",
    "a traditional village or cultural site",
    "a botanical garden or nature reserve",
    "a scenic mountain road or railway",
    "a coastal lighthouse or harbor",
    "a rock formation or geological landmark",
]

# Non-tourism labels (negative set)
NON_TOURISM_LABELS = [
    "a selfie or portrait photo of a person",
    "a screenshot of a phone or computer screen",
    "a document or text on paper",
    "a meme or internet image with text overlay",
    "food photography on a plate or table",
    "a pet or domestic animal indoors",
    "an office or workspace interior",
    "a car or vehicle close-up",
    "a commercial product or advertisement",
    "a diagram or technical drawing",
    "a blurry or unrecognizable image",
    "a logo or brand graphic",
    "an urban city scene with no landmarks",
    "inappropriate or adult content",
]

# ---------------------------------------------------------------------------
# Rejection Messages (user-friendly)
# ---------------------------------------------------------------------------

REJECTION_MESSAGES = {
    "not_tourism": (
        "This image doesn't appear to be a tourism destination. "
        "Please upload photos of tourist locations, landmarks, scenic spots, "
        "or cultural sites in Sri Lanka. Examples include temples, beaches, "
        "historical ruins, waterfalls, or national parks."
    ),
    "invalid_format": (
        "Unsupported image format. Please upload a JPEG, PNG, or WebP image."
    ),
    "too_large": (
        "Image file is too large. Please upload an image smaller than {max_size_mb} MB."
    ),
    "corrupt": (
        "The uploaded image appears to be corrupted or cannot be opened. "
        "Please try uploading a different image."
    ),
    "empty": (
        "No image data received. Please upload a valid image file."
    ),
}


class ImageValidationResult:
    """
    Result of image validation.

    Attributes:
        is_valid: Whether the image passed all validation checks.
        rejection_reason: Key identifying why the image was rejected (None if valid).
        rejection_message: User-friendly rejection message (None if valid).
        top_positive_label: Best matching tourism label (for debugging).
        top_positive_score: Score of the best tourism match.
        top_negative_label: Best matching non-tourism label (for debugging).
        top_negative_score: Score of the best non-tourism match.
    """

    def __init__(
        self,
        is_valid: bool,
        rejection_reason: Optional[str] = None,
        rejection_message: Optional[str] = None,
        top_positive_label: Optional[str] = None,
        top_positive_score: float = 0.0,
        top_negative_label: Optional[str] = None,
        top_negative_score: float = 0.0,
    ):
        self.is_valid = is_valid
        self.rejection_reason = rejection_reason
        self.rejection_message = rejection_message
        self.top_positive_label = top_positive_label
        self.top_positive_score = top_positive_score
        self.top_negative_label = top_negative_label
        self.top_negative_score = top_negative_score

    @property
    def message(self) -> str:
        """User-friendly message (valid or rejection reason)."""
        if self.is_valid:
            return "Image appears to be a valid Sri Lankan tourism destination."
        return self.rejection_message or "Image validation failed."

    @property
    def positive_score(self) -> float:
        """Alias for top_positive_score."""
        return self.top_positive_score

    @property
    def negative_score(self) -> float:
        """Alias for top_negative_score."""
        return self.top_negative_score

    def to_dict(self) -> Dict:
        return {
            "is_valid": self.is_valid,
            "message": self.message,
            "rejection_reason": self.rejection_reason,
            "positive_score": round(self.top_positive_score, 4),
            "negative_score": round(self.top_negative_score, 4),
            "top_positive_label": self.top_positive_label,
            "top_negative_label": self.top_negative_label,
        }


class ImageValidator:
    """
    Validates uploaded images for Sri Lanka tourism relevance.

    Uses CLIP zero-shot classification to determine whether an image
    depicts a tourism destination or a non-tourism subject.

    Attributes:
        clip_service: CLIPEmbeddingService instance for embeddings.
        threshold: Minimum positive score to accept (default: 0.25).
        max_size_bytes: Maximum allowed file size in bytes.

    Example:
        >>> validator = ImageValidator()
        >>> result = validator.validate_image(pil_image)
        >>> if result.is_valid:
        ...     print("Image accepted!")
        >>> else:
        ...     print(result.rejection_message)
    """

    def __init__(
        self,
        threshold: float = 0.25,
        max_size_mb: int = 10,
    ):
        """
        Initialize the Image Validator.

        Args:
            threshold: Minimum CLIP similarity score for positive labels
                      to accept the image. Lower = more permissive.
            max_size_mb: Maximum upload size in megabytes.
        """
        self.threshold = threshold
        self.max_size_mb = max_size_mb
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._clip_service = None

        logger.info(
            f"ImageValidator configured: threshold={threshold}, "
            f"max_size_mb={max_size_mb}"
        )

    def _get_clip_service(self):
        """Lazy-load the CLIP service."""
        if self._clip_service is None:
            from .clip_embedding_service import get_clip_service
            self._clip_service = get_clip_service()
        return self._clip_service

    def validate_file_format(
        self,
        image_bytes: bytes,
        filename: Optional[str] = None,
    ) -> Optional[ImageValidationResult]:
        """
        Validate image file format and size.

        Args:
            image_bytes: Raw image file bytes.
            filename: Original filename (optional, for extension check).

        Returns:
            ImageValidationResult with rejection if invalid, None if OK.
        """
        # Check empty
        if not image_bytes:
            return ImageValidationResult(
                is_valid=False,
                rejection_reason="empty",
                rejection_message=REJECTION_MESSAGES["empty"],
            )

        # Check file size
        if len(image_bytes) > self.max_size_bytes:
            return ImageValidationResult(
                is_valid=False,
                rejection_reason="too_large",
                rejection_message=REJECTION_MESSAGES["too_large"].format(
                    max_size_mb=self.max_size_mb
                ),
            )

        # Check file extension if filename provided
        if filename:
            ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            if ext and ext not in ACCEPTED_EXTENSIONS:
                return ImageValidationResult(
                    is_valid=False,
                    rejection_reason="invalid_format",
                    rejection_message=REJECTION_MESSAGES["invalid_format"],
                )

        # Check magic bytes (file signature)
        is_known_format = False
        for mime_type, signatures in ACCEPTED_FORMATS.items():
            for sig in signatures:
                if image_bytes[:len(sig)] == sig:
                    is_known_format = True
                    break
            if is_known_format:
                break

        if not is_known_format:
            return ImageValidationResult(
                is_valid=False,
                rejection_reason="invalid_format",
                rejection_message=REJECTION_MESSAGES["invalid_format"],
            )

        return None  # Passed format checks

    def validate_content(
        self,
        image: "Image.Image",
    ) -> ImageValidationResult:
        """
        Validate image content using CLIP zero-shot classification.

        Compares the image against tourism-positive and non-tourism-negative
        label sets. Accepts if the best positive score exceeds both the
        best negative score and the configured threshold.

        Args:
            image: PIL Image object.

        Returns:
            ImageValidationResult with classification details.
        """
        clip = self._get_clip_service()

        if not clip.is_available:
            # If CLIP is unavailable, accept all images (graceful degradation)
            logger.warning(
                "CLIP not available for image validation — accepting by default"
            )
            return ImageValidationResult(is_valid=True)

        try:
            # Embed the image
            image_embedding = clip.embed_image(image)

            # Embed all labels
            positive_embeddings = clip.embed_texts(TOURISM_POSITIVE_LABELS)
            negative_embeddings = clip.embed_texts(NON_TOURISM_LABELS)

            # Compute similarities
            positive_scores = [
                clip.compute_similarity(image_embedding, emb)
                for emb in positive_embeddings
            ]
            negative_scores = [
                clip.compute_similarity(image_embedding, emb)
                for emb in negative_embeddings
            ]

            # Find best matches
            best_pos_idx = max(range(len(positive_scores)), key=lambda i: positive_scores[i])
            best_neg_idx = max(range(len(negative_scores)), key=lambda i: negative_scores[i])

            best_pos_score = positive_scores[best_pos_idx]
            best_neg_score = negative_scores[best_neg_idx]
            best_pos_label = TOURISM_POSITIVE_LABELS[best_pos_idx]
            best_neg_label = NON_TOURISM_LABELS[best_neg_idx]

            logger.info(
                f"Image validation scores: "
                f"positive={best_pos_score:.4f} ({best_pos_label}), "
                f"negative={best_neg_score:.4f} ({best_neg_label})"
            )

            # Decision: positive must beat negative AND exceed threshold
            is_valid = (
                best_pos_score > best_neg_score
                and best_pos_score >= self.threshold
            )

            if is_valid:
                return ImageValidationResult(
                    is_valid=True,
                    top_positive_label=best_pos_label,
                    top_positive_score=best_pos_score,
                    top_negative_label=best_neg_label,
                    top_negative_score=best_neg_score,
                )
            else:
                return ImageValidationResult(
                    is_valid=False,
                    rejection_reason="not_tourism",
                    rejection_message=REJECTION_MESSAGES["not_tourism"],
                    top_positive_label=best_pos_label,
                    top_positive_score=best_pos_score,
                    top_negative_label=best_neg_label,
                    top_negative_score=best_neg_score,
                )

        except Exception as e:
            logger.error(f"Image content validation failed: {e}")
            # On error, accept (fail open) to avoid blocking users
            return ImageValidationResult(is_valid=True)

    def validate_image(
        self,
        image: "Image.Image",
        image_bytes: Optional[bytes] = None,
        filename: Optional[str] = None,
    ) -> ImageValidationResult:
        """
        Full validation pipeline: file format + content analysis.

        Args:
            image: PIL Image object.
            image_bytes: Raw bytes (for size/format checks). Optional.
            filename: Original filename. Optional.

        Returns:
            ImageValidationResult with full validation details.
        """
        # Step 1: File-level validation (if bytes provided)
        if image_bytes is not None:
            format_result = self.validate_file_format(image_bytes, filename)
            if format_result is not None:
                return format_result

        # Step 2: Content validation via CLIP
        return self.validate_content(image)

    def validate_base64_image(
        self,
        base64_string: str,
        filename: Optional[str] = None,
    ) -> ImageValidationResult:
        """
        Validate a base64-encoded image (full pipeline).

        Decodes the base64 string, runs format and content validation.

        Args:
            base64_string: Base64-encoded image data.
            filename: Original filename (optional).

        Returns:
            ImageValidationResult with validation details.
        """
        from .clip_embedding_service import decode_base64_image

        if not base64_string:
            return ImageValidationResult(
                is_valid=False,
                rejection_reason="empty",
                rejection_message=REJECTION_MESSAGES["empty"],
            )

        # Decode base64 to bytes for format checking
        try:
            import base64 as b64

            # Strip data URI prefix if present
            raw_b64 = base64_string
            if "," in raw_b64 and raw_b64.startswith("data:"):
                raw_b64 = raw_b64.split(",", 1)[1]

            image_bytes = b64.b64decode(raw_b64)
        except Exception:
            return ImageValidationResult(
                is_valid=False,
                rejection_reason="corrupt",
                rejection_message=REJECTION_MESSAGES["corrupt"],
            )

        # Format validation
        format_result = self.validate_file_format(image_bytes, filename)
        if format_result is not None:
            return format_result

        # Decode to PIL Image
        try:
            image = decode_base64_image(base64_string)
        except ValueError:
            return ImageValidationResult(
                is_valid=False,
                rejection_reason="corrupt",
                rejection_message=REJECTION_MESSAGES["corrupt"],
            )

        # Content validation
        return self.validate_content(image)


# ---------------------------------------------------------------------------
# Singleton Access
# ---------------------------------------------------------------------------

_image_validator: Optional[ImageValidator] = None


def get_image_validator(
    threshold: Optional[float] = None,
    max_size_mb: Optional[int] = None,
) -> ImageValidator:
    """
    Get or create the ImageValidator singleton.

    Args:
        threshold: CLIP similarity threshold (default from settings).
        max_size_mb: Max upload size in MB (default from settings).

    Returns:
        ImageValidator singleton instance.
    """
    global _image_validator
    if _image_validator is None:
        from ..config import settings

        _image_validator = ImageValidator(
            threshold=threshold or settings.IMAGE_VALIDATION_THRESHOLD,
            max_size_mb=max_size_mb or settings.IMAGE_UPLOAD_MAX_SIZE_MB,
        )
    return _image_validator
