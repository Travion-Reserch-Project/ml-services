"""
CLIP Embedding Service: Multimodal Embeddings for Image & Text.

This module wraps OpenAI's CLIP model (clip-vit-base-patch32) to generate
unified embeddings for both images and text in the same 512-dimensional
vector space. This enables cross-modal search:
  - Text-to-Image: user types "sunset at Sigiriya" -> finds matching images
  - Image-to-Image: user uploads a photo -> finds visually similar locations

Research Note:
    CLIP (Contrastive Language-Image Pre-Training) learns visual concepts
    from natural language supervision. It maps images and text into a shared
    embedding space where semantically similar content clusters together,
    regardless of modality.

Model Details:
    - Model: openai/clip-vit-base-patch32
    - Embedding Dimension: 512
    - Image Input: 224x224 RGB (auto-resized by preprocessor)
    - Text Input: max 77 tokens (CLIP tokenizer limit)
"""

import io
import logging
import base64
from typing import List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports for heavy ML libraries
_clip_model = None
_clip_processor = None
_clip_tokenizer = None
_device = None

# Track availability
CLIP_AVAILABLE = False
TORCH_AVAILABLE = False
PIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available. Install with: pip install torch torchvision")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    logger.warning("Pillow not available. Install with: pip install Pillow")

try:
    from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast
    CLIP_AVAILABLE = True
except ImportError:
    logger.warning(
        "Transformers not available. Install with: pip install transformers"
    )


class CLIPEmbeddingService:
    """
    Singleton service for generating CLIP embeddings.

    Loads the CLIP model once at first use and caches it in memory.
    Supports both CPU and CUDA inference.

    Attributes:
        model_name: HuggingFace model identifier
        device: torch device (cpu or cuda)
        embedding_dim: Output embedding dimensionality (512)

    Example:
        >>> service = CLIPEmbeddingService()
        >>> text_emb = service.embed_text("Sigiriya Rock Fortress at sunrise")
        >>> img_emb = service.embed_image(pil_image)
        >>> # Both are 512-dim vectors in the same space
        >>> similarity = np.dot(text_emb, img_emb)
    """

    EMBEDDING_DIM = 512

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
    ):
        """
        Initialize the CLIP Embedding Service.

        Args:
            model_name: HuggingFace CLIP model identifier.
            device: "cpu" or "cuda". Auto-detected if None.
        """
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._device = None
        self._initialized = False

        # Resolve device
        if device:
            self._target_device = device
        elif TORCH_AVAILABLE and torch.cuda.is_available():
            self._target_device = "cuda"
        else:
            self._target_device = "cpu"

        logger.info(
            f"CLIPEmbeddingService configured: model={model_name}, "
            f"target_device={self._target_device}"
        )

    def _ensure_loaded(self) -> None:
        """Lazy-load the CLIP model on first use."""
        if self._initialized:
            return

        if not (TORCH_AVAILABLE and CLIP_AVAILABLE and PIL_AVAILABLE):
            missing = []
            if not TORCH_AVAILABLE:
                missing.append("torch")
            if not CLIP_AVAILABLE:
                missing.append("transformers")
            if not PIL_AVAILABLE:
                missing.append("Pillow")
            raise RuntimeError(
                f"CLIP dependencies not available. Missing: {', '.join(missing)}. "
                f"Install with: pip install {' '.join(missing)}"
            )

        try:
            import torch as _torch

            logger.info(f"Loading CLIP model: {self.model_name} ...")
            self._device = _torch.device(self._target_device)
            self._model = CLIPModel.from_pretrained(self.model_name).to(self._device)
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._tokenizer = CLIPTokenizerFast.from_pretrained(self.model_name)
            self._model.eval()
            self._initialized = True
            logger.info(
                f"CLIP model loaded successfully on {self._device} "
                f"(embedding_dim={self.EMBEDDING_DIM})"
            )
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise RuntimeError(f"CLIP model loading failed: {e}") from e

    @staticmethod
    def _extract_tensor(features):
        """
        Extract the raw tensor from CLIP model output.

        Transformers >= 5.x returns BaseModelOutputWithPooling instead of
        a raw tensor from get_text_features / get_image_features. This
        helper handles both formats.
        """
        import torch as _torch

        if isinstance(features, _torch.Tensor):
            return features
        # Transformers 5.x: BaseModelOutputWithPooling
        if hasattr(features, "pooler_output") and features.pooler_output is not None:
            return features.pooler_output
        if hasattr(features, "last_hidden_state"):
            # Use CLS token (first token) as fallback
            return features.last_hidden_state[:, 0, :]
        raise TypeError(
            f"Unexpected CLIP output type: {type(features)}. "
            "Expected torch.Tensor or BaseModelOutputWithPooling."
        )

    def embed_text(self, text: str) -> List[float]:
        """
        Encode a text string into a CLIP embedding.

        Args:
            text: Input text (max ~77 tokens after tokenization).

        Returns:
            List of 512 floats representing the text embedding.

        Raises:
            RuntimeError: If CLIP model is not available.
            ValueError: If text is empty.
        """
        if not text or not text.strip():
            raise ValueError("Text input cannot be empty")

        self._ensure_loaded()

        import torch as _torch

        try:
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            ).to(self._device)

            with _torch.no_grad():
                raw_features = self._model.get_text_features(**inputs)

            text_features = self._extract_tensor(raw_features)
            # L2 normalize for cosine similarity
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            embedding = text_features.squeeze().cpu().numpy().tolist()

            return embedding

        except Exception as e:
            logger.error(f"Text embedding failed: {e}")
            raise RuntimeError(f"Failed to generate text embedding: {e}") from e

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Batch-encode multiple text strings into CLIP embeddings.

        Args:
            texts: List of input texts.

        Returns:
            List of embeddings, each a list of 512 floats.
        """
        if not texts:
            return []

        self._ensure_loaded()

        import torch as _torch

        try:
            inputs = self._tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            ).to(self._device)

            with _torch.no_grad():
                raw_features = self._model.get_text_features(**inputs)

            text_features = self._extract_tensor(raw_features)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu().numpy().tolist()

        except Exception as e:
            logger.error(f"Batch text embedding failed: {e}")
            raise RuntimeError(f"Failed to generate batch text embeddings: {e}") from e

    def embed_image(self, image: "Image.Image") -> List[float]:
        """
        Encode a PIL Image into a CLIP embedding.

        The image is automatically resized and preprocessed by the
        CLIP processor (center crop to 224x224).

        Args:
            image: PIL Image object (any size, RGB or RGBA).

        Returns:
            List of 512 floats representing the image embedding.

        Raises:
            RuntimeError: If CLIP model is not available.
            ValueError: If image is None.
        """
        if image is None:
            raise ValueError("Image input cannot be None")

        self._ensure_loaded()

        import torch as _torch

        try:
            # Convert RGBA to RGB if needed
            if image.mode == "RGBA":
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                image = background
            elif image.mode != "RGB":
                image = image.convert("RGB")

            inputs = self._processor(
                images=image,
                return_tensors="pt",
            ).to(self._device)

            with _torch.no_grad():
                raw_features = self._model.get_image_features(**inputs)

            image_features = self._extract_tensor(raw_features)
            # L2 normalize for cosine similarity
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            embedding = image_features.squeeze().cpu().numpy().tolist()

            return embedding

        except Exception as e:
            logger.error(f"Image embedding failed: {e}")
            raise RuntimeError(f"Failed to generate image embedding: {e}") from e

    def embed_image_from_base64(self, base64_string: str) -> List[float]:
        """
        Encode a base64-encoded image into a CLIP embedding.

        Args:
            base64_string: Base64-encoded image data (JPEG, PNG, or WebP).
                          May include the data URI prefix
                          (e.g., "data:image/jpeg;base64,...").

        Returns:
            List of 512 floats representing the image embedding.

        Raises:
            ValueError: If base64 string is invalid or image cannot be decoded.
        """
        if not base64_string:
            raise ValueError("Base64 string cannot be empty")

        image = decode_base64_image(base64_string)
        return self.embed_image(image)

    def compute_similarity(
        self,
        embedding_a: List[float],
        embedding_b: List[float],
    ) -> float:
        """
        Compute cosine similarity between two CLIP embeddings.

        Args:
            embedding_a: First embedding (512 floats).
            embedding_b: Second embedding (512 floats).

        Returns:
            Cosine similarity score between -1.0 and 1.0.
        """
        a = np.array(embedding_a)
        b = np.array(embedding_b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    @property
    def is_available(self) -> bool:
        """Check if all CLIP dependencies are available."""
        return TORCH_AVAILABLE and CLIP_AVAILABLE and PIL_AVAILABLE

    @property
    def is_loaded(self) -> bool:
        """Check if the CLIP model is loaded in memory."""
        return self._initialized


# ---------------------------------------------------------------------------
# Utility: Base64 Image Decoding
# ---------------------------------------------------------------------------

def decode_base64_image(base64_string: str) -> "Image.Image":
    """
    Decode a base64 string into a PIL Image.

    Handles both raw base64 and data URI formatted strings
    (e.g., "data:image/jpeg;base64,/9j/4AAQ...").

    Args:
        base64_string: Base64-encoded image data.

    Returns:
        PIL Image object.

    Raises:
        ValueError: If the string cannot be decoded into a valid image.
    """
    if not PIL_AVAILABLE:
        raise RuntimeError("Pillow is required. Install with: pip install Pillow")

    try:
        # Strip data URI prefix if present
        if "," in base64_string and base64_string.startswith("data:"):
            base64_string = base64_string.split(",", 1)[1]

        # Decode base64
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_bytes))
        return image

    except base64.binascii.Error as e:
        raise ValueError(f"Invalid base64 encoding: {e}") from e
    except Exception as e:
        raise ValueError(f"Failed to decode image from base64: {e}") from e


# ---------------------------------------------------------------------------
# Singleton Access
# ---------------------------------------------------------------------------

_clip_service: Optional[CLIPEmbeddingService] = None


def get_clip_service(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> CLIPEmbeddingService:
    """
    Get or create the CLIPEmbeddingService singleton.

    Args:
        model_name: HuggingFace model identifier (default: clip-vit-base-patch32).
        device: "cpu" or "cuda" (auto-detected if None).

    Returns:
        CLIPEmbeddingService singleton instance.
    """
    global _clip_service
    if _clip_service is None:
        from ..config import settings

        _clip_service = CLIPEmbeddingService(
            model_name=model_name or settings.CLIP_MODEL_NAME,
            device=device or settings.CLIP_DEVICE,
        )
    return _clip_service
