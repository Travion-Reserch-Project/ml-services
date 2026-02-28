# Pillar 3: Visual Vibe Matcher

## Research Problem
**The Semantic Gap**: Users struggle to describe visual preferences in words.

## Solution Logic
**Cross-Modal Retrieval**: Uses OpenAI CLIP to map user-uploaded photos to the nearest Sri Lankan visual equivalent (Vector Proximity).

## Technology
- **Model**: OpenAI CLIP (ViT-B/32)
- **Vector DB**: ChromaDB
- **Embeddings**: text-embedding-3-small (1536 dimensions)

## Workflow
1. User uploads an image (e.g., "Misty Mountain")
2. CLIP converts image to vector
3. Search ChromaDB for nearest visual match
4. Return: "Riverston Gap" (94% Match)

## Status: PARTIALLY IMPLEMENTED (Knowledge base ready, CLIP integration pending)

## Files
```
pillar_3_visual_matcher/
├── notebooks/
│   └── knowledge_base_builder.ipynb    # ChromaDB setup & embedding
└── data/
    └── sri_lanka_knowledge_base.json   # 80 locations × 6 aspects
```

## Knowledge Base Structure
- 480 documents (80 locations × 6 aspects)
- Aspects: `_history`, `_adventure`, `_nature`, `_culture`, `_logistics`, `_vibe`

## Vector DB Location
`research/vector_db/` (ChromaDB persistent storage)
