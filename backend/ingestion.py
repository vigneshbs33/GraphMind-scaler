"""File ingestion pipeline for GraphMind."""

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import aiofiles
from fastapi import UploadFile

from .config import settings
from .parsers import MultiFormatParser
from .storage import GraphMindStorage

logger = logging.getLogger(__name__)


class ParallelIngestionPipeline:
    """Parallel ingestion pipeline with concurrent embedding and graph building."""

    def __init__(self, storage: GraphMindStorage):
        self.storage = storage
        self.parser = MultiFormatParser()

    async def _persist_upload(self, upload: UploadFile) -> Path:
        """Save uploaded file to disk asynchronously."""
        suffix = Path(upload.filename or "upload.txt").suffix
        file_id = f"{uuid.uuid4().hex}{suffix}"
        destination = settings.UPLOAD_DIR / file_id
        async with aiofiles.open(destination, "wb") as output:
            while chunk := await upload.read(1024 * 1024):
                await output.write(chunk)
        await upload.close()
        return destination

    async def ingest_upload(
        self, upload: UploadFile, file_type: str, metadata: Optional[Dict] = None
    ) -> Dict:
        """Complete ingestion flow directly from an UploadFile."""
        file_path = await self._persist_upload(upload)
        return await self.ingest_file(file_path, file_type, metadata)

    async def ingest_file(
        self, file_path: Path, file_type: str, metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Main ingestion pipeline with parallel processing.

        Steps:
        1. Parse file to extract content
        2. Split content into chunks
        3. Launch two parallel tasks:
           - Task A: Generate embeddings → ChromaDB
           - Task B: Build graph structure → NetworkX
        4. Wait for both tasks
        5. Run merge algorithm
        6. Return ingestion summary

        Return:
        {
            "ingestion_id": "uuid",
            "nodes_created": 10,
            "edges_created": 8,
            "processing_time": 2.5,
            "status": "success"
        }
        """
        start_time = time.time()

        ingestion_id = str(uuid.uuid4())
        metadata = metadata or {}
        metadata["ingestion_id"] = ingestion_id
        metadata["file_path"] = str(file_path)
        metadata["file_type"] = file_type

        try:
            # Step 1: Parse file
            logger.info(f"Parsing file: {file_path}")
            parsed = self.parser.parse_file(file_path, file_type)

            if parsed.get("error"):
                return {"status": "error", "error": parsed["error"]}

            content = parsed["content"]

            # Step 2: Chunk content
            chunks = self.parser.chunk_text(content, chunk_size=settings.CHUNK_SIZE)
            logger.info(f"Created {len(chunks)} chunks")

            # Step 3: Parallel processing
            task_a = self._embed_and_store(chunks, metadata)
            task_b = self._build_graph(chunks, content, metadata)

            nodes_created, edges_created = await asyncio.gather(task_a, task_b)

            # Step 4: Merge algorithm
            logger.info("Running merge algorithm...")
            self.storage.merge_algorithm()

            processing_time = time.time() - start_time

            return {
                "ingestion_id": ingestion_id,
                "nodes_created": nodes_created,
                "edges_created": edges_created,
                "processing_time": round(processing_time, 2),
                "status": "success",
            }

        except Exception as e:
            logger.error(f"Ingestion failed: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}

    async def _embed_and_store(self, chunks: List[str], metadata: Dict) -> int:
        """
        Task A: Generate embeddings and store in ChromaDB.
        """
        nodes_created = 0

        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 10:  # Skip tiny chunks
                continue

            node_id = f"{metadata['ingestion_id']}_chunk_{i}"
            chunk_metadata = {**metadata, "chunk_index": i}

            self.storage.add_node(node_id, chunk, chunk_metadata)
            nodes_created += 1

        logger.info(f"Embedded {nodes_created} chunks")
        return nodes_created

    async def _build_graph(self, chunks: List[str], full_content: str, metadata: Dict) -> int:
        """
        Task B: Extract entities and build graph structure.
        """
        edges_created = 0

        # Extract entities from full content
        entities = self.parser.extract_entities(full_content)
        logger.info(f"Extracted {len(entities)} entities")

        # Create entity nodes
        for entity in entities[:20]:  # Limit to top 20
            entity_id = f"{metadata['ingestion_id']}_entity_{entity.lower().replace(' ', '_')}"
            entity_metadata = {**metadata, "entity_type": "extracted"}

            if not self.storage.graph.has_node(entity_id):
                self.storage.add_node(entity_id, entity, entity_metadata)

        # Create edges between co-occurring entities in chunks
        for i, chunk in enumerate(chunks):
            chunk_entities = [e for e in entities if e.lower() in chunk.lower()]

            # Connect entities that appear together
            for j in range(len(chunk_entities)):
                for k in range(j + 1, len(chunk_entities)):
                    source = (
                        f"{metadata['ingestion_id']}_entity_{chunk_entities[j].lower().replace(' ', '_')}"
                    )
                    target = (
                        f"{metadata['ingestion_id']}_entity_{chunk_entities[k].lower().replace(' ', '_')}"
                    )

                    if self.storage.graph.has_node(source) and self.storage.graph.has_node(target):
                        self.storage.add_edge(source, target, "co_occurs_with", 0.5)
                        edges_created += 1

        # Also create sequential edges between chunks
        chunk_ids = [f"{metadata['ingestion_id']}_chunk_{i}" for i in range(len(chunks))]
        for i in range(len(chunk_ids) - 1):
            if (
                self.storage.graph.has_node(chunk_ids[i])
                and self.storage.graph.has_node(chunk_ids[i + 1])
            ):
                self.storage.add_edge(chunk_ids[i], chunk_ids[i + 1], "sequence", 0.5)
                edges_created += 1

        logger.info(f"Created {edges_created} edges")
        return edges_created


# Alias for backward compatibility
IngestionPipeline = ParallelIngestionPipeline


def parse_metadata(metadata_raw: Optional[str]) -> Dict:
    """Utility to decode metadata payloads supplied as JSON strings."""
    if not metadata_raw:
        return {}
    try:
        data = json.loads(metadata_raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return data

