"""
Document chunking with recursive character splitting for contracts and legal documents.
"""

from typing import List, Dict, Any
from dataclasses import dataclass
import hashlib

from config.settings import settings


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""

    id: str
    text: str
    chunk_index: int
    document_id: str
    metadata: Dict[str, Any]


class DocumentChunker:
    """
    Recursive character text splitter optimized for legal documents.

    Splits text recursively using a list of separators to preserve
    document structure (paragraphs, sentences, etc.).
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        separators: List[str] = None,
    ):
        """
        Initialize the chunker.

        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
            separators: List of separators to split on (tried in order)
        """
        self.chunk_size = chunk_size or settings.chunking.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunking.chunk_overlap
        self.separators = separators or settings.chunking.separators

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

    def _generate_chunk_id(self, document_id: str, chunk_index: int, text: str) -> str:
        """
        Generate a unique ID for a chunk.

        Args:
            document_id: Parent document ID
            chunk_index: Index of chunk within document
            text: Chunk text

        Returns:
            MD5 hash as chunk ID
        """
        id_string = f"{document_id}:{chunk_index}:{text[:50]}"
        return hashlib.md5(id_string.encode()).hexdigest()

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split text using the provided separators.

        Args:
            text: Text to split
            separators: List of separators to try

        Returns:
            List of text splits
        """
        if not separators:
            # Base case: no more separators, return text as-is
            return [text] if text else []

        # Try first separator
        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            # Empty separator means split into characters
            splits = list(text)
        else:
            splits = text.split(separator)

        # Keep separator with splits (except empty separator)
        if separator != "":
            splits = [
                (split + separator if i < len(splits) - 1 else split)
                for i, split in enumerate(splits)
                if split
            ]
        else:
            splits = [s for s in splits if s]

        # Recursively process splits that are too large
        final_splits = []
        for split in splits:
            if len(split) <= self.chunk_size:
                final_splits.append(split)
            else:
                # Split is too large, recurse with next separator
                final_splits.extend(self._split_text(split, remaining_separators))

        return final_splits

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """
        Merge splits into chunks of appropriate size with overlap.

        Args:
            splits: List of text splits

        Returns:
            List of chunks
        """
        chunks = []
        current_chunk = []
        current_length = 0

        for split in splits:
            split_length = len(split)

            # If adding this split would exceed chunk_size, finalize current chunk
            if current_length + split_length > self.chunk_size and current_chunk:
                chunk_text = "".join(current_chunk)
                chunks.append(chunk_text)

                # Start new chunk with overlap
                # Find splits that fit in overlap window
                overlap_length = 0
                overlap_splits = []
                for prev_split in reversed(current_chunk):
                    if overlap_length + len(prev_split) <= self.chunk_overlap:
                        overlap_splits.insert(0, prev_split)
                        overlap_length += len(prev_split)
                    else:
                        break

                current_chunk = overlap_splits
                current_length = overlap_length

            # Add split to current chunk
            current_chunk.append(split)
            current_length += split_length

        # Add final chunk if not empty
        if current_chunk:
            chunk_text = "".join(current_chunk)
            chunks.append(chunk_text)

        return chunks

    def chunk_text(
        self, text: str, document_id: str, document_metadata: Dict[str, Any] = None
    ) -> List[Chunk]:
        """
        Split text into chunks with metadata.

        Args:
            text: Text to chunk
            document_id: ID of the source document
            document_metadata: Metadata from source document

        Returns:
            List of Chunk objects
        """
        if not text:
            return []

        # Split text recursively
        splits = self._split_text(text, self.separators)

        # Merge into appropriately-sized chunks
        chunk_texts = self._merge_splits(splits)

        # Create Chunk objects
        chunks = []
        for i, chunk_text in enumerate(chunk_texts):
            chunk_metadata = {
                "chunk_size": len(chunk_text),
                "total_chunks": len(chunk_texts),
                **(document_metadata or {}),
            }

            chunk = Chunk(
                id=self._generate_chunk_id(document_id, i, chunk_text),
                text=chunk_text.strip(),
                chunk_index=i,
                document_id=document_id,
                metadata=chunk_metadata,
            )
            chunks.append(chunk)

        return chunks

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Chunk]:
        """
        Chunk multiple documents.

        Args:
            documents: List of documents with 'id', 'text', and 'metadata' keys

        Returns:
            List of all chunks from all documents
        """
        all_chunks = []

        for doc in documents:
            doc_id = doc.get("id")
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})

            if not doc_id or not text:
                print(f"Skipping document with missing id or text: {doc}")
                continue

            chunks = self.chunk_text(text, doc_id, metadata)
            all_chunks.extend(chunks)

        return all_chunks


def main():
    """Test the chunker."""
    chunker = DocumentChunker()

    print("=== Testing Document Chunker ===\n")
    print(f"Chunk size: {chunker.chunk_size}")
    print(f"Chunk overlap: {chunker.chunk_overlap}")
    print(f"Separators: {chunker.separators}\n")

    # Test document (sample contract text)
    test_text = """
    MASTER SERVICE AGREEMENT

    This Master Service Agreement ("Agreement") is entered into as of January 1, 2024
    ("Effective Date") by and between ABC Corporation ("Company") and XYZ Services Inc.
    ("Service Provider").

    1. SERVICES
    Service Provider agrees to provide professional consulting services as described in
    individual Statements of Work ("SOW") executed under this Agreement. Each SOW will
    specify the scope, deliverables, timeline, and fees for the services.

    2. PAYMENT TERMS
    Company agrees to pay Service Provider the fees specified in each SOW. Payment is
    due within thirty (30) days of invoice date. Late payments will incur interest at
    1.5% per month or the maximum rate permitted by law, whichever is less.

    3. CONFIDENTIALITY
    Each party agrees to maintain the confidentiality of the other party's Confidential
    Information and use it only for purposes of performing under this Agreement.
    Confidential Information does not include information that: (a) is publicly known
    through no breach of this Agreement; (b) was rightfully received by the receiving
    party from a third party; or (c) was independently developed without use of
    Confidential Information.

    4. TERM AND TERMINATION
    This Agreement commences on the Effective Date and continues for one (1) year,
    automatically renewing for successive one-year terms unless either party provides
    sixty (60) days written notice of non-renewal. Either party may terminate this
    Agreement for cause upon thirty (30) days written notice if the other party
    materially breaches this Agreement and fails to cure such breach within the notice
    period.

    5. GOVERNING LAW
    This Agreement shall be governed by and construed in accordance with the laws of
    the State of California, without regard to its conflict of law provisions.
    """

    # Chunk the text
    document_metadata = {
        "source_file": "test_contract.txt",
        "file_type": "txt",
        "document_type": "Master Service Agreement",
    }

    chunks = chunker.chunk_text(
        text=test_text, document_id="test_doc_001", document_metadata=document_metadata
    )

    # Display results
    print("=== Chunking Results ===")
    print(f"Generated {len(chunks)} chunks\n")

    for i, chunk in enumerate(chunks, 1):
        print(f"--- Chunk {i} ---")
        print(f"ID: {chunk.id}")
        print(f"Index: {chunk.chunk_index}/{chunk.metadata['total_chunks']-1}")
        print(f"Length: {chunk.metadata['chunk_size']} characters")
        print(f"Text preview:\n{chunk.text[:200]}...")
        print()


if __name__ == "__main__":
    main()
