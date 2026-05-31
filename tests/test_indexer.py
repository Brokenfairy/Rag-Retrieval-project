import unittest
from unittest.mock import patch, MagicMock
import os
import shutil
import tempfile
from agents.indexer import MilvusIndexerAgent

class TestMilvusIndexerAgent(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for the indexer
        self.test_dir = tempfile.mkdtemp()
        self.patcher_env = patch.dict(os.environ, {"ZILLIZ_CLOUD_URI": "", "ZILLIZ_CLOUD_TOKEN": "", "MILVUS_COLLECTION": "test_collection"})
        self.patcher_env.start()
        
    def tearDown(self):
        self.patcher_env.stop()
        shutil.rmtree(self.test_dir)

    @patch("agents.indexer.MilvusClient")
    @patch("agents.indexer.OllamaEmbeddings")
    def test_initialization(self, mock_embeddings, mock_client):
        """Test that MilvusIndexerAgent initializes correctly."""
        # Mock embed_query behavior for dimension inference
        mock_embeddings.return_value.embed_query.return_value = [0.1] * 768
        
        indexer = MilvusIndexerAgent()
        
        # Verify client and collection initialization
        mock_client.assert_called_once()
        self.assertEqual(indexer.collection_name, "test_collection")
        self.assertEqual(indexer.dimension, 768)
        self.assertTrue(indexer.is_ready())

    @patch("agents.indexer.MilvusClient")
    @patch("agents.indexer.OllamaEmbeddings")
    def test_index_document(self, mock_embeddings, mock_client):
        """Test indexing a document."""
        # Mock embed_query behavior for dimension inference
        mock_embeddings.return_value.embed_query.return_value = [0.1] * 768
        
        indexer = MilvusIndexerAgent()
        
        # Mock embed_documents behavior for indexing
        mock_embeddings.return_value.embed_documents.return_value = [[0.1] * 768]
        
        document_text = "This is a test document."
        metadata = {"filename": "test.txt", "upload_date": "2023-10-27"}
        
        count = indexer.index_document(document_text, metadata)
        
        # Verify chunking and insertion
        self.assertEqual(count, 1) # Should result in 1 chunk
        mock_client.return_value.insert.assert_called_once()
        
        # Inspect arguments to insert
        args, kwargs = mock_client.return_value.insert.call_args
        self.assertEqual(kwargs['collection_name'], "test_collection")
        self.assertEqual(len(kwargs['data']), 1)
        self.assertEqual(kwargs['data'][0]['text'], "This is a test document.")

    @patch("agents.indexer.MilvusClient")
    @patch("agents.indexer.OllamaEmbeddings")
    def test_search(self, mock_embeddings, mock_client):
        """Test searching for documents."""
        # Mock embed_query behavior for dimension inference
        mock_embeddings.return_value.embed_query.return_value = [0.1] * 768
        
        indexer = MilvusIndexerAgent()
        
        # Mock embed_query behavior for search query
        mock_embeddings.return_value.embed_query.return_value = [0.2] * 768
        
        # Mock search results
        mock_client.return_value.search.return_value = [[{"id": 1, "entity": {"text": "result"}}]]
        
        results = indexer.search("query")
        
        # Verify search call
        mock_client.return_value.search.assert_called_once()
        self.assertEqual(len(results), 1)

if __name__ == "__main__":
    unittest.main()
