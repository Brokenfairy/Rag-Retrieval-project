import unittest
import os
import json
import tempfile
import shutil
from data.state_manager import StateManager, FileStatus

class TestStateManager(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for the state file
        self.test_dir = tempfile.mkdtemp()
        self.state_file_path = os.path.join(self.test_dir, "test_state.json")
        self.state_manager = StateManager(state_file=self.state_file_path)

    def tearDown(self):
        # Remove the temporary directory after the test
        shutil.rmtree(self.test_dir)

    def test_initialization_creates_file(self):
        """Test that initializing StateManager creates the state file."""
        self.assertTrue(os.path.exists(self.state_file_path))
        with open(self.state_file_path, "r") as f:
            data = json.load(f)
        self.assertEqual(data, {})

    def test_add_new_file(self):
        """Test adding a new file to the state."""
        filename = "document.pdf"
        added = self.state_manager.add_file(filename)
        self.assertTrue(added)
        
        state = self.state_manager.get_all_files()
        self.assertIn(filename, state)
        self.assertEqual(state[filename]["status"], FileStatus.PENDING.value)

    def test_add_existing_file(self):
        """Test adding a file that already exists in the state."""
        filename = "document.pdf"
        self.state_manager.add_file(filename)
        added_again = self.state_manager.add_file(filename)
        self.assertFalse(added_again)

    def test_update_status(self):
        """Test updating the status of a file."""
        filename = "document.pdf"
        self.state_manager.add_file(filename)
        
        self.state_manager.update_status(filename, FileStatus.CONVERTED)
        status = self.state_manager.get_file_status(filename)
        self.assertEqual(status, FileStatus.CONVERTED.value)

    def test_update_status_with_details(self):
        """Test updating the status with additional details."""
        filename = "document.pdf"
        self.state_manager.add_file(filename)
        
        details = {"pages": 10}
        self.state_manager.update_status(filename, FileStatus.PARSED, details=details)
        
        record = self.state_manager.get_file_record(filename)
        self.assertEqual(record["status"], FileStatus.PARSED.value)
        self.assertEqual(record["details"]["pages"], 10)

    def test_update_nonexistent_file(self):
        """Test updating a file that wasn't added first (should auto-create)."""
        filename = "ghost_file.pdf"
        self.state_manager.update_status(filename, FileStatus.FAILED)
        
        state = self.state_manager.get_all_files()
        self.assertIn(filename, state)
        self.assertEqual(state[filename]["status"], FileStatus.FAILED.value)

if __name__ == "__main__":
    unittest.main()
