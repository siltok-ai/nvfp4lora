"""交付复制的实际二进制/覆盖与损坏检查，不触碰GPU或已验证模型。"""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import archive_training_delivery as delivery


class DeliveryCopyTests(unittest.TestCase):
    def test_binary_identity_and_existing_directory_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            data = bytes(range(256)) * 4096
            source.write_bytes(data)
            output = root / "new-artifact"
            hashes = delivery.verified_copy({"payload/weights.bin": source}, output)
            self.assertEqual((output / "payload/weights.bin").read_bytes(), data)
            self.assertEqual(source.read_bytes(), data)
            self.assertEqual(hashes["payload/weights.bin"], delivery.file_hash(source))
            with self.assertRaisesRegex(AssertionError, "new artifact"):
                delivery.verified_copy({"payload/weights.bin": source}, output)
            self.assertEqual((output / "payload/weights.bin").read_bytes(), data)

    def test_corrupt_copy_is_not_certified_and_evidence_is_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.bin", root / "failed-artifact"
            source.write_bytes(b"original source bytes")
            original_copy = delivery.shutil.copy2
            def corrupt(source_path, target_path):
                original_copy(source_path, target_path)
                Path(target_path).write_bytes(b"bad copy")
            with patch.object(delivery.shutil, "copy2", side_effect=corrupt):
                with self.assertRaisesRegex(AssertionError, "Copy changed"):
                    delivery.verified_copy({"weights.bin": source}, output)
            self.assertEqual(source.read_bytes(), b"original source bytes")
            self.assertEqual((output / "weights.bin").read_bytes(), b"bad copy")
            self.assertFalse((output / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
