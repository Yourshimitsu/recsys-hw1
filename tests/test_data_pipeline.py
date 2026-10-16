import gzip
import hashlib
import io
from contextlib import redirect_stdout
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import data_pipeline as pipeline


class DataProtocolTests(unittest.TestCase):
    def test_streaming_extraction_checks_source_and_discards_private_attributes(self):
        event = {
            "type": "WatchEvent", "payload": {"action": "started"}, "public": True,
            "actor": "ExampleUser", "actor_attributes": {"email": "must-not-be-retained@example.invalid"},
            "created_at": "2014-10-01T12:05:37-07:00",
            "repository": {"id": 11, "language": "Python", "owner": {"login": "ExampleOwner"}, "description": "not retained"},
        }
        missing = dict(event, actor=None)
        source_bytes = gzip.compress((json.dumps(event) + "\n" + json.dumps(missing) + "\n" + json.dumps({"type": "PushEvent"}) + "\n").encode(), mtime=0)
        response = io.BytesIO(source_bytes)
        response.headers = {"Content-Length": str(len(source_bytes)), "ETag": "sample-etag"}
        with tempfile.TemporaryDirectory() as directory, patch("urllib.request.urlopen", return_value=response):
            cache = Path(directory)
            manifest = pipeline._extract_hour("fake", "https://example.invalid/fake", cache, pipeline._config())
            compact = gzip.decompress((cache / "hours" / "fake.stars.jsonl.gz").read_bytes()).decode()
            row = json.loads(compact)
            self.assertEqual(manifest["source_sha256"], hashlib.sha256(source_bytes).hexdigest())
            self.assertEqual(manifest["source_compressed_bytes"], len(source_bytes))
            self.assertEqual(manifest["counts"]["raw_events"], 3)
            self.assertEqual(manifest["counts"]["retained_star_events"], 1)
            self.assertEqual(row[0], hashlib.sha256(b"exampleuser").hexdigest())
            self.assertEqual(row[1:], [11, pipeline._timestamp("2014-10-01T19:05:37Z"), "Python", "exampleowner"])
            self.assertNotIn("must-not-be-retained", compact)
            self.assertNotIn("description", compact)

    def test_timestamps_respect_offsets(self):
        self.assertEqual(
            pipeline._timestamp("2014-10-01T12:05:37-07:00"),
            pipeline._timestamp("2014-10-01T19:05:37Z"),
        )
        with self.assertRaises(ValueError):
            pipeline._timestamp("2014-10-01T12:05:37")

    def test_preparation_excludes_future_features_and_prior_seen_pairs(self):
        config = pipeline._config()
        config.update({
            "train_cutoff_utc": "1970-01-01T00:00:10Z",
            "validation_cutoff_utc": "1970-01-01T00:00:20Z",
            "min_train_user_interactions": 2,
            "min_train_item_interactions": 2,
            "max_items": 3,
            "min_final_interactions": 9,
            "max_final_interactions": 20,
            "min_users": 3,
            "min_items": 3,
        })
        rows = [
            ["a", 1, 1, "Python", "owner1"], ["a", 2, 2, "Ruby", "owner2"],
            ["b", 2, 3, "Ruby", "owner2"], ["b", 3, 4, "C", "owner3"],
            ["c", 1, 5, "Python", "owner1"], ["c", 3, 6, "C", "owner3"],
            ["a", 3, 12, "FUTURE_LANGUAGE", "FUTURE_OWNER"],
            ["b", 1, 13, "Python", "owner1"],
            ["a", 1, 14, "Python", "owner1"],
            ["cold_user", 1, 15, "Python", "owner1"],
            ["a", 99, 16, "NEW_ITEM_LANGUAGE", "new_owner"],
            ["b", 99, 17, "NEW_ITEM_LANGUAGE", "new_owner"],
            ["c", 99, 18, "NEW_ITEM_LANGUAGE", "new_owner"],
            ["c", 2, 22, "Ruby", "owner2"],
        ]
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            (cache / "hours").mkdir()
            compact_path = cache / "hours" / "fake.stars.jsonl.gz"
            with gzip.open(compact_path, "wt") as output:
                for row in rows:
                    output.write(json.dumps(row) + "\n")
            pipeline._json_write(compact_path.with_suffix(compact_path.suffix + ".manifest.json"), {
                "source_url": "https://example.invalid/fake", "extracted_sha256": pipeline._sha256_file(compact_path),
                "counts": {"raw_events": len(rows), "retained_star_events": len(rows)},
            })
            toy_config_path = cache / "data_config.json"
            toy_config_path.write_text(json.dumps(config), encoding="utf-8")
            configuration_patch = patch.object(pipeline, "CONFIG_PATH", toy_config_path)
            configuration_patch.start()
            self.addCleanup(configuration_patch.stop)
            with patch.object(pipeline, "_hours", return_value=[("fake", "https://example.invalid/fake")]), redirect_stdout(io.StringIO()):
                pipeline.prepare_data(cache, download=False)
            result = pipeline.load_data(cache)
            self.assertTrue(pipeline.processed_cache_valid(cache))
            with patch.object(pipeline, "_config_sha256", return_value="changed configuration"):
                self.assertFalse(pipeline.processed_cache_valid(cache))
                with self.assertRaisesRegex(ValueError, "stale"):
                    pipeline.load_data(cache)
            with patch.object(pipeline, "_preprocessing_sha256", return_value="changed preprocessing"):
                self.assertFalse(pipeline.processed_cache_valid(cache))
                with self.assertRaisesRegex(ValueError, "stale"):
                    pipeline.load_data(cache)
            self.assertEqual(result["train"].shape, (3, 3))
            self.assertEqual([result[name].nnz for name in ("train", "validation", "test")], [6, 2, 1])
            self.assertEqual(result["train"].multiply(result["validation"]).nnz, 0)
            self.assertEqual((result["train"] + result["validation"]).multiply(result["test"]).nnz, 0)
            for name in ("train", "validation", "test"):
                matrix, timestamps = result[name], result[name + "_timestamps"]
                np.testing.assert_array_equal(matrix.indices, timestamps.indices)
                np.testing.assert_array_equal(matrix.indptr, timestamps.indptr)
                self.assertTrue(np.all(matrix.data == 1))
            self.assertTrue(np.all(result["train_timestamps"].data < 10))
            self.assertTrue(np.all((result["validation_timestamps"].data >= 10) & (result["validation_timestamps"].data < 20)))
            self.assertTrue(np.all(result["test_timestamps"].data >= 20))
            self.assertEqual(result["summary"]["raw"]["duplicate_star_events"], 1)
            self.assertEqual(result["summary"]["splits"]["validation"]["dropped_unknown_user"], 1)
            self.assertEqual(result["summary"]["splits"]["validation"]["dropped_unknown_item"], 3)
            self.assertTrue(all(result["summary"]["requirements_met"].values()))
            mappings = json.loads((cache / "processed" / "private_mappings.json").read_text())
            self.assertNotIn("FUTURE_LANGUAGE", mappings["language_categories"])
            self.assertNotIn("FUTURE_OWNER", mappings["owner_categories"])
            self.assertNotIn(99, mappings["repository_ids"])
            self.assertEqual(result["item_features"]["language"].shape, (3,))
            self.assertEqual(sum(row["n_items"] for row in result["summary"]["language_distribution"]), 3)
            self.assertEqual(sum(row["train_interactions"] for row in result["summary"]["language_distribution"]), 6)
            self.assertEqual(result["summary"]["owners_with_multiple_items"], 0)
            changed_spec = dict(config, event_action="stopped")
            with patch.object(pipeline, "_config", return_value=changed_spec), patch.object(
                pipeline, "_hours", return_value=[("fake", "https://example.invalid/fake")]
            ):
                with self.assertRaisesRegex(ValueError, "different extraction spec"):
                    pipeline.prepare_data(cache, download=False)

    def test_complete_hour_cache_is_verified_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            target = cache / "hours" / "fake.stars.jsonl.gz"
            target.parent.mkdir()
            target.write_bytes(b"arbitrary verified compact payload")
            saved = {"source_url": "https://example.invalid/fake", "extracted_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
            target.with_suffix(target.suffix + ".manifest.json").write_text(json.dumps(saved))
            with patch("urllib.request.urlopen", side_effect=AssertionError("Unexpected network access")):
                migrated = pipeline._extract_hour("fake", saved["source_url"], cache, pipeline._config())
                self.assertEqual(migrated["source_url"], saved["source_url"])
                self.assertEqual(migrated["extraction_spec"], {"schema_version": 1, "event_type": "WatchEvent", "event_action": "started"})
            changed = dict(pipeline._config(), event_action="stopped", download_retries=1)
            with patch("urllib.request.urlopen", side_effect=OSError("different spec requires retrieval")) as fetch:
                with self.assertRaisesRegex(RuntimeError, "different spec requires retrieval"):
                    pipeline._extract_hour("fake", saved["source_url"], cache, changed)
                fetch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
