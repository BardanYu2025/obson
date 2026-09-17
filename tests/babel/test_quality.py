import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from obson.babel.data import validate_frame
from obson.babel.quality import audit_pairs, window_quality


class QualityTests(unittest.TestCase):
    def test_trace_and_reject_changed_source_or_chart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "rb").mkdir()
            df = pd.DataFrame({"datetime": ["2026-01-02 14:00", "2026-01-05 09:00"],
                               "open": [100., 130.], "high": [102., 132.],
                               "low": [99., 129.], "close": [101., 131.], "volume": [0., 10.]})
            source = root / "rb/rb2605_60m.csv"
            df.to_csv(source, index=False)
            validated = validate_frame(df)
            digest = hashlib.sha256(pd.util.hash_pandas_object(validated, index=False).values.tobytes()).hexdigest()
            meta = {"window": 2, "checkpoint": "test", "manifest": {"sources": [
                {"key": "rb/60/rb2605", "sha256": digest}]}}
            index = root / "index.npz"
            np.savez(index, metadata=json.dumps(meta), series=[0], row=[1])
            bars = df[["open", "high", "low", "close"]].to_dict("records")
            packet = {"schema": "babel-pairs-v1", "packet_id": "test", "cases": [
                {"id": "P001", "query": bars, "left": bars, "right": bars}]}
            key = {"schema": "babel-pairs-v1", "packet_id": "test", "checkpoint_sha256": "test",
                   "cases": {"P001": {"query_index": 0, "left": "model", "right": "path",
                                      "hit_ids": {"model": 0, "path": 0}}}}
            pairs, answer = root / "pairs.json", root / "key.json"
            pairs.write_text(json.dumps(packet))
            answer.write_text(json.dumps(key))
            result = audit_pairs(root, index, pairs, answer)
            self.assertEqual(result["unique_charts"], 1)
            self.assertEqual(result["charts"][0]["largest_price_gaps"][0]["signed_gap"], 29)
            self.assertEqual(result["charts"][0]["zero_volume_bars"], 1)
            packet["cases"][0]["query"][0]["open"] = 99
            pairs.write_text(json.dumps(packet))
            with self.assertRaisesRegex(ValueError, "Chart/source mismatch"):
                audit_pairs(root, index, pairs, answer)
            df.loc[0, "volume"] = 1
            df.to_csv(source, index=False)
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                audit_pairs(root, index, pairs, answer)

    def test_flat_window_has_no_infinite_ratio(self):
        df = pd.DataFrame({"datetime": pd.date_range("2026-01-01", periods=2, freq="h"),
                           "open": [1., 2.], "high": [1., 2.], "low": [1., 2.],
                           "close": [1., 2.], "volume": [0., 0.]})
        report = window_quality(df, 60)
        self.assertIsNone(report["largest_price_gaps"][0]["gap_over_median_range"])
        json.dumps(report, allow_nan=False)
