"""效果图 1:1 复量单元测试（合成图层）。运行：python -m unittest discover -s tests -v"""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("audit_render", ROOT / "scripts/audit_render.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

W, H, PPM = 400, 500, 2  # 200×250 mm 画布，2 px/mm


def rgba(path, box, alpha=255, size=(W, H)):
    a = np.zeros((size[1], size[0], 4), np.uint8)
    x0, y0, x1, y1 = box
    a[y0:y1, x0:x1] = (200, 200, 200, alpha)
    Image.fromarray(a, "RGBA").save(path)


class Render(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        d = self.d = Path(self.t.name)
        win = np.zeros((H, W), np.uint8)
        win[40:360, 20:380] = 255  # 窗：x 10–190mm，y 20–180mm
        Image.fromarray(win, "L").save(d / "win.png")
        rgba(d / "a.png", (40, 60, 80, 450))     # 件A：x 20–40mm，离窗左边 10mm，下部压纸
        rgba(d / "b.png", (100, 60, 140, 300))   # 件B：x 50–70mm，与 A 净距 10mm
        rgba(d / "deco.png", (0, 9, 400, 39))    # 装饰：y 4.5–19.5mm
        a = np.zeros((H, W, 4), np.uint8)
        a[7:9, :] = (0, 0, 0, 40)                # 抗锯齿边：y 3.5–4.5mm
        a[9:39, :] = (0, 0, 0, 255)
        Image.fromarray(a, "RGBA").save(d / "deco_aa.png")

    def test_items_and_gaps(self):
        r = m.audit(PPM, self.d / "win.png", [f"A={self.d/'a.png'}", f"B={self.d/'b.png'}"], min_item_gap=8)
        A, B = r["items"]
        self.assertEqual(A["bbox_mm"], [20.0, 30.0, 40.0, 225.0])
        self.assertAlmostEqual(A["inside_window_pct"], 100 * (360 - 60) / (450 - 60), places=1)
        self.assertEqual(A["window_clear_mm"]["top"], 10.0)      # 顶 y30 − 窗沿 y20
        self.assertEqual(A["window_clear_mm"]["left"], 10.0)
        self.assertEqual(A["window_clear_mm"]["bottom"], "出窗/压纸")
        self.assertEqual(B["inside_window_pct"], 100.0)
        self.assertAlmostEqual(r["adjacent_visible_gaps_mm"][0]["gap_mm"], 10.0, delta=0.6)
        self.assertEqual(r["summary"]["fail"], 0)

    def test_decor_fold_distance_solid_vs_aa(self):
        r = m.audit(PPM, None, [], [f"屋脊={self.d/'deco_aa.png'}"], folds_y_mm=[0])
        d = r["decor"][0]
        self.assertAlmostEqual(d["fold_clear_mm_solid"]["y=0"], 4.5)
        self.assertAlmostEqual(d["fold_clear_mm_any_alpha"]["y=0"], 3.5)
        self.assertEqual(r["summary"]["fail"], 1)

    def test_too_close_fails(self):
        r = m.audit(PPM, self.d / "win.png", [f"A={self.d/'a.png'}"], min_window_clear=12)
        self.assertTrue(any(c["status"] == "fail" and "离窗上边" in c["rule"] for c in r["checks"]))

    def test_source_ppi(self):
        r = m.audit(PPM, self.d / "win.png", sources=["茶叉=829:146", "大图=2000:146"])
        self.assertAlmostEqual(r["sources"][0]["effective_ppi"], 144.2, places=1)
        st = [c["status"] for c in r["checks"]]
        self.assertEqual(st, ["fail", "ok"])

    def test_offset_layer(self):
        small = self.d / "s.png"
        rgba(small, (0, 0, 20, 20), size=(20, 20))
        r = m.audit(PPM, self.d / "win.png", [f"S={small}@100,100"])
        self.assertEqual(r["items"][0]["bbox_mm"], [50.0, 50.0, 60.0, 60.0])

    def test_cli(self):
        md = self.d / "r.md"
        rc = m.main(["--px-per-mm", "2", "--window", str(self.d / "win.png"), "--item", f"A={self.d/'a.png'}",
                     "--decor", f"D={self.d/'deco.png'}", "--fold-y", "0", "--json", str(self.d / "r.json"), "--md", str(md)])
        self.assertEqual(rc, 1)  # 装饰离顶 4.5 < 5
        self.assertIn("离折线", md.read_text(encoding="utf-8"))
        self.assertEqual(m.main(["--px-per-mm", "2"]), 2)


if __name__ == "__main__":
    unittest.main()
