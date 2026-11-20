#!/usr/bin/env python3
"""1:1 效果图复量（只给数据，不判定好看或可印刷）。

在已知 px/mm 的效果图画布上，用透明图层量：
  * 餐具/产品（--item）：外包框 mm、在窗内比例、离窗边最近距离、相邻两件可见净距；
  * 装饰（--decor）：离折线（盒顶、侧边等）最近距离，分实心像素和含抗锯齿像素；
  * 源图有效分辨率（--source 名称=源像素:放置毫米）。
图层 PNG 与画布同尺寸，或写成 名称=路径@x,y 指定左上角像素偏移。
依赖：numpy、Pillow、scipy（distance_transform_edt）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt


SIDE_CN = {"top": "上", "bottom": "下", "left": "左", "right": "右"}


def parse_layer(spec):
    if "=" not in spec:
        raise ValueError(f"图层写成 名称=路径[@x,y]：{spec}")
    name, rest = spec.split("=", 1)
    off = (0, 0)
    if "@" in rest:
        rest, o = rest.rsplit("@", 1)
        off = tuple(int(v) for v in o.split(","))
    return name, Path(rest), off


def load_alpha(path, off, canvas):
    im = Image.open(path).convert("RGBA")
    a = np.array(im.getchannel("A"))
    full = np.zeros(canvas, dtype=np.uint8)
    x, y = off
    h, w = a.shape
    y0, x0 = max(0, y), max(0, x)
    y1, x1 = min(canvas[0], y + h), min(canvas[1], x + w)
    if y1 > y0 and x1 > x0:
        full[y0:y1, x0:x1] = a[y0 - y:y1 - y, x0 - x:x1 - x]
    return full


def bbox_mm(mask, ppm):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [round(float(v) / ppm, 2) for v in (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)]


def runs(win):
    """每个窗内像素到四个方向窗边的连续窗内像素数（含自身）。"""
    h, w = win.shape
    up = np.zeros((h, w), np.int32); dn = np.zeros((h, w), np.int32)
    lf = np.zeros((h, w), np.int32); rt = np.zeros((h, w), np.int32)
    wi = win.astype(np.int32)
    up[0] = wi[0]
    for r in range(1, h):
        up[r] = (up[r - 1] + 1) * wi[r]
    dn[-1] = wi[-1]
    for r in range(h - 2, -1, -1):
        dn[r] = (dn[r + 1] + 1) * wi[r]
    lf[:, 0] = wi[:, 0]
    for c in range(1, w):
        lf[:, c] = (lf[:, c - 1] + 1) * wi[:, c]
    rt[:, -1] = wi[:, -1]
    for c in range(w - 2, -1, -1):
        rt[:, c] = (rt[:, c + 1] + 1) * wi[:, c]
    return {"top": up, "bottom": dn, "left": lf, "right": rt}


def min_gap_mm(a, b, ppm):
    if not a.any() or not b.any():
        return None
    d = distance_transform_edt(~a)
    return round(float(d[b].min()) / ppm, 2)


def audit(px_per_mm, window=None, items=(), decors=(), folds_y_mm=(), folds_x_mm=(), sources=(),
          canvas=None, alpha_any=1, alpha_solid=128, min_window_clear=5.0, min_fold_clear=5.0,
          min_ppi=300.0, min_item_gap=None, clear_sides=("top", "left", "right")):
    ppm = float(px_per_mm)
    win = None
    if window:
        wm = np.array(Image.open(window).convert("L")) > 127
        canvas = canvas or wm.shape
        win = wm
    layers = [parse_layer(s) for s in items] + [parse_layer(s) for s in decors]
    if canvas is None:
        if not layers:
            raise ValueError("至少给窗口蒙版或一个图层")
        canvas = Image.open(layers[0][1]).size[::-1]
    res = {"canvas_px": [canvas[1], canvas[0]], "px_per_mm": ppm,
           "canvas_mm": [round(canvas[1] / ppm, 2), round(canvas[0] / ppm, 2)],
           "items": [], "decor": [], "sources": [], "checks": [],
           "verdict": "仅测量数据；好不好看由审美员判断，能否印刷由技术员判断"}
    if win is not None:
        res["window_bbox_mm"] = bbox_mm(win, ppm)
        rr = runs(win)
    masks = []
    for spec in items:
        name, path, off = parse_layer(spec)
        a = load_alpha(path, off, canvas)
        m_any = a >= alpha_any
        rec = {"name": name, "bbox_mm": bbox_mm(m_any, ppm), "pixels": int(m_any.sum())}
        if win is not None and m_any.any():
            inside = m_any & win
            rec["inside_window_pct"] = round(100.0 * inside.sum() / m_any.sum(), 2)
            rec["window_clear_mm"] = {}
            if inside.any():
                outside = m_any & ~win
                for side, arr in rr.items():
                    v = float(arr[inside].min() - 1) / ppm
                    rec["window_clear_mm"][side] = "出窗/压纸" if (v == 0 and outside.any()) else round(v, 2)
        masks.append((name, m_any))
        res["items"].append(rec)
    # Keep full-layer geometry diagnostic separate from window-visible geometry.
    # EDT measures nearest occupied pixel centres; retain that existing convention.
    for (n1, a1), (n2, a2) in zip(masks, masks[1:]):
        raw_gap = min_gap_mm(a1, a2, ppm)
        res.setdefault("adjacent_full_layer_gaps_mm", []).append(
            {"pair": f"{n1}-{n2}", "gap_mm": raw_gap})
        if win is None:
            g, reason = None, "未提供窗口蒙版，窗内可见净距未测量"
        else:
            g = min_gap_mm(a1 & win, a2 & win, ppm)
            reason = "至少一件在窗内无可见像素，无法量相邻可见净距" if g is None else None
        rec = {"pair": f"{n1}-{n2}", "gap_mm": g,
               "scope": "item alpha intersected with window", "reason": reason}
        res.setdefault("adjacent_visible_gaps_mm", []).append(rec)
        if g is None:
            res["checks"].append({"rule": f"{n1}-{n2} 可见净距", "status": "warn", "detail": reason})
        elif min_item_gap is not None:
            res["checks"].append({"rule": f"{n1}-{n2} 可见净距 ≥{min_item_gap}mm", "status": "ok" if g >= min_item_gap else "fail", "detail": f"{g} mm"})
    for spec in decors:
        name, path, off = parse_layer(spec)
        a = load_alpha(path, off, canvas)
        rec = {"name": name}
        for label, thr in (("solid", alpha_solid), ("any_alpha", alpha_any)):
            m = a >= thr
            rec[f"bbox_mm_{label}"] = bbox_mm(m, ppm)
            if not m.any():
                continue
            ys, xs = np.nonzero(m)
            dists = []
            for fy in folds_y_mm:
                dists.append(("y=%g" % fy, float(np.maximum(0, np.maximum(ys / ppm - fy, fy - (ys + 1) / ppm)).min())))
            for fx in folds_x_mm:
                dists.append(("x=%g" % fx, float(np.maximum(0, np.maximum(xs / ppm - fx, fx - (xs + 1) / ppm)).min())))
            rec[f"fold_clear_mm_{label}"] = {k: round(v, 2) for k, v in dists}
        for k, v in rec.get("fold_clear_mm_solid", {}).items():
            res["checks"].append({"rule": f"{name} 离折线 {k} ≥{min_fold_clear}mm（实心）", "status": "ok" if v >= min_fold_clear else "fail", "detail": f"{v} mm"})
        for k, v in rec.get("fold_clear_mm_any_alpha", {}).items():
            if v < min_fold_clear:
                res["checks"].append({"rule": f"{name} 离折线 {k}（含抗锯齿）", "status": "warn", "detail": f"{v} mm"})
        res["decor"].append(rec)
    for s in sources:
        name, val = s.split("=", 1)
        px, mm = val.split(":")
        ppi = float(px) / (float(mm) / 25.4)
        res["sources"].append({"name": name, "source_px": float(px), "placed_mm": float(mm), "effective_ppi": round(ppi, 1)})
        res["checks"].append({"rule": f"{name} 源图有效 ≥{min_ppi:g}ppi", "status": "ok" if ppi >= min_ppi else "fail",
                              "detail": f"{ppi:.1f} ppi；量产至少需 {int(np.ceil(float(mm) / 25.4 * min_ppi))} px" + ("" if ppi >= min_ppi else "，效果图可用，量产需高清原图/实拍或负责人批准")})
    res["min_window_clear_rule_mm"] = min_window_clear
    for it in res["items"]:
        for side in clear_sides:
            c = it.get("window_clear_mm", {}).get(side)
            if c is None:
                continue
            ok = isinstance(c, float) and c >= min_window_clear
            res["checks"].append({"rule": f"{it['name']} 离窗{SIDE_CN[side]}边 ≥{min_window_clear}mm", "status": "ok" if ok else "fail",
                                  "detail": f"{c}" + (" mm" if isinstance(c, float) else "")})
    st = [c["status"] for c in res["checks"]]
    res["summary"] = {k: st.count(k) for k in ("ok", "warn", "fail")}
    return res


def to_markdown(r):
    L = ["# 效果图 1:1 复量", "", f"> {r['verdict']}。AI 测量，待人工确认。", "",
         f"- 画布 {r['canvas_px'][0]}×{r['canvas_px'][1]} px = {r['canvas_mm'][0]}×{r['canvas_mm'][1]} mm（{r['px_per_mm']} px/mm）"]
    if r.get("window_bbox_mm"):
        b = r["window_bbox_mm"]
        L.append(f"- 窗口外包框 x {b[0]}–{b[2]}、y {b[1]}–{b[3]} mm（宽 {round(b[2] - b[0], 2)}）")
    for it in r["items"]:
        L.append(f"- {it['name']}：外包框 {it['bbox_mm']}，窗内 {it.get('inside_window_pct')}%，离窗边 {it.get('window_clear_mm')}")
    for g in r.get("adjacent_visible_gaps_mm", []):
        L.append(f"- 相邻可见净距 {g['pair']}：" + (f"{g['gap_mm']} mm" if g["gap_mm"] is not None else g["reason"]))
    for d in r["decor"]:
        L.append(f"- 装饰 {d['name']}：离折线 实心 {d.get('fold_clear_mm_solid')}，含抗锯齿 {d.get('fold_clear_mm_any_alpha')}")
    for s in r["sources"]:
        L.append(f"- 源图 {s['name']}：{s['source_px']:g}px 放 {s['placed_mm']:g}mm → {s['effective_ppi']} ppi")
    L += ["", "## 检查"] + [f"- [{c['status']}] {c['rule']}：{c['detail']}" for c in r["checks"]]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="效果图 1:1 复量")
    ap.add_argument("--px-per-mm", type=float, required=True)
    ap.add_argument("--window", help="窗口蒙版 PNG（白=窗）")
    ap.add_argument("--item", action="append", default=[], help="名称=图层.png[@x,y]，按从左到右顺序给")
    ap.add_argument("--decor", action="append", default=[], help="名称=装饰图层.png[@x,y]")
    ap.add_argument("--fold-y", type=float, action="append", default=[], help="水平折线位置 mm（盒顶=0）")
    ap.add_argument("--fold-x", type=float, action="append", default=[], help="竖直折线位置 mm")
    ap.add_argument("--source", action="append", default=[], help="名称=源像素:放置毫米，如 茶叉=829:146")
    ap.add_argument("--min-window-clear", type=float, default=5.0)
    ap.add_argument("--min-fold-clear", type=float, default=5.0)
    ap.add_argument("--min-item-gap", type=float)
    ap.add_argument("--clear-sides", default="top,left,right", help="要检查离窗边距离的方向，逗号分隔：top,bottom,left,right")
    ap.add_argument("--min-ppi", type=float, default=300.0)
    ap.add_argument("--json"); ap.add_argument("--md")
    a = ap.parse_args(argv)
    try:
        r = audit(a.px_per_mm, a.window, a.item, a.decor, a.fold_y, a.fold_x, a.source,
                  min_window_clear=a.min_window_clear, min_fold_clear=a.min_fold_clear,
                  min_ppi=a.min_ppi, min_item_gap=a.min_item_gap,
                  clear_sides=tuple(x.strip() for x in a.clear_sides.split(",") if x.strip()))
    except Exception as e:  # noqa: BLE001
        print(f"错误：{e}", file=sys.stderr)
        return 2
    t = json.dumps(r, ensure_ascii=False, indent=2)
    if a.json:
        Path(a.json).write_text(t, encoding="utf-8")
    else:
        print(t)
    if a.md:
        Path(a.md).write_text(to_markdown(r), encoding="utf-8")
    return 1 if r["summary"]["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())

