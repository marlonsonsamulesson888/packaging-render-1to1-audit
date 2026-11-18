# 效果图 1:1 复量

对已知 px/mm 的 1:1 彩盒效果图，用窗口蒙版和透明图层量餐具离窗边距离、相邻净距、装饰离折线距离和源照片有效 ppi。只给数据，不评审美，不代替白样。

入口：[SKILL.md](SKILL.md)。需要 Python 3.10+、numpy、Pillow、scipy。

```sh
python scripts/audit_render.py --px-per-mm 10 --window 窗.png --item 刀=刀.png --decor 屋脊=屋脊.png --fold-y 0 --source 茶叉=829:146 --md 复量记录.md
python -m unittest discover -s tests -v
```

测试全部用合成图层。透视/3D 角度的效果图不适用。

MIT，Copyright (c) 2026 Marlonson Samulesson，见 [LICENSE](LICENSE)。
