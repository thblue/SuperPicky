# -*- coding: utf-8 -*-
"""SuperPicky 正式测试包。

含 __init__.py 是有意为之：pytest 据此向上寻找到不含 __init__.py 的
仓库根作为 basedir 并插入 sys.path，使 `from tools.xxx import`、
`from core.xxx import` 等与生产代码一致的导入路径在测试中直接可用。

The __init__.py is intentional: pytest walks up from test modules to the
first directory without __init__.py (the repo root) and inserts it into
sys.path, so test imports mirror production imports exactly.
"""
