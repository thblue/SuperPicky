# -*- coding: utf-8 -*-
"""tools — 运行时支撑包 / Runtime support package.

分层规则 / Layering rules（详见 dev-docs/ARCHITECTURE.md）：

- 本包是**运行时模块**（被 core/、ui/、入口脚本 import），不是脚本目录。
  一次性/运维脚本放 `scripts/`（构建发布类）与 `scripts_dev/`（直接读写
  真实数据的运维类），不要加回本包。
  This package holds runtime modules imported by core/, ui/ and the root
  entry points — not scratch scripts. Build helpers go to scripts/, real-data
  ops scripts to scripts_dev/.

- 依赖方向 / Dependency direction:
  入口层与 ui/ → core/ → tools/ → （根级 config/constants/advanced_config）。
  core 反向 import tools 是既定事实（report_db/exiftool_manager/find_bird_util/
  resume_state/i18n）；tools 反向 import core/config 时一律用**函数内延迟
  import** 缓解循环（先例：merged_report_db.py）。新增跨包引用请沿用此惯例。
  core→tools back-references are established; any tools→core/config import
  must stay a function-local delayed import to avoid cycles.

- 数据安全 / Data safety:
  exiftool_manager 是所有 EXIF/XMP 写入的唯一闸门；report_db 是 report.db
  唯一管理者（schema v13 冻结）。二者行为改动前必读 dev-docs/INTERFACE_CONTRACTS.md。
"""
