# -*- coding: utf-8 -*-
"""
被 BirdIndex 子进程调用的 CLI 参数面契约测试 / CLI Surface Contract Test

守护 dev-docs/INTERFACE_CONTRACTS.md §4/§5：`spb_rename_species.py` 与
`spb_pixel_art.py` 被 BirdIndex 以**固定仓库根路径**子进程调用，并以
`--json` 末行结果、`--apply` 落盘等参数面耦合。本测试对两者的 --help
参数面做快照——选项消失/改名会让 BirdIndex 的 fixer.py / sprite.py
直接瘫痪，因此必须先行双仓同步评审。

Guards §4/§5: both CLIs are invoked by BirdIndex as fixed-path
subprocesses; this test snapshots their --help surface so a removed or
renamed option fails loudly here instead of breaking the website.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

ROOT = Path(__file__).resolve().parents[1]
HELP_TIMEOUT_SECONDS = 120

# §4 冻结：spb_rename_species.py 的子命令与选项 / Frozen surface (§4)
RENAME_SPECIES_REQUIRED = [
    "photo", "species",
    "--apply", "--json",
    "--to-cn", "--to-en", "--to-sci",
    "--old-cn", "--old-en", "--old-sci",
    "--wipe",
]

# §5 冻结：spb_pixel_art.py 的选项（含 BirdIndex sprite.py 实际使用集）
PIXEL_ART_REQUIRED = [
    "--batch", "--image", "--bbox", "--label", "--name",
    "--workflow", "--image-node", "--host", "--out",
    "--padding", "--min-crop-px", "--timeout", "--seed",
    "--overwrite", "--white-bg", "--pixel-grid", "--prompt", "--json",
]


def _help_surface(script_name: str, args: list = None) -> str:
    """
    以子进程运行脚本的 --help，返回帮助文本。

    参数:
    script_name (str): 仓库根下的脚本文件名
    args (list): 追加参数（如子命令名，用于探测子解析器的参数面）

    返回:
    str: stdout + stderr 合并文本（argparse 帮助可能在任一通道输出）

    Runs `<script> --help` in a subprocess and returns merged output.
    Optional `args` probes a subparser's own help (subcommand options are
    invisible to the top-level help).
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.run(
        [sys.executable, str(ROOT / script_name), *(args or []), "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=HELP_TIMEOUT_SECONDS,
        env=env,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0, (
        f"{script_name} {' '.join(args or [])} --help 退出码 {proc.returncode}："
        f"脚本可能已无法启动。\nstderr: {proc.stderr[-500:]}"
    )
    return proc.stdout + proc.stderr


@pytest.mark.parametrize(
    "script_name, required",
    [
        ("spb_rename_species.py", RENAME_SPECIES_REQUIRED),
        ("spb_pixel_art.py", PIXEL_ART_REQUIRED),
    ],
    ids=["rename-species", "pixel-art"],
)
def test_cli_surface_frozen(script_name: str, required: list):
    """
    参数面快照：冻结选项必须全部出现在 --help 中。

    spb_rename_species 的 --to-*/--old-*/--wipe 挂在 photo/species 子解析器
    上，顶层 --help 看不到，因此对顶层与两个子命令分别探测。

    All frozen options must appear in --help. Subcommand-scoped options are
    probed via the subparsers' own help surfaces.
    """
    surfaces = [_help_surface(script_name)]
    if script_name == "spb_rename_species.py":
        surfaces += [
            _help_surface(script_name, ["photo"]),
            _help_surface(script_name, ["species"]),
        ]
    missing = [
        opt for opt in required
        if not any(opt in surface for surface in surfaces)
    ]
    assert not missing, (
        f"{script_name} 缺少冻结的 CLI 选项 {missing}——"
        "这些选项被 BirdIndex（indexer/fixer.py、indexer/sprite.py）子进程调用，"
        "请先评审 dev-docs/INTERFACE_CONTRACTS.md §4/§5 并双仓同步"
    )


def test_rename_species_stdout_json_protocol_doc():
    """
    §4 输出协议冒烟：--json 模式「stdout 末行 JSON」约定在文档串中可发现。

    Smoke for the stdout-last-line-JSON protocol promise (§4): BirdIndex
    parses the last stdout line, so the contract must stay documented and
    honoured by the implementation.
    """
    source = (ROOT / "spb_rename_species.py").read_text(encoding="utf-8")
    assert "--json" in source
    assert " BirdIndex" in source or "BirdIndex 的 /api/fix" in source, (
        "spb_rename_species.py 的 docstring 丢失 BirdIndex 消费方说明——"
        "该脚本是 /api/fix 的写后端，契约注释不可删"
    )
