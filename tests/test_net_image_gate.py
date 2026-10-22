# -*- coding: utf-8 -*-
"""非相机原图门槛（网图 EXIF 三空判定）测试。

Net-image metadata gate tests (all-empty camera EXIF detection).

背景 / Background:
2026-10-21 用户定则的元数据层延伸：文件名前缀黑名单管不住伪装相机命名的
转存下载图（IMG_8475.JPG 煤山雀高清网图、IMG_1689-91 雪鸮网图），这类图
唯一稳定特征是 EXIF 相机型号/拍摄时间/GPS 三项全空。管线据此整体按无鸟
处理（不评分/不识鸟/不入鸟统计）；存量库用
scripts_dev/backfill_net_image_gate.py 清洗。

Metadata counterpart of the filename-prefix blacklist: transferred net
images masquerading as camera names carry NO camera model / capture time
/ GPS at all. The pipeline treats them as birdless end to end; existing
libraries are cleaned by scripts_dev/backfill_net_image_gate.py.
"""
import pytest

from constants import is_non_camera_original_exif


class TestIsNonCameraOriginalExif:
    """is_non_camera_original_exif 判定真值表 / truth table."""

    def test_all_three_empty_is_net_image(self):
        """相机/时间/GPS 三全空 = 非相机原图。All empty → net image."""
        assert is_non_camera_original_exif({
            'camera_model': None,
            'date_time_original': None,
            'gps_latitude': None,
        }) is True

    def test_camera_model_present_passes(self):
        """有相机型号 = 相机原图。Camera model present → camera original."""
        assert is_non_camera_original_exif({
            'camera_model': 'iPhone 11 Pro',
            'date_time_original': None,
            'gps_latitude': None,
        }) is False

    def test_datetime_only_passes(self):
        """仅拍摄时间（如无 GPS 的 90D 室内片）也放行。Capture time alone passes."""
        assert is_non_camera_original_exif({
            'camera_model': None,
            'date_time_original': '2020:10:03 15:27:53',
            'gps_latitude': None,
        }) is False

    def test_gps_only_passes(self):
        """仅 GPS（时间被工具抹掉的图）也放行。GPS alone passes."""
        assert is_non_camera_original_exif({
            'camera_model': None,
            'date_time_original': None,
            'gps_latitude': 51.4965,
        }) is False

    def test_missing_keys_treated_as_none(self):
        """键缺失等同 None（EXIF 读取器对缺失标签不落键）。Missing keys = None."""
        assert is_non_camera_original_exif({
            'camera_model': None,
            'date_time_original': None,
        }) is True

    @pytest.mark.parametrize('bad_input', [None, {}, ''])
    def test_unverifiable_input_passes(self, bad_input):
        """None/空输入（预取禁用/读取失败）保守放过，不误杀。Unverifiable → pass."""
        assert is_non_camera_original_exif(bad_input) is False

    def test_wired_into_photo_processor(self):
        """管线与清洗脚本共用同一判定函数，防两处实现漂移。

        The pipeline and the backfill script must share one predicate so
        their criteria can never drift apart.
        """
        import subprocess
        import sys
        import os

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for module in ('core.photo_processor', 'scripts_dev.backfill_net_image_gate'):
            code = (
                f"import sys; sys.path.insert(0, {project_root!r}); "
                f"from {module} import is_non_camera_original_exif; "
                "print('OK')"
            )
            result = subprocess.run([sys.executable, '-c', code],
                                    capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
            assert 'OK' in result.stdout
