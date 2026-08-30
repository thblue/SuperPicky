# SuperPicky CLI 命令行工具参考 | CLI Reference

[中文](#中文) | [English](#english)

---

## 中文

### 简介

SuperPicky V4.0.0 提供完整的命令行工具，支持批量处理和脚本自动化。包含两个主要 CLI：
- `superpicky_cli.py` - 鸟类照片选片与评分
- `birdid_cli.py` - 独立的鸟类识别工具

### superpicky_cli.py 命令

#### process - 处理照片目录

```bash
# 基本用法
python superpicky_cli.py process ~/Photos/Birds

# 自定义阈值
python superpicky_cli.py process ~/Photos/Birds -s 600 -n 5.2

# 启用自动鸟种识别
python superpicky_cli.py process ~/Photos/Birds --auto-identify

# 指定国家进行 eBird 过滤
python superpicky_cli.py process ~/Photos/Birds --auto-identify --birdid-country AU
```

**参数说明**（默认值优先级：CLI 显式参数 > `advanced_config.json`（与 GUI 同源）> 内置 skill 预设；速查见 [PROCESS_QUICKSTART.md](PROCESS_QUICKSTART.md)）：
| 参数 | 默认值 | 说明 |
|------|-------|------|
| `-s, --sharpness` | 跟随 skill 预设/配置¹ | 锐度阈值 (200-600) |
| `-n, --nima-threshold` | 跟随 skill 预设/配置¹ | 美学阈值 (4.0-7.0) |
| `-c, --confidence` | 配置 `min_confidence`×100 | AI置信度阈值 |
| `--flight / --no-flight` | 跟随配置 `flight_check` | 飞鸟检测 |
| `--burst / --no-burst` | 跟随配置 `burst_check` | 连拍检测 |
| `-i, --auto-identify` | 关闭 | 自动识别鸟种 |
| `--birdid-country` | - | eBird 国家代码 |
| `--birdid-region` | - | eBird 区域代码 |
| `--birdid-threshold` | 配置 `birdid_confidence` | 识别置信度阈值 |

> ¹ skill 预设：beginner 300/4.5 · intermediate 380/4.8 · master 520/5.5 · custom 读配置文件 `custom_sharpness`/`custom_aesthetics`。

#### reset - 重置目录

```bash
# 交互式重置
python superpicky_cli.py reset ~/Photos/Birds

# 跳过确认
python superpicky_cli.py reset ~/Photos/Birds -y
```

#### restar - 重新评星

```bash
# 使用新阈值重新评星
python superpicky_cli.py restar ~/Photos/Birds -s 700 -n 5.5
```

#### identify - 识别单张照片

```bash
# 基本识别
python superpicky_cli.py identify ~/Photos/bird.jpg

# 返回更多候选
python superpicky_cli.py identify ~/Photos/bird.NEF --top 10

# 识别并写入 EXIF
python superpicky_cli.py identify bird.jpg --write-exif
```

#### info - 查看目录信息

```bash
python superpicky_cli.py info ~/Photos/Birds
```

#### burst - 连拍检测

```bash
# 预览连拍组
python superpicky_cli.py burst ~/Photos/Birds

# 执行连拍分组
python superpicky_cli.py burst ~/Photos/Birds --execute
```

---

### birdid_cli.py 命令

独立的鸟类识别 CLI，支持批量识别和按鸟种分类。

#### identify - 批量识别

```bash
# 单张识别
python birdid_cli.py identify ~/Photos/bird.jpg

# 批量识别（支持通配符）
python birdid_cli.py identify ~/Photos/*.jpg --batch

# 指定国家过滤
python birdid_cli.py identify ~/Photos/*.jpg --country AU --region AU-SA

# 写入 EXIF
python birdid_cli.py identify ~/Photos/*.jpg --write-exif
```

**参数说明：**
| 参数 | 说明 |
|------|------|
| `--no-yolo` | 禁用 YOLO 裁剪 |
| `--no-gps` | 禁用 GPS 自动检测 |
| `--no-ebird` | 禁用 eBird 过滤 |
| `-c, --country` | 国家代码 (如 AU, CN, US) |
| `-r, --region` | 区域代码 (如 AU-SA, CN-GD) |
| `--top` | 返回前 N 个结果 |
| `-w, --write-exif` | 写入 EXIF |
| `--threshold` | 写入阈值 (默认 70%) |
| `-b, --batch` | 批量模式 |

#### organize - 按鸟种分类整理

```bash
# 识别并按鸟种分目录
python birdid_cli.py organize ~/Photos/Birds

# 跳过确认
python birdid_cli.py organize ~/Photos/Birds -y
```

目录结构示例：
```
Birds/
├── 彩虹蜂虎_Rainbow Bee-eater/
│   ├── DSC_0001.NEF
│   └── DSC_0002.NEF
├── 笑翠鸟_Laughing Kookaburra/
│   └── DSC_0003.NEF
└── .birdid_manifest.json
```

#### reset - 恢复原始目录

```bash
# 将分类后的照片移回原位
python birdid_cli.py reset ~/Photos/Birds

# 跳过确认
python birdid_cli.py reset ~/Photos/Birds -y
```

#### list-countries - 查看支持的国家代码

```bash
python birdid_cli.py list-countries
```

---

## English

### Introduction

SuperPicky V4.0.0 provides full command-line tools for batch processing and automation. Two main CLIs:
- `superpicky_cli.py` - Bird photo selection and rating
- `birdid_cli.py` - Standalone bird identification tool

### superpicky_cli.py Commands

#### process - Process Photo Directory

```bash
# Basic usage
python superpicky_cli.py process ~/Photos/Birds

# Custom thresholds
python superpicky_cli.py process ~/Photos/Birds -s 600 -n 5.2

# Enable auto bird identification
python superpicky_cli.py process ~/Photos/Birds --auto-identify

# Specify country for eBird filtering
python superpicky_cli.py process ~/Photos/Birds --auto-identify --birdid-country AU
```

**Parameters** (default precedence: explicit CLI flag > `advanced_config.json` (same source as GUI) > built-in skill preset; see [PROCESS_QUICKSTART.md](PROCESS_QUICKSTART.md)):
| Parameter | Default | Description |
|-----------|---------|-------------|
| `-s, --sharpness` | skill preset/config¹ | Sharpness threshold (200-600) |
| `-n, --nima-threshold` | skill preset/config¹ | Aesthetics threshold (4.0-7.0) |
| `-c, --confidence` | config `min_confidence`×100 | AI confidence threshold |
| `--flight / --no-flight` | config `flight_check` | Flying bird detection |
| `--burst / --no-burst` | config `burst_check` | Burst detection |
| `-i, --auto-identify` | Off | Auto bird species ID |
| `--birdid-country` | - | eBird country code |
| `--birdid-region` | - | eBird region code |
| `--birdid-threshold` | config `birdid_confidence` | ID confidence threshold |

> ¹ Skill presets: beginner 300/4.5 · intermediate 380/4.8 · master 520/5.5 · custom reads `custom_sharpness`/`custom_aesthetics` from the config file.

#### reset - Reset Directory

```bash
# Interactive reset
python superpicky_cli.py reset ~/Photos/Birds

# Skip confirmation
python superpicky_cli.py reset ~/Photos/Birds -y
```

#### restar - Re-rate Photos

```bash
# Re-rate with new thresholds
python superpicky_cli.py restar ~/Photos/Birds -s 700 -n 5.5
```

#### identify - Identify Single Photo

```bash
# Basic identification
python superpicky_cli.py identify ~/Photos/bird.jpg

# Return more candidates
python superpicky_cli.py identify ~/Photos/bird.NEF --top 10

# Identify and write to EXIF
python superpicky_cli.py identify bird.jpg --write-exif
```

#### info - View Directory Info

```bash
python superpicky_cli.py info ~/Photos/Birds
```

#### burst - Burst Detection

```bash
# Preview burst groups
python superpicky_cli.py burst ~/Photos/Birds

# Execute burst grouping
python superpicky_cli.py burst ~/Photos/Birds --execute
```

---

### birdid_cli.py Commands

Standalone bird identification CLI with batch processing and species organization.

#### identify - Batch Identification

```bash
# Single image
python birdid_cli.py identify ~/Photos/bird.jpg

# Batch (supports wildcards)
python birdid_cli.py identify ~/Photos/*.jpg --batch

# With country filter
python birdid_cli.py identify ~/Photos/*.jpg --country AU --region AU-SA

# Write to EXIF
python birdid_cli.py identify ~/Photos/*.jpg --write-exif
```

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| `--no-yolo` | Disable YOLO cropping |
| `--no-gps` | Disable GPS auto-detection |
| `--no-ebird` | Disable eBird filtering |
| `-c, --country` | Country code (e.g., AU, CN, US) |
| `-r, --region` | Region code (e.g., AU-SA, CN-GD) |
| `--top` | Return top N results |
| `-w, --write-exif` | Write to EXIF |
| `--threshold` | Write threshold (default 70%) |
| `-b, --batch` | Batch mode |

#### organize - Organize by Species

```bash
# Identify and organize into species folders
python birdid_cli.py organize ~/Photos/Birds

# Skip confirmation
python birdid_cli.py organize ~/Photos/Birds -y
```

Directory structure example:
```
Birds/
├── 彩虹蜂虎_Rainbow Bee-eater/
│   ├── DSC_0001.NEF
│   └── DSC_0002.NEF
├── 笑翠鸟_Laughing Kookaburra/
│   └── DSC_0003.NEF
└── .birdid_manifest.json
```

#### reset - Restore Original Directory

```bash
# Move photos back to original location
python birdid_cli.py reset ~/Photos/Birds

# Skip confirmation
python birdid_cli.py reset ~/Photos/Birds -y
```

#### list-countries - List Supported Country Codes

```bash
python birdid_cli.py list-countries
```

---

© 2024-2025 James Yu · [SuperPicky](https://superpicky.jamesphotography.com.au)
