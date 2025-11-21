# 递归网页爬取功能使用指南

## 功能概述

`langchain_web_ingest.py` 脚本现已支持递归爬取功能，可以自动跟踪网页中的链接并递归抓取相关页面，大幅提升 RAG 知识库的覆盖范围。

## 核心特性

- ✅ **广度优先搜索 (BFS)**: 按深度逐层爬取，同一深度的页面并行处理
- ✅ **智能域名过滤**: 默认只爬取起始 URL 的域名，可配置多域名白名单
- ✅ **深度控制**: 通过 `max_depth` 参数控制爬取层数
- ✅ **页面数限制**: 通过 `max_pages` 参数防止过度爬取
- ✅ **智能去重**: 基于 URL 自动去重，避免重复访问
- ✅ **防止重复注入**: 使用确定性 ID，相同文档不会重复添加到 Milvus
- ✅ **链接规范化**: 自动处理相对路径、移除锚点、过滤无效链接
- ✅ **错误容错**: 单个页面失败不影响整体流程
- ✅ **幂等性**: 多次运行相同命令不会产生重复数据

## 新增参数说明

| 参数 | 短参数 | 默认值 | 说明 |
|------|--------|--------|------|
| `--recursive` | `-r` | False | 启用递归爬取模式 |
| `--max_depth` | `-d` | 2 | 最大爬取深度（0=仅起始URL） |
| `--allowed_domains` | - | 自动提取 | 允许爬取的域名列表 |
| `--max_pages` | `-m` | 100 | 最大爬取页面数 |
| `--force` | `-f` | False | 强制重新爬取，忽略缓存 |

## 使用示例

### 1. 基础递归爬取（深度 2，最多 100 页）

```bash
cd /localhome/local-jilei/NeMo-Agent-Toolkit
source .venv/bin/activate
cd scripts

python langchain_web_ingest.py \
  --urls https://docs.nvidia.com/cuda/ \
  --recursive \
  --collection_name cuda_docs_recursive
```

### 2. 自定义深度和页面限制

```bash
python langchain_web_ingest.py \
  --urls https://docs.nvidia.com/cuda/ \
  --recursive \
  --max_depth 3 \
  --max_pages 50 \
  --collection_name cuda_docs_deep
```

### 3. 多起始 URL 递归爬取

```bash
python langchain_web_ingest.py \
  --urls https://docs.nvidia.com/cuda/cuda-c-programming-guide/ \
  --urls https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/ \
  --recursive \
  --max_depth 2 \
  --collection_name cuda_guides
```

### 4. 跨多个域名爬取

```bash
python langchain_web_ingest.py \
  --urls https://docs.nvidia.com/cuda/ \
  --recursive \
  --allowed_domains docs.nvidia.com \
  --allowed_domains developer.nvidia.com \
  --max_depth 2 \
  --max_pages 200 \
  --collection_name nvidia_docs_full
```

### 5. 强制重新爬取（忽略缓存）

```bash
# 如果 URL 已在缓存中，使用 --force 强制重新爬取
python langchain_web_ingest.py \
  --urls https://docs.nvidia.com/nim/large-language-models/latest/introduction.html \
  --recursive \
  --max_depth 3 \
  --max_pages 100 \
  --force \
  --collection_name nim_docs
```

**重要提示**：在递归模式下，如果起始 URL 已在缓存中，必须使用 `--force` 参数才能触发递归爬取！

### 6. 非递归模式（原有功能保持不变）

```bash
python langchain_web_ingest.py \
  --urls https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html \
  --urls https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html \
  --collection_name cuda_specific_docs
```

## 爬取深度说明

- **depth 0**: 仅爬取起始 URL（等同于非递归模式）
- **depth 1**: 起始 URL + 起始页面中的所有链接
- **depth 2**: depth 1 + 第一层链接页面中的所有链接
- **depth 3+**: 以此类推

**示例**：
```
起始 URL: https://example.com/docs/
  ↓ depth 0
  
页面 A: https://example.com/docs/intro.html
页面 B: https://example.com/docs/guide.html
  ↓ depth 1 (从起始页面提取的链接)
  
页面 A1: https://example.com/docs/intro/chapter1.html
页面 A2: https://example.com/docs/intro/chapter2.html
页面 B1: https://example.com/docs/guide/setup.html
  ↓ depth 2 (从 depth 1 页面提取的链接)
```

## 防止重复注入机制

### 问题：为什么需要防重复？

在递归爬取模式下，可能会多次运行脚本来更新知识库。如果不处理重复，会导致：
- Milvus 中存在大量重复文档
- 浪费存储空间
- 检索时返回重复结果
- 影响 RAG 系统质量

### 解决方案：确定性文档 ID

脚本使用**确定性 ID 生成**机制，基于以下因素生成唯一 ID：
1. **源 URL**: 文档来源
2. **Chunk 位置**: 文档块在原文档中的位置
3. **内容哈希**: 文档块内容的 MD5 哈希（前8位）

**ID 生成公式**：
```
ID = SHA256(源URL + ":chunk_" + chunk位置 + ":" + 内容哈希)
```

### 工作原理

**第一次运行**：
```bash
python langchain_web_ingest.py --urls https://example.com --recursive
# 结果：添加 100 个新文档到 Milvus
```

**第二次运行相同命令**：
```bash
python langchain_web_ingest.py --urls https://example.com --recursive --force
# 结果：尝试添加 100 个文档，但因为 ID 相同，Milvus 会更新而不是重复添加
```

### 测试验证

运行测试脚本验证：
```bash
cd scripts
python test_deterministic_ids.py
```

**测试结果**：
- ✓ 相同文档 → 相同 ID（防止重复）
- ✓ 不同文档 → 不同 ID（保持唯一性）
- ✓ 多次运行 → 相同 ID（幂等性）

### 实际效果

| 场景 | 行为 | 结果 |
|------|------|------|
| 首次爬取 100 页 | 添加 100 个文档 | ✓ 集合包含 100 个文档 |
| 再次爬取相同 100 页 | 尝试添加 100 个文档 | ✓ 集合仍然只有 100 个文档（被更新） |
| 爬取新增 20 页 | 添加 20 个新文档 | ✓ 集合包含 120 个文档 |

**注意**：Milvus 的 upsert 行为取决于配置，可能是覆盖或保留旧数据。

## 域名过滤策略

### 默认行为（推荐）
不指定 `--allowed_domains` 时，只爬取起始 URL 的域名：

```bash
# 只爬取 docs.nvidia.com 域名下的页面
python langchain_web_ingest.py \
  --urls https://docs.nvidia.com/cuda/ \
  --recursive
```

### 多域名白名单
允许爬取多个相关域名：

```bash
python langchain_web_ingest.py \
  --urls https://docs.nvidia.com/cuda/ \
  --recursive \
  --allowed_domains docs.nvidia.com \
  --allowed_domains developer.nvidia.com \
  --allowed_domains nvidia.github.io
```

## 实际效果对比

### 非递归模式
```bash
# 只爬取 1 个指定页面
python langchain_web_ingest.py \
  --urls https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
```
**结果**: ~1 个文档，~10-50 个文档块

### 递归模式（depth 2）
```bash
# 爬取起始页面 + 所有链接页面 + 二级链接页面
python langchain_web_ingest.py \
  --urls https://docs.nvidia.com/cuda/cuda-c-programming-guide/ \
  --recursive \
  --max_depth 2
```
**结果**: ~50-100 个文档，~500-2000 个文档块

## 性能建议

1. **初次测试**: 使用小的 `max_pages` 值（如 10-20）进行测试
2. **生产环境**: 根据实际需求调整 `max_depth` 和 `max_pages`
3. **大规模爬取**: 建议分批次进行，每批次限制页面数
4. **网络友好**: 脚本已实现异步并发，但建议不要设置过大的深度

## 日志输出示例

```
INFO - Starting recursive crawl, max_depth: 2, allowed_domains: {'docs.nvidia.com'}
INFO - Depth 0: crawling 1 pages
INFO - Extracted 25 new links from https://docs.nvidia.com/cuda/
INFO - Depth 1: crawling 25 pages
INFO - Extracted 15 new links from https://docs.nvidia.com/cuda/chapter1.html
INFO - Extracted 18 new links from https://docs.nvidia.com/cuda/chapter2.html
...
INFO - Depth 2: crawling 45 pages
INFO - Recursive crawl completed, visited 71 pages
INFO - Recursive crawl completed: fetched 68 pages, visited 71 URLs
INFO - Adding 152 document chunks to Milvus collection cuda_docs_recursive
INFO - Ingested 152 document chunks
```

## 技术实现细节

### 新增模块 (`web_utils.py`)

1. **`extract_links_from_html()`**
   - 从 HTML 中提取所有有效链接
   - 自动处理相对路径转绝对路径
   - 过滤无效链接（锚点、JavaScript、mailto 等）
   - 支持域名白名单过滤

2. **`scrape_recursive()`**
   - 实现广度优先搜索算法
   - 按深度批量并发爬取
   - 智能去重和状态追踪
   - 返回所有成功响应和访问记录

### 修改说明 (`langchain_web_ingest.py`)

- 添加 4 个新的命令行参数
- 在主函数中添加递归/非递归模式分支
- 保持向后兼容，不影响现有使用方式

## 故障排查

### 问题：递归模式只爬取了起始 URL（页面数很少）

**症状**：日志显示 "Loading X files from cache" 和很少的文档块

**原因**：起始 URL 已在缓存中，脚本跳过了递归爬取

**解决方案**：
```bash
# 方案 1: 使用 --force 参数强制重新爬取
python langchain_web_ingest.py \
  --urls https://example.com \
  --recursive \
  --force

# 方案 2: 清理缓存目录
rm -rf ./.tmp/data/*
python langchain_web_ingest.py \
  --urls https://example.com \
  --recursive
```

### 问题：爬取的页面数少于预期

**可能原因**：
- 域名过滤限制
- `max_pages` 设置过小
- 网页中的链接较少

**解决方案**：
- 检查 `allowed_domains` 配置
- 增加 `max_pages` 值
- 增加 `max_depth` 值

### 问题：爬取时间过长

**可能原因**：
- `max_depth` 或 `max_pages` 设置过大
- 网络速度慢

**解决方案**：
- 减小 `max_depth` 和 `max_pages` 值
- 使用更快的网络环境

### 问题：某些页面无法访问

**正常行为**: 脚本会跳过失败的页面并继续处理其他页面

**查看日志**: 检查 WARNING 级别的日志了解失败详情

## 最佳实践

1. **渐进式爬取**: 先用 depth 1 测试，再逐步增加深度
2. **监控日志**: 观察实际爬取的页面数和提取的链接数
3. **缓存利用**: 脚本会自动缓存已下载的页面，避免重复下载
4. **定期更新**: 定期重新爬取以更新知识库内容
5. **域名限制**: 严格控制 `allowed_domains`，避免爬取无关域名

## 更新日期

2025-11-19

