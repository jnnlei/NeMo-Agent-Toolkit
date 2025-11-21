# Milvus Collection 清理指南

## 快速开始

### 1. 查看所有 Collections

```bash
cd /localhome/local-jilei/NeMo-Agent-Toolkit
source .venv/bin/activate
cd scripts

python clean_milvus_collection.py --list
```

**输出示例**：
```
======================================================================
Available Collections in Milvus
======================================================================

  📦 nim_docs
     Documents: 850

  📦 cuda_docs
     Documents: 320

======================================================================
```

### 2. 删除整个 Collection（推荐）

**完全删除 collection，包括结构和所有数据**：

```bash
# 删除 nim_docs collection
python clean_milvus_collection.py --drop nim_docs

# 跳过确认提示（自动确认）
python clean_milvus_collection.py --drop nim_docs --yes
```

**适用场景**：
- ✅ 有大量重复数据需要清理
- ✅ 想要重新开始，重新爬取数据
- ✅ 最彻底的清理方式

### 3. 清空 Collection 数据（保留结构）

**删除所有数据，但保留 collection 结构**：

```bash
# 清空 nim_docs collection 的所有数据
python clean_milvus_collection.py --clear nim_docs

# 跳过确认提示
python clean_milvus_collection.py --clear nim_docs --yes
```

**适用场景**：
- 需要保持 collection 配置
- 只想删除数据不想重新创建结构

⚠️ **注意**：某些情况下 `--clear` 可能失败，建议使用 `--drop`

## 完整清理和重新注入流程

### 步骤 1: 查看现有 Collections

```bash
python clean_milvus_collection.py --list
```

### 步骤 2: 删除有重复数据的 Collection

```bash
# 假设要清理 nim_docs
python clean_milvus_collection.py --drop nim_docs --yes
```

### 步骤 3: 重新递归爬取并注入（使用确定性 ID）

```bash
# 使用 --force 确保递归爬取
python langchain_web_ingest.py \
  --urls https://docs.nvidia.com/nim/large-language-models/latest/introduction.html \
  --recursive \
  --force \
  --max_depth 3 \
  --max_pages 100 \
  --collection_name nim_docs
```

### 步骤 4: 验证结果

```bash
# 再次查看 collection
python clean_milvus_collection.py --list
```

## 命令参数说明

| 参数 | 短参数 | 说明 |
|------|--------|------|
| `--list` | `-l` | 列出所有 collections |
| `--drop COLLECTION` | - | 删除指定 collection |
| `--clear COLLECTION` | - | 清空指定 collection 的数据 |
| `--uri URI` | `-u` | Milvus 连接地址（默认: http://localhost:19530）|
| `--yes` | `-y` | 跳过确认提示 |

## 常见使用场景

### 场景 1: 清理所有 Collections 并重新开始

```bash
# 列出所有 collections
python clean_milvus_collection.py --list

# 删除所有 collections（逐个删除）
python clean_milvus_collection.py --drop nim_docs --yes
python clean_milvus_collection.py --drop cuda_docs --yes

# 重新爬取
python langchain_web_ingest.py --urls ... --recursive --force
```

### 场景 2: 只清理特定 Collection

```bash
# 删除 nim_docs，保留其他 collections
python clean_milvus_collection.py --drop nim_docs --yes

# 重新注入
python langchain_web_ingest.py \
  --urls https://docs.nvidia.com/nim/large-language-models/latest/introduction.html \
  --recursive \
  --force \
  --collection_name nim_docs
```

### 场景 3: 使用远程 Milvus 服务器

```bash
# 连接到远程 Milvus
python clean_milvus_collection.py \
  --list \
  --uri http://192.168.1.100:19530

# 删除远程 collection
python clean_milvus_collection.py \
  --drop nim_docs \
  --uri http://192.168.1.100:19530 \
  --yes
```

## 安全提示

⚠️ **重要警告**：

1. **不可逆操作**：删除操作无法撤销，请谨慎使用
2. **确认提示**：默认会要求输入 'yes' 确认，使用 `--yes` 参数会跳过确认
3. **备份建议**：如果数据重要，建议先备份再删除
4. **生产环境**：在生产环境中使用时要格外小心

## 故障排查

### 问题：连接失败

**错误**：`Error: Cannot connect to Milvus`

**解决方案**：
```bash
# 1. 检查 Milvus 是否运行
docker ps | grep milvus

# 2. 检查端口
netstat -tuln | grep 19530

# 3. 使用正确的 URI
python clean_milvus_collection.py --list --uri http://localhost:19530
```

### 问题：Collection 不存在

**错误**：`Collection 'xxx' does not exist`

**解决方案**：
```bash
# 先列出所有 collections，确认名称
python clean_milvus_collection.py --list
```

### 问题：权限不足

**错误**：`Permission denied`

**解决方案**：
- 检查 Milvus 连接权限
- 确保使用的用户有删除权限

## 最佳实践

1. **定期清理**：定期检查并清理不再使用的 collections
2. **命名规范**：使用有意义的 collection 名称，便于管理
3. **测试先行**：在测试环境验证后再在生产环境执行
4. **日志记录**：记录每次清理操作，便于追溯
5. **验证确认**：删除后使用 `--list` 确认操作成功

## 完整工作流示例

```bash
#!/bin/bash
# 清理并重新注入 NIM 文档的完整脚本

cd /localhome/local-jilei/NeMo-Agent-Toolkit
source .venv/bin/activate
cd scripts

echo "=========================================="
echo "Step 1: 查看现有 collections"
echo "=========================================="
python clean_milvus_collection.py --list

echo ""
echo "=========================================="
echo "Step 2: 删除旧的 nim_docs collection"
echo "=========================================="
python clean_milvus_collection.py --drop nim_docs --yes

echo ""
echo "=========================================="
echo "Step 3: 递归爬取并重新注入"
echo "=========================================="
python langchain_web_ingest.py \
  --urls https://docs.nvidia.com/nim/large-language-models/latest/introduction.html \
  --recursive \
  --force \
  --max_depth 3 \
  --max_pages 100 \
  --collection_name nim_docs

echo ""
echo "=========================================="
echo "Step 4: 验证结果"
echo "=========================================="
python clean_milvus_collection.py --list

echo ""
echo "✅ 完成！"
```

保存为 `reset_nim_docs.sh` 并运行：
```bash
chmod +x reset_nim_docs.sh
./reset_nim_docs.sh
```

