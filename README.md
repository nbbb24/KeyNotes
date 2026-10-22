<h1 align="center">KeyNotes</h1>

<p align="center">Infra 与 ML 知识笔记</p>

<p align="center">
| <a href="docs/zh/index.md">中文文档</a> |
</p>

文档顶层分为 `Infra` 和 `ML`。已有 NPU 知识点归入 `Infra`；`ML` 按 Notion 原始层级导入。

## 本地预览

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/mkdocs.txt
mkdocs serve
```

Windows PowerShell 使用 `.venv\Scripts\Activate.ps1` 激活虚拟环境。服务启动后打开
`http://127.0.0.1:8000`。

## 构建

```bash
mkdocs build --strict
python scripts/verify_migration.py
```

生成的网站位于 `site/`，该目录已加入 Git 忽略列表。

## 部署到 Read the Docs

1. 使用能够访问本仓库的 GitHub 账号登录 Read the Docs。
2. 选择 **Add project**，导入 `nbbb24/KeyNotes`。
3. 保持默认分支为 `main`，Read the Docs 会自动读取 `.readthedocs.yaml`。
4. 触发第一次构建并打开生成的网站地址。
5. 首次构建成功后，可以在 **Admin → Domains** 中绑定自定义域名。

不需要提交生成的 HTML。后续推送到 `main` 分支即可触发文档构建。

## 编写约定

- 中文页面统一放在 `docs/zh/` 下。
- 新增页面后，需要同步更新 `mkdocs.yml` 中的 `nav`。
- 缩写首次出现时给出全称，例如 `H2D (Host to Device)`。
- 非通用结论需要注明硬件型号或架构代际。
- 优先用拓扑、数据路径和可观测现象解释概念。

## Notion ML 迁移

ML 内容来自 Notion 页面 `2689c9ac32ee8033a03ecae02ca75d65` 的官方 **Markdown & CSV** 导出，
包含子页面和所有内容，并保留子页面目录。导出包含 104 个 Markdown 页面、215 张 PNG 图片；按用户要求排除 ECGBYTE 后，站点保留 103 个页面和 215 张图片。

- 除按用户要求删除 ECGBYTE 页面及 ML 根页面中的对应链接外，导出的 Markdown 和图片以原始字节保存于 `docs/zh/ml/`，未润色或改写正文，也未压缩或重绘图片。
- 根页面以 `ml/index.md` 保存；其余导出文件名和相对目录保持原样。
- 侧边栏按原始页面顺序、父子关系和原有章节标题组织；`Infra` 和 `ML` 为同级入口。
- `migration/notion-ml-manifest.json` 记录导出来源、文件映射、大小和 SHA-256；
  `migration/infra-manifest.json` 记录原仓库内容的移动映射和 SHA-256。
- `scripts/notion_links.py` 仅在构建时编码链接中的括号，修复 Notion 截断文件名导致的图片/页面链接解析问题；
  不会修改源文件。
- 公式使用 MathJax 显示；任务列表使用 Markdown 扩展显示。网页字体与排版由 MkDocs 主题提供。

核验命令会检查迁移清单中的文件和图片哈希（ML 根页面单独记录移除 ECGBYTE 前后的哈希）、构建产物中的原图、每张图片的显示引用以及站内链接。
修改迁移内容前，请保留原始清单，避免把内容改动误认为无损迁移。
