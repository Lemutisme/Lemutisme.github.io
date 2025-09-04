# GitHub Pages 个人主页模板

一个干净、响应式、无依赖（纯 HTML/CSS/JS）的个人主页，适合直接发布到 GitHub Pages。内含：
- 导航栏（移动端菜单）与暗色模式切换
- Hero 简介、项目、博客、联系与页脚
- 可选自托管分析（Matomo）代码片段
- 可选隐私说明区块

## 使用步骤（个人主页）
1. 将本模板文件放入名为 `your-username.github.io` 的公开仓库（将 `your-username` 替换为你的 GitHub 用户名）。
2. 仓库 Settings → Pages：Build and deployment 选择 *Deploy from a branch*，Branch 选 `main`，Folder 选 `/`。
3. 打开 `https://your-username.github.io/` 访问。

## 自定义
- 修改 `index.html` 中的文字、链接与社交图标。
- `styles.css` 可调整主题色变量（`--accent` 等）。
- `main.js` 包含简单交互逻辑（菜单、主题、隐私开关）。

## 集成自托管分析（记录 IP 的开源方案）
> **合规提示**：如需记录访客 IP，请在你的隐私政策中告知，并遵守所在地区法律（如 GDPR/CCPA）。推荐开启 IP 匿名化。

### 选项 A：Matomo（开源、功能完备）
1. 自行部署 Matomo（示例 docker-compose 见 `analytics/matomo-docker-compose.yml`）。
2. 在 Matomo 中创建站点，获得 `SITE_ID`。
3. 将 `index.html` 底部的 Matomo 脚本解除注释，并替换：
   - `MATOMO_HOST` 为你的 Matomo 服务域名（如 `analytics.example.com`）。
   - `SITE_ID` 为你的站点 ID。
4. 若仅需匿名统计，请在 Matomo 后台启用 **Anonymize IP** 插件（推荐）。若业务**确需**记录完整 IP，请评估合法性与必要性。

### 选项 B：其他开源方案
- **Umami**（轻量、现代、开源 AGPL，默认不存原始 IP，更注重隐私）
- **Ackee / Shynet**（均可自托管，不同程度的隐私取向）

## 目录结构
```
.
├── index.html
├── styles.css
├── main.js
├── .nojekyll               # 如需禁用 Jekyll 处理，可保留
└── analytics/
    └── matomo-docker-compose.yml
```

## 许可证
MIT（站点模板部分）。Matomo/其他分析工具遵循各自的许可证。
