# 官网部署

官网静态文件位于 `site/`，托管在 Vercel。

- 生产地址：https://bandbuddy-xi.vercel.app/
- 项目：`dourgeys-projects/bandbuddy`
- 自定义域名：`bandbuddy.lonelyme.cn`，DNS 由域名管理员配置；使用前需在 Vercel 项目的 Domains 中添加域名并按提示设置 DNS。
- GitHub Pages 已停用，原 `pages.yml` 工作流已移除。

## 更新官网

在仓库根目录执行：

```powershell
npx vercel login
npx vercel link --cwd site --project bandbuddy --scope dourgeys-projects
npx vercel deploy --cwd site --prod --scope dourgeys-projects
```

仅上传 `site/` 静态内容，不运行桌面应用构建。`site/vercel.json` 固定无框架、无安装和构建命令，输出目录为当前目录；本地关联信息 `.vercel/` 不提交。

当前通过 CLI 发布。首次部署的 GitHub 自动关联未成功，因此推送代码本身不会自动触发 Vercel 部署。若后续在 Vercel 接入仓库，请将 Root Directory 设置为 `site`，框架设为 Other，保持安装与构建命令为空、输出目录为 `.`。

部署后检查首页、隐私页、下载入口、版本记录展开操作，以及桌面和手机布局。
