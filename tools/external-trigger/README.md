# 第三层触发：Cloudflare Worker

前两层都靠不住过：GitHub schedule 实测迟到 97 分钟 / 10 小时 25 分 /
整段不触发；本机计划任务在机器睡着或关机时形同虚设。这一层跑在
Cloudflare 边缘网络，和上面两者完全独立，常年在线。

免费额度足够：每天两次定时触发，远低于 Workers 免费版每天 10 万次请求。

## 一次性配置，约 5 分钟

### 1. 建一个权限最小的 GitHub token

GitHub → 右上角头像 → Settings → Developer settings →
Personal access tokens → **Fine-grained tokens** → Generate new token

- Token name：随便，比如 `daily-report-trigger`
- Expiration：建议 1 年（到期要换，日历上记一笔）
- Repository access：**Only select repositories** → 只勾 `daily-report`
- Permissions → Repository permissions → **Actions: Read and write**
- 其余权限**一个都不要给**

生成后把那串 `github_pat_...` 复制下来，页面关掉就看不到了。

这个 token 的能力上限就是「派发这个仓库的 workflow」，拿不到代码以外的
任何东西，也改不了仓库内容。

### 2. 部署 Worker

装了 Node 之后在本目录执行：

```bash
npx wrangler login          # 浏览器里授权 Cloudflare 账号
npx wrangler secret put GH_TOKEN    # 粘贴上一步那串 token，回车
npx wrangler deploy
```

### 3. 验证

部署完会给一个 `https://daily-report-trigger.<你的子域>.workers.dev` 地址，
浏览器打开它的 `/check`：

```
https://daily-report-trigger.<子域>.workers.dev/check
```

返回 `"status": 200` 和六个 workflow 的名字，就说明 token 有效、
链路通了。**这个自检不会真的派发**，可以随时点。

想立刻实测一次真的派发：Cloudflare 控制台 → Workers → 这个 Worker →
Settings → Trigger Events → Cron Triggers 那里可以手动触发一次。

### 4. 出问题去哪看

Cloudflare 控制台 → Workers → daily-report-trigger → Logs（实时日志）。
每次定时触发都会打一行 `cron=... -> xxx.yml : {...}`，
`"ok":true` 就是成功。

## 不想用 Cloudflare 的替代

[cron-job.org](https://cron-job.org)，免费，纯网页配置，不用写代码：

- URL：`https://api.github.com/repos/SophiaSha2026/daily-report/actions/workflows/auction.yml/dispatches`
- Method：POST
- Headers：
  - `Authorization: Bearer <你的token>`
  - `Accept: application/vnd.github+json`
  - `X-GitHub-Api-Version: 2022-11-28`
- Body：`{"ref":"main"}`
- 时间：UTC 23:30，周日到周四

形态那条同理，把 `auction.yml` 换成 `pullback.yml`，时间改成 UTC 09:05
周一到周五。

## 时间对照

| 北京 | UTC cron | 触发 |
|---|---|---|
| 周一至周五 07:30 | `30 23 * * 0-4` | 竞价（job 内自旋等到 09:19:40 采 T1） |
| 周一至周五 17:05 | `5 9 * * 1-5` | 形态（收盘后，数据已定型） |

竞价那条跨了 UTC 午夜，所以 UTC 的星期是 0-4（周日到周四），
对应北京的周一到周五。**这个地方最容易写错，写成 1-5 会漏掉北京周一。**
