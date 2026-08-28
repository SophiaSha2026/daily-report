/**
 * 第三层触发：Cloudflare Worker Cron。
 *
 * 前两层都出过问题：
 *   GitHub schedule —— 实测迟到 97 分钟 / 10 小时 25 分 / 整段不触发
 *   本机计划任务   —— 机器睡着或关机时形同虚设
 * 这一层跑在 Cloudflare 的边缘网络上，和上面两者完全独立，常年在线。
 *
 * 它只做一件事：到点调 GitHub 的 workflow_dispatch API。
 * 重复触发无害，两条流水线各自有串行组和幂等检查。
 *
 * 需要一个 Secret：GH_TOKEN
 *   GitHub -> Settings -> Developer settings -> Personal access tokens
 *   -> Fine-grained tokens -> Generate new token
 *   Repository access: 只选 SophiaSha2026/daily-report
 *   Permissions -> Repository permissions -> Actions: Read and write
 *   其余权限一概不给。这个 token 只能派发 workflow，做不了别的。
 */

const REPO = "SophiaSha2026/daily-report";

// UTC cron -> 要派发的 workflow。和 wrangler.toml 里的 crons 一一对应。
//   "30 23 * * *" = 北京次日 07:30
//   "5 9 * * *"   = 北京当日 17:05
//
// 为什么不限定星期：Cloudflare 的 cron 解析器拒绝 "0-4" 这种星期范围
// （invalid cron string, code 10100），而且「UTC 星期几」和「北京星期几」
// 跨午夜时还会差一天，最容易写错。干脆每天都触发，让流水线自己判交易日
// —— 两条线都查新浪交易日历，非交易日直接退出，不发邮件、不留数据。
// 代价是周末两次空跑，每次不到一分钟。
const ROUTES = {
  "30 23 * * *": "auction.yml",
  "5 9 * * *": "pullback.yml",
};

async function dispatch(workflow, token) {
  const r = await fetch(
    `https://api.github.com/repos/${REPO}/actions/workflows/${workflow}/dispatches`,
    {
      method: "POST",
      headers: {
        Accept: "application/vnd.github+json",
        Authorization: `Bearer ${token}`,
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "daily-report-external-trigger",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ ref: "main" }),
    }
  );
  // 204 No Content 才是成功
  return { ok: r.status === 204, status: r.status, body: await r.text() };
}

export default {
  async scheduled(event, env, ctx) {
    const wf = ROUTES[event.cron];
    if (!wf) {
      console.log(`未知 cron ${event.cron}，忽略`);
      return;
    }
    const res = await dispatch(wf, env.GH_TOKEN);
    console.log(`cron=${event.cron} -> ${wf} : ${JSON.stringify(res)}`);
    if (!res.ok) throw new Error(`派发 ${wf} 失败: ${res.status} ${res.body}`);
  },

  // 手动自检：浏览器打开 https://<worker>.workers.dev/check
  // 只验证 token 能不能用，不会真的派发。
  async fetch(req, env) {
    const u = new URL(req.url);
    if (u.pathname !== "/check") {
      return new Response("ok. 自检请访问 /check", { status: 200 });
    }
    const r = await fetch(`https://api.github.com/repos/${REPO}/actions/workflows`, {
      headers: {
        Accept: "application/vnd.github+json",
        Authorization: `Bearer ${env.GH_TOKEN}`,
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "daily-report-external-trigger",
      },
    });
    const j = await r.json();
    const names = (j.workflows || []).map((w) => `${w.name} [${w.path}]`);
    return new Response(
      JSON.stringify({ status: r.status, workflows: names }, null, 2),
      { headers: { "content-type": "application/json; charset=utf-8" } }
    );
  },
};
