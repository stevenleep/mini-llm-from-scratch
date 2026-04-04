/**
 * 续写一律走 POST /api/chat/stream（SSE），按事件逐字更新界面。
 */

const $ = (id) => document.getElementById(id);

const maxEl = $("maxNewTokens");
const tempEl = $("temperature");
const topKEl = $("topK");
const outMax = $("outMax");
const outTemp = $("outTemp");
const outTopK = $("outTopK");

const sync = () => {
  outMax.textContent = maxEl.value;
  outTemp.textContent = (parseInt(tempEl.value, 10) / 100).toFixed(2);
  outTopK.textContent = topKEl.value;
};
maxEl.addEventListener("input", sync);
tempEl.addEventListener("input", sync);
topKEl.addEventListener("input", sync);
sync();

const btn = $("btn");
const statusEl = $("status");
const outPrompt = $("outPrompt");
const outGen = $("outGen");

/**
 * 解析 fetch 返回的 SSE 字节流，逐个 yield 解析好的 JSON 对象。
 * @param {ReadableStreamDefaultReader<Uint8Array>} reader
 */
async function* sseEvents(reader) {
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let sep;
    while ((sep = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, sep);
      buf = buf.slice(sep + 2);
      for (const rawLine of block.split("\n")) {
        const line = rawLine.replace(/\r$/, "");
        if (line.startsWith(":")) continue;
        if (!line.startsWith("data: ")) continue;
        yield JSON.parse(line.slice(6));
      }
    }
  }
  if (buf.trim()) {
    for (const rawLine of buf.split("\n")) {
      const line = rawLine.replace(/\r$/, "");
      if (line.startsWith(":")) continue;
      if (!line.startsWith("data: ")) continue;
      yield JSON.parse(line.slice(6));
    }
  }
}

async function run() {
  const prompt = $("prompt").value || "在";
  const maxNewTokens = parseInt(maxEl.value, 10);
  const temperature = parseInt(tempEl.value, 10) / 100;
  const topK = parseInt(topKEl.value, 10);
  const topP = parseFloat($("topP").value) || 0.92;
  const repetitionPenalty = parseFloat($("repetitionPenalty").value) || 1.12;
  const greedy = $("greedy").checked;
  const seedRaw = $("seed").value.trim();
  const seed = seedRaw === "" ? undefined : parseInt(seedRaw, 10);

  statusEl.textContent = "生成中（流式）…";
  statusEl.classList.remove("err");
  btn.disabled = true;
  outPrompt.textContent = prompt;
  outGen.textContent = "";

  const body = JSON.stringify({
    prompt,
    maxNewTokens,
    temperature,
    topK,
    topP,
    repetitionPenalty,
    greedy,
    seed,
  });

  try {
    const res = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || res.statusText);
    }
    const reader = res.body?.getReader();
    if (!reader) throw new Error("浏览器不支持 ReadableStream");

    let seedUsed = null;
    for await (const ev of sseEvents(reader)) {
      if (ev.error) throw new Error(ev.error);
      if (ev.token) outGen.textContent += ev.token;
      if (ev.done) seedUsed = ev.seedUsed;
    }
    statusEl.textContent =
      greedy || seedUsed == null ? "完成" : `完成 · seed ${seedUsed}`;
  } catch (e) {
    statusEl.textContent = e.message || String(e);
    statusEl.classList.add("err");
  } finally {
    btn.disabled = false;
  }
}

btn.addEventListener("click", run);
