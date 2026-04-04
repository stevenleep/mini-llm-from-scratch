/**
 * 训练语料里身份问答多为「问句带全角问号，再接 我是… / 我叫…」。
 * 用户常打成「你是谁」不带问号，下一步预测会与语料分布错位；补上后与训练一致。
 *
 * @param {string} prompt
 * @returns {string}
 */
export function alignIdentityPrompt(prompt) {
  const t = prompt.trim();
  if (t === '你是谁') return '你是谁？';
  if (t === '你叫什么') return '你叫什么？';
  if (t === '你叫什么名字') return '你叫什么名字？';
  return prompt;
}

/**
 * 训练语料里这些问句后面紧跟「我…」（我是 / 我叫）。字级小模型里句号 logits 常整体偏高，
 * 第一步可对「我」加微小偏置（见 generate.js），默认可用环境变量 IDENTITY_LOGIT_BIAS=0 关闭。
 */
export function shouldBiasIdentityAnswer(prompt) {
  const t = prompt.trim();
  return (
    t.endsWith('你是谁？') || t.endsWith('你叫什么？') || t.endsWith('你叫什么名字？')
  );
}

/**
 * 与语料一致的标准续写（用于引导解码 / 早停）。非身份问句返回 null。
 */
export function getIdentityCanonicalContinuation(prompt) {
  const t = prompt.trim();
  if (t.endsWith('你是谁？')) return '我是李小烨。';
  if (t.endsWith('你叫什么？')) return '我叫李小烨。';
  if (t.endsWith('你叫什么名字？')) return '我叫李小烨。';
  return null;
}
