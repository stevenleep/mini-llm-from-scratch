/**
 * =============================================================================
 * 本文件里的英文词是什么意思（全部写在代码里，方便对照）
 * =============================================================================
 * tokenizer      分词器：把文字切成「单位」并编号的工具；这里是「字」级。
 * buildCharTokenizer  函数名：build=建立，Char=字符，合起来=建立字符级分词器。
 * text           入参：一段字符串原文。
 * chars          语料里出现过的字去重、排序后的列表。
 * stoi           string-to-index：字 → 整数编号（对象：键是字，值是号）。
 * itos           index-to-string：用数组下标当编号，值是对应的字（编号 → 字）。
 * encode         编码：字符串 → 编号数组。
 * decode         解码：编号数组 → 拼回字符串。
 * vocabSize      vocabulary size：词表大小，有几个不同的字。
 * idx            index 的缩写：某个位置上的编号。
 * Uint32Array    JS 内置类型：存「非负整数」的紧凑数组；不必记英文名，知是「编号串」即可。
 * Record         类型注解：表示「字符串键 → 数字」这种映射表。
 * s / c / i      循环里常用：s 常表示 string，c 表示 character，i 表示 index。
 *
 * =============================================================================
 * 字符级分词（tokenizer）
 * =============================================================================
 *
 * 【问题】模型只能算数字；必须把文本变成整数序列。
 *
 * 【为什么用「字」而不是词】
 * 中文词切分要额外规则或统计；字符级词表小、代码短，适合先搞懂整条管线。
 * 真正大模型常用子词（BPE 等），词表更大、更省空间，但「编号→嵌入」本质一样。
 *
 * 【stoi / itos】
 * string-to-index：字符 → 编号；index-to-string：编号 → 字符。decode 用于生成后把编号转回可读文本。
 *
 * 【排序】
 * chars 排序后编号固定，同样语料每次 build 词表一致，便于复现实验。
 *
 * 【未知字符】
 * encode 时若语料里没出现过的字符会报错；真实系统会加「未知字」特殊 token 或字节级编码。
 */
export const buildCharTokenizer = (text) => {
  const chars = [...new Set(text)].sort();
  /** @type {Record<string, number>} */
  const stoi = {};
  chars.forEach((c, i) => {
    stoi[c] = i;
  });
  const itos = chars;

  const encode = (s) => {
    const out = new Uint32Array(s.length);
    for (let i = 0; i < s.length; i++) {
      const idx = stoi[s[i]];
      if (idx === undefined) throw new Error(`未知字符: ${s[i]}`);
      out[i] = idx;
    }
    return out;
  };

  const decode = (ids) => {
    let s = '';
    for (let i = 0; i < ids.length; i++) s += itos[ids[i]];
    return s;
  };

  return {
    vocabSize: chars.length,
    encode,
    decode,
    stoi,
    itos,
  };
};

/**
 * 从「保存下来的字表」恢复分词器（和 buildCharTokenizer 用起来一样）。
 * itos 就是「第 0 号是什么字、第 1 号是什么字…」的列表，必须和训练导出时完全一致，不能少字、不能重复。
 */
export const tokenizerFromItos = (itos) => {
  if (!Array.isArray(itos) || itos.length === 0) {
    throw new Error('字表 itos 必须是非空数组');
  }
  /** @type {Record<string, number>} */
  const stoi = {};
  for (let i = 0; i < itos.length; i++) {
    const c = itos[i];
    if (stoi[c] !== undefined) throw new Error(`字表里同一个字出现了两次: ${c}`);
    stoi[c] = i;
  }
  const encode = (s) => {
    const out = new Uint32Array(s.length);
    for (let i = 0; i < s.length; i++) {
      const idx = stoi[s[i]];
      if (idx === undefined) throw new Error(`未知字符: ${s[i]}`);
      out[i] = idx;
    }
    return out;
  };
  const decode = (ids) => {
    let s = '';
    for (let i = 0; i < ids.length; i++) s += itos[ids[i]];
    return s;
  };
  return {
    vocabSize: itos.length,
    encode,
    decode,
    stoi,
    itos,
  };
};
