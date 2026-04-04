/**
 * =============================================================================
 * 这个文件在干什么（人话）
 * =============================================================================
 * SafeTensors 是一种「把很多张权重表打包成一个二进制文件」的公开格式，网上大模型常用。
 * 这里只实现我们用到的一小部分：每张表都是 32 位浮点（和内存里的 float32 一样）、二维表。
 * 写出来的文件，给 Python 的 safetensors 库读，一般也能打开。
 *
 * 文件大概长什么样：先 8 个字节说「后面说明书有多长」→ 一段 JSON 说明书（每张表叫什么、
 * 多长、在文件里从哪字节到哪字节）→ 说明书按规则用空格凑成 8 的倍数 → 再往后就是一张张表
 * 的二进制数据紧挨着放。
 * =============================================================================
 */

/** 行×列，一共多少个格子（用来核对字节数对不对）。 */
const numel = (shape) => shape.reduce((a, b) => a * b, 1);

/**
 * 把「名字 → 一张表」打成 SafeTensors 二进制，返回 Node 里的 Buffer。
 * 表会按名字**字母顺序**排好再写进文件，和常见实现习惯一致。
 */
export const encodeSafetensors = (named) => {
  const keys = Object.keys(named).sort();
  /** 说明书里：每张表的名字对应 dtype、形状、在数据区里的起止字节 */
  const header = {};
  let offset = 0;
  for (const name of keys) {
    const { data, shape } = named[name];
    const byteLen = data.byteLength;
    if (numel(shape) * 4 !== byteLen) {
      throw new Error(`safetensors: ${name} 格子数和 Float32 字节数对不上`);
    }
    const start = offset;
    offset += byteLen;
    header[name] = { dtype: 'F32', shape, data_offsets: [start, offset] };
  }
  // 给别的工具看的备注，不影响我们读权重
  header.__metadata__ = { format: 'pt', framework: 'mini-llm-from-scratch' };

  let headerJson = JSON.stringify(header);
  let headerBuf = Buffer.from(headerJson, 'utf8');
  // 规定：说明书占的字节数必须是 8 的倍数，不够就末尾加空格凑
  while (headerBuf.length % 8 !== 0) {
    headerJson += ' ';
    headerBuf = Buffer.from(headerJson, 'utf8');
  }

  const lenBuf = Buffer.allocUnsafe(8);
  lenBuf.writeBigUInt64LE(BigInt(headerBuf.length), 0);
  const parts = [lenBuf, headerBuf];
  for (const name of keys) {
    const { data } = named[name];
    parts.push(Buffer.from(data.buffer, data.byteOffset, data.byteLength));
  }
  return Buffer.concat(parts);
};

/**
 * 把一个 SafeTensors 文件从 Buffer 读回来，得到「名字 → { 类型、形状、一维 float 数组 }」。
 * __metadata__ 那种说明段会跳过，只还原真正的权重表。
 */
export const decodeSafetensors = (buf) => {
  const headerLen = Number(buf.readBigUInt64LE(0));
  const headerJson = buf.subarray(8, 8 + headerLen).toString('utf8');
  const header = JSON.parse(headerJson.trimEnd());
  const dataBase = 8 + headerLen;
  /** @type {Record<string, { dtype: string, shape: number[], data: Float32Array }>} */
  const out = {};
  for (const [name, info] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    const [s, e] = info.data_offsets;
    const slice = buf.subarray(dataBase + s, dataBase + e);
    const n = slice.length / 4;
    const copy = new Float32Array(n);
    copy.set(new Float32Array(slice.buffer, slice.byteOffset, n));
    out[name] = { dtype: info.dtype, shape: info.shape, data: copy };
  }
  return out;
};
