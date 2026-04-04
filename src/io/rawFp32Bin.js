/**
 * =============================================================================
 * 这个文件在干什么（人话）
 * =============================================================================
 * 有些环境只认「纯二进制 .bin」：里面没有任何 JSON 头，就是一段接一段的 float32（小端），
 * 和内存里的数组顺序一致。光靠一个 .bin 不知道每段多长，所以再配一个小清单
 * model.bin.manifest.json：每张表叫什么名字、几行几列、在文件里从第几个字节到第几个字节。
 * 张量排列顺序和 SafeTensors 那份一样：按名字字母排序，再挨个拼。
 * =============================================================================
 */

import { writeFileSync } from 'node:fs';
import { join } from 'node:path';

const MANIFEST_VERSION = 1;

/**
 * @param {Record<string, import('../tensor/Tensor.js').Tensor>} stateDict
 */
export const buildRawFp32Bin = (stateDict) => {
  const keys = Object.keys(stateDict).sort();
  const parts = [];
  let offset = 0;
  /** @type {{ name: string, shape: [number, number], offset: number, length: number }[]} */
  const tensors = [];
  for (const name of keys) {
    const t = stateDict[name];
    const buf = Buffer.from(t.data.buffer, t.data.byteOffset, t.data.byteLength);
    tensors.push({
      name,
      shape: [t.rows, t.cols],
      offset,
      length: buf.length,
    });
    parts.push(buf);
    offset += buf.length;
  }
  const manifest = {
    format: 'mini-llm-fp32-bin',
    version: MANIFEST_VERSION,
    dtype: 'float32',
    little_endian: true,
    tensors,
    total_bytes: offset,
  };
  return { buffer: Buffer.concat(parts), manifest };
};

/**
 * @param {string} dir
 * @param {Record<string, import('../tensor/Tensor.js').Tensor>} stateDict
 */
export const writeRawFp32BinFiles = (dir, stateDict) => {
  const { buffer, manifest } = buildRawFp32Bin(stateDict);
  writeFileSync(join(dir, 'model.bin'), buffer);
  writeFileSync(join(dir, 'model.bin.manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`, 'utf8');
};

/**
 * 把 .bin + 清单读成和 decodeSafetensors 一样的那种「名字 → F32 数据」，好共用后面的灌权重逻辑。
 * @param {Buffer} fileBuf
 * @param {ReturnType<typeof buildRawFp32Bin>['manifest']} manifest
 */
export const decodeRawFp32BinToLoaded = (fileBuf, manifest) => {
  if (manifest.format !== 'mini-llm-fp32-bin') {
    throw new Error(`不认识的 bin 清单 format: ${manifest.format}`);
  }
  if (manifest.version !== MANIFEST_VERSION) {
    throw new Error(`不支持的 bin 清单版本: ${manifest.version}`);
  }
  /** @type {Record<string, { dtype: string, shape: number[], data: Float32Array }>} */
  const out = {};
  for (const t of manifest.tensors) {
    const slice = fileBuf.subarray(t.offset, t.offset + t.length);
    const n = slice.length / 4;
    if (!Number.isInteger(n) || n * 4 !== slice.length) {
      throw new Error(`张量 ${t.name} 字节数不是 4 的倍数`);
    }
    const data = new Float32Array(n);
    data.set(new Float32Array(slice.buffer, slice.byteOffset, n));
    out[t.name] = { dtype: 'F32', shape: t.shape, data };
  }
  return out;
};
