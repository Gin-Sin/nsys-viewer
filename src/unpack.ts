import { unzipSync } from "fflate";
import { decompress as zstdDecompress } from "fzstd";
import { ungzip } from "pako";

const SQLITE_MAGIC = "SQLite format 3\0";

function startsWith(bytes: Uint8Array, magic: number[], offset = 0): boolean {
  return magic.every((value, index) => bytes[offset + index] === value);
}

export function isSqlite(bytes: Uint8Array): boolean {
  return startsWith(bytes, [...SQLITE_MAGIC].map((char) => char.charCodeAt(0)));
}

function isTar(bytes: Uint8Array): boolean {
  // POSIX and GNU tar both write "ustar" at offset 257 of each header block.
  return bytes.length >= 512 && startsWith(bytes, [0x75, 0x73, 0x74, 0x61, 0x72], 257);
}

function looksLikeSqliteName(name: string): boolean {
  return /\.(sqlite|sqlite3|db)$/i.test(name);
}

async function nativeDecompress(bytes: Uint8Array, format: string): Promise<Uint8Array | null> {
  if (typeof DecompressionStream === "undefined") return null;
  let stream: DecompressionStream;
  try {
    stream = new DecompressionStream(format as CompressionFormat);
  } catch {
    return null; // Format not supported by this browser.
  }
  try {
    const output = new Blob([bytes as BlobPart]).stream().pipeThrough(stream);
    return new Uint8Array(await new Response(output).arrayBuffer());
  } catch {
    return null; // Fall back to the JS decoder.
  }
}

async function gunzip(bytes: Uint8Array): Promise<Uint8Array> {
  const native = await nativeDecompress(bytes, "gzip");
  // Browsers stop after the first gzip member, so compare the output with the
  // trailer's size (mod 2^32) and use pako, which handles concatenated members.
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const trailerSize = view.getUint32(bytes.byteLength - 4, true);
  if (native && native.byteLength % 2 ** 32 === trailerSize) return native;
  return ungzip(bytes);
}

function extractTar(bytes: Uint8Array): Uint8Array {
  let fallback: Uint8Array | null = null;
  for (let offset = 0; offset + 512 <= bytes.length;) {
    const header = bytes.subarray(offset, offset + 512);
    if (header.every((value) => value === 0)) break;
    const sizeText = new TextDecoder().decode(header.subarray(124, 136)).replace(/\0.*$/, "").trim();
    const size = parseInt(sizeText || "0", 8);
    const type = String.fromCharCode(header[156] || 0x30);
    const body = bytes.subarray(offset + 512, offset + 512 + size);
    if (type === "0" || type === "\0") {
      if (isSqlite(body)) return body;
      if (!fallback || body.length > fallback.length) fallback = body;
    }
    offset += 512 + Math.ceil(size / 512) * 512;
  }
  if (fallback) return fallback;
  throw new Error("The tar archive does not contain any files.");
}

function extractZip(bytes: Uint8Array): Uint8Array {
  const entries: { name: string; size: number }[] = [];
  unzipSync(bytes, {
    filter: (file) => {
      if (!file.name.endsWith("/")) entries.push({ name: file.name, size: file.originalSize });
      return false;
    },
  });
  if (!entries.length) throw new Error("The zip archive does not contain any files.");
  const chosen = entries.find((entry) => looksLikeSqliteName(entry.name))
    ?? entries.reduce((a, b) => (b.size > a.size ? b : a));
  const files = unzipSync(bytes, { filter: (file) => file.name === chosen.name });
  return files[chosen.name];
}

/**
 * Returns raw SQLite bytes from a plain, gzip, zstd, zip, or tar(.gz/.zst)
 * file. Formats are detected from magic bytes, not the file extension.
 */
export async function unpack(
  input: Uint8Array,
  name: string,
  progress: (message: string) => void = () => {},
): Promise<Uint8Array> {
  let bytes = input;
  for (let depth = 0; depth < 3; depth++) {
    if (isSqlite(bytes)) return bytes;
    if (startsWith(bytes, [0x1f, 0x8b])) {
      progress("Decompressing gzip…");
      bytes = await gunzip(bytes);
    } else if (startsWith(bytes, [0x28, 0xb5, 0x2f, 0xfd])) {
      progress("Decompressing zstd…");
      bytes = await nativeDecompress(bytes, "zstd") ?? zstdDecompress(bytes);
    } else if (startsWith(bytes, [0x50, 0x4b, 0x03, 0x04])) {
      progress("Extracting zip…");
      bytes = extractZip(bytes);
    } else if (isTar(bytes)) {
      progress("Extracting tar…");
      bytes = extractTar(bytes);
    } else if (startsWith(bytes, [0xfd, 0x37, 0x7a, 0x58, 0x5a, 0x00]) ||
               startsWith(bytes, [0x42, 0x5a, 0x68]) ||
               startsWith(bytes, [0x37, 0x7a, 0xbc, 0xaf, 0x27, 0x1c])) {
      throw new Error("xz, bzip2, and 7z are not supported. Compress with gzip, zstd, or zip instead.");
    } else {
      break;
    }
  }
  if (isSqlite(bytes)) return bytes;
  if (/\.(nsys-rep|qdrep)$/i.test(name)) {
    throw new Error("This is an .nsys-rep report. Convert it first: nsys export --type sqlite <file>.");
  }
  throw new Error("Not an SQLite database. Open an Nsight SQLite export (optionally .gz, .zst, .zip, or .tar.gz).");
}
