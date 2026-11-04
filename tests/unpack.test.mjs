import assert from "node:assert/strict";
import test from "node:test";
import { gzipSync, zipSync } from "fflate";
import { unpack } from "../src/unpack.ts";

const sqlite = new Uint8Array(4096);
sqlite.set(new TextEncoder().encode("SQLite format 3\0"));
for (let i = 16; i < sqlite.length; i++) sqlite[i] = i % 251;

function tar(name, body) {
  const header = new Uint8Array(512);
  const text = new TextEncoder();
  header.set(text.encode(name));
  header.set(text.encode(body.length.toString(8).padStart(11, "0")), 124);
  header[156] = 0x30;
  header.set(text.encode("ustar"), 257);
  const padded = Math.ceil(body.length / 512) * 512;
  const out = new Uint8Array(512 + padded + 1024);
  out.set(header);
  out.set(body, 512);
  return out;
}

test("passes plain SQLite through", async () => {
  assert.deepEqual(await unpack(sqlite, "a.sqlite"), sqlite);
});

test("decompresses gzip, including multi-member streams", async () => {
  assert.deepEqual(await unpack(gzipSync(sqlite), "a.sqlite.gz"), sqlite);
  const half = sqlite.length / 2;
  const a = gzipSync(sqlite.subarray(0, half));
  const b = gzipSync(sqlite.subarray(half));
  const joined = new Uint8Array(a.length + b.length);
  joined.set(a);
  joined.set(b, a.length);
  assert.deepEqual(await unpack(joined, "a.sqlite.gz"), sqlite);
});

test("extracts the SQLite entry from zip", async () => {
  const zip = zipSync({ "notes.txt": new Uint8Array(9000), "out/trace.sqlite": sqlite });
  assert.deepEqual(await unpack(zip, "trace.zip"), sqlite);
});

test("extracts SQLite from tar.gz", async () => {
  const archive = gzipSync(tar("trace.sqlite", sqlite));
  assert.deepEqual(await unpack(archive, "trace.tar.gz"), sqlite);
});

test("explains unsupported inputs", async () => {
  await assert.rejects(unpack(new Uint8Array([0xfd, 0x37, 0x7a, 0x58, 0x5a, 0, 1]), "a.xz"), /xz/);
  await assert.rejects(unpack(new Uint8Array(32), "a.nsys-rep"), /nsys export/);
  await assert.rejects(unpack(new Uint8Array(32), "a.bin"), /Not a trace/);
});
