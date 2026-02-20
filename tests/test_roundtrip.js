import { unpackBinary, packBinary } from "../src/index.js";
import { readFileSync } from 'fs';

// Load the systems msgpack (fetched from http://localhost:8000/systems)
const binary = readFileSync('./systems.msgpack');

// Decode
const decoded = unpackBinary(binary);
console.log("=== Decoded ===");
console.log(JSON.stringify(decoded, (key, val) => {
  if (val instanceof Uint32Array) return `Uint32Array(${val.length}): [${Array.from(val.slice(0, 6)).join(', ')}...]`;
  if (val?.constructor?.name?.includes('Array') && ArrayBuffer.isView(val)) return `${val.constructor.name}(${val.length})`;
  return val;
}, 2));

// Re-encode
const reencoded = packBinary(decoded);
console.log(`\nOriginal size:  ${binary.length} bytes`);
console.log(`Re-encoded size: ${reencoded.byteLength} bytes`);

// Decode the re-encoded data
const redecoded = unpackBinary(reencoded);

// Compare: walk both decoded trees and check values match
let errors = 0;
function compare(a, b, path) {
  const ta = a?.constructor?.name;
  const tb = b?.constructor?.name;

  if (ta !== tb) {
    console.error(`MISMATCH at ${path}: type ${ta} vs ${tb}`);
    errors++;
    return;
  }

  if (ArrayBuffer.isView(a)) {
    if (a.length !== b.length) {
      console.error(`MISMATCH at ${path}: length ${a.length} vs ${b.length}`);
      errors++;
      return;
    }
    for (let i = 0; i < a.length; i++) {
      if (a[i] !== b[i]) {
        console.error(`MISMATCH at ${path}[${i}]: ${a[i]} vs ${b[i]}`);
        errors++;
        return;
      }
    }
    return;
  }

  // UnicodeArray
  if (a?.codePoints instanceof Uint32Array) {
    compare(a.codePoints, b.codePoints, path + '.codePoints');
    if (a.dtype !== b.dtype) {
      console.error(`MISMATCH at ${path}.dtype: ${a.dtype} vs ${b.dtype}`);
      errors++;
    }
    return;
  }

  if (Array.isArray(a)) {
    if (a.length !== b.length) { console.error(`MISMATCH at ${path}: array length ${a.length} vs ${b.length}`); errors++; return; }
    a.forEach((v, i) => compare(v, b[i], `${path}[${i}]`));
    return;
  }

  if (a !== null && typeof a === 'object') {
    const keysA = Object.keys(a).sort();
    const keysB = Object.keys(b).sort();
    if (JSON.stringify(keysA) !== JSON.stringify(keysB)) {
      console.error(`MISMATCH at ${path}: keys ${keysA} vs ${keysB}`);
      errors++;
      return;
    }
    keysA.forEach(k => compare(a[k], b[k], `${path}.${k}`));
    return;
  }

  if (a !== b) {
    console.error(`MISMATCH at ${path}: ${a} vs ${b}`);
    errors++;
  }
}

compare(decoded, redecoded, 'root');

if (errors === 0) {
  console.log("\nRound-trip OK: decoded === re-decoded");
} else {
  console.error(`\n${errors} mismatch(es) found`);
  process.exit(1);
}
