import msgpack from "msgpack-lite";
import _ from "underscore";

if (typeof BigInt64Array === "undefined") {
  const BigInt64Array = window.BigInt64Array;
}
if (typeof BigUint64Array === "undefined") {
  const BigUint64Array = window.BigUint64Array;
}

// Browser-compatible Buffer utilities
const bufferFrom = (input) => {
  if (typeof input === 'string') {
    return new TextEncoder().encode(input);
  } else if (Array.isArray(input)) {
    return new Uint8Array(input);
  } else if (input instanceof Uint8Array) {
    return input;
  } else {
    return new Uint8Array(input);
  }
};

const bufferConcat = (arrays) => {
  const totalLength = arrays.reduce((sum, arr) => sum + arr.length, 0);
  const result = new Uint8Array(totalLength);
  let offset = 0;
  for (const arr of arrays) {
    result.set(arr, offset);
    offset += arr.length;
  }
  return result;
};

const isBuffer = (obj) => {
  return obj instanceof Uint8Array || (typeof Buffer !== 'undefined' && obj instanceof Buffer);
};

const isLittleEndian =
  new Uint8Array(new Uint32Array([0x11223344]).buffer)[0] === 0x44;

function byteswap(typecode, data) {
  if (typecode === "|") return data; // Not much we need to do here...
  if (typecode === "=") return data; // Not much we can do here :P
  if (typecode === "<" && isLittleEndian) return data;
  if (typecode === ">" && !isLittleEndian) return data;

  throw new Error("Endianness swapping not implemented");
}

const npsplitfirstdim = (data, dim) => {
  const restdim = data.length / dim;
  const res = [];
  for (var i = 0; i < restdim; i++) {
    (function (currentI) {
      res.push(data.filter((d, di) => (di - currentI) % restdim === 0));
    })(i);
  }

  return res;
};
const npsplitdims = (data, dims) => {
  for (var i = 0; i < dims.length - 1; i++) {
    data = npsplitfirstdim(data, dims[i]);
  }
  return data;
};

const npmergefirstdim = (data) => {
  return _.zip
    .apply(
      [],
      data.map((a) => Array.from(a))
    )
    .flat();
};

const npmergedims = (data) => {
  if (data.buffer !== undefined) {
    return Array.from(data);
  } else {
    return npmergefirstdim(data.map(npmergedims));
  }
};

// Represents a numpy unicode array (e.g. dtype <U8).
// Each character is a 32-bit UCS-4 code point, stored as Uint32Array.
// Shape is derived from dtype (chars per string) and codePoints.length.
class UnicodeArray {
  constructor(dtype, codePoints) {
    this.dtype = dtype;   // e.g. "<U8"
    this.codePoints = codePoints instanceof Uint32Array
      ? codePoints
      : new Uint32Array(codePoints.buffer, codePoints.byteOffset, codePoints.byteLength / 4);
  }

  get shape() {
    const charsPerString = parseInt(this.dtype.match(/\d+/)[0]);
    return [this.codePoints.length / charsPerString];
  }
}

export function unpackNumpy(v) {
  if (v === null) {
    return v;
  } else if (Array.isArray(v)) {
    var res = [];
    for (var i in v) {
      res.push(unpackNumpy(v[i]));
    }
    return res;
  } else if (typeof v === "object") {
    var constr;
    var isnd = v["nd"] || v["110,100"];
    var t = v["type"] || v["116,121,112,101"];
    var d = v["data"] || v["100,97,116,97"];
    var s = v["shape"] || v["115,104,97,112,101"];

    if (isnd !== undefined && typeof t === "string") {
      const originalDtype = t;
      d = byteswap(t[0], d);
      t = t.slice(1);

      // Unicode (U*): each character is a 32-bit UCS-4 code point.
      if (t[0] === "U") {
        const codePoints = new Uint32Array(d.buffer, d.byteOffset, d.byteLength / 4);
        return new UnicodeArray(originalDtype, codePoints);
      }

      // documentation is here https://numpy.org/doc/stable/reference/arrays.dtypes.html
      constr = {
        b: Int8Array,
        B: Int8Array,

        i1: Int8Array,
        i2: Int16Array,
        i4: Int32Array,
        i8: BigInt64Array,

        I1: Uint8Array,
        I2: Uint16Array,
        I4: Uint32Array,
        I8: BigUint64Array,

        u1: Uint8Array,
        u2: Uint16Array,
        u4: Uint32Array,
        u8: BigUint64Array,

        f4: Float32Array,
        f8: Float64Array,
      }[t];

      if (constr !== undefined) {
        if (d.byteOffset !== undefined) {
          // NodeJS
          v = new constr(
            d.buffer,
            d.byteOffset,
            d.length / constr.BYTES_PER_ELEMENT
          );
        } else {
          v = new constr(d.buffer);
        }
        v = npsplitdims(v, s);
        return v;
      } else {
        console.warn("Unknown datatype in msgpack binary", t);
      }
    }

    res = {};
    for (i in v) {
      res[i] = unpackNumpy(v[i]);
    }
    return res;
  } else {
    return v;
  }
}

const multidimTypedArrayType = function (v) {
  while (v && v.length !== undefined && v.buffer === undefined) {
    v = v[0];
  }
  if (v && v.buffer !== undefined && v.length !== undefined)
    return v.constructor;
  return null;
};

const multidimTypedArrayShape = function (v) {
  if (!v) return false;
  if (v.buffer !== undefined && v.length !== undefined) return [v.length];
  if (!Array.isArray(v)) return false;
  if (v.length === 0) return false;
  const dims = v.map(multidimTypedArrayShape).reduce((a, b) => {
    if (a === false || b === false) return false;
    if (JSON.stringify(a) !== JSON.stringify(b)) return false;
    return b;
  });
  if (dims === false) return false;
  return dims.concat([v.length]);
};

const enc = new TextEncoder();

// Marker class to identify numpy metadata that needs special encoding
class NumpyMap extends Map {
  constructor(entries) {
    super(entries);
  }
}

// Reuse buffer keys to ensure Map key identity
const KEY_ND = bufferFrom("nd");
const KEY_TYPE = bufferFrom("type");
const KEY_SHAPE = bufferFrom("shape");
const KEY_DATA = bufferFrom("data");

export function packNumpy(v) {
  if (v === null) {
    return v;
  } else if (v instanceof UnicodeArray) {
    // Re-encode with original dtype and shape for a faithful round-trip
    const res = new NumpyMap();
    res.set(KEY_ND, true);
    res.set(KEY_TYPE, v.dtype);
    res.set(KEY_SHAPE, v.shape);
    res.set(KEY_DATA, v.codePoints.buffer.slice(v.codePoints.byteOffset, v.codePoints.byteOffset + v.codePoints.byteLength));
    return res;
  } else if (Array.isArray(v)) {
    const dims = multidimTypedArrayShape(v);
    if (dims !== false) {
      const data = npmergedims(v, dims);
      const constr = multidimTypedArrayType(v);
      const packed = packNumpy(new constr(data));
      packed.set(KEY_SHAPE, dims);  // dims is already in the right order
      return packed;
    } else {
      var res = [];
      for (var i in v) {
        res.push(packNumpy(v[i]));
      }
      return res;
    }
  } else if (typeof v === "object") {
    if (v.buffer !== undefined) {
      const type = {
        Uint8Array: "u1",  // uint8, 1 byte/element — compatible with shape = Uint8Array.length

        Int8Array: "b",

        Int16Array: "i2",
        Int32Array: "i4",
        BigInt64Array: "i8",

        Uint16Array: "I2",
        Uint32Array: "I4",
        BigUint64Array: "I8",

        Float32Array: "f4",
        Float64Array: "f8",
      }[v.constructor.name];

      // Use NumpyMap marker class with reused Buffer keys
      const res = new NumpyMap();
      res.set(KEY_ND, true);
      res.set(KEY_TYPE, (isLittleEndian ? "<" : ">") + type);
      res.set(KEY_SHAPE, [v.length]);
      // Use slice to get only the relevant portion of the buffer (TypedArray may be a view)
      res.set(KEY_DATA, v.buffer.slice(v.byteOffset, v.byteOffset + v.byteLength));
      return res;
    } else if (v.constructor === ArrayBuffer) {
      return v;
    } else if (v.constructor === Map || v instanceof NumpyMap) {
      return new Map(
        Array.from(v.entries()).map(([vk, vv]) => [
          packNumpy(vk),
          packNumpy(vv),
        ])
      );
    } else {
      res = {};
      for (i in v) {
        res[i] = packNumpy(v[i]);
      }
      return res;
    }
  } else {
    return v;
  }
}

// Manually encode NumpyMap as msgpack map with binary keys
// This avoids the usemap extension which causes issues
function manualEncodeNumpyMap(numpyMap, codec) {
  const entries = Array.from(numpyMap.entries());
  const buffers = [];

  // Write map header
  const len = entries.length;
  if (len <= 15) {
    buffers.push(bufferFrom([0x80 + len])); // fixmap
  } else if (len <= 65535) {
    buffers.push(bufferFrom([0xde, len >> 8, len & 0xff])); // map16
  } else {
    buffers.push(bufferFrom([0xdf, (len >> 24) & 0xff, (len >> 16) & 0xff, (len >> 8) & 0xff, len & 0xff])); // map32
  }

  // Write each key-value pair
  for (const [key, value] of entries) {
    // Write key as bin (binary string)
    const keyBuf = isBuffer(key) ? key : bufferFrom(key);
    if (keyBuf.length <= 255) {
      buffers.push(bufferFrom([0xc4, keyBuf.length])); // bin8
    } else if (keyBuf.length <= 65535) {
      buffers.push(bufferFrom([0xc5, keyBuf.length >> 8, keyBuf.length & 0xff])); // bin16
    } else {
      buffers.push(bufferFrom([0xc6, (keyBuf.length >> 24) & 0xff, (keyBuf.length >> 16) & 0xff, (keyBuf.length >> 8) & 0xff, keyBuf.length & 0xff])); // bin32
    }
    buffers.push(keyBuf);

    // Write value - use encodeWithNumpySupport for consistency
    const valueBuf = encodeWithNumpySupport(value, codec);
    buffers.push(valueBuf);
  }

  return bufferConcat(buffers);
}

// Recursively replace NumpyMaps with manually encoded buffers
// But we can't do this directly - we need to encode the whole structure at once
// So instead, we'll convert NumpyMaps to a special marker that we handle during encoding
function prepareForEncoding(v) {
  if (v instanceof NumpyMap) {
    // Mark this as needing special encoding
    return { __numpymap__: v };
  } else if (Array.isArray(v)) {
    return v.map(prepareForEncoding);
  } else if (v && typeof v === "object" && v.constructor === Object) {
    const result = {};
    for (const [key, value] of Object.entries(v)) {
      result[key] = prepareForEncoding(value);
    }
    return result;
  } else {
    return v;
  }
}

// Recursively encode structures, handling NumpyMaps manually
function encodeWithNumpySupport(v, codec) {
  if (v instanceof NumpyMap) {
    return manualEncodeNumpyMap(v, codec);
  } else if (v && v.constructor === ArrayBuffer) {
    // ArrayBuffer - encode directly with msgpack
    return msgpack.encode(v, { codec });
  } else if (Array.isArray(v)) {
    // Encode array
    const encodedItems = v.map(item => encodeWithNumpySupport(item, codec));
    const len = encodedItems.length;
    const buffers = [];

    // Array header
    if (len <= 15) {
      buffers.push(bufferFrom([0x90 + len])); // fixarray
    } else if (len <= 65535) {
      buffers.push(bufferFrom([0xdc, len >> 8, len & 0xff])); // array16
    } else {
      buffers.push(bufferFrom([0xdd, (len >> 24) & 0xff, (len >> 16) & 0xff, (len >> 8) & 0xff, len & 0xff])); // array32
    }

    buffers.push(...encodedItems);
    return bufferConcat(buffers);
  } else if (v && typeof v === "object" && v.constructor === Object) {
    // Encode plain object as map
    const entries = Object.entries(v);
    const len = entries.length;
    const buffers = [];

    // Map header
    if (len <= 15) {
      buffers.push(bufferFrom([0x80 + len])); // fixmap
    } else if (len <= 65535) {
      buffers.push(bufferFrom([0xde, len >> 8, len & 0xff])); // map16
    } else {
      buffers.push(bufferFrom([0xdf, (len >> 24) & 0xff, (len >> 16) & 0xff, (len >> 8) & 0xff, len & 0xff])); // map32
    }

    // Encode each key-value pair
    for (const [key, value] of entries) {
      // Encode key as string
      const keyBuf = msgpack.encode(key, { codec });
      buffers.push(keyBuf);

      // Recursively encode value
      const valueBuf = encodeWithNumpySupport(value, codec);
      buffers.push(valueBuf);
    }

    return bufferConcat(buffers);
  } else {
    // Primitive value - use msgpack
    return msgpack.encode(v, { codec });
  }
}

export const packBinary = (data) => {
  const processed = packNumpy(data);
  const codec = msgpack.createCodec({ binarraybuffer: true });

  return encodeWithNumpySupport(processed, codec);
};

export const unpackBinary = (data) => {
  return unpackNumpy(msgpack.decode(data));
};
