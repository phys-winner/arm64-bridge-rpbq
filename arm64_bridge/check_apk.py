"""ARM64 package gate and independently reproducible raw-deflate reference."""
from pathlib import Path
import hashlib
import struct
import sys
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
ENTRY = "assets/bin/Data/Managed/Resources/mscorlib.dll-resources.dat"
RAW_SHA = "ff2f46780683757c23543069a8addff8ec0f4410330093c900f6238f16a4946e"
DATA_SHA = "c5baa176a5b72cd545266340e42102d393a5e43d38c95796bc828918bb95277f"
GUEST_SHA = {
    "libmain.so": "8245bb1ee053cd4c75019799bbe53be1789dbfe9455d87d1c9d94290759034a8",
    "libunity.so": "6d86b679dc2004df498e23dc3fa1feaef8ef6e90d405aec2fd3515ab6b40ba93",
    "libil2cpp.so": "f4ea589bfe0a833181993dd1dc03c2bb9ac8a690bec06bac2a5b3ff583e73982",
    "libcri_ware_unity.so": "77cf0ec641b9428d9d04b010ece8e60722104fa3304059a6c542a46ccab68087",
    "libovrplatformloader.so": "8c393c4de1b6d4e557f02bb7115e13b57ed079270f74ae55cda838e8bd086a62",
}


def elf_loads(data):
    assert data[:6] == b"\x7fELF\x02\x01", "Expected little-endian ELF64"
    assert struct.unpack_from("<H", data, 18)[0] == 183, "Expected AArch64"
    phoff = struct.unpack_from("<Q", data, 32)[0]
    stride, count = struct.unpack_from("<HH", data, 54)
    assert stride == 56 and phoff + stride * count <= len(data)
    loads = []
    for i in range(count):
        kind, flags, offset, va, pa, filesz, memsz, align = struct.unpack_from("<II6Q", data, phoff + i * stride)
        if kind == 1:
            assert filesz <= memsz and offset + filesz <= len(data)
            assert align >= 0x4000 and (offset - va) % 0x4000 == 0, (offset, va, align)
            loads.append((offset, va, align))
    assert loads, "Missing PT_LOAD"
    return loads


def raw_entry(apk):
    with zipfile.ZipFile(apk) as archive, open(apk, "rb") as source:
        info = archive.getinfo(ENTRY)
        source.seek(info.header_offset)
        header = source.read(30)
        assert header[:4] == b"PK\x03\x04"
        name_len, extra_len = struct.unpack_from("<HH", header, 26)
        offset = info.header_offset + 30 + name_len + extra_len
        source.seek(offset)
        raw = source.read(info.compress_size)
        assert info.compress_type == 8 and len(raw) == 126076 and info.file_size == 337563
        assert hashlib.sha256(raw).hexdigest() == RAW_SHA
        plain = zlib.decompress(raw, -15)
        assert len(plain) == info.file_size and hashlib.sha256(plain).hexdigest() == DATA_SHA
        assert zlib.crc32(plain) == info.CRC
        return offset, raw


def check(apk, bridge_reference=None):
    if bridge_reference is None:
        # The stripped dist copy is disposable; the staged app library is the
        # canonical bridge input retained by the project cleanup policy.
        bridge_reference = ROOT / "app/lib/arm64-v8a/libmain.so"
    with zipfile.ZipFile(apk) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)), "Duplicate ZIP names"
        assert not any(Path(n).suffix.lower() in {".i64", ".id0", ".id1", ".id2", ".nam", ".til"} for n in names)
        libraries = [n for n in names if n.startswith("lib/") and not n.endswith("/")]
        assert libraries == ["lib/arm64-v8a/libmain.so"], libraries
        for name in libraries:
            data = archive.read(name)
            assert data == bridge_reference.read_bytes()
            print(name, "PT_LOAD", elf_loads(data))
        guest_names = {"libmain.so", "libunity.so", "libil2cpp.so", "libcri_ware_unity.so", "libovrplatformloader.so"}
        assert {n.removeprefix("assets/guestlib/") for n in names if n.startswith("assets/guestlib/") and not n.endswith("/")} == guest_names
        for name in sorted(guest_names):
            entry = f"assets/guestlib/{name}"
            assert archive.getinfo(entry).compress_type == zipfile.ZIP_STORED
            data = archive.read(entry)
            source = (ROOT / "app/assets/guestlib" / name).read_bytes()
            assert hashlib.sha256(source).hexdigest() == GUEST_SHA[name], entry
            assert data == source, entry
            assert data[:6] == b"\x7fELF\x01\x01" and struct.unpack_from("<H", data, 18)[0] == 40
            print(entry, len(data), hashlib.sha256(data).hexdigest(), "STORED/source-matched")
        assert archive.getinfo("assets/bin/Data/unity default resources").compress_type == 0
        assert archive.getinfo("assets/bin/Data/Managed/Metadata/global-metadata.dat").compress_type == 0
    offset, raw = raw_entry(apk)
    print(f"RAW_REFERENCE offset={offset} compressed={len(raw)} sha256={RAW_SHA} decompressed_sha256={DATA_SHA}")
    print("PHASE7_PACKAGE_GATE=PASS (not a runtime acceptance)")


def self_check():
    data = bytearray(120)
    data[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<H", data, 18, 183)
    struct.pack_into("<Q", data, 32, 64)
    struct.pack_into("<HH", data, 54, 56, 1)
    struct.pack_into("<II6Q", data, 64, 1, 5, 0, 0, 0, 120, 120, 0x4000)
    assert elf_loads(data) == [(0, 0, 0x4000)]
    for field, value in ((112, 0x1000), (80, 0x1000)):
        broken = bytearray(data)
        struct.pack_into("<Q", broken, field, value)
        try:
            elf_loads(broken)
        except AssertionError:
            pass
        else:
            raise AssertionError("ELF gate missed invalid alignment/congruence")
    print("ELF_AUDIT_SELF_CHECK=PASS")


if __name__ == "__main__":
    self_check()
    if len(sys.argv) > 1:
        reference = Path(sys.argv[2]) if len(sys.argv) > 2 else None
        check(Path(sys.argv[1]), reference)
