"""Static NEON/VFP coverage check: guest ARM32 libraries vs Dynarmic decoder."""
import os
import re
import struct
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DYNSRC = Path(os.environ["DYNARMIC_SOURCE_DIR"]) / "src/dynarmic/frontend/A32/decoder"
GUEST = ROOT / "app/assets/guestlib"

DECODERS = ["arm.inc", "asimd.inc", "vfp.inc"]


def parse_decoder():
    implemented = []   # (mask, expect, name)
    missing = []       # (mask, expect, name)
    for name in DECODERS:
        text = (DYNSRC / name).read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            m = re.match(r"^\s*(//)?INST\(\s*(\w+)\s*,\s*\"[^\"]*\"\s*,\s*\"([^\"]*)\"\s*\)", line)
            if not m:
                continue
            commented, inst, pattern = m.group(1), m.group(2), m.group(3)
            pattern = pattern.strip()
            if len(pattern) != 32 or any(c not in "01x-_" and not c.isalnum() for c in pattern):
                # sanity: unexpected chars are fine (field names), only 0/1 fix bits
                pass
            mask = expect = 0
            ok = True
            for ch in pattern:
                mask <<= 1
                expect <<= 1
                if ch == "0":
                    mask |= 1
                elif ch == "1":
                    mask |= 1
                    expect |= 1
                elif ch in "01x-_":
                    pass
                elif ch.isalnum():
                    pass  # variable field
                else:
                    ok = False
                    break
            if not ok:
                continue
            if commented:
                missing.append((mask, expect, inst, name))
            else:
                implemented.append((mask, expect, inst, name))
    return implemented, missing


def load_exec_words(path):
    """Yield all 4-byte aligned words from executable PT_LOAD segments."""
    data = path.read_bytes()
    if data[:4] != b"\x7fELF":
        return
    e_phoff = struct.unpack_from("<I", data, 0x1C)[0]
    e_phentsize = struct.unpack_from("<H", data, 0x2A)[0]
    e_phnum = struct.unpack_from("<H", data, 0x2C)[0]
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        p_type, p_offset, p_vaddr, _pa, p_filesz, _memsz, p_flags, _align = struct.unpack_from("<8I", data, off)
        if p_type != 1 or not (p_flags & 1):
            continue
        seg = data[p_offset:p_offset + p_filesz]
        for pos in range(0, len(seg) - 3, 4):
            yield struct.unpack_from("<I", seg, pos)[0]


def in_neon_vfp_space(w):
    """ASIMD unconditional space and VFP/coproc conditional space (cp10/cp11)."""
    if (w & 0xFE000000) == 0xF2000000:      # unconditional ASIMD data-proc
        return True
    if (w & 0xFE000000) == 0xF4000000:      # ASIMD load/store/struct
        return True
    if (w & 0xF0000000) != 0xF0000000 and (w & 0x0F000000) in (0x0C000000, 0x0D000000, 0x0E000000):
        if ((w >> 8) & 0xF) in (10, 11):    # VFP cp10/cp11
            return True
    return False


def main():
    implemented, missing = parse_decoder()
    print(f"decoder entries: implemented={len(implemented)} commented-out(missing)={len(missing)}")

    report = {}
    for lib in sorted(GUEST.glob("*.so")):
        gaps = Counter()
        gap_names = {}
        checked = 0
        for w in load_exec_words(lib):
            if not in_neon_vfp_space(w):
                continue
            checked += 1
            hits_impl = any((w & mask) == expect for mask, expect, _n, _f in implemented)
            if hits_impl:
                continue
            gaps[w] += 1
            # name the closest commented-out entry if any
            names = [nm for mask, expect, nm, _f in missing if (w & mask) == expect]
            gap_names[w] = names[0] if names else "<no decoder entry>"
        report[lib.name] = (checked, gaps, gap_names)

    total_gap_words = 0
    for lib_name, (checked, gaps, gap_names) in sorted(report.items()):
        print(f"\n=== {lib_name}: neon/vfp words checked={checked}, distinct gap words={len(gaps)}, "
              f"gap occurrences={sum(gaps.values())}")
        for w, cnt in sorted(gaps.items(), key=lambda kv: -kv[1])[:40]:
            print(f"  0x{w:08X} x{cnt:6d}  {gap_names[w]}")
        total_gap_words += len(gaps)
    print(f"\nTOTAL distinct gap words: {total_gap_words}")


if __name__ == "__main__":
    main()
