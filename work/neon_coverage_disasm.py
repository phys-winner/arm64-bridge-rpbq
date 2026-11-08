"""Ground-truth check: parse objdump disassembly, flag instructions the Dynarmic decoder cannot handle."""
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DYNSRC = Path(os.environ["DYNARMIC_SOURCE_DIR"]) / "src/dynarmic/frontend/A32/decoder"
DECODERS = ["arm.inc", "asimd.inc", "vfp.inc"]


def parse_decoder():
    implemented = []
    missing = []
    for name in DECODERS:
        text = (DYNSRC / name).read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            m = re.match(r"^\s*(//)?INST\(\s*(\w+)\s*,\s*\"[^\"]*\"\s*,\s*\"([^\"]*)\"\s*\)", line)
            if not m:
                continue
            commented, inst, pattern = m.group(1), m.group(2), m.group(3)
            mask = expect = 0
            for ch in pattern.strip():
                mask <<= 1
                expect <<= 1
                if ch == "0":
                    mask |= 1
                elif ch == "1":
                    mask |= 1
                    expect |= 1
            if commented:
                missing.append((mask, expect, inst))
            else:
                implemented.append((mask, expect, inst))
    return implemented, missing


def mnemonic_of(line):
    # objdump line: "  f04d3c: f3c25480 \tvraddhn.i16\td21, q9, q0"
    m = re.match(r"^\s*[0-9a-f]+:\s+([0-9a-f]{8})\s+\t(.+)$", line)
    if not m:
        return None, None
    word = int(m.group(1), 16)
    text = m.group(2)
    return word, text


def main():
    implemented, missing = parse_decoder()

    for disasm_path in sys.argv[1:]:
        gaps = Counter()
        gap_mnemonics = {}
        lines = 0
        with open(disasm_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if "\t" not in line or ":" not in line:
                    continue
                word, text = mnemonic_of(line)
                if word is None:
                    continue
                lines += 1
                mnem0 = text.split()[0] if text else ""
                # NEON/VFP mnemonics start with 'v' (or unknown pool junk, kept for review)
                if not mnem0.startswith("v"):
                    continue
                if any((word & mask) == expect for mask, expect, _n in implemented):
                    continue
                mnem = text.split()[0] if text else "<none>"
                gaps[(word, mnem)] += 1
                gap_mnemonics.setdefault((word, mnem), text[:70])
        print(f"=== {disasm_path}: instruction lines={lines}")
        for (word, mnem), cnt in sorted(gaps.items(), key=lambda kv: -kv[1])[:30]:
            names = [nm for mask, expect, nm in missing if (word & mask) == expect]
            dyn_name = names[0] if names else "<no decoder entry>"
            print(f"  0x{word:08X} x{cnt:5d}  {mnem:30s} dyn: {dyn_name}   | {gap_mnemonics[(word, mnem)]}")
        print(f"  total distinct gap encodings: {len(gaps)}, occurrences: {sum(gaps.values())}\n")


if __name__ == "__main__":
    main()
