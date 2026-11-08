#!/usr/bin/env python3
"""Empty a smali class while keeping the surface the native side resolves by name.

Why not just delete the class: libraries of this era reach Java through JNI FindClass +
GetMethodID, i.e. by *name*. A deleted class turns that lookup into a null and a pending
exception somewhere deep in the guest; an empty method returns the "nothing happened"
answer and the guest carries on (lesson 6.1 of LEGACY_GAME_PORTING_LESSONS.md).

Why emptying alone is not enough: a signature is a list of type references, so the class
still drags in whatever its parameters and return types mention. In one port a stubbed
analytics entry point kept 2071 classes alive through the parameter types of four methods
nobody called. Hence `keep_only`: name the methods the native side actually calls and
delete the rest outright, signature and all.

Three rules bought with debugging:

  1. The return type decides whether a stub is safe. `void` always is. A method that used
     to answer "" will now answer null, and the guest may walk off with that into
     GetStringUTFChars -- read the original before stubbing anything that returns a value.
  2. Never gut a class the native side reads fields from (GetFieldID): dropping fields
     breaks it silently.
  3. `.implements` is a type reference too. An emptied class that still declares the
     interfaces of the subsystem you are deleting keeps it alive by its header alone.
     Dropping them is safe only once nothing assigns an instance to those types -- which is
     exactly what emptying the bodies guarantees.

Where to get the method list: the JNI name/signature pairs sit in .rodata right next to the
class name. Dump the strings of the guest .so, find the class name, read the next few:

    com/example/Analytics
    init             (Landroid/content/Context;Ljava/lang/String;)V
    logEvent         (Ljava/lang/String;)I

Use as a library (`from smali_gut import gut`) or from the command line:

    python smali_gut.py <file.smali> [--drop-implements] [--keep-only m1,m2,...]
"""
import re
import sys

METHOD_RE = re.compile(r"^\.method\s+(.*?)\s*$")
SIG_RE = re.compile(r"^(?P<flags>[\w$ ]*?)\s*(?P<name>[^ (]+)\((?P<params>[^)]*)\)(?P<ret>.+)$")


def _stub_body(ret, super_cls, name):
    """Minimal verifiable body for a method returning `ret`."""
    if name == "<init>":
        # A constructor must reach a super constructor before returning. Every class worth
        # gutting so far derived from Object or from a framework class with a no-arg
        # constructor; if yours does not, this is where it breaks loudly.
        return ["    .locals 0", "", "    invoke-direct {p0}, %s-><init>()V" % super_cls,
                "", "    return-void"]
    if ret == "V":
        return ["    .locals 0", "", "    return-void"]
    if ret in ("J", "D"):
        return ["    .locals 2", "", "    const-wide/16 v0, 0x0", "", "    return-wide v0"]
    if ret in ("Z", "B", "S", "C", "I", "F"):
        return ["    .locals 1", "", "    const/4 v0, 0x0", "", "    return v0"]
    return ["    .locals 1", "", "    const/4 v0, 0x0", "", "    return-object v0"]


def gut(path, drop_implements=False, keep_only=None):
    """Rewrite `path` in place.

    drop_implements  also remove the `.implements` lines (see rule 3 above).
    keep_only        set of method names; every other method is deleted rather than
                     emptied, so its parameter and return types stop being references.

    Returns (methods_kept_verbatim, methods_stubbed, fields_dropped).
    """
    raw = open(path, "rb").read().decode("utf-8")
    crlf = "\r\n" in raw
    lines = raw.replace("\r\n", "\n").split("\n")

    super_cls = "Ljava/lang/Object;"
    for line in lines:
        if line.startswith(".super "):
            super_cls = line.split(None, 1)[1].strip()
            break

    out, i, n = [], 0, len(lines)
    kept = stubbed = fields = 0
    while i < n:
        line = lines[i]
        if drop_implements and line.startswith(".implements"):
            i += 1
            continue
        if line.startswith(".field"):
            # Fields go: a field descriptor is a type reference, and it is usually the
            # field types -- not the code -- that pin the SDK you are trying to remove.
            fields += 1
            i += 1
            while i < n and lines[i].strip().startswith(".annotation"):
                while i < n and lines[i].strip() != ".end annotation":
                    i += 1
                i += 1
            if i < n and lines[i].strip() == ".end field":
                i += 1
            continue
        if line.startswith(".method"):
            start = i
            while i < n and lines[i].strip() != ".end method":
                i += 1
            block = lines[start:i + 1]
            i += 1
            m = SIG_RE.match(METHOD_RE.match(block[0]).group(1))
            if m is None:
                out += block
                kept += 1
                continue
            if keep_only is not None and m.group("name") not in keep_only:
                continue
            if "native" in m.group("flags").split() or "abstract" in m.group("flags").split():
                out += block          # no body to remove
                kept += 1
                continue
            out.append(block[0])
            out += _stub_body(m.group("ret"), super_cls, m.group("name"))
            out.append(".end method")
            stubbed += 1
            continue
        out.append(line)
        i += 1

    text = re.sub(r"\n{3,}", "\n\n\n", "\n".join(out)).rstrip("\n") + "\n"
    open(path, "wb").write((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))
    return kept, stubbed, fields


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit(__doc__)
    only = None
    for a in sys.argv[1:]:
        if a.startswith("--keep-only="):
            only = set(a.split("=", 1)[1].split(","))
    k, s, f = gut(args[0], "--drop-implements" in sys.argv, only)
    print("%s: %d kept verbatim, %d stubbed, %d fields dropped" % (args[0], k, s, f))
