#!/usr/bin/env python3
"""Delete the classes nothing can reach, in an apktool tree.

The engine is trivial -- index every .smali, close over every `Lpkg/Class;` descriptor the
kept files mention, delete the rest. All the difficulty is in the ROOT SET, and that is
where the two expensive mistakes live (lesson 6.6, 6.9-6.11):

  * Rooting a whole package "because native calls it by name" keeps hundreds of classes
    that native never names. The names it does use are enumerable -- they are string
    literals in the .so.
  * Collecting only the JNI spelling. FindClass takes com/foo/Bar, Class.forName takes
    com.foo.Bar, and games reach the second through a Java helper. A scanner that knew one
    form put two live classes on the delete list in a real port.

Roots gathered automatically:
  - every component named in AndroidManifest.xml
  - every class-shaped string in the given .so files, BOTH spellings
  - every dotted class name in res/**.xml (LayoutInflater instantiates those by name and
    there is nothing to catch a failure)

Roots you must add by hand (`--keep-class`, `--keep-prefix`) when something is reached in a
way no string reveals. Deny a res/manifest-derived root with `--deny-prefix` when the layout
that names it belongs to a library you are removing.

Reflective names inside smali are deliberately NOT roots: a Class.forName in a try/catch is
a soft dependency, the code is written to survive the class being absent. They are listed at
the end instead of being dropped silently.

    python smali_reachability.py <tree> --lib assets/libGame.so --lib assets/libSdk.so
    python smali_reachability.py <tree> ... --apply

Order matters: run this AFTER cleaning the manifest (a declared component is a root) and
BEFORE cleaning resources (library R$* classes hold a literal for every resource id, so a
resource pass run first reports that everything is used).
"""
import argparse
import os
import re
import sys

TYPE_RE = re.compile(rb"L([a-zA-Z][a-zA-Z0-9_$]*(?:/[a-zA-Z0-9_$]+)*);")
DOTTED_RE = re.compile(r"\b((?:com|org|net|io|android|androidx|de|rx)(?:\.[a-zA-Z0-9_$]+){2,})\b")
SLASH_OK = re.compile(r"^[a-z][a-zA-Z0-9_]*(/[a-zA-Z0-9_$]+)+$")
DOT_OK = re.compile(r"^[a-z][a-zA-Z0-9_]*(\.[a-zA-Z0-9_$]+)+$")
NOT_A_CLASS = ("usr/", "proc/", "sys/", "data/", "dev/", "system/", "vendor/", "apex/",
               "assets/", "lib/", "res/")


def index_classes(tree):
    """name -> path, over every smali* root in the tree."""
    idx = {}
    for d in sorted(os.listdir(tree)):
        if not d.startswith("smali"):
            continue
        root = os.path.join(tree, d)
        if not os.path.isdir(root):
            continue
        for dp, _, fs in os.walk(root):
            for f in fs:
                if f.endswith(".smali"):
                    p = os.path.join(dp, f)
                    idx[os.path.relpath(p, root).replace("\\", "/")[:-6]] = p
    return idx


def names_in_binaries(libs):
    """Class names the native side resolves by string, in both spellings."""
    out = set()
    pat = re.compile(rb"[ -~]{5,}")
    for lib in libs:
        if not os.path.exists(lib):
            sys.exit("!! missing %s" % lib)
        for m in pat.findall(open(lib, "rb").read()):
            t = m.decode()
            if SLASH_OK.match(t) and not t.startswith(NOT_A_CLASS):
                out.add(t)
            elif DOT_OK.match(t):
                out.add(t.replace(".", "/"))
    return out


def names_in_xml(tree):
    """Dotted class names in the manifest and in res/**.xml -- hard references."""
    out = set()
    files = [os.path.join(tree, "AndroidManifest.xml")]
    res = os.path.join(tree, "res")
    if os.path.isdir(res):
        for dp, _, fs in os.walk(res):
            files += [os.path.join(dp, f) for f in fs if f.endswith(".xml")]
    for p in files:
        if not os.path.exists(p):
            continue
        body = open(p, "rb").read().decode("utf-8", "ignore")
        for m in DOTTED_RE.finditer(body):
            out.add(m.group(1).replace(".", "/"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tree")
    ap.add_argument("--lib", action="append", default=[],
                    help="guest .so to read root names from (repeatable)")
    ap.add_argument("--keep-class", action="append", default=[],
                    help="extra root, slash form (repeatable)")
    ap.add_argument("--keep-prefix", action="append", default=[],
                    help="root every class under this prefix (repeatable); use sparingly")
    ap.add_argument("--deny-prefix", action="append", default=[],
                    help="refuse res/manifest-derived roots under this prefix (repeatable)")
    ap.add_argument("--apply", action="store_true", help="actually delete")
    a = ap.parse_args()

    idx = index_classes(a.tree)
    print("[i] %d classes" % len(idx))

    libs = [os.path.join(a.tree, l) if not os.path.isabs(l) else l for l in a.lib]
    named = names_in_binaries(libs) | names_in_xml(a.tree)
    hit = {n for n in named if n in idx}
    refused = sorted(n for n in hit if n.startswith(tuple(a.deny_prefix)) and a.deny_prefix)
    if refused:
        print("[i] refused as roots (--deny-prefix): %s" % ", ".join(refused))
        hit -= set(refused)

    roots = set(hit)
    roots |= {c for c in a.keep_class if c in idx}
    if a.keep_prefix:
        roots |= {n for n in idx if n.startswith(tuple(a.keep_prefix))}
    print("[i] roots: %d (%d from binaries/manifest/res)" % (len(roots), len(hit)))
    if not roots:
        sys.exit("!! empty root set -- pass --lib and/or --keep-class")

    keep, work = set(), list(roots)
    while work:
        n = work.pop()
        if n in keep or n not in idx:
            continue
        keep.add(n)
        for m in TYPE_RE.finditer(open(idx[n], "rb").read()):
            t = m.group(1).decode()
            if t in idx and t not in keep:
                work.append(t)

    dead = sorted(set(idx) - keep)
    print("[i] reachable %d, unreachable %d" % (len(keep), len(dead)))
    by_pkg = {}
    for n in dead:
        k = "/".join(n.split("/")[:3])
        by_pkg[k] = by_pkg.get(k, 0) + 1
    for k, v in sorted(by_pkg.items(), key=lambda kv: -kv[1])[:20]:
        print("      %-46s %5d" % (k, v))

    # No silent caps: name every guarded reflective lookup that will now miss.
    soft = set()
    for n in keep:
        body = open(idx[n], "rb").read().decode("utf-8", "ignore")
        for m in DOTTED_RE.finditer(body):
            t = m.group(1).replace(".", "/")
            if t in idx and t not in keep:
                soft.add("%s -> %s" % (n, t))
    if soft:
        print("[i] reflective targets that will now be absent (verify each is inside a catch):")
        for s in sorted(soft):
            print("      %s" % s)

    if not a.apply:
        print("[i] dry run; pass --apply to delete")
        return
    for n in dead:
        os.remove(idx[n])
    for d in sorted(os.listdir(a.tree)):
        root = os.path.join(a.tree, d)
        if d.startswith("smali") and os.path.isdir(root):
            for dp, _, _ in os.walk(root, topdown=False):
                if dp != root and not os.listdir(dp):
                    os.rmdir(dp)
    print("[+] deleted %d classes. Re-run: it must now report 0 unreachable." % len(dead))


if __name__ == "__main__":
    main()
