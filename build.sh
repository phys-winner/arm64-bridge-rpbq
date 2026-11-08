#!/usr/bin/env bash
set -euo pipefail

GAME_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${ANDROID_PORT_PROJECT_ROOT:-$GAME_ROOT}"
while [ ! -f "$PROJECT_ROOT/tools/android/toolchain.sh" ] && [ "$PROJECT_ROOT" != "$(dirname "$PROJECT_ROOT")" ]; do
  PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
done
TOOL_ROOT="$PROJECT_ROOT/tools/android"
test -f "$TOOL_ROOT/toolchain.sh" || { printf '%s\n' 'Missing project-local toolchain' >&2; exit 1; }
MSYS2_ARG_CONV_EXCL='*' cmd.exe /d /c "$(cygpath -w "$TOOL_ROOT/verify-tools.cmd")" || exit 1
source "$TOOL_ROOT/toolchain.sh"

SIGN_ROOT="${APK_SIGN_ROOT:?set APK_SIGN_ROOT to your external APK signer directory}"
EXPECTED_CERT_SHA256="${APK_SIGN_CERT_SHA256:?set APK_SIGN_CERT_SHA256 to your signing certificate SHA-256}"
[[ "$EXPECTED_CERT_SHA256" =~ ^[[:xdigit:]]{64}$ ]] || { printf '%s\n' 'Expected a 64-digit certificate SHA-256' >&2; exit 1; }
APK_SIGNER="$SIGN_ROOT/apksigner.jar"
APK_SIGN_KEY="$SIGN_ROOT/testkey.pk8"
APK_SIGN_CERT="$SIGN_ROOT/testkey.x509.pem"
test -f "$APK_SIGNER" || { printf 'Missing required portable signer: %s\n' "$APK_SIGNER" >&2; exit 1; }
test -f "$APK_SIGN_KEY" || { printf 'Missing required test key: %s\n' "$APK_SIGN_KEY" >&2; exit 1; }
test -f "$APK_SIGN_CERT" || { printf 'Missing required test certificate: %s\n' "$APK_SIGN_CERT" >&2; exit 1; }

FINAL="${FINAL_APK:-$GAME_ROOT/out/republique_arm64_bridge.apk}"
case "$(basename -- "$FINAL")" in
  republique.apk|republique_arm32_LASTGOOD.apk|snapshot_arm32_*)
    printf 'Protected rollback output: %s\n' "$FINAL" >&2; exit 1 ;;
esac
test ! -e "$FINAL" || { printf 'Output exists; choose a new FINAL_APK: %s\n' "$FINAL" >&2; exit 1; }
mkdir -p -- "$(dirname -- "$FINAL")" "$GAME_ROOT/work"

# Always package a clean copy of app/. Never touch locked IDA files in source inputs.
PACKAGE_WORK="$(mktemp -d "$GAME_ROOT/work/phase7-package.XXXXXX")"
trap 'rm -rf -- "$PACKAGE_WORK"' EXIT
python - "$GAME_ROOT/app" "$PACKAGE_WORK/app" <<'PY'
from pathlib import Path
import shutil
import sys
source, destination = map(Path, sys.argv[1:])
ida = {'.i64', '.id0', '.id1', '.id2', '.nam', '.til'}
def ignore(directory, names):
    relative = Path(directory).relative_to(source)
    skipped = {'build'} if relative == Path('.') else set()
    if relative == Path('assets/guestlib') or relative.parts[:1] == ('lib',):
        skipped.update(n for n in names if '.so.' in n and Path(n).suffix.lower() in ida)
    return skipped
shutil.copytree(source, destination, ignore=ignore)
PY
if [ -n "${BRIDGE_LIBRARY:-}" ]; then
  test -f "$BRIDGE_LIBRARY" || { printf 'Missing bridge override: %s\n' "$BRIDGE_LIBRARY" >&2; exit 1; }
  cp -- "$BRIDGE_LIBRARY" "$PACKAGE_WORK/app/lib/arm64-v8a/libmain.so"
fi
for file in "$PACKAGE_WORK"/app/lib/*/*; do
  [ -f "$file" ] || continue
  case "$file" in
    *.so) ;;
    *) printf 'Refusing to package non-library artifact: %s\n' "$file" >&2; exit 1 ;;
  esac
done

UNSIGNED="$PACKAGE_WORK/unsigned.apk"
ALIGNED="$PACKAGE_WORK/aligned.apk"
tool_apktool b "$PACKAGE_WORK/app" -o "$UNSIGNED"
tool_zipalign -f -P 16 4 "$UNSIGNED" "$ALIGNED"
java -jar "$APK_SIGNER" sign --v1-signing-enabled true --v2-signing-enabled true --v3-signing-enabled true --key "$APK_SIGN_KEY" --cert "$APK_SIGN_CERT" --out "$FINAL" "$ALIGNED"
VERIFY="$(java -jar "$APK_SIGNER" verify --verbose --print-certs "$FINAL")"
VERIFY_V1="$(java -jar "$APK_SIGNER" verify --verbose --print-certs --min-sdk-version 23 "$FINAL")"
printf '%s\n' "$VERIFY"
printf '%s\n' "$VERIFY_V1" | grep -q "Verified using v1 scheme .*: true"
printf '%s\n' "$VERIFY" | grep -Fqi -- "$EXPECTED_CERT_SHA256"
for version in 2 3; do
  printf '%s\n' "$VERIFY" | grep -q "Verified using v$version scheme .*: true"
done
tool_zipalign -c -P 16 4 "$FINAL"
BADGING="$(tool_aapt2 dump badging "$FINAL")"
test "$(printf '%s\n' "$BADGING" | grep -c '^native-code:')" -eq 1
test "$(printf '%s\n' "$BADGING" | grep '^native-code:')" = "native-code: 'arm64-v8a'"
printf '%s\n' "$BADGING" | grep -E "^(package:|sdkVersion:|targetSdkVersion:|native-code:)"
if [ -n "${BRIDGE_LIBRARY:-}" ]; then
  python "$GAME_ROOT/arm64_bridge/check_apk.py" "$FINAL" "$BRIDGE_LIBRARY"
else
  python "$GAME_ROOT/arm64_bridge/check_apk.py" "$FINAL"
fi
printf 'APK=%s\n' "$FINAL"
sha256sum "$FINAL"
