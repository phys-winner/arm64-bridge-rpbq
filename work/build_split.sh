#!/usr/bin/env bash
set -euo pipefail
GAME_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TOOL_ROOT="${ANDROID_TOOL_ROOT:?set ANDROID_TOOL_ROOT to the external tools/android directory}"
source "$TOOL_ROOT/toolchain.sh"
SIGN_ROOT="${APK_SIGN_ROOT:?set APK_SIGN_ROOT to your external APK signer directory}"
EXPECTED_CERT_SHA256="${APK_SIGN_CERT_SHA256:?set APK_SIGN_CERT_SHA256 to your signing certificate SHA-256}"
[[ "$EXPECTED_CERT_SHA256" =~ ^[[:xdigit:]]{64}$ ]] || { printf '%s\n' 'Expected a 64-digit certificate SHA-256' >&2; exit 1; }
APK_SIGNER="$SIGN_ROOT/apksigner.jar"
APK_SIGN_KEY="$SIGN_ROOT/testkey.pk8"
APK_SIGN_CERT="$SIGN_ROOT/testkey.x509.pem"

build_split() {
  local dir="$1" out="$2"
  local unsigned="$WORK_DIR/$(basename "$out" .apk)_unsigned.apk"
  local aligned="$WORK_DIR/$(basename "$out" .apk)_aligned.apk"
  tool_apktool b "$GAME_ROOT/splits/$dir" -o "$unsigned"
  tool_zipalign -f -P 16 4 "$unsigned" "$aligned"
  java -jar "$APK_SIGNER" sign --v1-signing-enabled true --v2-signing-enabled true --v3-signing-enabled true \
    --key "$APK_SIGN_KEY" --cert "$APK_SIGN_CERT" --out "$GAME_ROOT/splits/$dir/$out" "$aligned"
  local verification
  verification="$(java -jar "$APK_SIGNER" verify --verbose --print-certs "$GAME_ROOT/splits/$dir/$out")"
  printf '%s\n' "$verification"
  printf '%s\n' "$verification" | grep -Fqi -- "$EXPECTED_CERT_SHA256"
  tool_zipalign -c -P 16 4 "$GAME_ROOT/splits/$dir/$out"
  sha256sum "$GAME_ROOT/splits/$dir/$out"
}

WORK_DIR="$(mktemp -d)"
trap 'rm -rf -- "$WORK_DIR"' EXIT
build_split obb_main republique-obb-main_signed.apk
build_split obb_patch republique-obb-patch_signed.apk
echo SPLITS_DONE
