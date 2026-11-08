# ARM64 Bridge for République

A game-specific ARM32-to-ARM64 compatibility bridge for the Android release of **République**. It runs ARMv7 native libraries inside an AArch64 Android process using Dynarmic's A32 JIT and forwards supported Android, JNI, graphics, audio, and C runtime calls to the host.

This is a source release of the bridge and supporting tools. It targets République 6.1 (version code `1139955`), Unity `2017.4.17f1`, and the guest-library fingerprints checked by `arm64_bridge/check_apk.py`. Native entry points and optimized hot paths are specific to those binaries. Other releases require a separate analysis and validation.

## What is included

- An ELF32 loader, ARM relocation handling, guest memory management, signal handling, and Dynarmic integration.
- AAPCS32 argument translation, JNI object and method handling, and libc/pthread thunks.
- OpenGL ES/EGL forwarding, OpenSL ES audio callbacks, and Android performance hints.
- Native replacements for selected game hot paths, LZ4 decompression, and APK-backed data access.
- Reproducible patches for the pinned Dynarmic ARM64 backend.
- Bridge build scripts, APK packaging helpers, import checks, smali tools, and host regression tests.

Game APKs, OBBs, guest libraries, decoded Java/smali, game assets, signing keys, third-party source trees, build products, device logs, and private project notes are excluded. You must supply your own game files and separately prepared application and split source trees. This repository alone does **not** reconstruct the modified game application from an original APK.

## Build the bridge

The Windows build helper requires PowerShell, Git, Android NDK `27.2.12479018`, Android SDK CMake `3.22.1` with Ninja, and external **Boost 1.83.0** headers. Set `ANDROID_SDK_ROOT` to your SDK directory, or pass `-SdkRoot` explicitly.

Obtain Boost 1.83.0 from the [official release archive](https://archives.boost.io/release/1.83.0/source/). Set `BOOST_ROOT` to its extracted directory containing `boost/version.hpp`. No Boost source is tracked here.

Dependencies downloaded by CMake are pinned:

| Dependency | Revision | License |
| --- | --- | --- |
| [Dynarmic](https://github.com/azahar-emu/dynarmic) | `e77b1ba0b7da7cbe93021b01a663acfe7c4dd516` | 0BSD |
| [LZ4](https://github.com/lz4/lz4) library | `ebb370ca83af193212df4dcbadcc5d87bc0de2f0` (1.10.0) | BSD-2-Clause |
| [Boost](https://www.boost.org/) headers | 1.83.0 | BSL-1.0 |

Dynarmic's own external dependencies are obtained with its Git submodules. Their sources and license notices remain in the external dependency tree. Preserve their applicable notices when distributing a linked binary.

With `BOOST_ROOT` and `ANDROID_SDK_ROOT` configured, allow the first dependency fetch:

```powershell
pwsh -File arm64_bridge/build_arm64.ps1 -AllowNetwork -ReleaseLogs
```

Later builds use the cached dependencies without `-AllowNetwork`. For pre-fetched sources, including Dynarmic's recursive submodules:

```powershell
pwsh -File arm64_bridge/build_arm64.ps1 `
  -BoostSourceDirectory $env:BOOST_ROOT `
  -DynarmicSourceDirectory $env:DYNARMIC_SOURCE_DIR `
  -Lz4SourceDirectory $env:LZ4_SOURCE_DIR `
  -ReleaseLogs
```

Use disposable, revision-matched Dynarmic sources: configuration applies the checked-in patches to that dependency tree. The output is `arm64_bridge/dist/libmain.so`. The helper checks its ELF64/AArch64 identity and 16 KiB load-segment alignment. `-AndroidLogPrint` enables Android logging; it is disabled by default.

The equivalent CMake interface supports `BOOST_ROOT`, `FETCHCONTENT_SOURCE_DIR_DYNARMIC`, and `FETCHCONTENT_SOURCE_DIR_LZ4`, together with the NDK toolchain, `ANDROID_ABI=arm64-v8a`, `ANDROID_PLATFORM=android-26`, and `ANDROID_STL=c++_static`.

## Application integration and packaging

The prepared application loads the bridge as `lib/arm64-v8a/libmain.so`. Its matching ARM32 libraries belong in `assets/guestlib/`, rather than an Android native-library directory. The existing bridge expects the matching Unity/IL2CPP/CRIWARE libraries and Java interfaces; merely copying the bridge into an arbitrary APK is insufficient.

The optional packaging scripts require Bash with MSYS utilities, Python 3.10+, Java, an external `tools/android/toolchain.sh`, and the matching prepared `app/` and `splits/` trees. That external toolchain must provide `tool_apktool`, `tool_zipalign`, and `tool_aapt2`; base packaging also invokes `verify-tools.cmd`. Set `ANDROID_PORT_PROJECT_ROOT` to the directory containing those tools. Set `APK_SIGN_ROOT` to an external directory containing `apksigner.jar`, `testkey.pk8`, and `testkey.x509.pem`, and set `APK_SIGN_CERT_SHA256` to the expected 64-digit certificate fingerprint.

```bash
BRIDGE_LIBRARY=arm64_bridge/dist/libmain.so \
  FINAL_APK=out/republique_arm64_bridge.apk bash build.sh
```

`build.sh` refuses to overwrite an existing output, checks APK signatures and alignment, and runs the game-specific package verifier. To build the prepared OBB splits, also set `ANDROID_TOOL_ROOT` to the external `tools/android` directory and run `bash work/build_split.sh`. Signing material and toolchain configuration must stay outside Git.

## Checks and supporting tools

Run the source contract and ELF import regression tests without game files:

```powershell
python -m unittest discover -s arm64_bridge/tests -p "test_*.py"
python arm64_bridge/check_apk.py
```

The C++ host regression runner requires Visual Studio C++ Build Tools. Supply your local compiler environment script explicitly:

```powershell
python arm64_bridge/tests/run_host_regressions.py `
  --vcvars $env:VCVARS64 --output work/host_regressions
```

Additional tools:

- `check_symbols.py`: check relocation-backed guest imports against bridge providers.
- `check_apk.py`: check a prepared APK, matching guest fingerprints, and optional bridge-library equality.
- `smali_reachability.py` and `smali_gut.py`: inspect and transform locally supplied smali source.
- `memcheck.sh`: collect guest-arena memory samples; device and package are supplied through `DEV` and `PKG`.
- `work/neon_coverage*.py`: inspect ARM32 instruction coverage using the decoder under `DYNARMIC_SOURCE_DIR`.

## Compatibility and limitations

This is a game-specific user-space bridge with partial Android/JNI API coverage. Historical gameplay testing does not establish compatibility across every device, Android version, or game revision. A successful build and 16 KiB ELF alignment do not prove operation on an ARM64-only device with a 16 KiB runtime page size. Validate rendering, audio, input, loading, saves, and sustained gameplay on the intended hardware.

## License

The bridge, supporting scripts, and repository documentation are licensed under **GNU GPL version 3 only** (`GPL-3.0-only`); see [LICENSE](LICENSE). This is a strong copyleft license. External dependencies retain their own licenses. No license to République, Unity, CRIWARE, or other proprietary game components is granted by this repository. République and associated names belong to their respective owners.
