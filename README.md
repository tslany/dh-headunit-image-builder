# Genesis DH Headunit SSD Image Builder

Builds a patched 120 GB SSD image for Genesis G80 DH 9.2" head units from an
owner-supplied navigation update.

## Disclaimer

Most of this repository was written by LLM, so the code quality... reflects that.

## Requirements

- Linux with Python 3.10+
- e2fsprogs, util-linux, Partclone, a C compiler, `pkg-config`, and OpenSSL
  development files
- 32 GiB of free space

## Patch and feature catalogs

Byte patches and feature packages are maintained separately:

- [dh-headunit-patches](https://github.com/tslany/dh-headunit-patches)
- [dh-headunit-packages](https://github.com/tslany/dh-headunit-packages)

Copy their catalogs into this checkout before building:

```bash
make -C ../dh-headunit-packages
cp -a ../dh-headunit-patches/byte_patches/. byte_patches/
cp -a ../dh-headunit-packages/feature_packages/. feature_packages/
```

The contents of both local directories are ignored by Git. All populated
recipes and packages are applied by default. Their filenames or directory
names are the IDs accepted by `--exclude-patch` and `--exclude-feature`.
Use `--no-patches` or `--no-features` to disable either catalog.

## Build

Point `--update-dir` at the update directory containing the root-level `.ver`
file and `HU/`.

```bash
python3 -m image_builder build \
  --update-dir /path/to/2017_Genesis_G80_EU \
  --output outputs/headunit.img
```
