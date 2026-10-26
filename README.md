# zx0-ultra

Exact multi-threaded ZX0 compressor and a family of 16-bit x86 real-mode (67–84 bytes) decompressors.

## Build

Run `make` at the repository root to compile `zx0`, `dzx0`, and `zx0_ultra` into `bin/`:

```bash
make all
```

## Acknowledgments
- **Einar Saukas** — Creator of the original [ZX0](https://github.com/einar-saukas/ZX0) compression algorithm.
- **Emmanuel Marty** — Author of the reference [unzx0_8088](https://github.com/emmanuel-marty/unzx0_x86) decompressor.
