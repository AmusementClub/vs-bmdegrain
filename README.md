# vs-bmdegrain
Denoising inspired by bm3d and mvdegrain. Under development, **no backward compatibility guarantee**.

## Usage
Prototype:

`core.bmdegrain.BMDegrain(clip clip[, float[] th_sse = 3.0, int block_size = 8, int block_step = 8, int group_size = 8, int bm_range = 7, int radius = 0, int ps_num = 2, int ps_range = 4, bool residual = false, clip rclip = None])`

- clip:

    The input clip. Must be of 32 bit float format. Each plane is denoised separately.

- sigma:

    Denoising strength of each plane. `block_size`-invariant.

- block_size, block_step, group_size, bm_range, radius, ps_num, ps_range:

    Same as those in [VapourSynth-BM3D](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-BM3D).

- residual:

    Whether to center blocks before collaborative filtering. Default: `False`.

- rclip:

    Reference clip for block matching. Must be of the same dimensions and format as `clip`.

## Compilation
[Vector class library](https://github.com/vectorclass/version2) is required when compiling with AVX2.

```bash
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release

cmake --build build

cmake --install build
```

Example build process can be found in [workflows](https://github.com/AmusementClub/vs-bmdegrain/tree/master/.github/workflows).

## Reference
1. [mvtools](http://avisynth.nl/index.php/MVTools)

2. [VapourSynth-WNNM](https://github.com/WolframRhodium/VapourSynth-WNNM)

