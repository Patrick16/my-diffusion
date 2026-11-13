# Checkpoints and guidance

## Supported formats

All formats found in `models/checkpoints` are supported:

| Format | How it is stored |
|---|---|
| `mxfp8` | fp8 + a 2^e multiplier per block of 32 numbers |
| `float8_e4m3fn` | fp8 + one multiplier per layer |
| `int8_tensorwise` | int8 + a multiplier per row |
| bare fp8 | fp8 without multipliers |
| `q8_0` (GGUF) | int8 + a float16 multiplier per block of 32 numbers |

The loader is strict: an unknown format or an "extra" tensor is an error, not silent garbage.

Checkpoints can be switched at runtime with `POST /checkpoints/load` (or `--checkpoint` in the CLI)
without restarting the server.

## Turbo or raw

The kind of checkpoint cannot be determined from the weights, and the settings depend on it:
turbo - 8 steps without CFG, raw - 28 steps and CFG 4.5. By default the kind is taken from the file
name (`raw` in the name -> raw, otherwise turbo); it can be overridden with the
`MY_DIFFUSION_CHECKPOINT_KIND` variable (see [configuration](configuration.md)) or with the `steps` /
`guidance_scale` parameters in the request.

`guidance_scale` here is the usual CFG, where 1.0 means "off". In Wan2GP the scale is shifted by one:
their `guidance 3.5` is our `4.5`.

## Guidance interval

CFG calls the network twice per step. `guidance_start` / `guidance_end` (fractions of the steps,
default `0` / `1`) limit CFG to a part of the run; steps outside it call the network once.
`"guidance_end": 0.6` on 28 steps applies CFG to the first 17 steps and cuts sampling by about 20%
(measured: 20.9 s -> 16.9 s at 480). It changes the image, so it is off by default, and it only matters
when `guidance_scale` is not 1 - that is, on raw checkpoints.

Not validated for image quality: there was no raw checkpoint to test it on, only the timing and the
mechanics were checked (with CFG forced on a turbo checkpoint).
