# Image modes

Beyond text-to-image, the service supports image-to-image, inpaint, reference editing and
outpainting. The mode is chosen by which fields are present in the request; the response
contains `mode` (`text2image`, `image2image`, `inpaint`, `edit`).

| Mode | Request fields | CLI flags |
|---|---|---|
| Image-to-image | `init_image`, `strength` (default 0.6) | `--init-image`, `--strength` |
| Inpaint | `init_image`, `mask`, `feather` (default 2), `strength` (default 1.0), `inpaint_iterations` (default 0) | `--init-image`, `--mask`, `--feather`, `--strength`, `--inpaint-iterations` |
| Reference editing | `references` (1 or 2 pictures) | `--reference` (repeat it), `--aspect-ratio auto` |

Pictures in the API are base64 PNG or JPEG (a `data:image/...;base64,` prefix is accepted).

## Image-to-image

The picture is cropped to the requested size, encoded, mixed with noise and denoised from there.
`strength` is how much changes: 1.0 starts from pure noise, which is the same as text-to-image.

```bash
uv run my-diffusion-generate "the same street at night, rain" --init-image street.png --strength 0.6
```

```json
{"prompt": "the same street at night, rain", "init_image": "<base64>", "strength": 0.6}
```

## Inpaint

White in the mask = regenerate, black = keep. It is not a trained inpaint model: at every step the
kept area is blended back from the original latent. `feather` softens the mask edge, in latent
pixels (1 latent pixel = 8 image pixels).

```bash
uv run my-diffusion-generate "a red umbrella" --init-image street.png --mask mask.png --feather 2
```

```json
{"prompt": "a red umbrella", "init_image": "<base64>", "mask": "<base64>", "feather": 2, "strength": 1.0}
```

### Completing part of an object: `inpaint_iterations`

The plain inpaint gives the masked area one network call per noise level, and that is often too
little for it to agree with what is kept. On a subject cut in half the network may draw a second
head; on a band through a body, an extra leg. `inpaint_iterations` (CLI: `--inpaint-iterations`)
adds that many network calls before every step but the last: the picture is moved around at the same
noise level (Langevin dynamics, the LanPaint method), so the two areas settle on one picture before
the step is taken. The kept area is still exactly the original at the end.

| Value | Use | Sampling time (turbo, 8 steps) |
|---|---|---|
| 0 (default) | the plain blend; filling a hole with background, removing an object | 1x |
| 2 | completing a part of an object from the part that is kept | about 2.5x |
| 3-5 | allowed, but measured no better than 2 | up to 5x |

It does not remove the failure, only makes it rarer, and it varies by seed: on a fox cut in half, 2
iterations gave a clean picture in about 6 seeds of 9 against none of 3 without them. Try another
`seed` if a second head still appears.

For removing an object (a large hole to fill with background) keep it at 0: with 5 and more the
model tends to draw a separate scene inside the mask, with its own horizon. To repeat the
comparison on your pictures: `scripts/check_inpaint_iterations.py`.

```bash
uv run my-diffusion-generate "a red fox walking through the snow, side view" --init-image fox.png --mask right_half.png --inpaint-iterations 2
```

```json
{"prompt": "a red fox walking through the snow, side view", "init_image": "<base64>", "mask": "<base64>", "inpaint_iterations": 2}
```

## Reference editing

One or two reference pictures: 1 = the scene (it becomes the output frame), 2 = a subject or person.
Refer to them in the prompt as "image 1" and "image 2". Three parts work together: the LoRA
`krea2_identity_edit_v1_2` (added automatically at weight 1.0 unless the request names it), the
vision encoder of Qwen3-VL (the pictures go into the prompt) and the reference tokens (the pictures'
latents go into the sequence). None of them works alone. `"aspect_ratio": "auto"` takes the exact
proportions of reference 1 at the pixel budget of the resolution (see [resolutions](resolutions.md)).
References can be combined with `init_image` / `mask`, see below.

```bash
uv run my-diffusion-generate "put the person from image 2 into the scene of image 1" --reference scene.png --reference person.png --aspect-ratio auto
```

```json
{"prompt": "put the person from image 2 into the scene of image 1", "references": ["<base64>", "<base64>"], "aspect_ratio": "auto"}
```

Needs the edit LoRA in `models/loras` and the vision weights next to the Qwen3-VL text encoder.

### Reference modes (`reference_mode`)

Two ways to read the references, as in Wan2GP ("KI" and "I"):

- `"scene"` (default): reference 1 is the main scene, the others are people or objects.
  `aspect_ratio: "auto"` takes the exact proportions of reference 1 at the pixel budget of `resolution`
  (sides rounded to 16), so the scene is not cropped.
- `"subjects"`: every reference is a person or an object, there is no scene. The size comes from
  `resolution` + `aspect_ratio`; `"auto"` is not allowed.

```bash
uv run my-diffusion-generate "the two people at a cafe table" --reference-mode subjects --reference a.png --reference b.png --aspect-ratio 4:3
```

```json
{"prompt": "the two people at a cafe table", "references": ["<base64>", "<base64>"], "reference_mode": "subjects", "aspect_ratio": "4:3"}
```

### Limits of reference editing

- Every size up to 1440p runs on a 16 GB card, because the engine streams part of the network from RAM when the
  sequence is long. What differs is time: an edit with one reference takes about 30 s at 720, about 80 s at 1080 and
  about 170 s at 1440; two references at 1080 about 110 s.
- The old limits (one reference up to 720, two at 480) are now a policy, `MY_DIFFUSION_EDIT_LIMITS`: `strict` refuses
  larger sizes, `warn` (the default) accepts them and the job's `warnings` says it is large and about how long it
  takes, `off` accepts silently. `MY_DIFFUSION_EDIT_MAX_PIXELS` caps the output frame in every mode. See
  [configuration](configuration.md). The CUDA out-of-memory error is the second line of defence: the job fails with a
  clear message and the next job runs.
- A GPU check of the features is in `scripts/check_outpaint.py` (it saves pictures to `outputs/`).

## Outpainting (`outpaint`)

Extends reference 1 (the scene) on each side and generates the new area around it. Margins are
percentages of reference 1's own height (top, bottom) or width (left, right). Scene mode only; the
frame size replaces `aspect_ratio`: the frame is reference 1 plus the margins at the resolution's
pixel budget. Reference 1 is placed at a 16-pixel token offset and the rest is generated. All zeros
means off.

```bash
uv run my-diffusion-generate "the same pier, the sea and the sky continue to the right" --reference pier.png --outpaint 0 0 0 25
```

```json
{"prompt": "the same pier, the sea and the sky continue to the right", "references": ["<base64>"], "outpaint": {"top": 0, "bottom": 0, "left": 0, "right": 25}}
```

By default the model may lightly redraw the original area too. Add `"keep_original": true`
(CLI: `--keep-original`) to keep it: after every step the original is put back, as in inpaint, so
only the margins are generated (a 16 px band along inner edges is blended in latent space to hide the
seam). After the VAE decode the original pixels are pasted back with an 8 px feathered seam, so the kept
area is pixel-exact (resized to its box) except inside that seam. With two references the second
one then only fits into the margins.

## References together with `init_image` / `mask`

`init_image` (image-to-image) or `init_image` + `mask` (inpaint) can be combined with `references`, as
Wan2GP does with its control image. The init picture is **image 1** in the prompt and is also what is
noised or inpainted; the references follow it as **image 2**. So the total is limited by `max_references`
(2): with an init picture you can add one reference. Typical use: inpaint a region and bring in a person
or object from a reference.

```json
{"prompt": "the person from image 2 stands in the masked area of image 1", "init_image": "<base64>", "mask": "<base64>", "references": ["<base64>"]}
```

```bash
uv run my-diffusion-generate "the person from image 2 stands in the masked area of image 1" --init-image scene.png --mask mask.png --reference person.png
```

Rules (a bad combination is a clear 400 / CLI error before any model work):
- `mask` still needs `init_image`; `strength` defaults as before (0.6 image-to-image, 1.0 inpaint).
- `reference_mode` must be `scene` (the init picture is the scene); `outpaint` / `keep_original` cannot be
  used with `init_image`.
- Result size and crop follow the init picture exactly as without references (`aspect_ratio` / `auto`).
- The edit LoRA is added as for any reference edit, and the edit size limits count the init picture too.
- The result's `mode` stays `image2image` / `inpaint`; `references` counts only the references you sent.

## Background removal (`remove_background`)

`"subjects"` cuts the background out of the people or objects before use (u2net, a salient-object
model run by onnxruntime on the CPU, so the GPU stays free). In scene mode reference 1 is never cut.
Off by default (`"none"`).

```bash
uv run my-diffusion-generate "the person from image 2 in the scene of image 1" --reference scene.png --reference person.png --remove-background
```

```json
{"prompt": "the person from image 2 in the scene of image 1", "references": ["<base64>", "<base64>"], "remove_background": "subjects"}
```

The model file is `models/rembg/u2net.onnx`. `scripts/link_models.ps1` links it from the Wan2GP
install (`ckpts/rembg`); on another machine put the file there or set `MY_DIFFUSION_REMBG_MODEL`.
