# Resolutions and aspect ratios

The resolution sets a **pixel budget** - the area of a 16:9 frame at that height. The other formats
get the same area, so the time and memory are the same for all formats of one resolution. The sides
are multiples of 16.

| | 480 | 720 | 1080 |
|---|---|---|---|
| 16:9 | 848x480 | 1280x720 | 1920x1088 |
| 4:3 | 736x560 | 1104x832 | 1664x1248 |
| 3:2 | 784x528 | 1168x784 | 1760x1168 |
| 21:9 | 976x416 | 1472x624 | 2192x944 |
| 1:1 | 640x640 | 960x960 | 1440x1440 |

Portrait formats (9:16, 3:4, 2:3, 9:21) use the same numbers with the sides swapped.

To print the table: `uv run python -m my_diffusion.resolutions`, or `GET /resolutions` on the server.

Special case: `aspect_ratio: "auto"` (reference editing in scene mode) takes the exact proportions of
reference 1 at the same pixel budget, with sides rounded to 16. See [image modes](image-modes.md).
