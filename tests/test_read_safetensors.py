"""read_safetensors: a small hand-made file is read back, and a damaged file is reported by name instead of
failing on a header of a nonsense length (a half-downloaded checkpoint chosen through the config API)."""

from __future__ import annotations

import json
import struct

import pytest
import torch

from my_diffusion.weights import read_safetensors


def write_file(path, tensors: dict[str, torch.Tensor], metadata: dict | None = None) -> None:
    header: dict = {"__metadata__": metadata} if metadata else {}
    body = b""
    for name, tensor in tensors.items():
        raw = tensor.numpy().tobytes()
        header[name] = {"dtype": "F32", "shape": list(tensor.shape), "data_offsets": [len(body), len(body) + len(raw)]}
        body += raw
    text = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(text)) + text + body)


def test_a_file_is_read_back(tmp_path):
    weights = {"a.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3), "b.bias": torch.ones(4)}
    write_file(tmp_path / "small.safetensors", weights, {"format": "pt"})
    tensors, metadata = read_safetensors(tmp_path / "small.safetensors")
    assert metadata == {"format": "pt"}
    assert set(tensors) == set(weights)
    assert all(torch.equal(tensors[name], weights[name]) for name in weights)


@pytest.mark.parametrize("content", [
    b"",                                                   # empty
    b"abc",                                                # shorter than the length field
    b"this is not a safetensors file",                     # the "length" is far beyond the file
    struct.pack("<Q", 0),                                  # no header
    struct.pack("<Q", 12) + b"not json at.",               # the header is not JSON
    struct.pack("<Q", 4) + b"\xff\xfe\xfd\xfc",            # ... not even text
    struct.pack("<Q", 2) + b"[]",                          # JSON, but not a header
    struct.pack("<Q", 5000) + b'{"a": 1}',                 # cut short inside the header
])
def test_a_damaged_file_is_named_in_the_error(tmp_path, content):
    path = tmp_path / "broken.safetensors"
    path.write_bytes(content)
    with pytest.raises(ValueError, match="broken.safetensors is not a .safetensors file"):
        read_safetensors(path)
