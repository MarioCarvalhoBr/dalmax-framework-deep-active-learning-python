"""`dalmax.embeddings.registry` — the embedding-provider counterpart of
`.claude/rules/code-quality.md`'s "registry pattern over if/elif chains".

Also carries `VCTexProvider`'s only functional test in this Phase 2 batch
(no dedicated `test_embeddings_vctex.py` file was allocated to the
`embeddings` owner): see `test_vctex_provider_runs_on_cpu` below, which
verifies the CPU-capability finding documented in
`dalmax/embeddings/vctex_provider.py`'s docstring — `core.tools.VCTex`'s
`RNN` only ever does generic `.to(self._device)` calls, never a hardcoded
`.cuda()`, so it runs correctly with `device="cpu"`.
"""

from __future__ import annotations

import numpy as np
import pytest

from dalmax.embeddings.registry import EMBEDDING_REGISTRY, get_embedding_provider
from dalmax.embeddings.resnet_imagenet_provider import ResNetImageNetProvider
from dalmax.embeddings.ssrae_provider import SSRAEProvider
from dalmax.embeddings.vctex_provider import VCTexProvider


@pytest.mark.parametrize(
    "name, expected_cls",
    [
        ("ssrae", SSRAEProvider),
        ("vctex", VCTexProvider),
        ("resnet_imagenet", ResNetImageNetProvider),
    ],
)
def test_get_embedding_provider_resolves_known_names(name, expected_cls):
    assert get_embedding_provider(name) is expected_cls


def test_registry_contains_exactly_the_three_documented_providers():
    assert set(EMBEDDING_REGISTRY) == {"ssrae", "vctex", "resnet_imagenet"}


def test_unknown_name_raises_key_error_listing_valid_names():
    with pytest.raises(KeyError) as excinfo:
        get_embedding_provider("bogus")

    message = str(excinfo.value)
    assert "bogus" in message
    assert "ssrae" in message
    assert "vctex" in message
    assert "resnet_imagenet" in message


def test_every_registered_class_has_a_name_attribute_matching_its_key():
    for key, cls in EMBEDDING_REGISTRY.items():
        assert cls.name == key


def test_vctex_provider_runs_on_cpu_and_returns_expected_shape(tiny_rgb_image):
    q = [2, 3]
    provider = VCTexProvider(q=q, device="cpu")
    images = np.stack([tiny_rgb_image], axis=0)

    embedded = provider.embed(images)

    assert embedded.shape[0] == 1
    assert embedded.dtype == np.float32
    assert embedded.shape[1] > 0
