"""The four einops rearrangements used by the pinned MoViNet port, in torch.

No global shims or monkey-patches are installed. Unsupported patterns fail.
"""


def rearrange(x, pattern, **axes):
    if pattern == "b c t h w -> (b t) c h w":
        b, c, t, h, w = x.shape
        return x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    if pattern == "(b t) c h w -> b c t h w":
        bt, c, h, w = x.shape
        t = axes["t"]
        return x.reshape(bt // t, t, c, h, w).permute(0, 2, 1, 3, 4)
    if pattern == "b c t h w -> b c t (h w)":
        return x.flatten(3)
    if pattern == "b c t (h w) -> b c t h w":
        b, c, t, hw = x.shape
        w = axes["w"]
        return x.reshape(b, c, t, hw // w, w)
    raise ValueError(f"Unsupported pinned MoViNet rearrangement: {pattern}")
