"""E1 model factory pinned to Wolny's pytorch-3dunet implementation."""


def build_unet3d():
    from pytorch3dunet.unet3d.model import UNet3D

    return UNet3D(
        in_channels=1,
        out_channels=4,
        f_maps=[32, 64, 128],
        layer_order="gcr",
        num_groups=8,
        final_sigmoid=False,
        is_segmentation=True,
    )

