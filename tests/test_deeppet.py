"""deepPET: the 2D projector, the noise model, the resampler, the network's shapes."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("parallelproj")


@pytest.fixture(scope="module")
def sc():
    from deepPET.scanner2d import Scanner2D

    return Scanner2D(128)


def disk(grid, r_mm, value=1.0, cx=0.0, cy=0.0):
    from deepPET.scanner2d import FOV_MM

    v = FOV_MM / grid
    c = (np.arange(grid) - (grid - 1) / 2.0) * v
    return np.where(np.hypot(c[None, :] - cx, c[:, None] - cy) <= r_mm, value, 0.0).astype(np.float32)


def test_crop_is_the_bins_inside_the_bore():
    from deepPET.scanner2d import N_TANG, TANG, tangential_distance_mm

    s = tangential_distance_mm()
    assert (TANG.start, TANG.stop, N_TANG) == (5, 376, 371)
    assert np.abs(s[TANG]).max() <= 350.0 < np.abs(s[[4, 376]]).min()


def test_adjoint(sc):
    rng = np.random.default_rng(0)
    x = rng.random((128, 128)).astype(np.float32)
    y = rng.random(sc.shape).astype(np.float32)
    lhs = float(np.vdot(sc.fwd(x).astype(np.float64), y))
    rhs = float(np.vdot(x.astype(np.float64), sc.back(y)))
    assert abs(lhs - rhs) / abs(lhs) < 1e-4


def test_line_integral_of_a_disk(sc):
    """A 100 mm radius uniform disk: the central bins integrate to its 200 mm chord."""
    from deepPET.scanner2d import N_TANG

    p = sc.fwd(disk(128, 100.0))
    assert p[:, N_TANG // 2].mean() == pytest.approx(200.0, rel=0.03)


def test_subset_projection_matches_full(sc):
    x = disk(128, 150.0)
    v = np.arange(3, 288, 16)
    assert np.allclose(sc.fwd(x, v), sc.fwd(x)[v], atol=1e-4)


@pytest.mark.parametrize("mode", ["paper", "attn", "pure"])
def test_noiseless_input_is_the_line_integral(sc, mode):
    """E[x_in] = P x: precorrection undoes attenuation, randoms and scatter exactly."""
    from deepPET.simulate import simulate

    suv = disk(128, 150.0, 2.0, cx=30.0)
    mu = disk(128, 180.0, 0.0096)
    x, t, info = simulate(suv, mu, sc, np.random.default_rng(1), mode, counts=1e6,
                          psf_mm=0.0, noiseless=True, return_raw=True)
    assert np.allclose(x, info["px"], rtol=1e-4, atol=1e-3 * info["px"].max())
    assert np.allclose(t, suv)
    if mode == "pure":
        assert np.allclose(info["mult"], info["s"])
    else:
        assert info["mult"].min() < 0.1 * info["mult"].max()


def test_counts_and_fractions(sc):
    from deepPET.simulate import simulate

    suv, mu = disk(128, 150.0, 2.0), disk(128, 180.0, 0.0096)
    _, _, i = simulate(suv, mu, sc, np.random.default_rng(2), "paper", counts=2e6,
                       noiseless=True, return_raw=True)
    assert i["y"].sum() == pytest.approx(2e6, rel=1e-3)
    assert i["gamma"].sum() == pytest.approx((i["rf"] + i["sf"]) * 2e6, rel=1e-3)


def test_nothing_outside_the_bore_reaches_the_sinogram(sc):
    from deepPET.simulate import simulate

    corner = disk(128, 20.0, 5.0, cx=320.0, cy=320.0)
    x, t, i = simulate(corner, corner * 0, sc, np.random.default_rng(3), "pure", counts=1e6,
                       noiseless=True, psf_mm=0.0)
    assert t.sum() == 0 and i["empty"]


def test_downsample_is_the_2x2_mean():
    from deepPET.simulate import downsample

    a = np.random.default_rng(4).random((256, 256)).astype(np.float32)
    assert np.allclose(downsample(a, 128)[5, 7], a[10:12, 14:16].mean())
    assert downsample(a, 256) is a


@pytest.mark.parametrize("pixel,n", [(500.0 / 256, 256), (700.0 / 256, 256)])
def test_to_grid_puts_a_point_at_its_world_position(pixel, n):
    """A hot pixel at patient (x, y) = (+60, +40) mm lands at STIR (+60, -40) on both FOVs."""
    from deepPET import nifti
    from deepPET.scanner2d import FOV_MM

    x0 = y0 = -(n - 1) / 2.0 * pixel
    a = np.zeros((1, n, n), np.float32)
    a[0, int(round((40 - y0) / pixel)), int(round((60 - x0) / pixel))] = 1.0
    g = nifti.to_grid(a, x0, y0, pixel, 256)[0]
    r, c = np.unravel_index(g.argmax(), g.shape)
    v = FOV_MM / 256
    assert abs((c - 127.5) * v - 60) < 1.5 * v and abs((r - 127.5) * v + 40) < 1.5 * v

    cc = x0 + np.arange(n) * pixel
    blob = (np.hypot(cc[None, :] - 30, cc[:, None] + 20) <= 100.0).astype(np.float32)[None]
    gb = nifti.to_grid(blob, x0, y0, pixel, 256)[0]
    assert gb.sum() * v * v == pytest.approx(blob.sum() * pixel * pixel, rel=0.02)


def test_nifti_las_affine_reads_as_lps(tmp_path):
    """H108 files are LAS; a voxel at a known RAS position must come out at its LPS one."""
    import nibabel as nib

    from deepPET import nifti

    a = np.zeros((8, 8, 3), np.float32)
    a[1, 6, 2] = 7.0
    aff = np.array([[-2.0, 0, 0, 10.0], [0, 2.0, 0, -5.0], [0, 0, 3.0, 100.0], [0, 0, 0, 1]])
    nib.save(nib.Nifti1Image(a, aff), tmp_path / "v.nii.gz")
    v = nifti.load(tmp_path / "v.nii.gz")
    ras = aff @ np.array([1, 6, 2, 1.0])
    z, r, c = np.argwhere(v.data == 7.0)[0]
    assert v.x0 + c * v.pixel_mm == pytest.approx(-ras[0])
    assert v.y0 + r * v.pixel_mm == pytest.approx(-ras[1])
    assert v.z[z] == pytest.approx(ras[2])


@pytest.mark.parametrize("grid", [128, 256])
def test_model_shapes(grid):
    torch = pytest.importorskip("torch")
    from deepPET.model import DeepPET

    m = DeepPET(grid).eval()
    with torch.no_grad():
        out = m(torch.zeros(2, 1, 288, 371))
    assert out.shape == (2, 1, grid, grid)
    assert m.n_conv() == (31 if grid == 128 else 34)


@pytest.fixture
def tiny_data(tmp_path):
    """Two studies of three slices each, in the `prepare` layout."""
    d = tmp_path / "data"
    (d / "slices").mkdir(parents=True)
    rows = []
    for k, sid in enumerate(("0001_20200101", "0002_20200101")):
        suv = np.stack([disk(256, 80.0 + 20 * i, 1.0 + k) for i in range(3)]).astype(np.float16)
        mu = np.stack([disk(256, 150.0, 0.0096)] * 3).astype(np.float16)
        np.save(d / "slices" / f"{sid}_suv.npy", suv)
        np.save(d / "slices" / f"{sid}_mu.npy", mu)
        np.save(d / "slices" / f"{sid}_z.npy", np.arange(3, dtype=np.float32))
        rows.append({"sid": sid, "pid": sid[:4], "n": 3})
    (d / "index.json").write_text(json.dumps(rows))
    (d / "split.json").write_text(json.dumps({"train": ["0001_20200101"],
                                              "val": ["0002_20200101"], "test": []}))
    return d


def test_validation_items_are_deterministic(tiny_data):
    from deepPET.dataset import SinoDataset

    ds = SinoDataset(tiny_data, "val", 128, "paper", train=False)
    a, b = ds.sample(1), ds.sample(1)
    assert np.array_equal(a[0], b[0]) and a[2]["counts"] == b[2]["counts"]
    assert not np.array_equal(ds.sample(0)[0], a[0])


def test_training_items_are_fresh(tiny_data):
    from deepPET.dataset import SinoDataset

    ds = SinoDataset(tiny_data, "train", 128, "paper", train=True)
    assert not np.array_equal(ds.sample(0)[0], ds.sample(0)[0])
    x, t, c = ds[0]
    assert tuple(x.shape) == (1, 288, 371) and tuple(t.shape) == (1, 128, 128)


def test_osem_recovers_a_disk(sc):
    """High counts, the model OSEM assumes: the disk's mean comes back within a few %."""
    from deepPET.osem2d import osem
    from deepPET.simulate import simulate

    suv, mu = disk(128, 120.0, 3.0), disk(128, 160.0, 0.0096)
    _, t, i = simulate(suv, mu, sc, np.random.default_rng(5), "paper", counts=1e7,
                       psf_mm=0.0, return_raw=True)
    img = osem(i["y"], i["mult"], i["gamma"], sc, 3, 8, post_fwhm_mm=0.0)
    inner = disk(128, 90.0) > 0
    assert img[inner].mean() == pytest.approx(3.0, rel=0.05)
