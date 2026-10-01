from __future__ import annotations

import types

import numpy as np
import pytest

from simulation import pp, singles, sss
from utils.scanner import GANTRY_XY_MM, NDET, NRINGS, NXTAL


def _ellipse_phantom(shape=(40, 40, 6), dr=(6.0, 6.0, 6.0), seed=3):
    rng = np.random.default_rng(seed)
    nx, ny, nz = shape
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    body = ((x - nx / 2) ** 2 / (nx / 3) ** 2 + (y - ny / 2) ** 2 / (ny / 4) ** 2) < 1
    mu = np.repeat(np.where(body, 0.0096, 0.0)[:, :, None], nz, 2).astype(np.float32)
    act = ((mu > 0) * (1.0 + rng.random(shape))).astype(np.float32)
    origin = ((-np.array(shape) / 2 + 0.5) * np.array(dr)).astype(np.float32)
    return act, mu, origin, np.array(dr, np.float32)


def test_sss_kernel_equals_pytomography_summed_not_averaged(monkeypatch):
    torch = pytest.importorskip("torch")
    psss = pytest.importorskip("pytomography.utils.sss")
    import pytomography
    from pytomography.metadata import ObjectMeta

    monkeypatch.setattr(pytomography, "device", torch.device("cpu"))
    act, mu, origin, vox = _ellipse_phantom()
    lut = pp.crystal_lut()
    got = {}

    def grab(ids, info, weights=None, tof_meta=None):
        got["ids"], got["w"] = ids, weights

    monkeypatch.setattr(psss.shared, "listmode_to_sinogram", grab)
    monkeypatch.setattr(psss.torch, "rand", lambda *a, **k: torch.full(a or (3,), 0.5))
    meta = types.SimpleNamespace(info={"NrCrystalsPerRing": NDET, "NrRings": NRINGS},
                                 scanner_lut=torch.from_numpy(lut))
    psss.compute_sss_sparse_sinogram(ObjectMeta(dr=tuple(vox.tolist()), shape=act.shape), meta,
                                     torch.from_numpy(act), torch.from_numpy(mu),
                                     image_stepsize=4, attenuation_cutoff=0.004,
                                     sinogram_interring_stepsize=8,
                                     sinogram_intraring_stepsize=16)
    rings, crys = sss.sample_rings(8), sss.sample_crystals(16)
    f, info = sss.simulate_sparse((act, origin, vox), (mu, origin, vox), lut, rings, crys,
                                  image_step=4, cutoff=0.004, e_low=430.0, e_res=0.15,
                                  seed=None, out=lambda s: None)
    det = sss.sample_detectors(rings, crys)
    i, j = np.triu_indices(det.size, k=1)
    ids = got["ids"].numpy()
    assert np.array_equal(ids[:, 0], det[i]) and np.array_equal(ids[:, 1], det[j])
    theirs = got["w"].double().numpy() * info["scatter_points"] * 4 ** 3
    ours = f.reshape(det.size, det.size)[i, j].astype(np.float64)
    assert info["scatter_points"] > 10
    assert np.max(np.abs(ours - theirs)) <= 1e-5 * np.max(np.abs(theirs))


def test_sss_interpolation_is_exact_at_the_nodes():
    rings, crys = sss.sample_rings(4), sss.sample_crystals(4)
    rng = np.random.default_rng(0)
    f = rng.random((len(rings), len(crys), len(rings), len(crys))).astype(np.float32)
    for ia, ib in ((0, 0), (2, 5), (len(rings) - 1, 1)):
        t1 = crys[[0, 7, 100]]
        t2 = crys[[50, 3, 143]]
        v = sss.interpolate_ring_pair(f, rings, 4, int(rings[ia]), int(rings[ib]), t1, t2)
        want = f[ia, [0, 7, 100], ib, [50, 3, 143]]
        assert np.allclose(v, want, rtol=1e-6)


def test_sss_interpolation_wraps_the_crystal_axis():
    rings, crys = sss.sample_rings(4), sss.sample_crystals(4)
    f = np.zeros((len(rings), len(crys), len(rings), len(crys)), np.float32)
    f[0, -1, 0, 10] = 1.0
    f[0, 0, 0, 10] = 3.0
    v = sss.interpolate_ring_pair(f, rings, 4, 0, 0, np.array([NDET - 2]), np.array([40]))
    assert np.isclose(v[0], 0.5 * 1.0 + 0.5 * 3.0)


def test_ring_weights_hit_the_last_ring_exactly():
    rings = sss.sample_rings(4)
    i0, w = sss.ring_weights(NRINGS - 1, rings)
    assert rings[i0 + 1] == NRINGS - 1 and w == 1.0
    i0, w = sss.ring_weights(22, rings)
    assert rings[i0] == 20 and np.isclose(w, 2 / 3)


def test_stored_singles_permutation_is_a_bijection_in_modules_of_two_by_four_blocks():
    m = singles.stored_to_crystal()
    assert np.array_equal(np.sort(m), np.arange(NXTAL))
    assert m[:9].tolist() == list(range(9))
    assert m[9] == NDET
    assert m[54] == 9
    assert m[108] == 6 * NDET
    assert m[8 * 54] == 18
    ring, trans = np.divmod(m, NDET)
    blocks = ring // 6 * 64 + trans // 9
    assert all(len(np.unique(blocks[k * 54:(k + 1) * 54])) == 1 for k in range(256))


def test_measured_singles_are_reordered(tmp_path):
    raw = np.arange(NXTAL, dtype=np.uint32)
    case = types.SimpleNamespace(decoded=tmp_path, name="x")
    np.save(tmp_path / "bed1.singles.npy", raw)
    s = singles.measured(case, 1)
    m = singles.stored_to_crystal()
    assert np.array_equal(s[m], raw.astype(np.float64))


def test_randoms_from_singles_are_positive_and_symmetric():
    rng = np.random.default_rng(1)
    rate = 500.0 + 300.0 * rng.random(NXTAL)
    f = pp.singles_randoms(rate, 2.4545, 90.0, None)
    a = rng.integers(0, NXTAL, 1000)
    b = rng.integers(0, NXTAL, 1000)
    bins = np.zeros(1000, np.int64)
    ab, ba = f(a, b, bins), f(b, a, bins)
    assert np.all(ab > 0) and np.allclose(ab, ba)
    assert np.allclose(ab, 2 * 2.4545e-9 * 90.0 * rate[a] * rate[b], rtol=1e-6)


def test_geometric_singles_of_a_centred_point_are_uniform_round_each_ring():
    pytest.importorskip("parallelproj")
    A = np.zeros((5, 9, 9))
    A[2, 4, 4] = 1.0
    gx, gy = GANTRY_XY_MM
    mu = (np.zeros((9, 9, 5), np.float32), np.array([-40 + gx, -40 + gy, -20], np.float32),
          np.array([10, 10, 10], np.float32))
    G = singles.geometric_singles(A, mu, pp.crystal_lut(), crystal_step=3,
                                  out=lambda s: None).reshape(NRINGS, NDET)
    per_ring = G.mean(1)
    assert np.all(G.std(1) / per_ring < 1e-3)
    assert np.allclose(per_ring, per_ring[::-1], rtol=1e-4)
    assert per_ring[11] > per_ring[0]


def test_virtual_beds_cover_the_pet_with_at_least_the_template_overlap():
    from simulation import virtual

    step = 124.26
    pos = virtual.table_positions(-1000.0, 0.0, step)
    assert pos[0] == -1000.0
    assert np.isclose(pos[-1] + virtual.BED_SPAN_MM, 0.0)
    assert np.all(np.diff(pos) <= step + 1e-9)
    assert virtual.table_positions(0.0, virtual.BED_SPAN_MM, step) == [0.0]
    with pytest.raises(SystemExit):
        virtual.table_positions(0.0, 100.0, step)


def test_a_pet_spanning_whole_beds_plus_round_off_gets_no_extra_bed():
    from simulation import virtual

    step = 124.26
    pos = virtual.table_positions(0.0, virtual.BED_SPAN_MM + 6 * step + 1e-6, step)
    assert len(pos) == 7 and np.allclose(np.diff(pos), step)
    assert len(virtual.table_positions(0.0, virtual.BED_SPAN_MM + 6.5 * step, step)) == 8


def test_depostfilter_inverts_ges_post_filter():
    from osem.stitch import post_filter
    from simulation import phantom as ph

    vox = [3.27, 2.734, 2.734]
    z, y, x = np.mgrid[:24, :40, :40].astype(np.float64)
    zz = (z - 12) * vox[0] / vox[1]
    obj = (0.2 * ((y - 20) ** 2 + (x - 20) ** 2 < 15 ** 2)
           + 4.0 * (zz ** 2 + (y - 20) ** 2 + (x - 20) ** 2 < 4 ** 2)).astype(np.float32)

    def H(a):
        return post_filter(a, vox, verbose=False)

    g = H(obj)
    vol = ph.Volume(g, vox[0] * np.arange(24), 0.0, 0.0, vox[1], "t")
    assert ph.depostfilter(vol, iters=0) is vol
    out = ph.depostfilter(vol, iters=30)
    d = out.data
    r0 = np.linalg.norm(H(g) - g) / np.linalg.norm(g)
    r = np.linalg.norm(H(d) - g) / np.linalg.norm(g)
    assert r < r0 and r < 0.02
    assert out.meta["depostfilter"]["residual"] == pytest.approx(r, rel=1e-3)
    assert np.isclose(d.sum(dtype=np.float64), g.sum(dtype=np.float64), rtol=0.01)
    assert d.min() >= 0.0 and d.max() > g.max()


def test_psf_blur_of_the_bed_slab_equals_the_whole_grid_and_keeps_counts(tmp_path):
    from scipy.ndimage import gaussian_filter

    from simulation import phantom as ph
    from utils.scanner import DR_MM, NSEG0, PLANE_MM, PSF_FWHM_MM

    margin = 30.0
    first, n = ph.grid_planes(margin)
    rng = np.random.default_rng(4)
    act = np.zeros((n, 40, 40), np.float32)
    act[:, 10:30, 10:30] = rng.random((n, 20, 20))
    sigma = (np.array([PSF_FWHM_MM[2] / PLANE_MM, PSF_FWHM_MM[1] / DR_MM, PSF_FWHM_MM[0] / DR_MM])
             / (2.0 * np.sqrt(2.0 * np.log(2.0))))
    whole = gaussian_filter(act, sigma, mode="constant")[-first:-first + NSEG0]
    assert np.allclose(ph.psf_bed_planes(act, margin, PSF_FWHM_MM), whole, rtol=1e-5, atol=1e-7)

    inner = np.zeros_like(act)
    sl = slice(-first + 3, -first + NSEG0 - 3)
    inner[sl] = act[sl]
    ph.write_mhd(tmp_path / "act_bqml.mhd", inner)
    meta = {"margin_mm": margin, "decay_scan_to_bed": 1.0, "frame_integral_s": 1.0,
            "timing": {"positron_fraction": 1.0}}
    a = ph.decays_per_voxel(tmp_path, meta)
    b = ph.decays_per_voxel(tmp_path, meta, PSF_FWHM_MM)
    assert a.shape == b.shape == (NSEG0, 40, 40)
    assert np.isclose(b.sum(dtype=np.float64), a.sum(dtype=np.float64), rtol=1e-5)
    assert b.max() < a.max()


def test_shield_aperture_blocks_only_rays_that_leave_through_the_lead():
    lut = pp.crystal_lut().astype(np.float64)
    det = np.array([0, 11 * NDET + 100, 23 * NDET + 300])
    rdet = lut[det]
    inside = np.array([[0.0, 0.0, 0.0], [100.0, -50.0, 60.0]])
    assert singles.aperture(inside, rdet, 108.5, 350.0).all()
    far = np.array([[0.0, 0.0, 1500.0], [0.0, 0.0, -1500.0]])
    assert not singles.aperture(far, rdet, 108.5, 350.0).any()
    near_axis = np.array([[0.0, 0.0, 120.0]])
    ok = singles.aperture(near_axis, rdet, 108.5, 350.0)
    t = (108.5 - 120.0) / (rdet[:, 2] - 120.0)
    rho = t * np.hypot(rdet[:, 0], rdet[:, 1])
    assert np.array_equal(ok[0], rho <= 350.0)


def test_virtual_to_stir_keeps_layout_and_drops_the_template_exam(tmp_path):
    import json

    from simulation import virtual
    from utils.paths import Case

    T = Case("tmpl", tmp_path)
    T.work_bed(4).mkdir(parents=True)
    (T.work_bed(4) / "to_stir.json").write_text(json.dumps({
        "mapping": "m", "sensitivity_term": "s", "background_term": "b",
        "stats": {"randoms": {"sum": 1.0}},
        "estimate": {"ct": "/their/ct", "raw": "/their/raw", "norm": "/cal/norm"}}))
    w = tmp_path / "v"
    w.mkdir()
    virtual.write_to_stir(T, 4, w, "/new/ct.nii.gz", -500.0)
    m = json.loads((w / "to_stir.json").read_text())
    assert m["mapping"] == "m" and m["estimate"]["ct"] == "/new/ct.nii.gz"
    assert "stats" not in m and "raw" not in m["estimate"]
    assert m["virtual"] == {"template": "tmpl", "template_bed": 4}


def test_simulated_cases_and_caches_live_inside_the_case(tmp_path):
    from simulation import events_io as eio
    from utils.paths import Case

    R = Case("fdg1", tmp_path)
    assert eio.sim_case(R, "an", 1).root == tmp_path / "fdg1" / "sim_an_s1"
    assert eio.sim_case(R, "an", 1).name == "sim_an_s1"
    assert eio.sim_root(R) == tmp_path / "fdg1" / "sim_cache"
    V = Case(eio.VIRTUAL, tmp_path / "lympho2_x")
    assert eio.sim_case(V, "an", 2).root == tmp_path / "lympho2_x" / "sim_an_s2"
    assert eio.sim_root(V) == tmp_path / "lympho2_x" / "sim_cache"
    assert eio.sim_named(V, "gate_s1").root == tmp_path / "lympho2_x" / "sim_gate_s1"


def _lympho2_manifest(tmp_path, **suv):
    import json

    s = {"patient_weight_kg": 48.0, "radionuclide_total_dose_Bq": 262700000.0,
         "radionuclide_half_life_s": 6586.2001953125, "decay_correction": "START",
         "injection_time": "2026-03-11T10:00:00", "scan_time": "2026-03-11T11:05:16",
         "decayed_dose_Bq": 173970136.88729987, "suv_factor": 0.0002759094224952816}
    s.update(suv)
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({"SUV": s}))
    return p


def test_exam_from_manifest_is_utc_and_inverts_the_manifest_suv(tmp_path):
    import datetime as dt

    from simulation import phantom as ph
    from simulation import virtual

    e = virtual.exam_from_manifest(_lympho2_manifest(tmp_path), 7.0)
    assert e["radiopharm_start_datetime"] == "20260311030000.00"
    scan = dt.datetime(2026, 3, 11, 4, 5, 16, tzinfo=dt.timezone.utc).timestamp()
    assert e["bed_start_time"] == scan
    assert e["dose_mbq"] == 262.7 and e["residual_dose_mbq"] == 0.0
    assert np.isclose(e["bqml_per_suv"] * 0.0002759094224952816, 1.0, rtol=1e-6)
    timing = {"t_scan": e["bed_start_time"], "dose_bq": e["dose_mbq"] * 1e6,
              "t_inj": ph.utc_epoch(e["radiopharm_start_datetime"]),
              "weight_kg": e["patient_weight_kg"], "half_life_s": e["half_life_s"]}
    f, info = ph.bqml_scale(ph.Volume(np.ones((2, 2, 2), np.float32), np.array([0.0, 1.0]),
                                      0.0, 0.0, 1.0, "t"), "suv", timing)
    assert np.isclose(f, e["bqml_per_suv"], rtol=1e-9)
    assert np.isclose(info["uptake_min"], 65 + 16 / 60)


def test_exam_from_manifest_refuses_a_disagreeing_or_non_start_manifest(tmp_path):
    from simulation import virtual

    with pytest.raises(SystemExit):
        virtual.exam_from_manifest(_lympho2_manifest(tmp_path, decayed_dose_Bq=1.8e8))
    with pytest.raises(SystemExit):
        virtual.exam_from_manifest(_lympho2_manifest(tmp_path, decay_correction="ADMIN"))
    with pytest.raises(SystemExit):
        virtual.exam_from_manifest(_lympho2_manifest(tmp_path, scan_time="2026-03-11T09:00:00"))


def test_virtual_build_lands_in_sim_virtual_with_the_patient_exam(tmp_path, monkeypatch):
    import json

    from simulation import __main__ as cli
    from simulation import events_io as eio
    from simulation import phantom as ph
    from simulation import virtual
    from utils.paths import Case

    T = Case("tmpl", tmp_path)
    T.decoded.mkdir(parents=True)
    w = T.work_bed(4)
    w.mkdir(parents=True)
    for b in (3, 4):
        (T.decoded / f"bed{b}.json").write_text(json.dumps({
            "bed_start_time": 1.0e9 + 100.0 * b, "table_position_mm": 120.0 * b,
            "bed_start_ticks": 7, "dose_mbq": 185.0, "residual_dose_mbq": 3.7,
            "patient_weight_kg": 25.0, "half_life_s": 6586.2, "patient_name": "X",
            "radiopharm_start_datetime": "20260728024500.00"}))
        (T.decoded / f"bed{b}.hs").write_text(f"name of data file := bed{b}.s\n")
    for stem in ("normdt", "norm_only", "randoms"):
        (w / f"{stem}.hs").write_text(f"name of data file := {stem}.s\n")
        (w / f"{stem}.s").write_bytes(b"")
    z = np.arange(0.0, 400.0, 3.27)
    monkeypatch.setattr(ph, "load_volume",
                        lambda p, m: ph.Volume(np.zeros((len(z), 2, 2), np.float32), z,
                                               0.0, 0.0, 2.0, "stub"))
    monkeypatch.setattr(ph, "build", lambda *a, **k: None)
    exam = virtual.exam_from_manifest(_lympho2_manifest(tmp_path))
    V = virtual.build(T, eio.VIRTUAL, "ct.nii.gz", "pet.nii.gz", "suv",
                      tmp_path / "lympho2_x", exam=exam, out=lambda *a: None)
    assert V.root == tmp_path / "lympho2_x" / "sim_virtual"
    beds = V.decoded_beds()
    assert len(beds) >= 2
    for n in beds:
        h = V.header(n)
        assert h["dose_mbq"] == 262.7 and h["patient_weight_kg"] == 48.0
        assert h["residual_dose_mbq"] == 0.0
        assert h["radiopharm_start_datetime"] == "20260311030000.00"
        assert np.isclose(h["bed_start_time"], exam["bed_start_time"] + (n - 1) * 100.0)
        assert "patient_name" not in h
        assert (V.work_bed(n) / "normdt.s").resolve() == (w / "normdt.s").resolve()
    assert json.loads((V.root / "virtual.json").read_text())["exam"]["dose_mbq"] == 262.7
    assert cli.resolve("lympho2_x", tmp_path).root == V.root
    assert cli.resolve("tmpl", tmp_path).root == T.root
    assert eio.sim_case(cli.resolve("lympho2_x", tmp_path), "an", 1).root == \
        tmp_path / "lympho2_x" / "sim_an_s1"


def _needs_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA here")


def _sss_run(batch=None, device="cpu", shield=None):
    act, mu, origin, vox = _ellipse_phantom(shape=(40, 40, 10))
    return sss.simulate_sparse((act, origin, vox), (mu, origin, vox), pp.crystal_lut(),
                               sss.sample_rings(8), sss.sample_crystals(16), image_step=3,
                               seed=0, batch=batch, shield=shield, device=device,
                               out=lambda s: None)


def _singles_numpy_reference(A, mu, lut, crystal_step, shield):
    import parallelproj

    crystals = np.arange(0, NDET, crystal_step)
    det = (np.arange(NRINGS)[:, None] * NDET + crystals[None, :]).ravel()
    rdet = lut[det].astype(np.float64)
    rxy = rdet[:, :2] / np.linalg.norm(rdet[:, :2], axis=1, keepdims=True)
    zyx = np.argwhere(A > 0)
    act = A[tuple(zyx.T)]
    p = mu[1][None, :].astype(np.float64) + zyx[:, ::-1] * mu[2][None, :]
    xe = np.ascontiguousarray(rdet, np.float32)
    T = parallelproj.joseph3d_fwd(np.repeat(p.astype(np.float32), len(det), axis=0),
                                  np.tile(xe, (len(p), 1)), *mu).reshape(len(p), len(det))
    d = rdet[None] - p[:, None]
    r2 = (d ** 2).sum(-1)
    cos = np.clip((d[..., :2] * rxy[None]).sum(-1) / np.sqrt(r2), 0.0, None)
    g = act[:, None] * cos / r2 * np.exp(-T) * singles.aperture(p, rdet, *shield)
    return g.sum(0)


def _singles_phantom():
    rng = np.random.default_rng(5)
    A = np.zeros((6, 8, 8))
    A[1:5, 2:6, 2:6] = rng.random((4, 4, 4))
    gx, gy = GANTRY_XY_MM
    mu = (np.full((8, 8, 6), 0.0096, np.float32),
          np.array([-35 + gx, -35 + gy, -60], np.float32), np.array([10, 10, 24], np.float32))
    return A, mu


def test_sss_does_not_depend_on_the_batch_size():
    f8, i8 = _sss_run(batch=8, shield=(108.5, 350.0))
    f37, i37 = _sss_run(batch=37, shield=(108.5, 350.0))
    assert i8["batch"] == 8 and i37["batch"] == 37 and i8["device"] == "cpu"
    assert np.max(np.abs(f8 - f37)) <= 1e-6 * np.max(np.abs(f8))


def test_torch_geometric_singles_equal_the_numpy_formula():
    pytest.importorskip("parallelproj")
    A, mu = _singles_phantom()
    lut, shield = pp.crystal_lut(), (60.0, 350.0)
    ref = _singles_numpy_reference(A, mu, lut, 3, shield)
    G = singles.geometric_singles(A, mu, lut, crystal_step=3, keep=1.0, shield=shield,
                                  device="cpu", chunk=7, out=lambda s: None)
    got = G.reshape(NRINGS, NDET)[:, ::3].reshape(-1)
    assert np.allclose(got, ref, rtol=1e-6, atol=0.0)


def test_aperture_gives_the_same_answer_for_numpy_and_torch():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(2)
    p = rng.uniform(-400, 400, (50, 3))
    rdet = pp.crystal_lut().astype(np.float64)[::97]
    a = singles.aperture(p, rdet, 108.5, 350.0)
    b = singles.aperture(torch.from_numpy(p), torch.from_numpy(rdet), 108.5, 350.0)
    assert a.dtype == bool and np.array_equal(a, b.numpy())


def test_device_choice(monkeypatch):
    torch = pytest.importorskip("torch")
    from simulation import gpu

    monkeypatch.delenv(gpu.ENV, raising=False)
    assert gpu.device("cpu").type == "cpu"
    monkeypatch.setenv(gpu.ENV, "cpu")
    assert gpu.device(None).type == "cpu"
    assert gpu.batch_for(torch.device("cpu"), 1e6, 8, 256) == 8
    with pytest.raises(SystemExit):
        gpu.device("tpu")
    if not torch.cuda.is_available():
        with pytest.raises(SystemExit):
            gpu.device("cuda")
        monkeypatch.delenv(gpu.ENV)
        assert gpu.device(None).type == "cpu"


def test_sss_on_gpu_matches_cpu():
    _needs_cuda()
    fc, _ = _sss_run(device="cpu", shield=(108.5, 350.0))
    fg, ig = _sss_run(device="cuda", shield=(108.5, 350.0))
    assert ig["device"].startswith("cuda") and ig["batch"] >= 8
    assert np.max(np.abs(fg - fc)) <= 1e-5 * np.max(np.abs(fc))


def test_geometric_singles_on_gpu_match_cpu():
    _needs_cuda()
    A, mu = _singles_phantom()
    kw = dict(crystal_step=3, keep=1.0, shield=(60.0, 350.0), out=lambda s: None)
    gc = singles.geometric_singles(A, mu, pp.crystal_lut(), device="cpu", **kw)
    gg = singles.geometric_singles(A, mu, pp.crystal_lut(), device="cuda", **kw)
    assert np.allclose(gg, gc, rtol=1e-6, atol=0.0)
