# sino

Non-TOF OSEM on the decoded sinogram, with PyTomography and parallelproj. It
does what `d710 osem` does without SIRF or STIR, and is meant to replace it.

```bash
conda activate petct_recon
d710 attn   --case ped
d710 sino   --case ped            # -> recon_sino.npz, work/bed<n>/sino.npz
d710 export --case ped --sino     # K = utils.scanner.K_EXPORT_SINO
```

It reads the same five files per bed as `osem`: `decoded/bed<n>.hs` (prompts;
a TOF sinogram is summed over TOF) and `work/bed<n>/{normdt,attn,background}.hs`.

## The model

`y = S (G x) + b`, with `S = normdt x attn` and `b = background` per bin.

`G` projects every detector-ring pair of a bin with `parallelproj.joseph3d_*`
and takes their mean, so span 2 is modelled exactly: the 23 odd planes of
segment 0 hold two ring pairs, every other plane one, and GE's `normdt` already
carries the 2x. The endpoints come from `BinMap.ring_pairs_by_plane()`,
`det_pair_map` and `detector_xy_mm`, which are the ones `utils/attn_proj.py`
uses, so `d710 lm check` proves this geometry too. `--psf` defaults to GE's
PSF, as in `d710 lm recon`.

Subsets are views (`v % n_subsets`), and each subset projects only its own
views -- the thing STIR's parallelproj projector does not do.

## K

`parallelproj` returns path length in mm; STIR's projectors return it in
voxels. The image is therefore about `DR_MM` = 2.13x smaller than
`d710 osem`'s, and K is its own constant, `K_EXPORT_SINO`, measured like the
other two (`tools/compare_vendor.py --sino`, then `tools/calib_k.py`).

Measurements and design notes: `.claude/audit/sino/` at the project root.
