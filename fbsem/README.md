# fbsem

FBSEM-Net (Mehranian & Reader, IEEE NSS/MIC 2019, doi:10.1109/NSS/MIC42101.2019.9059998)
on the D710 sinogram. Each of the `n_it x n_sub` OSEM sub-iterations of `d710 sino` gets a
small 3D residual CNN and the paper's fusion step. The network parameters and the scalar
`gamma` are shared by every sub-iteration. The EM step is exactly `sino`'s, so `gamma = 0`
is OSEM.

```bash
conda activate petct_recon
d710 fbsem train --run R                                   # $D710_OUT/fbsem/runs/R/, val each epoch
d710 fbsem recon --case C --model R/best.pt                # recon_fbsem.npz
d710 export --case C --fbsem                               # C_fbsem_suvbw.nii.gz
d710 fbsem eval --sets thyr_testset                        # NRMSE vs x_true per bed
d710 fbsem recon --out $D710_OUT/X --case sim_an_s1 --check-osem   # gamma=0 vs sino.npz
```

Training reads simulated cases (`<case>/sim_an_s1/`). The label is
`raw_simulation/bed<n>_x_true.npy`. The drivers for sslab are `scripts/fbsem_*.sh` at the
project root.

Design, sources and measurements: `.claude/audit/fbsem/` at the project root.
