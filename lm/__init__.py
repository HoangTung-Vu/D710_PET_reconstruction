"""List-mode reconstruction: D710 events -> PyTomography, no file format in between.

`geom`   crystal ids <-> STIR sinogram bins, scanner LUT, TOF index
`terms`  per-event weights (norm x deadtime x attn) and additive term
`recon`  LM-OSEM / BSREM through PyTomography
"""
