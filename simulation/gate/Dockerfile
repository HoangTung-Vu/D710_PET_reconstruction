# GATE 10 runtime, for a host whose glibc is too old for the opengate wheels.
#
# `opengate-core` ships manylinux_2_34 wheels and no source distribution, so it
# cannot be installed on Ubuntu 20.04 (glibc 2.31): pip reports "No matching
# distribution found for opengate-core". Debian 12 has glibc 2.36, so the
# wheels install here and the image runs anywhere apptainer does.
#
# Only `python -m simulation.gate.run` runs inside this image -- the Geant4
# part. Everything after it (convert, lm recon, export, compare) stays on the
# host, where numpy, uproot and PyTomography already live.
#
# BUILD, on a machine with docker (as root on the workstation, or on the
# laptop followed by `docker save` / `docker load`):
#
#     docker build -t d710:gate D710/simulation/gate
#
# CONVERT to apptainer, as root on the workstation:
#
#     sif-convert d710:gate          # -> /home/shared/apptainer/images/d710_gate.sif
#
# DECLARE it once, in D710/.env:
#
#     D710_GATE_SIF=/home/shared/apptainer/images/d710_gate.sif
#
# Then `./d710_apptainer simulate gate ...` runs Geant4 inside it.
#
# The image carries the 2.2 GB of Geant4 data, so it needs no network at run
# time and nothing writable: opengate only downloads when the data are missing.
# Expect ~5 GB.

FROM python:3.12-slim-bookworm

ARG OPENGATE_VERSION=10.1.1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore

# opengate pulls in numpy, scipy, itk and uproot itself; `import opengate_core`
# downloads the Geant4 datasets into site-packages, which bakes them in here.
RUN pip install --no-cache-dir "opengate==${OPENGATE_VERSION}" \
 && python -c "import opengate_core" \
 && python -c "import opengate; opengate.Simulation(); print('opengate ${OPENGATE_VERSION} ready')"

WORKDIR /work

CMD ["python", "-c", "import opengate; opengate.Simulation(); print('ok')"]
