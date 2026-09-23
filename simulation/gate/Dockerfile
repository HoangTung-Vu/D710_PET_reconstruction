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
    PIP_ROOT_USER_ACTION=ignore \
    QT_QPA_PLATFORM=offscreen

# `opengate_core` links against Qt and OpenGL even when nothing is displayed,
# and the slim image has none of it: the first build died on
# "ImportError: libGL.so.1: cannot open shared object file". The wheel bundles
# Qt itself (opengate_core.libs) but not the driver-side libraries below.
# If another one is ever missing, the error names it; add it here and find its
# package with `apt-file search <name>`.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libxext6 \
        libxrender1 \
        libsm6 \
        libice6 \
        libxkbcommon0 \
        libdbus-1-3 \
        libfontconfig1 \
        libfreetype6 \
 && rm -rf /var/lib/apt/lists/*

# opengate pulls in numpy, scipy, itk and uproot itself.
RUN pip install --no-cache-dir "opengate==${OPENGATE_VERSION}"

# `import opengate_core` downloads the Geant4 datasets into site-packages,
# which bakes them into the image. A separate layer, so that a failure here
# does not repeat the install above.
RUN python -c "import opengate_core" \
 && python -c "import opengate; opengate.Simulation(); print('opengate ${OPENGATE_VERSION} ready')"

WORKDIR /work

CMD ["python", "-c", "import opengate; opengate.Simulation(); print('ok')"]
