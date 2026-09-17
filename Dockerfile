FROM debian:bookworm-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
        gdb python3 binutils libc6 libstdc++6 file procps \
    && rm -rf /var/lib/apt/lists/*

RUN dpkg --add-architecture i386 \
    && apt-get update && apt-get install -y --no-install-recommends \
        gcc gcc-multilib libc6:i386 libstdc++6:i386 libgcc-s1:i386 \
        python3-numpy python3-scipy python3-pydicom python3-pip \
    && pip install --break-system-packages --no-cache-dir petsird==0.9.1 \
    && rm -rf /var/lib/apt/lists/*

ENV PETSW_VLIB=/vendorlib

ENV PETSW_ROOT=/ \
    GERDF_LIBRDF=/usr/PET/lib/linux2/librdf.so.0 \
    GERDF_RDFX=/opt/custom_tool/native/rdfx \
    PYTHONPATH=/opt/custom_tool

RUN mkdir -p /petRDFS/OVLFILES /out /vendorlib

WORKDIR /out

FROM base AS full

COPY custom_tool/petsw/usr/PET /usr/PET
COPY custom_tool/petsw/usr/g   /usr/g
RUN chmod -R u+w /usr/PET/systemConfig && mkdir -p /usr/g/service/log

COPY custom_tool/petsw/usr/lib64/libreadcfg.so* \
     custom_tool/petsw/usr/lib64/libeventmgr.so* \
     custom_tool/petsw/usr/lib64/libmsghand.so* \
     custom_tool/petsw/usr/lib64/libcupipc.so* \
     custom_tool/petsw/usr/lib64/libstartup.so* \
     /vendorlib/

COPY custom_tool/gerdf          /opt/custom_tool/gerdf
COPY custom_tool/native         /opt/custom_tool/native
COPY custom_tool/tests          /opt/custom_tool/tests
COPY custom_tool/ge_rdf_tool.py custom_tool/README.md /opt/custom_tool/

RUN cd /opt/custom_tool/native \
    && bash build.sh \
    && bash build_stubs.sh >/dev/null \
    && ./rdfx 2>&1 | grep -q 'usage: rdfx'
