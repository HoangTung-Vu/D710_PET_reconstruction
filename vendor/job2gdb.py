#!/usr/bin/env python3
"""Turn a GE cmpclient .job text file into gdb 'set var IgJobReq.<field>' lines.

The .job format is one field per line:  <value> #<fieldName>
Array members are labelled  #name[i]  and 2-D ones  #name[i][j].
String fields are set with gdb's own string assignment so the char array is
NUL-terminated correctly.
"""
import re
import sys

# fields of IgJobReq that are char arrays (s8 name[...]); everything else is
# a scalar n32/s32/f32.  Taken from ptype IgJobReq.
STRING_FIELDS = {
    "inputEmissionFileName",
    "inputTransmissionFileName",
    "normalizationSinogramFile",
    "blankscanSinogramFile",
    "breakPointFile",
    "fileRead3dOverlap",
    "fileWrite3dOverlap",
}

# members of the cmpPackets[] sub-struct.  The .job file lists them flat, one
# group per packet, so they are re-indexed onto cmpPackets[k] here.  A new
# group starts each time cmpProcessingPacketID appears.
PACKET_FIELDS = {
    "cmpProcessingPacketID",
    "sliceNumber",
    "fileRead3dOverlap",
    "fileWrite3dOverlap",
    "cmpPacketDataType",
    "wellCounterValue",
}

LINE = re.compile(r"^(?P<val>.*?)\s*#(?P<name>[A-Za-z_][A-Za-z0-9_]*(?:\[\d+\])*)\s*$")


def parse(path):
    out = []
    packet = -1
    for lineno, raw in enumerate(open(path), 1):
        raw = raw.rstrip("\n")
        if not raw.strip():
            continue
        m = LINE.match(raw)
        if not m:
            continue
        val = m.group("val")
        name = m.group("name")
        base = name.split("[", 1)[0]
        if base in PACKET_FIELDS:
            if base == "cmpProcessingPacketID":
                packet += 1
            name = f"cmpPackets[{packet}].{name}"
        if base in STRING_FIELDS:
            val = val.strip()
            # gdb cannot assign a string literal to a char array directly;
            # write it byte by byte via a helper call instead.
            out.append(("str", name, val))
        else:
            val = val.strip()
            if not val:
                continue
            out.append(("num", name, val))
    return out


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit("usage: job2gdb.py <file.job> [overlap-dir]")
    # 3D recon packets stream axial-overlap files through /petRDFS/OVLFILES,
    # which does not exist off-console and cannot be created inside the mount
    # namespace (/ belongs to real root).  The paths come from the job, so
    # point them at a writable directory instead.
    ovl = sys.argv[2].rstrip("/") if len(sys.argv) == 3 else None
    for kind, name, val in parse(sys.argv[1]):
        if ovl and kind == "str" and "3dOverlap" in name and val.startswith("/petRDFS/OVLFILES/"):
            val = ovl + "/" + val.rsplit("/", 1)[1]
        if kind == "num":
            print(f"set var IgJobReq.{name} = {val}")
        else:
            # Poke the bytes straight into the inferior.  An inferior strcpy()
            # call would resume every thread (letting the parked comm thread
            # reach exit) and trips over glibc's strcpy ifunc besides.
            esc = val.replace("\\", "\\\\").replace('"', '\\"')
            print(f'python _s("IgJobReq.{name}", "{esc}")')


if __name__ == "__main__":
    main()
