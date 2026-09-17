#!/usr/bin/env python3
"""Turn a GE cmpclient `.job` file into gdb `set var IgJobReq.<field>` lines."""
import re
import sys

STRING_FIELDS = {
    "inputEmissionFileName",
    "inputTransmissionFileName",
    "normalizationSinogramFile",
    "blankscanSinogramFile",
    "breakPointFile",
    "fileRead3dOverlap",
    "fileWrite3dOverlap",
}

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
    ovl = sys.argv[2].rstrip("/") if len(sys.argv) == 3 else None
    for kind, name, val in parse(sys.argv[1]):
        if ovl and kind == "str" and "3dOverlap" in name and val.startswith("/petRDFS/OVLFILES/"):
            val = ovl + "/" + val.rsplit("/", 1)[1]
        if kind == "num":
            print(f"set var IgJobReq.{name} = {val}")
        else:
            esc = val.replace("\\", "\\\\").replace('"', '\\"')
            print(f'python _s("IgJobReq.{name}", "{esc}")')


if __name__ == "__main__":
    main()
