"""`vendor/job2gdb.py` -- GE's `.job` text -> `set var IgJobReq.<field>` lines.

This is what turned the vendor's own `selftest_kh_3dir.job` into `job.gdb`.
All 524 fields came through it, so a field silently dropped or mis-indexed
means the recon runs with a default instead of GE's value.
"""

from __future__ import annotations

import job2gdb


def write(tmp_path, *lines):
    p = tmp_path / "a.job"
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def test_a_scalar_becomes_a_set_var(tmp_path):
    assert job2gdb.parse(write(tmp_path, "3 #reconType")) == \
        [("num", "reconType", "3")]


def test_array_indices_survive(tmp_path):
    assert job2gdb.parse(write(tmp_path, "1.5 #zFilter[2]")) == \
        [("num", "zFilter[2]", "1.5")]


def test_two_dimensional_members_survive(tmp_path):
    assert job2gdb.parse(write(tmp_path, "7 #table[1][2]")) == \
        [("num", "table[1][2]", "7")]


def test_a_string_field_is_marked_as_one(tmp_path):
    got = job2gdb.parse(write(tmp_path, "/data/emission.rdf #inputEmissionFileName[0]"))
    assert got == [("str", "inputEmissionFileName[0]", "/data/emission.rdf")]


def test_an_empty_string_field_is_kept_but_an_empty_scalar_is_not(tmp_path):
    """An unset filename is meaningful; an unset number would poke a garbage value."""
    got = job2gdb.parse(write(tmp_path, " #breakPointFile", " #reconType"))
    assert got == [("str", "breakPointFile", "")]


def test_lines_without_a_field_name_are_ignored(tmp_path):
    got = job2gdb.parse(write(tmp_path, "", "just a comment", "4 #reconType"))
    assert got == [("num", "reconType", "4")]


def test_packet_members_are_regrouped_onto_cmp_packets(tmp_path):
    """The `.job` lists packet members flat; each ID starts the next packet."""
    got = job2gdb.parse(write(
        tmp_path,
        "0 #cmpProcessingPacketID", "47 #sliceNumber",
        "1 #cmpProcessingPacketID", "48 #sliceNumber"))
    assert got == [("num", "cmpPackets[0].cmpProcessingPacketID", "0"),
                   ("num", "cmpPackets[0].sliceNumber", "47"),
                   ("num", "cmpPackets[1].cmpProcessingPacketID", "1"),
                   ("num", "cmpPackets[1].sliceNumber", "48")]


def test_a_field_that_is_both_packet_and_string_gets_both(tmp_path):
    got = job2gdb.parse(write(
        tmp_path, "0 #cmpProcessingPacketID",
        "/petRDFS/OVLFILES/ovl0 #fileRead3dOverlap"))
    assert got[1] == ("str", "cmpPackets[0].fileRead3dOverlap",
                      "/petRDFS/OVLFILES/ovl0")


def test_non_packet_fields_are_not_reindexed(tmp_path):
    got = job2gdb.parse(write(tmp_path, "0 #cmpProcessingPacketID", "9 #reconType"))
    assert got[1] == ("num", "reconType", "9")


def test_main_emits_gdb_syntax(tmp_path, monkeypatch, capsys):
    job = write(tmp_path, "3 #reconType",
                "/data/emission.rdf #inputEmissionFileName[0]")
    monkeypatch.setattr("sys.argv", ["job2gdb.py", job])
    job2gdb.main()
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "set var IgJobReq.reconType = 3"
    assert out[1] == 'python _s("IgJobReq.inputEmissionFileName[0]", "/data/emission.rdf")'


def test_main_escapes_quotes_and_backslashes(tmp_path, monkeypatch, capsys):
    job = write(tmp_path, r'a"b\c #breakPointFile')
    monkeypatch.setattr("sys.argv", ["job2gdb.py", job])
    job2gdb.main()
    assert r'"a\"b\\c"' in capsys.readouterr().out


def test_overlap_paths_are_redirected_off_console(tmp_path, monkeypatch, capsys):
    """`/petRDFS/OVLFILES/` cannot be created inside the mount namespace."""
    job = write(tmp_path, "0 #cmpProcessingPacketID",
                "/petRDFS/OVLFILES/ovl_0 #fileWrite3dOverlap")
    monkeypatch.setattr("sys.argv", ["job2gdb.py", job, "/out/ovl/"])
    job2gdb.main()
    out = capsys.readouterr().out
    assert '"/out/ovl/ovl_0"' in out
    assert "/petRDFS/OVLFILES" not in out


def test_only_overlap_paths_are_redirected(tmp_path, monkeypatch, capsys):
    job = write(tmp_path, "/petRDFS/OVLFILES/x #inputEmissionFileName[0]")
    monkeypatch.setattr("sys.argv", ["job2gdb.py", job, "/out/ovl"])
    job2gdb.main()
    assert "/petRDFS/OVLFILES/x" in capsys.readouterr().out
