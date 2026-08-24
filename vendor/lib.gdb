# Helpers shared by every script in this directory.  Source AFTER boot.gdb.
#
# Two things gdb's batch mode does badly and that cost whole runs here:
#   * one failing command aborts the rest of a sourced file, so a 20-minute
#     boot is thrown away by a typo or a missing symbol;
#   * there is no way to get a buffer out of the inferior and onto disk.
# `try` and `dump_*` fix exactly those.
python
import gdb, json, os, array

OUT = "/out"

def say(s):
    """Write and flush.

    gdb buffers its own stdout while the inferior's printf goes straight to
    the terminal, so without the flush the log interleaves misleadingly and a
    run that hangs shows no sign of how far it got.  Every run lost to that
    ambiguity costs 10+ minutes, so everything here flushes.
    """
    gdb.write(s)
    gdb.flush()

class unlocked(object):
    """Let the other threads run for the duration of one call.

    boot.gdb sets scheduler-locking on, because an inferior call otherwise
    resumes all six processing threads and they race the allocator.  A few
    calls genuinely need the pool -- CIgManager::LoadRawData dispatches
    CRawDataLoad to it and would hang forever with the workers frozen -- so
    wrap just those:

        with unlocked():
            show("$ig->LoadRawData()")
    """
    def __enter__(self):
        gdb.execute("set scheduler-locking off")
    def __exit__(self, *a):
        gdb.execute("set scheduler-locking on")
        return False

def ex(cmd, quiet=False):
    """Run a gdb command, print the failure instead of aborting the script.

    Clears DF first: any command can turn into an inferior call (`set $p =
    malloc(...)`, `set $x = memset(...)`), and those inherit EFLAGS from the
    stop point exactly like an explicit `print f()` does.  Having clear_df()
    only in show() left the raw gdb.execute() sites unprotected, which showed
    up as "Couldn't write registers: Input/output error" on a plain malloc.
    """
    clear_df()
    try:
        out = gdb.execute(cmd, to_string=True)
        if not quiet:
            say(out)
        return out
    except gdb.error as e:
        if not quiet:
            say("!! %s  ->  %s\n" % (cmd, e))
        return None

def clear_df():
    """Clear the x86 direction flag before calling into the inferior.

    gdb builds a dummy frame from the register state at the stop point, EFLAGS
    included, so whatever DF happened to be when pet_recon stopped is what the
    called function starts with.  With DF=1, glibc's `rep stos` memset walks
    DOWNWARD: CCyclicMemBuffer::AllocateMem (CyclicMemBuffer.cpp:160) zeroing a
    fresh 121 MB buffer ran off the FRONT of it and took SIGSEGV 13 bytes in --
    rdi = m_pStartBuffer - 17, si_addr the page below.  The System V ABI
    guarantees DF=0 at every call boundary, so the compiler never emits a cld;
    nothing in the inferior repairs it.

    It looked nondeterministic only because DF depended on where the process
    happened to be stopped.  One `and` fixes it for good.
    """
    try:
        gdb.execute("set $eflags = $eflags & ~0x400", to_string=True)
    except gdb.error:
        pass

def show(expr):
    """print <expr>, tolerating symbols that do not exist in this build."""
    clear_df()
    try:
        out = gdb.execute("print " + expr, to_string=True)
    except gdb.error as e:
        # Say WHY.  A bare "<unavailable>" hid a plain syntax refusal for two
        # whole runs while it looked like a crash in the callee.
        say("   %-46s  !! %s\n" % (expr, e))
        return None
    if out is None:
        say("   %-46s  <unavailable>\n" % expr)
    else:
        say("   %-46s %s" % (expr, out.split("= ", 1)[-1]))
    return out

def val(expr):
    """Evaluate to a Python value, or None."""
    try:
        return gdb.parse_and_eval(expr)
    except gdb.error:
        return None

def num(expr, default=0):
    v = val(expr)
    try:
        return int(v)
    except Exception:
        return default

def deep(expr, *path):
    """Walk into a value by field name.

    GRE_IG_PARAMETER_STRUCT is anonymous in the DWARF, so gdb's CLI refuses
    `$ps->rawData` ("not a structure pointer").  The Python value API has no
    such trouble with anonymous types, which is the only way to read the
    geometry the allocators size themselves from.
    """
    v = val(expr)
    if v is None:
        return None
    try:
        if v.type.code == gdb.TYPE_CODE_PTR:
            v = v.dereference()
        for k in path:
            v = v[k]
        return v
    except Exception as e:
        say("!! deep(%s, %s): %s\n" % (expr, path, e))
        return None

def showdeep(expr, *path):
    v = deep(expr, *path)
    say("   %-46s %s\n" % (".".join(path) or expr, v))
    return v

def dump_mem(addr, nbytes, name, meta=None):
    """Copy a raw buffer out of the inferior into /out/<name>.

    This is the whole point of the exercise: the vendor kernels write their
    result into process memory and nothing ever writes it to disk unless the
    job completes, which it cannot here.  ptrace-read it instead.
    """
    addr, nbytes = int(addr), int(nbytes)
    if not addr or nbytes <= 0:
        say("!! dump %s: addr=0x%x size=%d -- skipped\n" % (name, addr, nbytes))
        return False
    path = os.path.join(OUT, name)
    inf = gdb.selected_inferior()
    # These run to ~240 MB per buffer; read in slices so neither gdb nor the
    # Python heap has to hold a second full copy.
    CHUNK = 8 << 20
    dt = meta.get("dtype", "f4") if meta else "f4"
    lo, hi, nz, tot = None, None, 0, 0
    with open(path, "wb") as f:
        off = 0
        while off < nbytes:
            k = min(CHUNK, nbytes - off)
            b = bytes(inf.read_memory(addr + off, k))
            f.write(b)
            if dt == "f4":
                a = array.array("f"); a.frombytes(b[:len(b) - len(b) % 4])
            elif dt == "u2":
                a = array.array("H"); a.frombytes(b[:len(b) - len(b) % 2])
            else:
                a = array.array("i"); a.frombytes(b[:len(b) - len(b) % 4])
            tot += len(a)
            for x in a:
                if x:
                    nz += 1
                    if lo is None or x < lo: lo = x
                    if hi is None or x > hi: hi = x
            off += k
    info = dict(meta or {})
    info.update(name=name, addr="0x%x" % addr, bytes=nbytes,
                elements=tot, nonzero=nz,
                min_nonzero=lo, max=hi)
    with open(path + ".json", "w") as f:
        json.dump(info, f, indent=2, sort_keys=True)
    say("== %s: %d bytes @0x%x  nonzero=%d/%d  min=%s max=%s\n"
        % (name, nbytes, addr, nz, tot, lo, hi))
    return True

def dump_expr(expr, nbytes, name, meta=None):
    """dump_mem() on the address an expression evaluates to."""
    v = val(expr)
    if v is None:
        say("!! dump %s: cannot evaluate %s\n" % (name, expr))
        return False
    try:
        addr = int(v)
    except Exception:
        addr = int(v.address)
    m = dict(meta or {}); m["expr"] = expr
    return dump_mem(addr, nbytes, name, m)

def dump_views(expr, name, dtype="f4", extra=None):
    """Dump a whole CViewBuffer: every view, in view order, one flat file.

    A CViewBuffer is a CCyclicMemBuffer of m_uiAllocElement views of
    m_uiElementSize bytes.  CCyclicMemBuffer::CalculateElementAddress puts
    view i at m_pFirstElement + (i - m_uiFirstElementVal) * m_uiElementSize,
    wrapping at m_pEndBuffer.  After a fresh Deallocate/Reallocate the ring is
    unrotated -- m_pFirstElement == m_pStartBuffer and m_uiFirstElementVal ==
    0 -- so the views are already contiguous and in order, and one flat read
    is exact.  Both assumptions are checked, not assumed.
    """
    cyc = expr + "->m_pCyclicMemBuff"
    n     = num(cyc + "->m_uiAllocElement")
    sz    = num(cyc + "->m_uiElementSize")
    first = num(cyc + "->m_uiFirstElementVal")
    start = val(cyc + "->m_pStartBuffer")
    felem = val(cyc + "->m_pFirstElement")
    if not n or not sz:
        say("!! %s: nothing allocated (views=%s viewSize=%s)\n" % (name, n, sz))
        return False
    if first != 0 or (start is not None and felem is not None
                      and int(start) != int(felem)):
        say("!! %s: ring is rotated (firstVal=%d start=%s firstElem=%s);"
            " a flat dump would be out of order -- skipped\n"
            % (name, first, start, felem))
        return False
    meta = dict(views=n, view_bytes=sz, dtype=dtype, source=expr)
    if extra:
        meta.update(extra)
    return dump_mem(int(start), n * sz, name, meta)

def setv(cmd):
    """`set`-style command that may call into the inferior; DF-safe, quiet."""
    return ex(cmd, quiet=True)

def banner(t):
    say("\n=== %s ===\n" % t)
end
