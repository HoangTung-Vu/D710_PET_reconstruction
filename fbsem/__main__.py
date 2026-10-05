from __future__ import annotations

import sys

USAGE = "usage: python -m fbsem {train,recon,eval} [options]   (--help after a command)"


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0 if argv else 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "train":
        from .train import main as run
    elif cmd == "recon":
        from .recon import main as run
    elif cmd == "eval":
        from .evaluate import main as run
    else:
        print(f"unknown command: {cmd}\n{USAGE}", file=sys.stderr)
        return 2
    return run(rest)


if __name__ == "__main__":
    raise SystemExit(main())
