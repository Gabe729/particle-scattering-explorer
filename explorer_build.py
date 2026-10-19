"""Assemble the self-contained particle scattering explorer.

Runs both validation suites (and refuses to build if either fails), regenerates
reference.json, then inlines plotly.min.js, mie.js, tmatrix.js and reference.json
into explorer_template.html so the result is a single file that works offline.

Usage
-----
    python explorer_build.py                     # -> dist/scattering_explorer.html
    python explorer_build.py --docs              # also -> docs/index.html (GitHub Pages)
    python explorer_build.py --plotly PATH       # inline a specific plotly.min.js
    python explorer_build.py --cdn               # load Plotly from a pinned CDN URL
                                                 # instead of inlining it (small file,
                                                 # but needs a network connection)

Where plotly.min.js comes from, in order:
  1. --plotly PATH, if given;
  2. the installed `plotly` Python package (`pip install plotly`), which ships the
     minified plotly.js bundle as plotly/package_data/plotly.min.js.
With --cdn neither is needed.
"""

import argparse
import datetime
import importlib.util
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
DIST = HERE / "dist" / "scattering_explorer.html"
DOCS = HERE / "docs" / "index.html"
PLOTLY_CDN_VERSION = "3.6.0"
PLOTLY_CDN = f"https://cdn.jsdelivr.net/npm/plotly.js-dist-min@{PLOTLY_CDN_VERSION}/plotly.min.js"


def guard(js: str) -> str:
    """A literal </script> inside an inlined script would close the tag early."""
    return js.replace("</script>", "<\\/script>")


def find_plotly(explicit):
    if explicit:
        p = pathlib.Path(explicit).expanduser()
        if not p.is_file():
            sys.exit(f"--plotly: no such file {p}")
        return p
    spec = importlib.util.find_spec("plotly")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("plotly.min.js not found: `pip install plotly`, pass --plotly PATH, "
                 "or build with --cdn")
    p = pathlib.Path(list(spec.submodule_search_locations)[0]) / "package_data" / "plotly.min.js"
    if not p.is_file():
        sys.exit(f"the installed plotly package has no {p.name}; pass --plotly PATH or use --cdn")
    return p


def plotly_version(js: str) -> str:
    m = re.search(r"plotly\.js v(\d+\.\d+\.\d+)", js[:400])
    return m.group(1) if m else "unknown version"


def validate(script: str) -> int:
    """Gate the build on a reference implementation passing. Returns the check count."""
    r = subprocess.run([sys.executable, script], cwd=HERE, capture_output=True, text=True)
    m = re.search(r"(\d+)/(\d+) checks passed", r.stdout)
    if r.returncode != 0 or not m or m.group(1) != m.group(2):
        sys.exit(f"{script} failed, refusing to build:\n" + r.stdout[-2000:] + r.stderr[-2000:])
    print(f"{script}: {m.group(0)}")
    return int(m.group(2))


def main():
    ap = argparse.ArgumentParser(description="Validate, then build the scattering explorer page.")
    ap.add_argument("--plotly", metavar="PATH", help="path to plotly.min.js to inline")
    ap.add_argument("--cdn", action="store_true",
                    help=f"reference plotly.js {PLOTLY_CDN_VERSION} from jsDelivr instead of inlining it")
    ap.add_argument("--docs", action="store_true", help="also write docs/index.html for GitHub Pages")
    args = ap.parse_args()

    if args.cdn:
        plotly_tag = f'<script src="{PLOTLY_CDN}" charset="utf-8"></script>'
        plotly_desc = f"plotly.js {PLOTLY_CDN_VERSION} from CDN"
    else:
        js = find_plotly(args.plotly).read_text(encoding="utf-8")
        plotly_tag = "<script>" + guard(js) + "</script>"
        plotly_desc = f"plotly.js {plotly_version(js)} inlined"

    nchecks = validate("validate.py")
    nchecks_tm = validate("validate_tmatrix.py")

    # regenerate the reference the browser self-test checks against
    subprocess.run([sys.executable, "crosscheck.py"], cwd=HERE, check=True)

    template = (HERE / "explorer_template.html").read_text(encoding="utf-8")
    assert "<script>__PLOTLY__</script>" in template
    html = (template
            .replace("<script>__PLOTLY__</script>", plotly_tag)
            .replace("__MIE_JS__", guard((HERE / "mie.js").read_text(encoding="utf-8")))
            .replace("__TMATRIX_JS__", guard((HERE / "tmatrix.js").read_text(encoding="utf-8")))
            .replace("__REFERENCE__", (HERE / "reference.json").read_text(encoding="utf-8"))
            .replace("__NCHECKS_TM__", str(nchecks_tm))
            .replace("__NCHECKS__", str(nchecks))
            .replace("__BUILT__", datetime.date.today().isoformat()))

    for tok in ("__PLOTLY__", "__MIE_JS__", "__TMATRIX_JS__", "__REFERENCE__",
                "__NCHECKS__", "__NCHECKS_TM__", "__BUILT__"):
        assert tok not in html, f"placeholder {tok} survived substitution"

    outs = [DIST] + ([DOCS] if args.docs else [])
    for out in outs:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        print(f"wrote {out.relative_to(HERE)}  ({out.stat().st_size / 1e6:.2f} MB, {plotly_desc})")
    if args.docs:
        (DOCS.parent / ".nojekyll").touch()


if __name__ == "__main__":
    main()
