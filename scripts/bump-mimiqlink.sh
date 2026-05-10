#!/usr/bin/env bash
# Bump the pinned `MIMIQLINK_REF` in .gitlab-ci.yml to a fresh commit
# of the sibling mimiqlink-python checkout.
#
# Usage:
#   bump-mimiqlink [options] [ref]
#
# Positional:
#   ref               Branch, tag, or 40-hex SHA to pin. Default: devel.
#                     A 40-hex SHA is used verbatim (no network lookup);
#                     anything else is resolved against the remote.
#
# Options:
#   -r, --remote NAME     Remote to fetch from (default: origin)
#   -d, --link-dir PATH   Path to sibling mimiqlink-python checkout
#                         (default: ../mimiqlink-python)
#   -h, --help            Show this help

set -euo pipefail

remote="origin"
link_dir=""
ref=""

usage() {
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
  case "$1" in
    -r|--remote)    remote="$2"; shift 2 ;;
    --remote=*)     remote="${1#*=}"; shift ;;
    -d|--link-dir)  link_dir="$2"; shift 2 ;;
    --link-dir=*)   link_dir="${1#*=}"; shift ;;
    -h|--help)      usage; exit 0 ;;
    --)             shift; break ;;
    -*)             echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    *)
      if [ -n "$ref" ]; then
        echo "too many positional arguments: $1" >&2
        exit 2
      fi
      ref="$1"; shift
      ;;
  esac
done

[ -n "$ref" ] || ref="devel"

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"
[ -n "$link_dir" ] || link_dir="$repo_root/../mimiqlink-python"

if [[ "$ref" =~ ^[0-9a-f]{40}$ ]]; then
  sha="$ref"
  echo "bumping to explicit SHA $sha"
else
  if [ ! -d "$link_dir/.git" ]; then
    echo "error: no mimiqlink-python checkout at $link_dir" >&2
    echo "       clone it next to this repo or pass --link-dir" >&2
    exit 1
  fi
  echo "fetching $remote in $link_dir …"
  git -C "$link_dir" fetch --quiet "$remote"
  sha="$(git -C "$link_dir" rev-parse "$remote/$ref")"
  echo "bumping mimiqlink-python -> $sha ($remote/$ref)"
fi

ci_file=".gitlab-ci.yml"
if ! grep -qE 'MIMIQLINK_REF: "[0-9a-f]{40}"' "$ci_file"; then
  echo "error: no SHA-pinned \`MIMIQLINK_REF: \"…\"\` entry found in $ci_file" >&2
  exit 1
fi

sed -i.bak -E "s/MIMIQLINK_REF: \"[0-9a-f]{40}\"/MIMIQLINK_REF: \"$sha\"/" "$ci_file"
rm -f "${ci_file}.bak"

git diff --stat "$ci_file"
echo
echo "done. Review the diff and commit with something like:"
echo
echo "    git add $ci_file"
echo "    git commit -m \"ci: bump mimiqlink-python to $(echo "$sha" | cut -c1-12)\""
