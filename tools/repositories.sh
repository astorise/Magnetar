#!/usr/bin/env bash
# The repositories Magnetar pins: Components, Providers, Loaders and Formats.
#
# They live in repositories of their own and are pinned by commit in
# `repositories.lock` (one `<path> <url> <commit>` per line). This puts each at
# its path, at exactly that commit, so the paths the build, the tests and the
# workflows use are the same ones a checkout with submodules gave. It needs
# only `bash` and `git`, which is all the CI containers have.
#
#   tools/repositories.sh fetch [path...]    put each pinned repository at its commit
#                                            (all of them without a path)
#   tools/repositories.sh status [path...]   say whether each one is at its pin
#   tools/repositories.sh bump <path> <commit>   move a pin (the repository must have the commit)
#   tools/repositories.sh check              check the lock file's own form (offline)
#
# `fetch` never overwrites work: a checkout that is not at its pin and has
# local changes is reported and left alone.
#
# MAGNETAR_REPOSITORIES_LOCK names another lock file (the tests use it).

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
lock="${MAGNETAR_REPOSITORIES_LOCK:-$root/repositories.lock}"
jobs="${MAGNETAR_REPOSITORIES_JOBS:-8}"

die() { echo "repositories: $*" >&2; exit 1; }

# Prints "<path> <url> <commit>" for every pin, or for the paths asked for.
pins() {
  [ -f "$lock" ] || die "no lock file at $lock"
  local wanted=" $* "
  local path url rev found=""
  while read -r path url rev; do
    case "$path" in ''|'#'*) continue ;; esac
    if [ "$#" -eq 0 ] || [[ "$wanted" == *" $path "* ]]; then
      echo "$path $url $rev"; found="$found $path"
    fi
  done < "$lock"
  local p
  for p in "$@"; do
    [[ " $found " == *" $p "* ]] || die "$p is not pinned in $lock"
  done
}

at_pin() { # <path> <commit>
  [ "$(git -C "$root/$1" rev-parse HEAD 2>/dev/null || true)" = "$2" ]
}

fetch_one() { # <path> <url> <commit>
  local path="$1" url="$2" rev="$3" dir="$root/$1"
  if [ -e "$dir/.git" ]; then
    if at_pin "$path" "$rev"; then echo "ok       $path"; return 0; fi
    if [ -n "$(git -C "$dir" status --porcelain 2>/dev/null)" ]; then
      echo "SKIPPED  $path: not at its pin, and it has local changes" >&2; return 1
    fi
  elif [ -d "$dir" ] && [ -n "$(ls -A "$dir")" ]; then
    echo "REFUSED  $path: exists, is not a git checkout and is not empty" >&2; return 1
  else
    mkdir -p "$dir"
    git -C "$dir" init -q
    git -C "$dir" remote add origin "$url"
  fi
  if ! git -C "$dir" fetch -q --depth 1 origin "$rev" 2>/dev/null; then
    # A server that will not serve a bare commit: fall back to the full history.
    git -C "$dir" fetch -q origin 2>/dev/null || { echo "FAILED   $path: cannot fetch $rev from $url" >&2; return 1; }
  fi
  git -C "$dir" checkout -q --detach "$rev" 2>/dev/null \
    || git -C "$dir" checkout -q --detach FETCH_HEAD 2>/dev/null \
    || { echo "FAILED   $path: $rev is not in $url" >&2; return 1; }
  at_pin "$path" "$rev" || { echo "FAILED   $path: checked out the wrong commit" >&2; return 1; }
  echo "fetched  $path @ ${rev:0:9}"
}

cmd_fetch() {
  local failed=0 running=0 status_dir list
  list="$(pins "$@")"
  status_dir="$(mktemp -d)"
  local line n=0
  while read -r line; do
    n=$((n + 1))
    # shellcheck disable=SC2086 # three words, by design
    ( fetch_one $line || touch "$status_dir/$n" ) &
    running=$((running + 1))
    if [ "$running" -ge "$jobs" ]; then wait -n 2>/dev/null || wait; running=$((running - 1)); fi
  done <<< "$list"
  wait
  if [ -n "$(ls -A "$status_dir")" ]; then failed=1; fi
  rm -rf "$status_dir"
  [ "$failed" -eq 0 ] || die "some repositories could not be put at their pin"
}

cmd_status() {
  local path url rev bad=0 list
  list="$(pins "$@")"
  while read -r path url rev; do
    if [ ! -e "$root/$path/.git" ]; then echo "absent   $path"; bad=1
    elif at_pin "$path" "$rev"; then
      if [ -n "$(git -C "$root/$path" status --porcelain 2>/dev/null)" ]; then echo "dirty    $path (at its pin, with local changes)"; else echo "ok       $path @ ${rev:0:9}"; fi
    else echo "moved    $path: at $(git -C "$root/$path" rev-parse --short HEAD), pinned ${rev:0:9}"; bad=1
    fi
  done <<< "$list"
  return "$bad"
}

cmd_bump() {
  [ "$#" -eq 2 ] || die "usage: bump <path> <commit>"
  local path="$1" rev="$2"
  [[ "$rev" =~ ^[0-9a-f]{40}$ ]] || die "a pin is a full 40-character commit, not '$rev'"
  pins "$path" > /dev/null
  local tmp; tmp="$(mktemp)"
  awk -v p="$path" -v r="$rev" '$1 == p { print $1, $2, r; next } { print }' "$lock" > "$tmp"
  cat "$tmp" > "$lock"; rm -f "$tmp"
  echo "$path pinned at $rev"
}

cmd_check() {
  local path url rev seen=" " ok=1 n=0
  while read -r path url rev; do
    case "$path" in ''|'#'*) continue ;; esac
    n=$((n + 1))
    [[ "$path" =~ ^[a-z0-9_-]+/[a-z0-9_-]+$ ]] || { echo "bad path: $path" >&2; ok=0; }
    [[ "$url" =~ ^(https://github\.com/|file://)[^[:space:]]+$ ]] || { echo "bad url for $path: $url" >&2; ok=0; }
    [[ "$rev" =~ ^[0-9a-f]{40}$ ]] || { echo "$path: the pin must be a full 40-character commit, not '$rev'" >&2; ok=0; }
    [[ "$seen" != *" $path "* ]] || { echo "$path is pinned twice" >&2; ok=0; }
    seen="$seen$path "
  done < "$lock"
  [ "$n" -gt 0 ] || die "$lock pins nothing"
  [ "$ok" -eq 1 ] || die "$lock is malformed"
  echo "$n repositories pinned, lock file well-formed"
}

case "${1:-}" in
  fetch)  shift; cmd_fetch "$@" ;;
  status) shift; cmd_status "$@" ;;
  bump)   shift; cmd_bump "$@" ;;
  check)  shift; cmd_check ;;
  *) sed -n '2,19p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
