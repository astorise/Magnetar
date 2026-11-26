#!/usr/bin/env bash
# Tests of tools/repositories.sh against throwaway local repositories.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
export GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@example.com GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@example.com
failures=0
expect() { # <description> <condition-as-exit-status...>
  local what="$1"; shift
  if "$@"; then echo "pass: $what"; else echo "FAIL: $what"; failures=$((failures + 1)); fi
}

# Two origins; "alpha" has two commits.
mkorigin() { # <name> <commits...>
  local dir="$work/origins/$1"; shift
  mkdir -p "$dir"; git -C "$dir" init -q -b main
  git -C "$dir" config uploadpack.allowAnySHA1InWant true
  local c
  for c in "$@"; do echo "$c" > "$dir/file"; git -C "$dir" add file; git -C "$dir" commit -q -m "$c"; done
}
mkorigin alpha one two
mkorigin beta only
first="$(git -C "$work/origins/alpha" rev-parse HEAD~1)"
second="$(git -C "$work/origins/alpha" rev-parse HEAD)"
beta="$(git -C "$work/origins/beta" rev-parse HEAD)"

# A Magnetar-shaped tree: the tool, and a lock file.
mkdir -p "$work/magnetar/tools"
cp "$here/repositories.sh" "$work/magnetar/tools/"
tool="$work/magnetar/tools/repositories.sh"
export MAGNETAR_REPOSITORIES_LOCK="$work/magnetar/repositories.lock"
cat > "$MAGNETAR_REPOSITORIES_LOCK" <<LOCK
# a comment
components/alpha file://$work/origins/alpha $first
providers/beta file://$work/origins/beta $beta
LOCK

at() { [ "$(git -C "$work/magnetar/$1" rev-parse HEAD)" = "$2" ]; }

expect "the lock file is well-formed" "$tool" check
"$tool" fetch > /dev/null
expect "fetch puts alpha at its pin, not at the tip" at components/alpha "$first"
expect "fetch puts beta at its pin" at providers/beta "$beta"
expect "fetch is idempotent" "$tool" fetch
expect "status is clean after a fetch" "$tool" status

"$tool" bump components/alpha "$second" > /dev/null
expect "bump rewrote only alpha's line" grep -q "components/alpha file://$work/origins/alpha $second" "$MAGNETAR_REPOSITORIES_LOCK"
expect "bump kept beta's line" grep -q "providers/beta file://$work/origins/beta $beta" "$MAGNETAR_REPOSITORIES_LOCK"
expect "bump kept the comment" grep -q '^# a comment' "$MAGNETAR_REPOSITORIES_LOCK"
expect "status says alpha moved" bash -c "! '$tool' status components/alpha > /dev/null"
"$tool" fetch components/alpha > /dev/null
expect "fetch moves a clean checkout to the new pin" at components/alpha "$second"

# A checkout off its pin with local changes is left alone.
"$tool" bump components/alpha "$first" > /dev/null
echo "my work" > "$work/magnetar/components/alpha/file"
expect "fetch refuses to overwrite local changes" bash -c "! '$tool' fetch components/alpha 2> /dev/null"
expect "the local change survived" grep -q "my work" "$work/magnetar/components/alpha/file"
git -C "$work/magnetar/components/alpha" checkout -q -- file
"$tool" fetch components/alpha > /dev/null
expect "once clean, fetch moves it" at components/alpha "$first"

# A directory that is not a checkout is not clobbered.
mkdir -p "$work/magnetar/components/gamma" && echo x > "$work/magnetar/components/gamma/x"
echo "components/gamma file://$work/origins/beta $beta" >> "$MAGNETAR_REPOSITORIES_LOCK"
expect "fetch refuses a non-empty directory that is not a checkout" bash -c "! '$tool' fetch components/gamma 2> /dev/null"
expect "that directory was not touched" test -f "$work/magnetar/components/gamma/x"

# Errors.
expect "an unpinned path is an error" bash -c "! '$tool' fetch components/nope 2> /dev/null"
git -C "$work/magnetar/providers/beta" checkout -q --detach HEAD  # keep it at its pin
"$tool" bump providers/beta 0000000000000000000000000000000000000000 > /dev/null
rm -rf "$work/magnetar/providers/beta"
expect "a commit the repository does not have fails" bash -c "! '$tool' fetch providers/beta 2> /dev/null"
"$tool" bump providers/beta "$beta" > /dev/null
cp "$MAGNETAR_REPOSITORIES_LOCK" "$work/good.lock"
echo "providers/delta file://$work/origins/beta main" >> "$MAGNETAR_REPOSITORIES_LOCK"
expect "a branch name is not a pin" bash -c "! '$tool' check 2> /dev/null"
echo "providers/beta file://$work/origins/beta $beta" >> "$work/good.lock"
MAGNETAR_REPOSITORIES_LOCK="$work/good.lock" "$tool" check > /dev/null 2>&1 && dup=ok || dup=refused
expect "a path pinned twice is refused" test "$dup" = refused

if [ "$failures" -ne 0 ]; then echo "$failures check(s) failed" >&2; exit 1; fi
echo "all repositories tool checks passed"
