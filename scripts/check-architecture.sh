#!/usr/bin/env bash
# Enforces the crate dependency rules of PLAN.md §2.7 (CLAUDE.md invariant 1):
# the integrator crate and the provider contract never depend on concrete force
# providers; only the composition root (ucmd-cli) knows them.
#
# Every workspace crate except ucmd-cli needs an entry in ALLOWED. A crate added
# without a rule fails the check, so new crates get a deliberate decision.
set -euo pipefail
cd "$(dirname "$0")/.."

# crate -> workspace crates it may depend on (directly or transitively)
declare -A ALLOWED=(
  [ucmd-core]=""
  [ucmd-force]="ucmd-core"
  [ucmd-md]="ucmd-core ucmd-force"
  [ucmd-io]="ucmd-core ucmd-force"
  [ucmd-providers-classical]="ucmd-core ucmd-force"
  [ucmd-providers-ipi]="ucmd-core ucmd-force"
)
UNRESTRICTED="ucmd-cli"

status=0
members=$(cargo tree --quiet --workspace --depth 0 --prefix none --format '{p}' | awk 'NF {print $1}' | sort -u)

for crate in $members; do
  if [[ " $UNRESTRICTED " == *" $crate "* ]]; then
    continue
  fi
  if [[ -z "${ALLOWED[$crate]+set}" ]]; then
    echo "architecture: no dependency rule for crate '$crate'; add one to scripts/check-architecture.sh" >&2
    status=1
    continue
  fi
  deps=$(cargo tree --quiet -p "$crate" -e normal,build --prefix none --format '{p}' \
    | awk '{print $1}' | grep '^ucmd-' | grep -vx "$crate" | sort -u || true)
  for dep in $deps; do
    if [[ " ${ALLOWED[$crate]} " != *" $dep "* ]]; then
      echo "architecture violation: $crate depends on $dep (allowed: ${ALLOWED[$crate]:-none})" >&2
      status=1
    fi
  done
done

if [[ $status -eq 0 ]]; then
  echo "architecture check passed: $(echo "$members" | wc -w) crates"
fi
exit $status
