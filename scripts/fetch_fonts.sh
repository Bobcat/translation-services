#!/usr/bin/env bash
# Provision the complex-script fonts the renderer routes to (app/replacement/fit.py).
#
# These Noto Sans faces are NOT vendored in the repo (OFL, a few MB each); they live in the
# per-user fonts dir next to the Google Fonts faces the renderer already uses (Arimo / Tinos /
# Cousine / NotoSans / NotoSansKR — provisioned separately). Run this on a fresh machine so
# Arabic / Devanagari (Hindi) / Bengali / Thai / Hebrew / Tamil render instead of tofu. Idempotent.
set -euo pipefail

DEST="${GF_FONTS_DIR:-$HOME/.local/share/fonts/gf}"
BASE="https://raw.githubusercontent.com/google/fonts/main/ofl"
mkdir -p "$DEST"

# google/fonts ofl directory -> output face name (saved as "<Name>[wght].ttf", the variable font).
FONTS=(
  "notosansarabic:NotoSansArabic"
  "notosansdevanagari:NotoSansDevanagari"
  "notosansbengali:NotoSansBengali"
  "notosansthai:NotoSansThai"
  "notosanshebrew:NotoSansHebrew"
  "notosanstamil:NotoSansTamil"
)

for entry in "${FONTS[@]}"; do
  dir="${entry%%:*}"
  name="${entry##*:}"
  out="$DEST/${name}[wght].ttf"
  if [[ -s "$out" ]]; then
    echo "  ${name}: present"
    continue
  fi
  ok=""
  # The face may ship as a single-axis [wght] or a [wdth,wght] variable font; try both.
  for ax in "%5Bwght%5D" "%5Bwdth%2Cwght%5D"; do
    if curl -fsSL -o "${out}.tmp" "${BASE}/${dir}/${name}${ax}.ttf" \
        && [[ "$(stat -c%s "${out}.tmp" 2>/dev/null || echo 0)" -gt 50000 ]]; then
      mv "${out}.tmp" "$out"
      echo "  ${name}: downloaded"
      ok=1
      break
    fi
    rm -f "${out}.tmp"
  done
  [[ -n "$ok" ]] || echo "  ${name}: FAILED" >&2
done

command -v fc-cache >/dev/null 2>&1 && fc-cache -f "$DEST" >/dev/null 2>&1 || true
echo "fonts in: $DEST"
