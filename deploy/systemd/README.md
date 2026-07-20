# systemd

This directory holds systemd user-service deploy files for `translation-services`.

Current scope:
- a user unit
- a start script that supports `DEFAULT_PORT` with optional `service.port` override from `config/settings.json`
- a repo-local `.venv` at `~/projects/translation-services/.venv`
- optional `TRANSLATION_SERVICES_VENV_DIR` override when a host needs a separate runtime venv

Expected layout on the target host:

```bash
~/projects/translation-services
```

Install or refresh the user service:

```bash
mkdir -p ~/.config/systemd/user
ln -sf ~/projects/translation-services/deploy/systemd/translation-services.service ~/.config/systemd/user/translation-services.service
systemctl --user daemon-reload
systemctl --user enable --now translation-services.service
```

Useful commands:

```bash
systemctl --user status translation-services.service
journalctl --user -u translation-services.service -f
systemctl --user restart translation-services.service
```

For deployments that use a separate venv:

```bash
systemctl --user edit translation-services.service
```

Add:

```ini
[Service]
Environment=TRANSLATION_SERVICES_VENV_DIR=/home/gunnar/projects/translation-services/.venv-image
```

## Fonts (render prerequisite)

The re-placement renderer loads fonts from `~/.local/share/fonts/gf/` (see
`app/replacement/fit.py`). These are **not** vendored in the repo; a missing file
degrades gracefully (Latin falls back to DejaVu, CJK/Korean to tofu), so they must be
provisioned per host for correct output:

- **Latin** — the Google Fonts metric-compatible faces `Arimo[wght].ttf`,
  `Tinos-Regular/Bold.ttf`, `Cousine-Regular/Bold.ttf` (Arial/Times/Courier metrics),
  plus their italic cuts `Arimo-Italic[wght].ttf`, `Tinos-Italic/BoldItalic.ttf`,
  `Cousine-Italic/BoldItalic.ttf` — a document whose text layer flags a line italic
  renders in them (a missing cut degrades to the roman face).
- **Han/Kana** — PingFang, fetched lazily by PaddleX to `~/.paddlex/fonts/` on the
  first CJK render (no action needed).
- **Korean (Hangul)** — Noto Sans KR; PingFang has no Hangul glyphs. Install with:

  ```bash
  mkdir -p ~/.local/share/fonts/gf
  curl -L -o "$HOME/.local/share/fonts/gf/NotoSansKR[wght].ttf" \
    "https://raw.githubusercontent.com/google/fonts/main/ofl/notosanskr/NotoSansKR%5Bwght%5D.ttf"
  ```

  The filename must stay `NotoSansKR[wght].ttf` (the path `fit.py` looks up). It is a
  variable font; the renderer pins a regular weight at load time.

