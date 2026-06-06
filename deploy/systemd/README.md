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

