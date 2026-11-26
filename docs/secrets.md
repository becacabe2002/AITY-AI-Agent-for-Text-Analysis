# Secrets: master key and JWT secret

The web service needs two secrets at startup. Both are read from files, never from environment variables, so they stay out of `docker inspect` and the process environment.

| Secret | Default file | Override | Rule |
|---|---|---|---|
| Master key (encrypts stored API keys) | `/run/secrets/master.key` | `AITY_MASTER_KEY_FILE` | a valid Fernet key: 32 bytes, url-safe base64 |
| JWT secret (signs login tokens) | `/run/secrets/jwt_secret` | `AITY_JWT_SECRET_FILE` | at least 32 bytes, different from the master key |

## Fail fast

- The service refuses to start if either file is missing, unreadable, empty or malformed. There is no fallback to a test cipher or a generated key.
- Surrounding whitespace (such as the trailing newline from `echo`) is ignored.
- Error messages name the file path and the reason, never the file's content.

## Production

The deploy job writes both files from the GitHub Actions `production` Environment to `/opt/aity/secrets/` (mode 0600) on Pouta, and docker compose mounts them read-only under `/run/secrets/`. See ADR-005.

## Development and tests

Generate a fresh pair (written to the gitignored `var/secrets/`, owner-only):

```
python scripts/generate_dev_secrets.py
```

Then point the app at them, for example in `.env`:

```
AITY_MASTER_KEY_FILE=./var/secrets/master.key
AITY_JWT_SECRET_FILE=./var/secrets/jwt_secret
```

The script refuses to replace existing files unless `--force` is given. Replacing the master key makes every stored API key unreadable (users enter them again); replacing the JWT secret logs everyone out.

Tests generate a new master key for each run and never use these files.

## Consequences of a change

- New master key: stored keys cannot be decrypted; each user sees "enter the key again". Rotation without this loss is future work (MultiFernet).
- New JWT secret: all login cookies become invalid. This is also the emergency switch for a leaked secret.
