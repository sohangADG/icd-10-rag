# Security

| Area | Control |
|---|---|
| Secrets | Read only from the environment or `.env` (git-ignored) as `SecretStr`. They never appear in `repr`, logs or error messages: DB URLs are never logged, and the embedding API key is never included in exceptions (tested). |
| Source data | `data/sources/*` and `data/synthetic/` are git-ignored and docker-ignored. Restricted PDFs are refused (`docs/licensing.md`). |
| Database | Docker exposes PostgreSQL on `127.0.0.1` only. Credentials come from env. Driver error text is never returned to clients. |
| Clinical notes | Processed in memory only: not persisted, not cached, not logged. Logs record the note length and counts only. Validation errors never echo submitted values. `LOG_CLINICAL_TEXT` (debug of extracted concepts) is off by default, and startup fails if it is enabled with `APP_ENV=production`. |
| Access logs | Uvicorn access logs are disabled (URLs can contain search text). The app logs method, route template, status and latency instead. |
| Correlation | `X-Request-ID`: client values are accepted only if they match `[A-Za-z0-9._-]{1,64}`; otherwise a value is generated. It is attached to every log record. |
| Request validation | Pydantic schemas with `extra="forbid"` on `/suggest`. Bounded `top_k`, `limit`, query and note lengths. |
| Admin operations | Disabled unless `ADMIN_API_TOKEN` is set; constant-time token comparison. There is no upload/import endpoint, and ingestion is CLI-only with a recorded licence basis. |
| XML | Parsed with defusedxml (no XXE or entity expansion). Tested with an external-entity payload. |
| Third parties | Remote embedding providers are refused for datasets that do not allow remote processing |
| Rate limiting | Every public ICD route depends on `rate_limit_hook` (`app/api/deps.py`), a no-op extension point for a gateway- or Redis-backed limiter. |
| Errors | Unhandled exceptions return a generic 500 and are logged server-side with the class name |

## Before committing

```bash
git status
git check-ignore -v data/sources/*.pdf .env
git diff --check
```
Then scan the diff for secrets (`docs/runtime-verification.md` §6 records the scan used).
