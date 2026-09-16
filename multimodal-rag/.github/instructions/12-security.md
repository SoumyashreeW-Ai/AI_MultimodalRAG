# Security Instructions

## Secrets

Never commit:

- API keys
- tokens
- passwords
- credentials

Use environment variables.

---

# File Uploads

Validate:

- extension
- MIME type
- file size
- filename

Never directly trust user-provided paths.

Prevent path traversal.

---

# File Storage

Generate safe internal filenames.

Do not use:

../../../etc/passwd

style paths.

---

# API

Do not expose internal stack traces.

Do not return secret configuration.

---

# Images

Do not allow arbitrary filesystem access through image URLs.

Images should be served through a controlled mechanism.

---

# Dependencies

Avoid unnecessary dependencies.

Keep dependencies pinned or constrained appropriately.

Regularly review dependencies for known vulnerabilities.