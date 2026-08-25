# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.2.x   | yes       |
| 0.1.x   | no — upgrade with `pip install --upgrade llmcalculator` |

## Reporting a vulnerability

**Do not open a public issue for a security problem.**

Report it privately through GitHub Security Advisories:

**https://github.com/BarakaSoka/llmcalculator/security/advisories/new**

That creates a private thread visible only to the maintainers. If you cannot
use advisories, open a normal issue saying only that you have a security report
and asking for a contact route — no details in the public issue.

### What to include

- What an attacker can do, and what they need in order to do it
- Steps to reproduce, ideally as a minimal command or script
- The version (`llmcalculator --version`) and platform
- Whether you intend to disclose publicly, and when

### What to expect

- Acknowledgement within **7 days**
- An assessment, with a fix or an explanation of why it is not a vulnerability,
  within **30 days**
- Credit in the advisory and the changelog, unless you prefer otherwise

This is a small volunteer-maintained project, so those are honest targets
rather than a contractual guarantee.

## What counts as a vulnerability here

It helps to know what this tool actually does. `llmcalculator` reads your
hardware, does arithmetic, and optionally fetches public `config.json` files
from Hugging Face. It has no runtime credentials, no database, no user
accounts, and does not execute model code.

**In scope:**

- Arbitrary code execution from parsing a malicious `config.json` or a crafted
  Hugging Face API response
- Path traversal or arbitrary file write via the cache (`~/.cache/llmcalculator`)
- The local web app (`llmcalculator app`) being reachable off-host, or serving
  files outside the package
- Command injection through any CLI argument
- Leaking a `HF_TOKEN` into logs, cache files, or network requests to any host
  other than Hugging Face
- Dependency-confusion or supply-chain issues in the published package

**Out of scope:**

- Inaccurate memory or speed estimates. These are bugs — sometimes serious
  ones — but they are not security issues. Use the "Wrong estimate" issue
  template.
- The local web app being reachable by other users of the *same* machine. It
  binds to `127.0.0.1` by default; binding it wider is an explicit choice via
  `--host`.
- Denial of service caused by pointing the tool at a deliberately enormous
  config file.
- Vulnerabilities in `rich` or `textual`. Report those upstream; tell us if a
  version pin is needed here.

## Notes for the security-minded

The core package has **zero required dependencies**, which is deliberate: it
keeps the supply-chain surface to the standard library plus whatever your
Python already ships. `rich` and `textual` are optional extras.

Releases are published from CI using **PyPI Trusted Publishing**, so no
long-lived API token exists in the repository or in GitHub secrets to be stolen.

Network access is limited to `huggingface.co` and only when you run `search`,
`trending`, or `check` against an uncatalogued model id. Everything else works
fully offline.
