# AI Setup

This document is for local AI tooling and validation setup.

## Required Setup

- Run `npm install`.
- Run `git lfs install --local`.
- Start the local dev server with `npm run dev` when you need a browser check.
- Set the repo Git hooks path with `git config core.hooksPath .githooks`.
- Read `AGENTS.md` before agent-driven work.

## Tooling Notes

- If `.agents/skills` exists, use it as the repo-level source of truth for skills.
- Serena is optional. If Serena is not configured, use `rg` and normal editor search tools.

## First Checks

- Run `npm run test:run`.
- Run `npm run build`.
- Run related tests by default instead of the full suite.
