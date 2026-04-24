# Project Goal

- Build a 2D web RPG.

# Guidance By Topic

- Read only the `docs/` files that match the current task.
- `docs/ai-setup.md`: Read this when you set up local AI tooling, Git hooks, or optional Serena usage.
- `docs/architecture.md`: Read this when you decide module ownership, dependency direction, or where new code should live.
- `docs/tech-stack.md`: Read this when you need to check the tech stack, supported platforms, build flow, or tool choice. Check this first for new libraries, build pipeline changes, or client/server boundary decisions.
- `docs/coding-standards.md`: Read this before writing or changing code. Check this first when you add a new source file or need to make style decisions.
- `docs/testing-strategy.md`: Read this when you plan validation work or add or change tests.
- `docs/git-rules.md`: Read this for branch, commit, merge, and release work.
- `docs/lua-controller-api.md`: Read this when you write or change Lua character controllers or the Lua controller bridge. Treat it and `src/game/lua/luaControllerApi.ts` as the Lua-facing contract.

# AI Agent Rules

- If the requester does not set a different language, reply in the user's language.
- For low-risk ambiguity, state your assumption and continue. Ask the requester only when the result could change a lot or would be hard to undo. If the requester clearly gives autonomy, you may act on your own.
- Write only the amount of code needed to satisfy the request. Do not add unnecessary abstraction or unnecessary error handling.
  - If you see a future need for abstraction or extra error cases, report it to the requester.
  - The requester may still ask for higher-level work, such as abstraction for future system growth.
- Do not change nearby code, comments, or formatting unless the change is needed for the requested work.
  - Do not do style-only refactors that are unrelated to behavior.
  - If you notice a useful improvement while working, do not change it on your own. Report it to the requester.
  - If your change makes a function or file truly unused, you may delete it.
- After writing or changing code, do validation that matches the size and risk of the change.
  - Prefer related tests, compile checks, and log checks when possible.
  - Run the full test suite only when the change has wide impact or the requester clearly asks for it.
  - If you could not validate, or could validate only part of the change, tell the requester what was not checked and why.
- Do not create Git commits unless the user clearly asks for them.
- Clear instructions from the requester override `AGENTS.md` and files under `docs/`. If something should stay as a lasting rule, report it to the requester so it can be written down.
- Update files under `docs/` when the real behavior contract, development rules, or workflow changes.
  - Keep documents clear and direct. For shared documents, prefer simple English.
- When work has many steps or lasts a long time, write working notes as Markdown files under `notes/` and use them as memory.
  - Git ignores that path, so you may use it freely.
  - Do not create `notes/` files for short, single-step tasks.

## AI Agent Tooling

- If the task clearly matches a skill under `.agents/skills`, read that skill first.
- The repo-level source of truth for skills is `.agents/skills`.
- If Serena is available, use it first for symbol search, reference tracking, and structural edits.
- If Serena is not available, use `rg` or other available search tools instead.
