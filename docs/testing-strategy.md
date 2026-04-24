# Testing Strategy

- The default test strategy for this project is risk-based, regression-test-first, and Vitest-first.
- Do not add tests mechanically for every new feature.
- Do not make large structural changes just to add tests.
- If automated tests are not a good fit, leave manual validation steps instead. Do not force brittle tests into the project.

## When To Add Tests

- Decide whether to add a test based on risk, separability, and stability.
- Add a test when the change is costly to break, can be isolated in a small scope, and can run in a stable way.
- For bug fixes, add a regression test first when the bug can be reproduced in a small case.
- Prioritize tests for important rules, edge cases, failure cases, and changes with regression risk.

## Test Scope

- Prefer small Vitest tests for pure logic, state transitions, calculations, parsing, serialization, and stable contracts.
- Add integration-style Vitest tests when multiple modules must work together and the flow can still run reliably in a test environment.
- Use manual validation for rendering, asset loading, input handling, timing-sensitive loops, and other browser or game runtime behavior that would be fragile to automate right now.

## Test Writing Notes

- You may use AI to write test code.
- A human should define the test inputs, expected results, and validation intent.
- If a test would effectively decide product policy, confirm the intent with a human first.
- This project is still being shaped, so prefer focused coverage over broad mechanical coverage.

## Test Placement

- Place tests next to the target module.
- Use `*.test.ts` for Vitest test files by default.
- If a shared test directory becomes clearly easier to maintain, document that structure before expanding it.
- Do not add tests under third-party code by default. Test the project module that depends on that code.

## Running Tests

- Run related Vitest tests by default instead of the full suite.
- Run the full suite only when the change has wide impact or the requester clearly asks for it.
- Lua bridge integration checks are also opt-in.
  Run `npm run lua:bridge:test` when you change `src/game/lua/createLuaCharacterControllerRuntime.ts`, the public `engine.*` Lua API, or the Lua return-value contract.
  Do not add `lua:bridge:test` to the normal `check` flow.
- Lua WebAssembly compatibility checks are opt-in.
  Run `npm run lua:test` only when you explicitly want to validate the Lua runtime against the official Lua 5.3 basic tests.
  Do not add `lua:test` to the normal `check` flow.
- For docs-only changes, a manual review is enough unless the document changes an executable workflow.
