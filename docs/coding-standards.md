# Coding Standards

- Keep all code as short, clear, and easy to read as possible.
- Write comments in simple English when needed.
- Do not over-engineer.

## Design Principles

- Prefer `const` over `let` when it makes the intent clearer. Use `let` only when reassignment is actually needed, such as inside a loop or when the value really changes.
- Do not add `| null` or `| undefined` just to avoid type errors.
  - If a value cannot logically be `null` or `undefined`, do not allow it in the type.
  - Use nullable types only when the runtime behavior can really produce that value. Do not use `any` by default.
  - Avoid unnecessary `null` or `undefined` checks.
  - Do not rely on a habit of "check for null and create later" just to avoid nullable values. If a value is always required, initialize it as early as possible.
- Keep variable scope as small as possible.
  - Do not reuse local variables or fields in ways that change their meaning.
  - Split values into new variables when each step has a different meaning, such as an intermediate result, a normalized value, or a final value.
  - Prefer passing a computed value to the next step over building flow through repeated reassignment.
- Do not create mutable state shared across a wide scope.
  - Local state inside a function is allowed, but keep the scope small and the intent clear.
  - In particular, do not write code that lets outside code directly change internal state.
  - For complex state management, consider explicit transition methods, state objects, or state machines.
- Avoid bidirectional coupling, circular dependencies, and structures where outside code pushes internal state backward.
- Do not use the singleton pattern as the default choice. If you need a single instance, define the owner and lifecycle first.
- Do not collect unrelated logic under generic names like Helper, Utils, or Manager. Place behavior close to the type that owns the responsibility.
  - If you split logic into a separate type, its responsibility should be clear from its name and location.
- Do not add trivial comments, but write comments when they help.
  - If the expected input or output is not obvious, first check whether the code itself can be clearer. If explanation is still needed, add a comment.

## General Conventions

- Use UTF-8 without BOM.
- Do not leave trailing whitespace, and end every text file with exactly one newline.
- Use Unix line endings.
- Follow the standard style guide of each language unless this document says otherwise.

## TypeScript / JavaScript Conventions

- Use two spaces for indentation.
- Do not use trailing semicolons.
