# Coding Standards

- Keep all code as short, clear, and easy to read as possible.
- Write comments in simple English when needed.
- Do not over-engineer.

## Design Principles

- 의도가 명확하게 보이도록 `let` 보다 `const` 를 우선 사용한다. 반복문 내부나 재할당이 실제로 필요한 경우에는 `let` 을 사용한다.
- 타입 오류를 피하려고 `| null` 또는 `| undefined` 를 임의로 추가하지 않는다.
  - 논리적으로 `null` 또는 `undefined` 가 될 수 없는 값이라면 타입에서 허용하지 않는다.
  - 실제 런타임 동작상 해당 값이 가능할 때만 nullable 타입으로 만든다. `any` 는 원칙적으로 사용하지 않는다.
  - 따라서 불필요한 `null` 또는 `undefined` 검사는 지양한다.
  - nullable 값을 피하려고 `null` 체크 후 생성하는 패턴을 습관적으로 넣지 않는다. 값이 항상 필요하다면 가능한 한 생성 시점에 초기화한다.
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
