# MujocoReplay: project rules

## Read first

- Before changing anything, read `README.md`, `plan.md`,
  `docs/recording-format.md` and `docs/design.md`.
- The sibling `../Centipede` produces the recordings this tool replays, and
  depends on this package to write them. It is reference only (the user's
  decision, 2026-10-07). Keep handoff files for it outside this repository.
  Ignore `../backup` entirely.

## Scope

- This is a general tool: it must work for any MuJoCo model and any recording
  that follows `docs/recording-format.md`. Never import from Centipede or build
  in Centipede-specific behaviour; everything about a run arrives inside the
  recording file.
- Keep the structure flat: one package, a handful of modules, one test file
  per module. The viewer must run with MuJoCo and NumPy alone.
- Importing `mujoco_replay` must never import GLFW or OpenGL. The window and
  video modules import them only when they run, so that a training machine
  without a display can still write recordings.
- Validate only what comes from outside: the recording file when it is read,
  and what MuJoCo returns when compiling the model. Trust the tool's own
  modules.
- Scene and rendering tests run offscreen, with a small recording built in the
  test. Skip them with a clear reason when no OpenGL context can be created.
- The assistant can't see the user's screen. Verify drawing by rendering
  frames offscreen to PNG files in the scratchpad and inspecting them, and
  verify the window by driving it under a virtual display (Xvfb with synthetic
  input). The user verifies the window on a real display and graphics card.
- Keep `README.md`'s status and the documents current. Design decisions go in
  `docs/design.md`, and the file contract in `docs/recording-format.md`. Keep
  one detailed source per topic and link to it.
- Move superseded material to the local `archive/` folder, and never import
  from it.
- Explain MuJoCo concepts (MjSpec, scenes, the render context) plainly when
  they come up. The user knows basic RL but has little MuJoCo experience.

## Git

- When a piece of work ends, leave the working folder clean: nothing modified
  or untracked, and no generated or stray files tracked (`.gitignore` covers
  them).
- Small, safe changes go straight to `main`. Longer work goes on its own
  branch, and `main` is fast-forwarded to it after each finished, tested
  piece, so that no branch drifts far from `main`. Push both.
- Before each commit, run the tests with and without a display, and `ruff`.
- Overrides the global credit rule: no AI co-author trailers or mentions
  anywhere in the repository, not even in cloud sessions.
