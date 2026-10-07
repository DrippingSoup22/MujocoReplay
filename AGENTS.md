# Working rules for coding assistants

## Read first

- Read this file, then `README.md`, `plan.md`, `docs/recording-format.md`, and
  `docs/design.md` before changing anything.
- The sibling `../Centipede` project produces the recordings this tool replays
  and depends on this package for writing them. It is reference only: the
  user works on it in sessions of its own, and nothing in it is changed from
  here (the user decided this on 2026-10-07). When something there needs
  changing, write a Markdown handoff file for the user to pass on to those
  sessions, outside this repository, and send it to the user. Ignore the
  sibling `../backup` folder entirely.

## Scope and boundaries

- This is a general tool: it must work for any MuJoCo model and any recording
  that follows `docs/recording-format.md`. Never import from Centipede or build
  centipede-specific behaviour in; everything about a run arrives inside the
  recording file.
- The assistant writes all of this project's code, tests, and documents. The
  user decided this on 2026-10-07; the authorship rules of Centipede do not
  apply here.
- Keep the structure flat and simple: one package, a handful of modules, one
  test file per module. Add a dependency only when the current stage needs it,
  and keep the viewer runnable with MuJoCo and NumPy alone. Importing
  `mujoco_replay` must never import GLFW or OpenGL; the window and the video
  modules import them when they run, so that a training machine without a
  display can write recordings.
- Check only what comes from outside: the recording file, when it is read, and
  what MuJoCo returns when compiling the model. Trust the tool's own modules.
- Tests are few and precise, one behaviour each. Scene and rendering tests work
  offscreen with a small recording built in the test; skip rendering tests with
  a clear reason when no OpenGL context can be created.
- An assistant cannot see the user's screen. Verify drawing by rendering
  frames offscreen to PNG files in the scratchpad and inspecting them, and the
  window by driving it under a virtual display (Xvfb with synthetic input);
  the user verifies the window on a real display and graphics card.
- Keep `README.md`'s status and the documents current. Design decisions belong
  in `docs/design.md`, the file contract in `docs/recording-format.md`; keep
  one detailed source per topic and link to it.
- Move superseded material to the local `archive/` folder; never import from
  it.
- Explain MuJoCo concepts (MjSpec, scenes, the render context) plainly when
  they come up; the user has basic RL knowledge and little MuJoCo experience.

## Git

- As the user set on 2026-10-07, the assistant commits and pushes on its own,
  and keeps the working folder and the repository clean: nothing left
  modified or untracked when a piece of work ends, and no generated or stray
  files tracked (`.gitignore` covers them). Small, safe changes may go
  straight to `main`; longer work goes on a branch of its own, and `main` is
  fast-forwarded to it after each finished and tested piece, so that no branch
  grows far from `main`. Push both.
- Run the tests, with and without a display, and `ruff` before each commit.
- Commit with the user's identity only: no AI co-author trailers or mentions
  anywhere in the repository.
