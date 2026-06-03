# Git Workflow for Agents

Use this checklist whenever the user asks for `git commit`, `git push`, release handoff, or branch cleanup.

## Standard Flow

1. Inspect repository state.
   - Run `git status --short`.
   - Run `git branch --show-current`.
   - Run `git log --oneline -8` to match local commit-message style.
   - Run `git remote -v` before push.

2. Review scope before staging.
   - Use `git diff --stat`.
   - Use focused `git diff -- <path>` for surprising files.
   - Do not revert unrelated user changes.
   - If unexpected changes appear that were not part of the current work, stop and ask before proceeding.

3. Verify before commit.
   - Run the relevant focused tests first.
   - Run the full available test suite when practical.
   - Run UTF-8/no-BOM checks for changed Markdown/Python files when documentation or source files changed.
   - Check staged diff for obvious secrets: `token`, `password`, `secret`, `api_key`.

4. Stage intentionally.
   - Prefer staging all files only after confirming the whole dirty tree belongs to the requested work.
   - Otherwise stage by explicit path.
   - Check `git diff --cached --stat` and `git diff --cached --name-status`.

5. Commit with a descriptive message.
   - Follow existing convention: `<type>: <summary>`.
   - Include a body when the change spans multiple concerns.
   - Mention key behavior changes and verification.

6. Push.
   - Push current branch to `origin`.
   - If sandbox/network blocks push, request escalation for `git push` with a narrow prefix rule.
   - Report the pushed branch and commit hash.

## Commit Message Template

```text
<type>: <short summary>

- Key change 1
- Key change 2
- Key change 3

Verification:
- <test command/result>
```

## Safety Rules

- Never use `git reset --hard`, `git checkout --`, or destructive cleanup unless the user explicitly asks and approves.
- Never amend unless the user explicitly asks.
- Never force-push unless the user explicitly asks and the target branch is confirmed.
- Do not commit generated output, local logs, model files, or credentials unless the repo explicitly tracks them.
