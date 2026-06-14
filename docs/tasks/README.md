# Tasks — 每輪任務規格

這個資料夾是所有 task 內容的單一家：

- [`BACKLOG.md`](./BACKLOG.md) — backlog / roadmap（接下來要做什麼，含優先序與最新發現）。
- `NNNN-<slug>.md` — [Claude × Codex 協作流程](../ai/AI_WORKFLOW.md) 中**每一輪要修改的任務**（一輪一檔，Claude 規劃、Codex 實作、Claude 審查）。
- 範本 [`_TEMPLATE.md`](./_TEMPLATE.md)；完成後歸檔到 [`../history/`](../history/)。

## 命名

```
docs/tasks/NNNN-<kebab-slug>.md      例如 0001-speaker-quality-gate.md
```

- `NNNN`：四位流水號，遞增。
- 範本：[`_TEMPLATE.md`](./_TEMPLATE.md)（底線開頭，不是任務）。

## 生命週期

1. **Claude** 複製 `_TEMPLATE.md` → 填寫目標 / 範圍 / 驗收標準 / 測試計畫，設 `Status: ready-for-codex`。
2. **Codex** 實作，回填 *Implementation Log*，設 `Status: ready-for-review`。
3. **Claude** 審查 + 跑測試/跑 app + 小修正，填 *Review*，設 `Status: done` 或 `changes-requested`。
4. 完成後把檔案移到 [`../history/tasks/`](../history/)，並更新根目錄 `task.md`。

## 規則

- **同一時間只有一個進行中的任務檔。**
- 該輪的所有決策寫在任務檔裡，不要散落在對話中。
- `Status:` 是交接的接力棒；狀態值定義見 [AI_WORKFLOW.md](../ai/AI_WORKFLOW.md#handoff-protocol-english)。
- 任務檔本身是 `.md`，目前被 `.gitignore` 忽略（本機工作檔）；若要納入版控需另加 `.gitignore` 例外。
