# Tamil OCR Debug Viewer — UI/UX Plan (Implementation Tracker)

Source requirements: `OCR/Doc/ui_updates.txt`  
Last updated: 2026-05-04

This plan converts the rough UI notes into an actionable checklist and defines what “done” means for each item. The intent is to **implement incrementally** and keep the app usable at every step.

---

## Goals (User Experience)

1. Make backend progress **visible in real-time** (streaming status).
2. Show a **post-run report** in the UI (not only in console logs).
3. Report must include **CER/WER** (when a reference is available) and a short explanation of what these metrics mean.
4. Keep the UI **structured, appealing, and user-friendly**.

Non-goals (for now):
- Production multi-user hosting, authentication, quota enforcement
- Full multilingual routing, queueing, LM correction (Phase 2+ items)

---

## UX / UI Structure (Proposed)

Single-page app with a clean split layout:

- **Left panel (Inputs)**
  - Upload document (PDF/Image/DOCX)
  - Optional: provide a **reference text** (paste or upload `.txt`) to compute CER/WER
  - Run button
  - Quick tips + constraints (e.g., PDF max pages)

- **Right panel (Outputs)**
  - Streaming “Processing Log”
  - “Run Report” (Markdown summary)
  - Download OCR output `.txt`

---

## Implementation Checklist (Do One-by-One)

### 1) Real-time streaming log (backend progress visibility)

Definition of done:
- Clicking “Run OCR” updates the log while processing pages (not only at the end).
- Log lines are short and explain what step is happening (load → OCR → layout filter → clean → save).

Implementation notes:
- Convert the Gradio callback to a **generator** and `yield` updated outputs each page.

### 2) Post-run report displayed in the window (CER/WER included)

Definition of done:
- A “Run Report” panel is shown after completion.
- The report contains:
  - Total pages processed
  - Output filename
  - CER and WER if a reference is available
  - A short explanation of CER/WER

Reference text strategy (practical constraint):
- CER/WER require a “reference” string (ground truth).
- The UI will compute CER/WER using the first available source:
  1) If the PDF has extractable embedded text, use that as a reference
  2) Else, if user provides reference text (paste or `.txt`), use it
  3) Else, show CER/WER as “N/A” with a note explaining why

### 3) Remove average confidence from UI output

Definition of done:
- Do not show “avg confidence …%” in the log or report.
- (Optional) internal confidence computations may still exist for future quality gating.

### 4) Make the UI structured and user-friendly

Definition of done:
- Clear headings, tips, and output areas.
- Report is readable and does not overwhelm users with internal debugging detail.

Optional enhancements (later):
- Advanced settings accordion (device `cuda/cpu`, max pages, language set)
- “Reset” button to clear outputs

---

## Test / Acceptance

- Run with a small PDF (≤ 3 pages): log streams page-by-page.
- Run with a digital-text PDF: report includes CER/WER against embedded text.
- Run with an image-only PDF + no reference: report shows CER/WER as N/A with explanation.
- Run with an image-only PDF + provided reference text: report includes CER/WER.

