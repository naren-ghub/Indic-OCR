# 📊 ByT5 Training Data Generation Report

## 🎯 Summary
- **Total Pairs Generated:** 75,000
- **Data Fits within 1024 bytes (ByT5 max):** 75,000 (100.0%)

---

## 📚 Source Distribution
The training data is carefully balanced across three primary registers:

| Source | Description | Pairs | Percentage |
|---|---|---|---|
| **IndicCorp** | Modern Tamil prose | 30,000 | 40.0% |
| **Project Madurai** | Classical Tamil | 24,750 | 33.0% |
| **Pre-Modern** | Late 19th/Early 20th century | 20,250 | 27.0% |

---

## 📏 Length Constraints

### Character Length
- **Minimum:** 20
- **Maximum:** 340 (Optimal target: ≤ 340)
- **Mean:** 114.1
- **Median:** 102.0

### UTF-8 Byte Length
- **Fits in 512 bytes:** 62,504 (83.3%)
- **Fits in 1024 bytes:** 75,000 (100.0%)

---

## 🔧 OCR Noise Model
Synthetic errors have been introduced dynamically based on register complexity:

- **Average Character Edits per Pair:** 3.5
- **HTML Leak Artifacts Injected:** 8,142 pairs (10.9%)
  *(Simulates Surya OCR injecting `<mark>`, `<sub>`, and `<sup>`)*

---
✅ *Report generated automatically from training data stats.*
