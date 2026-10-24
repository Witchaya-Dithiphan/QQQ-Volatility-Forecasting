# แผน implementation: SSL → DAPT → Fine-tune Transformer (NumPy from scratch)

> แทนที่ [แผน GRU SPY→QQQ](2026-10-24-spy-qqq-gru-transfer.md) — GRU กลายเป็น ablation ในแผนนี้
> Spec: [แผนการทดลอง](../../plan/TIME_SERIES_TRANSFER_LEARNING_PLAN.md)

**Goal:** ตอบคำถามอาจารย์ว่าโมเดลลำดับเวลาที่เรียนจากตลาดอื่นก่อน (self-supervised + domain-adaptive pretraining + transfer) ทำนาย `target_high_volatility` ของ QQQ ได้ดีกว่า MLP/persistence เดิมหรือไม่ และส่วนไหนของ pipeline ที่ช่วย

**Architecture:** mini-autograd ด้วย NumPy → Transformer ขนาดเล็ก (pre-LN, 2 ชั้น, d=32, 4 heads, lookback 60 วัน) และ GRU (hidden 32) → (1) masked-span reconstruction บน 10 ETF ก่อน cutoff → (2) masked reconstruction ต่อบน QQQ Train (DAPT) → (3) fine-tune BCE บน label QQQ พร้อม early stop จาก AP ของ 15% ท้าย Train (purge 5 วัน)

**Tech stack:** Python 3.14, NumPy, pandas, pytest — ไม่มี deep-learning framework

## กติกาที่ใช้ทุก task

- ไม่แตะ `data/**`, ไฟล์ data pipeline, `outputs/modeling/**`, `deliverables/slides.pptx`; ข้อมูล ETF เก็บที่ `outputs/experiments/transfer_learning/sources/`
- source ตัดที่ 2023-08-18 (วันสุดท้ายของ QQQ Train) **ก่อน** คำนวณ target ทำให้ไม่มี label ใดของ source ยาวเลยวันนั้น
- config ทั้งหมดล็อกใน `experiment.py:CONFIG` ก่อนรัน Validation ไม่มีการค้น hyperparameter
- RNG ใช้ `np.random.RandomState` เท่านั้น (ห้าม `default_rng`, `permutation(`, `shuffle=True` ตาม audit guard)
- ห้าม commit โดยไม่สรุปให้เจ้าของโปรเจกต์ดูก่อน; ห้ามเปิด Test จนกว่าเจ้าของโปรเจกต์อนุมัติ

## แผนที่ skill ต่อขั้น

| ขั้น | Skill หลัก | ใช้ทำอะไร |
| --- | --- | --- |
| ทุกขั้น | `qqq-model` | กฎ parity/RNG/TDD/DoD ของโปรเจกต์ |
| วางแผน | `superpowers:brainstorming`, `superpowers:writing-plans` | ถามทีละข้อ แล้วเขียนแผนนี้ |
| ข้อมูล | `fullstack-dev-skills:pandas-pro`, `validate-data` | ingest Nasdaq แล้วตรวจ overlap/cutoff ก่อนใช้ |
| autograd/โมเดล | `superpowers:test-driven-development`, `superpowers:systematic-debugging` (หรือ `fullstack-dev-skills:debugging-wizard`) | gradcheck ก่อน implement และ debug เมื่อ gradient ไม่ตรง |
| รันทดลอง | `fullstack-dev-skills:ml-pipeline`, `superpowers:dispatching-parallel-agents` | artifact/manifest และรัน seed ขนาน |
| อ่านผล | `data:statistical-analysis`, `fullstack-dev-skills:the-fool` | ตีความ bootstrap CI และหา leakage/overclaim ก่อนล็อก |
| gate ก่อน Test | `code-review`, `superpowers:verification-before-completion` | review diff และรัน suite ทั้งหมด |
| รายงาน | `dataviz`, `anthropic-skills:pptx` (ถ้าทำสไลด์) | กราฟ PR/AP/loss และสไลด์เสริม |
| ปิดงาน | `superpowers:finishing-a-development-branch` | สรุป diff ขออนุญาต commit/PR |

ถ้า skill ใดโหลดไม่ได้ (เช่น plugin superpowers เสีย) ให้ทำขั้นนั้นด้วยมือตามความหมายเดิม

## File map

- `src/ml/transfer_learning/sources.py` — ดึง Nasdaq, `featurize` (cleaning/features/target ของเดิม), `load_source` (cutoff + hash), `qqq_extended` (snapshot + แถวใหม่สำหรับ holdout)
- `autograd.py` — `Tensor` reverse-mode + ops + fused softmax/layer_norm/BCE/masked-MSE
- `nn.py` — `Model` (Transformer/GRU, recon head, prob head), `AdamW`
- `windows.py` — `timeline`, `make_windows`, `contiguous_windows`
- `train.py` — `pretrain` (span mask 2×5 วัน), `finetune` (tail AP early stop)
- `experiment.py` — CLI `train --backbone --seed`, `summarize` → `validation_summary.json`, `decision.json`
- `test_eval.py` — CLI `--allow-test` ครั้งเดียว → `test/`, `test_receipt.json` (`post_hoc: true`)
- `tests/ml/transfer_learning/test_{autograd,data,test_eval}.py`

## Task 1: ข้อมูล source และ holdout

- [x] ดึง SPY, DIA, IWM, XLK, XLF, XLE, XLV, XLY, EFA, EEM และ QQQ จาก Nasdaq (2016-10-24..2026-10-23) แล้วบันทึก manifest พร้อม SHA-256 และ adjustment policy
- [x] ตัด source ก่อนคำนวณ target แล้วตรวจว่า `label_end_date ≤ 2023-08-18` และไม่มีวันที่ |return_1d| > 25% (สงสัยว่าเป็น split)
- [x] QQQ ที่ดึงใหม่: ราคาตรง snapshot ทุกวัน แต่ RSI ต่างได้ถึง 0.35 (Wilder จำจุดเริ่ม) จึงต่อแถวใหม่ท้าย snapshot แล้วตรวจว่าฟีเจอร์ตรง `qqq_features.csv` ทั้ง 2,492 แถว (ต่างไม่เกิน 7e-15)
- [x] tests: cutoff ของ synthetic/ข้อมูลจริง และ target ของแถวที่เหลือต้องตรงกับแบบไม่ตัด

## Task 2: autograd + layers

- [x] เขียน gradcheck ก่อน: ทุก op และทั้ง Transformer และ GRU (float64, rel err < 1e-6)
- [x] implement จนผ่าน 22 tests, overfit batch เล็กได้, seed เดิมให้ผลซ้ำ, save/load ได้ output เดิม
- [x] เวลาฝึก: Transformer 0.38 s/batch 128 (19,665 พารามิเตอร์), GRU 0.18 s

## Task 3: windows + training loop

- [x] window ของวัน t คือ t-59..t โดย Val ใช้วัน purge เป็น context ได้ ทำให้ได้ครบ 371 วัน ส่วน Train เหลือ 1,674 window (positive 433)
- [x] tail split พร้อม purge 5 วัน และ probe ต้องไม่ขยับน้ำหนัก encoder
- [x] smoke test pipeline ครบทุก arm

## Task 4: Validation

- [x] เพิ่ม variant `non_spike` ให้เป็นมาตรฐานเดียวกับ 19 โมเดล โดย SSL บน ETF ใช้ร่วมกันต่อ seed ส่วน DAPT, scaler และ fine-tune แยกตาม variant ตรวจแล้วว่าโค้ดใหม่ให้ผล `with_spike` ตรงกับโค้ดเดิมทุกตัวเลข และ MLP/persistence ของ `non_spike` ได้ AP 0.2808/0.2474 ตรง leaderboard เดิม
- [x] รัน `train` ครบ 2 variants × 2 backbones × 3 seeds (ขนาน 12 process; 48 checkpoints ผ่าน reload check)
- [x] `summarize` → leaderboard, paired AP differences, criteria, `decision.json` (hash ของ code/source/model)
- [x] ตรวจ Transformer ที่แย่กว่า chance บน Validation: ไม่พบ bug (tail ROC-AUC 0.65–0.81 แต่ Validation 0.38–0.61 ทุก seed ทั้งสอง variant, best epoch ≤ 3 ใน 20/24 runs; `report/generalization_diagnostics.csv`) จึงรายงานตามจริงและไม่จูนจาก Validation

## Task 5: Gate + Test ครั้งเดียว + holdout

- [x] `pytest -q` ทั้ง suite ผ่าน 1,369 tests, `code-review` แก้ finding ที่เป็นจริง (label ของ baseline ใน CSV, ล็อก receipt ก่อนอ่าน Test, hash ของ QQQ refresh, บังคับ reload check)
- [x] เจ้าของโปรเจกต์อนุมัติแล้วจึงรัน `python -m src.ml.transfer_learning.test_eval --allow-test` ครั้งเดียว
- [x] คำนวณ AP/ROC-AUC ซ้ำจาก CSV ด้วย scikit-learn แล้วตรงกัน (ต่าง ≤ 2.2e-16) และ confusion matrix ตรงทุกโมเดล

## Task 6: รายงาน

- [x] `docs/models/transfer_learning/EXPERIMENT.md` + กราฟ (dataviz) + ลิงก์จาก `docs/README.md` และ `docs/models/PROGRESS_INDEX.md`
- [x] เพิ่มหัวข้อ 17 ใน notebook 04 และหัวข้อ 12 ใน notebook 05 (อ่าน artifact เท่านั้น) พร้อม test `test_transfer_supplement_has_ranking_and_rendered_graphs`
- [x] รายงาน: ภาคผนวก จ, ข้อเสนอแนะ 5.4.2 ข้อ 7, คำนำ, บรรณานุกรม 5 รายการใหม่ + Nasdaq (2026b) แล้วสร้าง `report.docx`/`report.pdf` ใหม่ (70 หน้า) ไม่ทำสไลด์ตามที่เจ้าของโปรเจกต์ขอ
- [ ] สรุป diff ให้เจ้าของโปรเจกต์ก่อน commit
