# ตารางเกณฑ์การให้คะแนน → หลักฐาน

ตารางนี้จับคู่เกณฑ์แต่ละข้อของโจทย์ (`_69-1-01076641_Project-description.pdf` หน้า 5) กับตำแหน่งในรายงาน สไลด์ และไฟล์หลักฐาน test `test_every_traceability_evidence_path_exists` ใน `tests/test_docs_links.py` ตรวจว่าไฟล์หลักฐานทุกไฟล์มีอยู่จริง

| เกณฑ์ | รายงาน | สไลด์ | หลักฐาน |
| --- | --- | --- | --- |
| 1. Problem setting | §1 | 2–3 | `notebooks/02_eda_and_features.ipynb`, `docs/guides/PROJECT_GUIDE_TH.md` |
| 2. Feature selection และ insight · why ML/DL | §2 | 4–6 | `outputs/reports/feature_report.json`, `outputs/figures/data_preparation/07_train_feature_correlation.png`, `outputs/modeling/report/with_spike/feature_importance.csv`, `outputs/figures/modeling/with_spike/feature_importance.png`, `outputs/modeling/report/with_spike/linear_coefficients.csv` |
| 3. Case analysis และ data preparation | §3 | 7–9 | `outputs/reports/data_split_report.json`, `outputs/reports/spike_analysis.json`, `outputs/reports/spike_event_audit.csv`, `docs/audit/DATA_PIPELINE_AUDIT.md`, `tests/test_audit_guards.py` |
| ใช้ฟีเจอร์อย่างน้อย 5 ตัว | §2.1 | 4 | `src/ml/core/contracts.py`, `src/build_features.py` |
| Regression from scratch | §5.1 | 10 | `src/ml/regression/multiple_linear.py`, `notebooks/03_regression.ipynb` |
| Classification from scratch | §5.2 | 11 | `src/ml/classification/logistic.py`, `notebooks/04_classification.ipynb` |
| โมเดล regression นอกห้องเรียน พร้อมกลไก | §6.1 | 12 | `src/ml/regression/elastic_net.py`, `outputs/modeling/report/cards/elastic_net.md` |
| โมเดล classification นอกห้องเรียน พร้อมกลไก | §6.2 | 13 | `src/ml/classification/adaboost.py`, `outputs/modeling/report/cards/adaboost.md` |
| Scratch เทียบ ready-to-use | §5.3 | 14 | `src/ml/core/compare.py`, `tests/ml/core/test_compare.py`, `docs/tasks/claude-model-progress.md` |
| Training-testing iterations, batch runs | §5.4 | 14 | `src/ml/core/trainer.py`, `configs/modeling.json`, `outputs/modeling/report/tables/model_inventory.md` |
| Save-load model แล้ว test | §5.5 | 14 | `tests/ml/core/test_artifacts.py`, `outputs/modeling/with_spike/classification/knn/load_verification.json` |
| Loss | §7.2 | 16 | `outputs/modeling/report/tables/classification_metrics_with_spike.md` |
| Confusion matrix | §7.2 | 16 | `outputs/figures/modeling/with_spike/confusion_matrices.png` |
| Accuracy, Precision, Sensitivity, Specificity, TNR, F1 | §7.2 | 16 | `outputs/modeling/report/tables/classification_metrics_with_spike.md`, `outputs/modeling/report/tables/test_classification_metrics_with_spike.md` |
| ROC และ AUC | §7.3 | 17 | `outputs/figures/modeling/with_spike/roc_overlay.png`, `outputs/figures/modeling/with_spike/pr_overlay.png` |
| Performance curve | §7.4 | 18 | `outputs/figures/modeling/with_spike/performance_curves.png` |
| R-Square | §7.1 | 15 | `outputs/modeling/report/with_spike/leaderboard_regression.md` |
| Benchmarking, analysis, discussion | §8 | 19–21 | `outputs/modeling/report/comparison_a_vs_b.md`, `outputs/figures/modeling/comparison_delta_classification.png`, `outputs/modeling/report/with_spike/walk_forward.csv`, `outputs/modeling/report/tables/test_spike_policy.md` |
| ผลบน Test (อ่านครั้งเดียว) | §7.6 | 19 | `outputs/modeling/selection_lock.json`, `outputs/modeling/finalize_receipt.json`, `outputs/modeling/final/with_spike/leaderboard_classification.md`, `outputs/modeling/final/segmented_metrics.csv`, `outputs/figures/modeling/final/test_roc_auc_by_segment.png` |
| +5 creative problem / data preparation | §1.3, §3.3 | 8–9, 20 | `outputs/figures/spike_analysis/daily_return_spikes.png`, `outputs/figures/spike_analysis/dataset_comparison.png` |
| +5 much better than taught models | §8.3 | 21 | `outputs/modeling/report/tables/validation_headline_vs_classroom.md`, `outputs/modeling/report/tables/test_headline_vs_classroom.md` |
| Report | ทั้งฉบับ | — | `docs/report/REPORT_DRAFT.md`, `deliverables/report.docx` |
| Slides | — | 22 แผ่น | `deliverables/slides.pptx` |
| `.ipynb` และ `.py` | — | — | `notebooks/05_results_comparison.ipynb`, `src/ml/run.py` |

## หมายเหตุสำคัญสองข้อ

**เรื่อง "+5 much better than taught models"** ผลของเราไม่สนับสนุนการอ้างข้อนี้ โมเดลตัวแทนดีกว่าโมเดลพื้นฐานบน Validation 3% ถึง 20% แต่ช่วงความเชื่อมั่น 95% ของผลต่างคร่อมศูนย์ทุกแถว และบน Test มีเพียง `mlp` ที่ยังดีกว่า `logistic` (ช่วงยังคร่อมศูนย์) รายงานจึงเขียนตามจริงว่า "ดีกว่าเล็กน้อยแต่ยังแยกจากความบังเอิญไม่ได้" ไม่อ้างว่าดีกว่ามาก

**เรื่อง "+5 creative"** สิ่งที่เสนอได้คือ การตั้งคำถามวิจัยที่ทดสอบได้ (spike ช่วยหรือทำร้ายการเรียนรู้), การออกแบบการทดลองแบบจับคู่, การตรึงข้อมูลพร้อม checksum, การล็อกโมเดลก่อนเปิด Test และการรายงานผลแยกช่วงตลาด
