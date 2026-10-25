# ตารางเกณฑ์การให้คะแนน → หลักฐาน

ตารางนี้จับคู่เกณฑ์แต่ละข้อของโจทย์ (`_69-1-01076641_Project-description.pdf` หน้า 5) กับตำแหน่งในรายงาน สไลด์ และไฟล์หลักฐาน test `test_every_traceability_evidence_path_exists` ใน `tests/test_docs_links.py` ตรวจว่าไฟล์หลักฐานทุกไฟล์มีอยู่จริง

| เกณฑ์ | รายงาน | สไลด์ | หลักฐาน |
| --- | --- | --- | --- |
| 1. Problem setting | บทที่ 1 | 2–3 | `notebooks/02_eda_and_features.ipynb`, `docs/guides/PROJECT_GUIDE_TH.md` |
| 2. Feature selection และ insight · why ML/DL | หัวข้อ 1.1, 3.3.2, 4.1 | 4–6 | `outputs/reports/feature_report.json`, `outputs/figures/data_preparation/07_train_feature_correlation.png`, `outputs/modeling/report/with_spike/feature_importance.csv`, `outputs/figures/modeling/with_spike/feature_importance.png`, `outputs/modeling/report/with_spike/linear_coefficients.csv` |
| 3. Case analysis และ data preparation | หัวข้อ 3.1, 3.3 | 7–11 | `outputs/reports/data_split_report.json`, `outputs/reports/spike_analysis.json`, `outputs/reports/spike_event_audit.csv`, `notebooks/06_spike_analysis.ipynb`, `docs/audit/DATA_PIPELINE_AUDIT.md`, `tests/test_audit_guards.py` |
| ใช้ฟีเจอร์อย่างน้อย 5 ตัว | หัวข้อ 3.3.2 | 4 | `src/ml/core/contracts.py`, `src/build_features.py` |
| Regression from scratch | หัวข้อ 2.1.2, 3.4.1 | 12 | `src/ml/regression/multiple_linear.py`, `notebooks/03_regression.ipynb` |
| Classification from scratch | หัวข้อ 2.1.4, 3.4.1 | 13 | `src/ml/classification/logistic.py`, `notebooks/04_classification.ipynb` |
| โมเดล regression นอกห้องเรียน พร้อมกลไก | หัวข้อ 2.1.3, 4.3.1 | 14 | `src/ml/regression/elastic_net.py`, `outputs/modeling/report/cards/elastic_net.md` |
| โมเดล classification นอกห้องเรียน พร้อมกลไก | หัวข้อ 2.1.5, 4.3.2 | 15 | `src/ml/classification/adaboost.py`, `outputs/modeling/report/cards/adaboost.md` |
| Scratch เทียบ ready-to-use | หัวข้อ 3.4.6 | 16 | `src/ml/core/compare.py`, `tests/ml/core/test_compare.py`, `docs/models/PROGRESS_INDEX.md` |
| Training-testing iterations, batch runs | หัวข้อ 3.4.7 | 16 | `src/ml/core/trainer.py`, `configs/modeling.json`, `outputs/modeling/report/tables/model_inventory.md` |
| Save-load model แล้ว test | หัวข้อ 3.4.7 | 16 | `tests/ml/core/test_artifacts.py`, `outputs/modeling/with_spike/classification/knn/load_verification.json` |
| Loss | หัวข้อ 4.2.2 | 18 | `outputs/modeling/report/tables/classification_metrics_with_spike.md` |
| Confusion matrix | หัวข้อ 4.2.2 | 18 | `outputs/figures/modeling/with_spike/confusion_matrices.png` |
| Accuracy, Precision, Sensitivity, Specificity, TNR, F1 | หัวข้อ 2.1.6, 4.2.2 | 18 | `outputs/modeling/report/tables/classification_metrics_with_spike.md`, `outputs/modeling/report/tables/test_classification_metrics_with_spike.md` |
| ROC และ AUC | หัวข้อ 4.2.3 | 19 | `outputs/figures/modeling/with_spike/roc_overlay.png`, `outputs/figures/modeling/with_spike/pr_overlay.png` |
| Performance curve | หัวข้อ 4.2.4 | 20 | `outputs/figures/modeling/with_spike/performance_curves.png` |
| R-Square | หัวข้อ 4.2.1 | 17 | `outputs/modeling/report/with_spike/leaderboard_regression.md` |
| Benchmarking, analysis, discussion | หัวข้อ 4.5–4.8, 5.2 | 21–23 | `outputs/modeling/report/comparison_a_vs_b.md`, `outputs/figures/modeling/comparison_delta_classification.png`, `outputs/modeling/report/with_spike/walk_forward.csv`, `outputs/modeling/report/tables/test_spike_policy.md` |
| ผลบน Test (อ่านครั้งเดียว) | หัวข้อ 4.4 | 21 | `outputs/modeling/selection_lock.json`, `outputs/modeling/finalize_receipt.json`, `outputs/modeling/final/with_spike/leaderboard_classification.md`, `outputs/modeling/final/segmented_metrics.csv`, `outputs/figures/modeling/final/test_roc_auc_by_segment.png` |
| +5 creative problem / data preparation | หัวข้อ 1.1, 3.3.4 | 8–11, 22 | `outputs/figures/spike_analysis/daily_return_spikes.png`, `outputs/figures/spike_analysis/dataset_comparison.png`, `notebooks/06_spike_analysis.ipynb`, `outputs/figures/modeling/spike_rule/spike_rule_sensitivity.png`, `outputs/figures/modeling/spike_rule/affected_window_example.png` |
| +5 much better than taught models | หัวข้อ 4.6 | 23 | `outputs/modeling/report/tables/validation_headline_vs_classroom.md`, `outputs/modeling/report/tables/test_headline_vs_classroom.md` |
| การทดลองเพิ่มเติมหลังส่งงานและหลังนำเสนอ (ผล Test เป็น post-hoc) | ภาคผนวก ง, จ, ฉ | — | `outputs/experiments/kernel_perceptron/test_receipt.json`, `outputs/experiments/transfer_learning/test_receipt.json`, `outputs/experiments/log_har/test_receipt.json`, `docs/models/log_har/EXPERIMENT.md` |
| Report | ทั้งฉบับ | — | `docs/report/REPORT_DRAFT.md`, `deliverables/report.docx` |
| Slides | — | 24 แผ่น | `deliverables/slides.pptx` |
| `.ipynb` และ `.py` | — | — | `notebooks/05_results_comparison.ipynb`, `notebooks/06_spike_analysis.ipynb`, `src/ml/run.py` |

## หมายเหตุสำคัญสองข้อ

**เรื่อง "+5 much better than taught models"** ผลของเราไม่สนับสนุนการอ้างข้อนี้ โมเดลตัวแทนดีกว่าโมเดลพื้นฐานบน Validation 3% ถึง 20% แต่ช่วงความเชื่อมั่น 95% ของผลต่างคร่อมศูนย์ทุกแถว และบน Test มีเพียง `mlp` ที่ยังดีกว่า `logistic` (ช่วงยังคร่อมศูนย์) รายงานจึงเขียนตามจริงว่า "ดีกว่าเล็กน้อยแต่ยังแยกจากความบังเอิญไม่ได้" ไม่อ้างว่าดีกว่ามาก

**เรื่อง "+5 creative"** สิ่งที่เสนอได้คือ การตั้งคำถามวิจัยที่ทดสอบได้ (spike ช่วยหรือทำร้ายการเรียนรู้), เกณฑ์ spike ที่กำหนดล่วงหน้าพร้อมเหตุผลและการเทียบกับเกณฑ์ทางเลือก (notebook 06), การออกแบบการทดลองแบบจับคู่, การตรึงข้อมูลพร้อม checksum, การล็อกโมเดลก่อนเปิด Test และการรายงานผลแยกช่วงตลาด
