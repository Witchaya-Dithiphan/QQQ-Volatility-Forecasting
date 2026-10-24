# แผนทดลอง Transfer Learning สำหรับการพยากรณ์ความผันผวนของ QQQ

Branch: `feat/time-series-transfer-learning`

สถานะ: แผนการทดลอง ยังไม่ได้เทรนโมเดลใหม่

## คำตัดสิน

เริ่มจาก **GRU ขนาดเล็กที่ฝึกแบบ supervised บน SPY แล้ว fine-tune บน QQQ** เพื่อทำนาย `target_high_volatility` ของ QQQ ในอีก 5 วันทำการ วิธีนี้ตรงกับข้อเสนอเรื่องเรียนจากตลาดอื่นก่อนและใช้โมเดลสำหรับลำดับเวลา โดยมีการทดลอง QQQ-only GRU ที่โครงสร้างเหมือนกันเป็นตัวควบคุม หาก GRU แบบ transfer ไม่ชนะตัวควบคุม เราจะไม่กล่าวว่า transfer learning ช่วย

SPY คือ ETF ที่ติดตาม S&P 500 และมี OHLCV ให้สร้างฟีเจอร์แบบเดียวกับ QQQ; ดัชนี S&P 500 โดยตรงไม่มีปริมาณซื้อขายที่เทียบกับฟีเจอร์ `volume_zscore_20` ได้ตรง ๆ ข้อมูล SPY **ยังไม่มีใน repository** จึงต้องจัดหาและบันทึกแหล่งที่มา/วัน snapshot ก่อนเริ่มเทรน โดยตรวจนโยบายปรับราคาของ SPY ให้เทียบได้กับ snapshot QQQ เดิม

คำว่า *spike* ในโครงการหมายถึงวันผลตอบแทนสุดโต่งแบบวันเดียว ส่วน label ที่ต้องทำนายคือความผันผวนสูงใน 5 วันข้างหน้า สองสิ่งนี้ไม่ใช่ label เดียวกัน แผนหลักจะ pretrain ให้รู้จัก **ภาวะความผันผวนสูงในอนาคต** บน SPY; จะทดสอบหัวทำนาย spike รายวันเป็นงานเสริมได้ภายหลัง แต่ห้ามใช้ผลนั้นแทนหลักฐานการทำนายเป้าหมาย QQQ

## หลักฐานและสมมติฐานตั้งต้น

- QQQ มี Train 1,733 วัน (positive 433), Validation 371 วัน (positive 34), Test 373 วัน (positive 73); แยกชุดตามเวลาและเว้น 5 วันทำการ ผล Validation จึงมีความไม่แน่นอนสูง
- เป้าหมายเดิมใช้ `target_volatility_5d > 0.2530580184684854` ซึ่งเป็น Q75 ที่ fit จาก QQQ Train เท่านั้น และมีฟีเจอร์ย้อนหลัง 8 ตัวตาม `src/ml/core/contracts.py`
- บน Validation เดิม MLP มี average precision (AP) 0.2668 และ persistence baseline 0.2474; บน Test เดิม MLP 0.5991 และ persistence 0.5365 ตัวเลข Test เหล่านี้เคยเปิดดูแล้ว จึงไม่ใช่ blind test สำหรับการออกแบบการทดลองใหม่นี้
- สมมติฐานที่ต้องตรวจ: SPY มีรูปแบบความผันผวนร่วมกับ QQQ มากพอให้ encoder ที่ pretrain ช่วย QQQ เมื่อเทียบกับ GRU ที่เริ่มจากน้ำหนักสุ่ม ผลลัพธ์อาจเป็น **negative transfer** ได้

## ทางเลือกที่พิจารณา

| วิธี | ใช้เมื่อไร | เหตุผล/ข้อจำกัด |
| --- | --- | --- |
| **Supervised SPY → QQQ GRU** | ทดลองหลัก | ฝึกตรงกับเป้าหมาย 5 วัน, พารามิเตอร์ไม่มาก, มี QQQ-only GRU เป็น ablation ที่ตีความได้ |
| QQQ-only GRU | ตัวควบคุมบังคับ | แยกประโยชน์จากการใช้ลำดับเวลาออกจากประโยชน์ของการ pretrain |
| Self-supervised masked reconstruction หรือ TS2Vec บนข้อมูลอดีต | ขั้นต่อไปเมื่อ supervised transfer ไม่ช่วย หรือข้อมูล source label ใช้ไม่ได้ | ใช้ข้อมูลที่ไม่มี label ได้ แต่ representation ที่เรียนอาจไม่สัมพันธ์กับอนาคต 5 วัน; เพิ่มงานจูนและความเสี่ยงเลือกจาก Validation ซ้ำ |
| TCN ขนาดเล็ก | ทางเลือกหาก GRU เรียนไม่เสถียร | เหมาะกับลำดับเวลาเช่นกัน แต่การเพิ่มสถาปัตยกรรมต้องมีเหตุผลจากผล Train/Validation ก่อน |
| Domain-adversarial adaptation (DANN) | เลื่อนออกไปจนกว่าจะวัด domain shift ได้ | มี source/target เพียงคู่เดียวและ QQQ Train เล็ก; การบังคับ feature ให้เหมือนกันอาจลบสัญญาณเฉพาะ QQQ |

ไม่เริ่มด้วย Transformer/foundation model ขนาดใหญ่หรือการค้น hyperparameter กว้าง เพราะจำนวน QQQ Train และ Validation เล็กเมื่อเทียบกับความซับซ้อนและต้นทุนการคัดเลือก

## ขั้นตอนลงมือทำ

### 1. ล็อกข้อมูลและตรวจสัญญาข้อมูล

1. บันทึก commit, data hash, ช่วงวัน และนิยาม target/feature เดิมก่อนทดลอง; ไม่แก้ `data/raw/qqq_daily.csv`, ชุด processed หลัก, `outputs/modeling/` หรือ selection lock ของ 19 โมเดลเดิม
2. จัดหา **SPY OHLCV รายวัน** จากแหล่งที่ระบุ provenance, license, timezone, adjustment policy และ snapshot hash ชัดเจน เก็บแยกใต้ `data/external/transfer_learning/` หรือพื้นที่ input ของการทดลอง ห้ามเขียนทับ QQQ และห้ามผสมราคาที่ปรับ dividend/split คนละแบบกับ QQQ โดยไม่ตรวจผลกระทบต่อฟีเจอร์ก่อน
3. ใช้สูตรฟีเจอร์เดียวกับ `src/build_features.py` และสูตร forward 5-day realized volatility เดียวกับ `src/build_targets.py`; ตรวจวันซื้อขายที่ไม่ตรงกัน, missing values, price adjustment และความต่างของ distribution ก่อนฝึก
4. ตัด SPY ที่ใช้ pretrain ให้สิ้นสุดก่อน QQQ Train cutoff พร้อมกันอย่างน้อย 5 วันทำการสำหรับ label ที่มองไปข้างหน้า ใน walk-forward แต่ละ fold ให้ตัด source ตาม cutoff ของ fold นั้นด้วย ห้ามใช้ SPY ในช่วง Validation/Test ของ QQQ เป็นข้อมูลช่วยเรียนรู้น้ำหนัก แม้จะไม่มี QQQ label
5. สร้าง source label ด้วย Q75 ของ SPY **เฉพาะ source Train ที่ใช้ในรอบนั้น** เพื่อเรียน “ภาวะสูงเมื่อเทียบกับ SPY”; QQQ ยังคงใช้ threshold เดิมของ QQQ Train ห้ามนำ QQQ Validation/Test ไป fit threshold หรือ scaler

**ประตูหยุด:** หากไม่มี SPY ที่มี provenance/adjustment น่าเชื่อถือ หรือสูตรฟีเจอร์และ label ทำให้เทียบกันไม่ได้ ให้หยุดขั้น source pretrain และรายงานข้อจำกัด ไม่ใช้ไฟล์สุ่มหรือข้อมูลอนาคตแทน

### 2. สร้างชุดลำดับเวลาอย่างปลอดภัย

1. ใช้ข้อมูลย้อนหลัง **20 วันทำการ** เป็นค่าตั้งต้น: input ของแถว `t` คือฟีเจอร์วัน `t-19..t`; label เป็นความผันผวนของ `t+1..t+5` ปฏิทินและการเรียงลำดับต้องตรงกับ pipeline เดิม
2. แยก Train/Validation/Test ตาม **วันที่ของ target row** เดิมและรักษา purge 5 วัน ห้ามให้แถว input ของ Validation ใช้ feature ในอนาคต แต่อนุญาตให้ใช้ประวัติราคาเก่าก่อน Validation เพื่อสร้าง context; ห้ามใช้ label ของช่วง purge เพื่อฝึก
3. fit scaler จากช่วงฝึกของ fold เท่านั้น เมื่อ fine-tune QQQ ให้กำหนด normalization policy ล่วงหน้าและใช้เหมือนกันทุก arm; บันทึก scaler ไว้กับ checkpoint
4. เนื่องจาก window ทำให้ต้นชุดบางส่วนใช้ไม่ได้ ให้เทียบ persistence, MLP, QQQ-only GRU และ transfer GRU บน **วันที่และ label ชุดเดียวกัน** ทุกครั้ง พร้อมแสดงจำนวนแถว/positive ที่เหลือ

**การตรวจขั้นต่ำ:** test หรือ assert ให้พิสูจน์ว่าไม่มี feature หลัง `t`, ไม่มี label window ข้าม cutoff, index ของผลทำนายเรียงและตรงกันทุก arm, source cutoff/threshold/scaler ไม่ใช้อนาคต

### 3. ฝึกและทำ ablation

ทดลอง `with_spike` ก่อน เพราะเป็นการใช้ Train หลักครบ; `non_spike` เป็น sensitivity check หลังได้ผลหลัก โดย Validation/Test ยังคงเดิมและต้องระบุว่าการตัดแถวเปลี่ยน class prior ด้วย

| Arm | น้ำหนักเริ่มต้น | การฝึกบน QQQ | คำถามที่ตอบ |
| --- | --- | --- | --- |
| Persistence, MLP เดิม | artifact เดิม | ไม่มี | ชนะ baseline เดิมหรือไม่ บนวันที่ตรงกัน |
| GRU scratch | random | train encoder + head | ลำดับเวลาเพียงอย่างเดียวช่วยหรือไม่ |
| GRU source-only | SPY supervised | ไม่มี | ใช้ตรง ๆ ข้าม domain ได้ไหม |
| GRU frozen | SPY supervised | train head เท่านั้น | representation เดิมถ่ายโอนได้ไหม |
| **GRU fine-tuned** | SPY supervised | train encoder + head ด้วย learning rate ต่ำ | การปรับให้เข้ากับ QQQ ช่วยเพิ่มไหม |

ใช้ encoder 1 ชั้น hidden size 16 หรือ 32 และ head แบบ binary probability เริ่มด้วยความจุเล็ก กำหนด seed อย่างน้อย 3 ค่าให้ทุก GRU arm เหมือนกันและรายงานราย seed; class weighting คำนวณจากช่วง Train ของ arm นั้นเท่านั้น การหยุดก่อนกำหนดใช้อดีตส่วนท้ายของ QQQ Train พร้อม purge ไม่ใช้ Test หากต้อง retrain บน Train เต็ม ให้ตรึง epoch/setting จากการทดลองภายใน Train ก่อน แล้วใช้ Validation สำหรับจัดอันดับ candidate

ใช้ PyTorch ใน environment แยกของการทดลองได้หากตรวจ runtime และข้อกำหนดงานอาจารย์แล้วอนุญาต; ระบุชัดว่าโมเดลเพิ่มนี้ใช้ framework และไม่ใช่หนึ่งใน 19 โมเดล NumPy ที่เขียนเองเดิม ไม่เพิ่ม dependency ให้ pipeline หลักโดยไม่จำเป็น

### 4. เลือกโมเดลและตัดสินผล

1. กำหนดรายการ candidate/seed/hyperparameter ล่วงหน้าให้เล็ก ใช้ expanding walk-forward ภายใน Train (`src/ml/core/walk_forward.py`) เพื่อตรวจความเสถียรโดยไม่ใช้ผล fold เดียวกันทั้งคัดเลือกและอ้างเป็นผลอิสระ จากนั้นใช้ Validation ที่ล็อกไว้จัดอันดับครั้งสุดท้าย
2. **Metric หลักคือ AP** เพราะ positive มีน้อย; รายงาน prevalence ของแต่ละ split, ROC-AUC, precision, recall, F1, confusion matrix, PR curve, calibration และจำนวนวันเตือนต่อปี Threshold สำหรับ metric แบบ binary เลือกจาก Train ภายในหรือ Validation แล้วล็อกก่อน Test
3. วัดผลต่าง AP แบบจับคู่บนวันที่เดียวกัน เทียบทั้ง QQQ-only GRU, MLP และ persistence; ใช้ moving-block bootstrap (block 20 วัน, 2,000 รอบ) เพื่อแสดงช่วงความไม่แน่นอน พร้อมรายงานผลราย seed และช่วงตลาดที่พลาด
4. เกณฑ์ “transfer ช่วย” คือ fine-tuned GRU มี AP สูงกว่า QQQ-only GRU บน Validation และผลไม่พึ่ง seed เดียว; เกณฑ์ “เหนือ baseline อย่างชัดเจน” ที่ตั้งไว้ก่อนดูผลคือ AP สูงกว่า **ทั้ง MLP และ persistence อย่างน้อย 0.05** บน Validation และช่วง 95% ของผลต่างแบบจับคู่ไม่คร่อมศูนย์ เกณฑ์นี้เป็นเป้าหมายการทดลอง ไม่ใช่คำรับประกัน หากไม่ผ่านให้รายงานผลจริงและ negative transfer ตรง ๆ
5. หาก supervised transfer ไม่ผ่าน ให้ตรวจความต่าง distribution/source-target และข้อผิดพลาดก่อน ค่อยเลือก **หนึ่ง** การทดลองต่อยอดที่มีเหตุผล: self-supervised masked reconstruction บน SPY และ QQQ Train เท่านั้น หรือ TCN ขนาดเล็ก ห้ามลองหลายแนวทางจนพบผล Test ที่ถูกใจ

### 5. เปิด Test และรายงาน

บันทึก source hash, code commit, seeds, configuration, model choice, Validation metrics, threshold และเหตุผลเลือก **ก่อน** เรียก `load_test(allow_test=True)` แล้วรัน Test ครั้งเดียวสำหรับ configuration ที่เลือก รวมทั้ง baselines บนวันที่ตรงกัน เก็บผลที่ `outputs/experiments/transfer_learning/` แยกจากผลเดิม

**ข้อจำกัดสำคัญ:** Test ช่วง 2025-02-27 ถึง 2026-08-21 เคยถูกเปิดในงานเดิมแล้ว การทดลองใหม่นี้จึงเป็นการประเมิน **post-hoc/exploratory** แม้จะล็อกการตั้งค่าใหม่ก่อนรัน Test ห้ามเขียนว่าเป็นการยืนยันอิสระหรือ blind test หากต้องการหลักฐานยืนยันทั่วไป ให้เก็บข้อมูล QQQ หลัง 2026-08-21 เป็น holdout อนาคตโดยไม่จูนกับช่วงนั้น

รายงาน Validation/Test พร้อม AP, ROC-AUC, precision, recall, confusion matrix, PR curve, ผลราย seed, confidence interval แบบจับคู่, กรณี spike/onset ที่โมเดลพลาด และข้อจำกัดเรื่อง source domain เขียนเป็น experiment เพิ่มใน notebook 04/05 และรายงานหลังได้ผลจริง โดยรักษาผล 19 โมเดลหลักตามเดิม

## ไฟล์และผลส่งมอบเมื่อเริ่ม implementation

- โค้ดการทดลองและ checkpoint แยกภายใต้ `src/ml/` และ `outputs/experiments/transfer_learning/`; ใช้ `src/ml/core/data.py`, `contracts.py`, `walk_forward.py` และสูตร feature/target เดิมให้มากที่สุด
- tests เฉพาะจุดสำหรับ window, cutoff, target horizon, scaler และความตรงกันของวันที่ประเมิน; การเทียบ baseline ใช้ artifact เดิมโดยไม่ retrain จาก Test
- experiment manifest (source URL/license/hash, data/code hash, dates, environment, seeds, config), table ของ ablation และรูป PR/confusion/calibration
- อัปเดต `notebooks/04_classification.ipynb`, `notebooks/05_results_comparison.ipynb`, เอกสารการทดลอง และรายงานฉบับส่ง **หลัง** ตรวจผลครบ โดยติดป้ายว่าเป็นการทดลองเพิ่มหลังงานหลัก

## งานวิจัยอ้างอิงสำหรับการเลือกวิธี

- GRU: [Chung et al., Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](https://arxiv.org/abs/1412.3555)
- TCN เทียบ RNN: [Bai et al., An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271)
- Self-supervised: [Yue et al., TS2Vec](https://ojs.aaai.org/index.php/AAAI/article/view/20881); [Zerveas et al., A Transformer-based Framework for Multivariate Time Series Representation Learning](https://arxiv.org/abs/2010.02803)
- Domain adaptation: [Ganin et al., Domain-Adversarial Training of Neural Networks](https://www.jmlr.org/papers/v17/15-239.html)

งานวิจัยสนับสนุนความเป็นไปได้ของวิธี ไม่ได้พิสูจน์ว่าจะเพิ่ม AP ในข้อมูล QQQ ชุดนี้ ผลตัดสินต้องมาจาก ablation และการประเมินตามเวลาในโครงการ
