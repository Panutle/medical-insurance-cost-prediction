# 🏥 Medical Insurance Cost Prediction (Regression Analysis)

โปรเจกต์นี้เป็นการนำข้อมูลค่าเบี้ยประกันสุขภาพ (Medical Insurance Cost) มาวิเคราะห์เชิงสำรวจ (Exploratory Data Analysis - EDA) และสร้างโมเดล Machine Learning เพื่อพยากรณ์ค่าใช้จ่ายทางการแพทย์ (`charges`) โดยเปรียบเทียบอัลกอริทึม Regression หลากหลายประเภท เพื่อหาโมเดลที่มีความแม่นยำสูงที่สุด

---

## 📌 สรุปขั้นตอนการดำเนินงาน (Workflow)

1. **Data Acquisition & Import**: ดึงชุดข้อมูล `mirichoi0218/insurance` จาก Kaggle รวมทั้งสิ้น 1,338 รายการ 7 ตัวแปร
2. **Data Cleaning & Encoding**:
   - ตรวจสอบค่าสูญหาย (Missing Values): ไม่พบค่าว่างในชุดข้อมูล
   - ทำ Label Encoding กับตัวแปร Category: `sex`, `smoker` และ `region` เพื่อเตรียมเข้ากระบวนการวิเคราะห์เชิงตัวเลข
3. **Exploratory Data Analysis (EDA)**:
   - ตรวจสอบค่าสถิติเบื้องต้น (Mean, Std, Min, Max)
   - วิเคราะห์สหสัมพันธ์ (Correlation Matrix) ด้วย Heatmap
   - สำรวจ Outliers ของตัวแปรต่างๆ ด้วย Boxplot
4. **Model Training & Evaluation**:
   - แบ่งข้อมูล Train/Test ในสัดส่วน 80:20 (Random State = 42)
   - ฝึกสอนและทดสอบโมเดล Regression ทั้งหมด 12 อัลกอริทึม
   - ประเมินผลลัพธ์ด้วยเมทริกซ์ 4 ด้าน: MSE, MAE, RMSE และ $R^2$ Score

---

## 📊 ผลการเปรียบเทียบโมเดล (Model Performance)

ผลที่บันทึกจากการทดลองใน notebook สำหรับข้อมูลเดิมก่อนตัด outlier เรียงลำดับตามประสิทธิภาพ $R^2$ Score (จากมากไปน้อย):

| อันดับ | Model | $R^2$ Score | RMSE | MAE | MSE |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 🥇 | **GradientBoostingRegressor** | **0.8790** | **4,333.52** | **2,397.93** | 1.88e+07 |
| 🥈 | **CatBoost Regressor** | **0.8702** | **4,488.80** | **2,515.95** | 2.01e+07 |
| 🥉 | **LightGBM** | **0.8666** | **4,550.62** | **2,570.57** | 2.07e+07 |
| 4 | Bagging Regressor | 0.8617 | 4,633.14 | 2,495.32 | 2.15e+07 |
| 5 | Random Forest Regressor | 0.8615 | 4,636.72 | 2,497.41 | 2.15e+07 |
| 6 | AdaBoost Regressor | 0.8333 | 5,086.68 | 3,968.59 | 2.59e+07 |
| 7 | Linear Regression | 0.7839 | 5,791.80 | 4,174.05 | 3.35e+07 |
| 8 | Decision Tree Regressor | 0.7170 | 6,628.09 | 3,198.25 | 4.39e+07 |
| 9 | K-Neighbors Regressor | 0.0820 | 11,938.34 | 7,514.11 | 1.43e+08 |
| 10 | MLP Regressor (Neural Network) | 0.0575 | 12,096.40 | 7,840.91 | 1.46e+08 |
| 11 | Support Vector Regressor (SVR) | -0.0723 | 12,902.37 | 8,592.03 | 1.66e+08 |
| 12 | Gaussian Process Regressor | -0.2367 | 13,856.07 | 8,709.33 | 1.92e+08 |

---

## 💡 สรุปผลการวิเคราะห์ (Key Insights)

* กลุ่มโมเดลประเภท **Ensemble / Boosting (Gradient Boosting, CatBoost, LightGBM, Random Forest)** ให้ประสิทธิภาพในการทำนายที่ดีที่สุด โดยทำค่า $R^2$ ได้สูงกว่า **86% - 87%**
* โมเดลที่ทำงานได้ดีที่สุดคือ **GradientBoostingRegressor** ด้วยค่า $R^2 \approx 0.8790$ และค่าความคลาดเคลื่อนเฉลี่ย MAE เพียง **2,397.93**
* โมเดลประเภท Neural Network และ SVR ที่ยังไม่ได้ผ่านการปรับจูน Hyperparameter หรือการทำ Feature Scaling ให้ผลลัพธ์ต่ำกว่าโมเดลกลุ่ม Tree-based อย่างเห็นได้ชัด

---

## 🛠️ เครื่องมือและไลบรารีที่ใช้ (Tech Stack)

* **Language**: Python 3
* **Libraries**:
  * Data Processing: `pandas`, `numpy`
  * Data Visualization: `matplotlib`, `seaborn`
  * Machine Learning: `scikit-learn`, `catboost`, `lightgbm`


## เริ่มต้นใช้งาน

```bash
git clone https://github.com/Panutle/medical-insurance-cost-prediction.git
cd medical-insurance-cost-prediction
```

เปิด [`notebooks/insurance_prediction.ipynb`](notebooks/insurance_prediction.ipynb) ใน Google Colab เพราะ notebook ใช้ `google.colab.drive` และ path `/content/` จากนั้น:

1. ติดตั้งไลบรารีที่ notebook ใช้ในเซลล์เริ่มต้น:

   ```python
   %pip install pandas numpy matplotlib seaborn scipy scikit-learn catboost lightgbm kaggle
   ```

2. Mount Google Drive และปรับ `KAGGLE_CONFIG_DIR` ให้ตรงกับโฟลเดอร์ที่เก็บ credential ของคุณ
3. รันส่วนดาวน์โหลด `mirichoi0218/insurance` แล้วตรวจว่าอ่าน `insurance.csv` ได้
4. รันส่วน EDA และ regression ตามลำดับ ก่อนทดลองส่วน cross-validation, การตัด outlier และ classification ที่อยู่ช่วงหลัง
5. ปรับ path ของ `to_csv(...)` ใน Google Drive ให้เป็นโฟลเดอร์ที่คุณสร้างไว้ก่อนรันส่วน export

หากรันผ่าน Jupyter บนเครื่อง ต้องแทนที่เซลล์ mount Drive และ path ของ Colab ก่อน repository นี้ยังไม่มี `requirements.txt` หรือ environment lock

## การอ่านผลและข้อจำกัด

- ชื่อ `XGBoostRegressor` ในรายการโมเดลของ notebook เป็นป้ายชื่อที่คลาดเคลื่อน: estimator ที่สร้างจริงคือ `sklearn.ensemble.GradientBoostingRegressor` ตารางด้านบนใช้ชื่อ estimator จริง
- คะแนนใน README เป็นผลการทดลองที่บันทึกไว้ ไม่ใช่ผลการรันทดสอบใหม่ และไม่รับประกันว่าจะได้ตัวเลขเดิมทุกครั้ง เพราะบาง estimator ไม่ได้กำหนด seed
- ผลหลังกรอง outlier จาก `charges` ประเมินบนประชากรข้อมูลที่เปลี่ยนไป จึงไม่ควรเทียบตรงกับผลข้อมูลเต็ม
- Notebook มีทั้ง regression และการแบ่งค่าใช้จ่ายเป็นช่วงเพื่อทำ classification ควรแยกอ่านผลของแต่ละการทดลอง
