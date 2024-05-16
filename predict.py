# predict.py
import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import model_training
from sklearn.model_selection import train_test_split

        # Fit and score the models
if st.button("Train Model"):
    model_training.model_training()

def load_model():
    model = joblib.load('best_model.pkl')
    # model = model_training.load_model()
    return model

def predict_Heart_Disease(model, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    prediction = model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    return prediction[0]

def predict():
    st.sidebar.title("Patient Attributes")
    age = st.sidebar.slider("Age", min_value=20, max_value=90, step=1)
    sex = st.sidebar.slider("Sex (0: Female, 1: Male)", min_value=0, max_value=1, step=1)
    cp = st.sidebar.slider("Chest Pain Type (cp)", min_value=0, max_value=3, step=1)
    trestbps = st.sidebar.slider("Resting Blood Pressure (trestbps)", min_value=90, max_value=200, step=1)
    chol = st.sidebar.slider("Serum Cholesterol (chol)", min_value=100, max_value=600, step=1)
    fbs = st.sidebar.slider("Fasting Blood Sugar (fbs)", min_value=0, max_value=1, step=1)
    restecg = st.sidebar.slider("Resting Electrocardiographic Results (restecg)", min_value=0, max_value=2, step=1)
    thalach = st.sidebar.slider("Maximum Heart Rate Achieved (thalach)", min_value=60, max_value=200, step=1)
    exang = st.sidebar.slider("Exercise Induced Angina (exang)", min_value=0, max_value=1, step=1)
    oldpeak = st.sidebar.slider("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=6.0, step=0.1)
    slope = st.sidebar.slider("Slope of the Peak Exercise ST Segment (slope)", min_value=0, max_value=2, step=1)
    ca = st.sidebar.slider("Number of Major Vessels (0-3) Colored by Flourosopy (ca)", min_value=0, max_value=3, step=1)
    thal = st.sidebar.slider("Thalassemia (thal)", min_value=0, max_value=3, step=1)

    model = load_model()
    st.markdown(
    """
    <style>
    .reportview-container .main .block-container{
        max-width: 100%;
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)
    
    st.title("Giải thích chi tiết các thuộc tính")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. age")
        st.write("Tuổi của bệnh nhân.")

        st.subheader("2. sex")
        st.write("Giới tính của bệnh nhân (1 = nam; 0 = nữ).")

        st.subheader("3. cp - chest pain type (Loại đau ngực)")
        st.write("""
        * **0: Typical angina (Đau thắt ngực điển hình)** - Đau ngực liên quan đến việc giảm lượng máu cung cấp cho tim.
        * **1: Atypical angina (Đau thắt ngực không điển hình)** - Đau ngực không liên quan đến tim.
        * **2: Non-anginal pain (Đau không đau thắt ngực)** - Điển hình là co thắt thực quản, không liên quan đến tim.
        * **3: Asymptomatic (Không có triệu chứng)** - Đau ngực nhưng không có dấu hiệu bệnh.
        """)

        st.subheader("4. trestbps - resting blood pressure (Huyết áp nghỉ)")
        st.write("""
        Huyết áp nghỉ (mm Hg khi nhập viện):
        * Bất kỳ giá trị nào trên 130-140 thường là nguyên nhân đáng lo ngại.
        """)

        st.subheader("5. chol - serum cholesterol (Lượng cholesterol trong huyết thanh)")
        st.write("""
        Lượng cholesterol trong huyết thanh (mg/dl):
        * Cholesterol huyết thanh = LDL + HDL + 0.2 * triglycerides.
        * Giá trị trên 200 là nguyên nhân đáng lo ngại.
        """)

        st.subheader("6. fbs - fasting blood sugar (Lượng đường trong máu khi đói)")
        st.write("""
        Lượng đường trong máu khi đói > 120 mg/dl (1 = đúng; 0 = sai):
        * '>126' mg/dl báo hiệu bệnh tiểu đường.
        """)
        st.subheader("7. restecg - resting electrocardiographic results (Kết quả điện tâm đồ khi nghỉ)")
        st.write("""
        * **0: Nothing to note** - Không có gì đáng chú ý.
        * **1: ST-T Wave abnormality** - Có thể từ các triệu chứng nhẹ đến các vấn đề nghiêm trọng, báo hiệu nhịp tim không bình thường.
        * **2: Possible or definite left ventricular hypertrophy** - Tăng kích thước của buồng bơm chính của tim.
        """)


    with col2:
        st.subheader("8. thalach - maximum heart rate achieved (Nhịp tim tối đa đạt được)")
        st.write("Nhịp tim tối đa đạt được.")

        st.subheader("9. exang - exercise induced angina (Đau thắt ngực khi tập thể dục)")
        st.write("Đau thắt ngực khi tập thể dục (1 = có; 0 = không).")

        st.subheader("10. oldpeak - ST depression (Sự suy giảm ST)")
        st.write("""
        Sự suy giảm ST do tập thể dục so với khi nghỉ:
        * Xem xét mức độ stress của tim khi tập thể dục, tim không khỏe sẽ stress nhiều hơn.
        """)

        st.subheader("11. slope - slope of the peak exercise ST segment (Độ dốc của đoạn ST đỉnh khi tập thể dục)")
        st.write("""
        * **0: Upsloping** - Tăng nhịp tim khi tập thể dục (không phổ biến).
        * **1: Flatsloping** - Thay đổi tối thiểu (tim khỏe mạnh điển hình).
        * **2: Downsloping** - Dấu hiệu của tim không khỏe mạnh.
        """)

        st.subheader("12. ca - number of major vessels (Số lượng các mạch chính)")
        st.write("""
        Số lượng các mạch chính được nhuộm màu bằng fluoroscopy (0-3):
        * Mạch được nhuộm màu có nghĩa là bác sĩ có thể thấy máu lưu thông qua.
        * Lưu thông máu nhiều hơn là tốt hơn (không có cục máu đông).
        """)

        st.subheader("13. thal - thalium stress result (Kết quả stress thallium)")
        st.write("""
        * **1,3: normal** - Bình thường.
        * **6: fixed defect** - Từng có khuyết tật nhưng hiện tại ổn.
        * **7: reversible defect** - Không có sự lưu thông máu đúng cách khi tập thể dục.
        """)

        st.subheader("14. target")
        st.write("Có bệnh hoặc không (1 = có; 0 = không) (đây là thuộc tính dự đoán).")


    st.title("Heart Disease Prediction")
    st.write("Please adjust the sliders on the left sidebar to input patient attributes.")

    if st.button("Predict"):
        prediction = predict_Heart_Disease(model, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
        st.write("Heart Disease Prediction:", "Positive" if prediction == 1 else "Negative")

        # # ROC Curve
        # X_test = np.load('X_test.npy')
        # y_test = np.load('y_test.npy')
        # y_prob = model.predict_proba(X_test)[:, 1]
        # fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        # roc_auc = auc(fpr, tpr)

        # st.title("Receiver Operating Characteristic (ROC) Curve")
        # st.write("Area under the curve (AUC):", roc_auc)

        # plt.figure()
        # plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver Operating Characteristic')
        # plt.legend(loc="lower right")
        # st.pyplot(plt)
