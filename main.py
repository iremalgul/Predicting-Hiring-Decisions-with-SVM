from fastapi import FastAPI
from pydantic import BaseModel
from candidate_selector import CandidateSelector
import numpy as np

# FastAPI uygulamasını başlatıyoruz
app = FastAPI()


# Kullanıcıdan alınacak veri için Pydantic modelini oluşturuyoruz
class CandidateInput(BaseModel):
    experience_years: float
    technical_score: float


# CandidateSelector'ı başlatıyoruz
selector = CandidateSelector()

# Veri oluşturuluyor
selector.generate_data()

# Veriyi eğitim ve test olarak ayırıyoruz ve ölçekliyoruz
selector.split_and_scale()

# Kernel değişikliği ve parametre tuning (isteğe bağlı)
# İlk önce kernel ve parametreler ayarlanabilir
# Eğer hyperparameter tuning yapmak isterseniz:
selector.tune_hyperparameters() # GridSearchCV ile

# Modeli eğitiyoruz (kernel tipi seçilebilir)
selector.train_model(kernel_type='linear', C=1.0, gamma='scale')  # Değiştirilebilir: 'linear', 'rbf', 'poly', vb.

# Model değerlendirmesi
selector.evaluate_model()

# Karar sınırını görselleştiriyoruz
selector.plot_decision_boundary()

# FastAPI endpoint'i
@app.post("/predict/")
async def predict_candidate(input: CandidateInput):
    # Kullanıcıdan gelen veriyi alıyoruz ve tahmin yapıyoruz
    input_data = np.array([[input.experience_years, input.technical_score]])
    input_scaled = selector.scaler.transform(input_data)
    prediction = selector.model.predict(input_scaled)

    if prediction[0] == 1:
        result = "The candidate will NOT be hired."
    else:
        result = "The candidate will be HIRED."

    return {"result": result}