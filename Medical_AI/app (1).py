import google.generativeai as genai
import pandas as pd
import gradio as gr

# --- API KEY doğrudan değişkene atanıyor ---
GOOGLE_API_KEY = "AIzaSyBh8GzRRZHlGG5gy8qtE3PX0lZhYYGNjSQ"  # buraya kendi key'ini yaz

# Ayarları yap
genai.configure(api_key=GOOGLE_API_KEY)

# Veri dosyasını oku
df = pd.read_csv("medquad.csv")
df = df[['question', 'answer']].head(100)

# Soru-cevapları metin olarak hazırla
medical_data_text = "\n".join([f"Soru: {q}\nCevap: {a}" for q, a in zip(df['question'], df['answer'])])

# Modeli başlat
model = genai.GenerativeModel("models/gemini-2.5-flash-preview-04-17")

# Modeli kullanan cevaplama fonksiyonu
def cevapla(soru):
    prompt = f"""
    Aşağıda tıbbi soru-cevaplar var:
    {medical_data_text}
    Bu verilere dayanarak aşağıdaki soruyu cevapla. Eğer veri setinde ilgili bilgi yoksa, genel bilginle veya internet araması yaparak cevapla.
    Cevap verirken **sadece** tıp alanına odaklan.
    Soru: {soru}
    Cevabını kısa, anlaşılır ve sadece ilgili tıbbi terimlerle yaz.
    """
    try:
        yanit = model.generate_content(prompt)
        return yanit.text
    except Exception as e:
        return f"Hata oluştu: {e}"

# Arayüzü başlat
gr.Interface(fn=cevapla, inputs="text", outputs="text",
             title="Tıbbi Soru-Cevap Asistanı",
             description="Bu yapay zekâ tıbbi sorulara Gemini API ile yanıt verir.").launch()
