# prompt_template_ornek.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct", 
    temperature=0.3, 
    groq_api_key=os.getenv("GROQ_API_KEY"))

# --- Örnek 1: Basit PromptTemplate ---
kod_review_prompt = ChatPromptTemplate.from_messages([
    ("system", """Sen kıdemli bir yazılım geliştiricisisin. 
    Verilen kodu {dil} dili standartlarına göre incele.
    Odak noktaları: {odak_noktalar}
    Yanıtını Türkçe ver."""),
    ("human", "İncelenecek kod:\n\n```{dil}\n{kod}\n```")
])
parser = StrOutputParser()

# Chain oluşturma (LCEL - LangChain Expression Language)
chain = kod_review_prompt | llm

# Çalıştırma
sonuc = chain.invoke({
    "dil": "Python",
    "odak_noktalar": "güvenlik, performans, okunabilirlik",
    "kod": """
def kullanici_sil(user_id):
    query = f"DELETE FROM users WHERE id = {user_id}"
    db.execute(query)
    return True
    """
})
# print(sonuc)

print("Content:", sonuc.content)
# Genel metadata
# print("Response metadata:", sonuc.response_metadata)

token_usage = sonuc.response_metadata.get("token_usage", {})
print("Prompt tokens:", token_usage.get("prompt_tokens"))
print("Completion tokens:", token_usage.get("completion_tokens"))
print("Total tokens:", token_usage.get("total_tokens"))
