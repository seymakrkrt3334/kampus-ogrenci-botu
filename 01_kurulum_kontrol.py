# kurulum_kontrol.py
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

load_dotenv()

# Model bağlantısı
llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # veya "llama3-8b-8192" (daha hızlı)
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Test mesajı
response = llm.invoke([
    SystemMessage(content="Sen yardımcı bir yazılım asistanısın."),
    HumanMessage(content="Python'da list comprehension nedir? Tek cümleyle açıkla.")
])

print(response.content)