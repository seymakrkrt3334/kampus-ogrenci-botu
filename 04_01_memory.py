import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

load_dotenv()


llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.7,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Tüm türlerde ortak prompt yapısı:
# MessagesPlaceholder → geçmiş mesajların enjekte edileceği yer
prompt = ChatPromptTemplate.from_messages([
    ("system", "Sen yardımcı bir yazılım asistanısın. Türkçe yanıt ver."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Session geçmişlerini RAM'de tutan sözlük
# Production'da Redis veya SQL ile değiştirilir
store: dict[str, InMemoryChatMessageHistory] = {}

def session_getir(session_id: str) -> InMemoryChatMessageHistory:
    """Session ID'ye göre geçmiş nesnesi döndürür, yoksa oluşturur."""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# ─────────────────────────────────────────────
# TÜR 1: Buffer Memory (Tüm geçmişi sakla)
# Avantaj: Eksiksiz bağlam
# Dezavantaj: Konuşma uzadıkça token tüketimi artar
# ─────────────────────────────────────────────
buffer_chain = prompt | llm

buffer_konusma = RunnableWithMessageHistory(
    buffer_chain,
    session_getir,
    input_messages_key="input",
    history_messages_key="history"
)

print("================= Buffer Memory ================")
yanit = buffer_konusma.invoke(
    {"input": "Merhaba, ben Ahmet. Python öğreniyorum."},
    config={"configurable": {"session_id": "buffer_session"}}
)
print(yanit.content)
print("================================================")
yanit = buffer_konusma.invoke(
    {"input": "Bana listeler hakkında bir şey anlat."},
    config={"configurable": {"session_id": "buffer_session"}}
)
print(yanit.content)
print("================================================")
yanit = buffer_konusma.invoke(
    {"input": "Az önce kendimden bahsettim, adım neydi?"},
    config={"configurable": {"session_id": "buffer_session"}}
)
print(yanit.content)
# → "Az önce adınızın Ahmet olduğunu söylediniz."