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
# TÜR 3: Summary Memory (Geçmişi özetle)
# Avantaj: Uzun konuşmalar için ideal
# Dezavantaj: Her özetleme ekstra LLM çağrısı demek
# ─────────────────────────────────────────────
from langchain_core.messages import SystemMessage

ozet_prompt_sablonu = ChatPromptTemplate.from_messages([
    ("system", """Aşağıdaki konuşmayı kısaca özetle.
Özet 2-3 cümleyi geçmesin. Türkçe yaz.

Konuşma:
{konusma}

Özet:""")
])

ozet_store: dict[str, str] = {}  # session_id → metin özeti

def ozetli_session_getir(session_id: str) -> InMemoryChatMessageHistory:
    """
    Geçmiş varsa özetini SystemMessage olarak enjekte eder,
    böylece LLM uzun geçmiş yerine kısa özet görür.
    """
    gecmis = InMemoryChatMessageHistory()
    if session_id in ozet_store:
        gecmis.add_message(
            SystemMessage(content=f"Önceki konuşma özeti: {ozet_store[session_id]}")
        )
    return gecmis

def gecmisi_ozetle(session_id: str, yeni_mesajlar: list) -> None:
    """Konuşmayı LLM ile özetleyip store'a kaydeder."""
    ozet_chain = ozet_prompt_sablonu | llm
    konusma_metni = "\n".join(
        f"{'Kullanıcı' if m.type == 'human' else 'Asistan'}: {m.content}"
        for m in yeni_mesajlar
    )
    ozet = ozet_chain.invoke({"konusma": konusma_metni})
    ozet_store[session_id] = ozet.content

summary_chain = prompt | llm

summary_konusma = RunnableWithMessageHistory(
    summary_chain,
    ozetli_session_getir,
    input_messages_key="input",
    history_messages_key="history"
)

print("\n=== Summary Memory ===")
r1 = summary_konusma.invoke(
    {"input": "Python'da dekoratörler ne işe yarar?"},
    config={"configurable": {"session_id": "summary_session"}}
)
print(r1.content)

gecmisi_ozetle("summary_session", [r1])

r2 = summary_konusma.invoke(
    {"input": "Az önce hangi konudan bahsettik?"},
    config={"configurable": {"session_id": "summary_session"}}
)
print(r2.content)
# → Özetteki bilgiye dayanarak yanıt verir