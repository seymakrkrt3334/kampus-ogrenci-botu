import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import trim_messages

load_dotenv()

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.7,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Session geçmişlerini RAM'de tutan sözlük
# Production'da Redis veya SQL ile değiştirilir
store: dict[str, InMemoryChatMessageHistory] = {}

def session_getir(session_id: str) -> InMemoryChatMessageHistory:
    """Session ID'ye göre geçmiş nesnesi döndürür, yoksa oluşturur."""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# ─────────────────────────────────────────────
# TÜR 2: Window Memory (Son N mesajı sakla)
# Avantaj: Token tasarrufu
# Dezavantaj: Eski mesajlar düşer
# ─────────────────────────────────────────────
# trim_messages: history'den yalnızca son k mesajı alır
WINDOW_SIZE = 3

window_trimmer = trim_messages(
    max_tokens=WINDOW_SIZE,  # token değil, mesaj sayısı için strategy="last" kullanılır
    strategy="last",     # en son mesajları tut
    token_counter=len,   # mesaj sayısını say (token sayacı yerine)
    include_system=False,
    allow_partial=False,
    start_on="human"     # human mesajıyla başla
)

window_prompt = ChatPromptTemplate.from_messages([
    ("system", "Sen yardımcı bir yazılım asistanısın. Türkçe yanıt ver."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# trim → prompt → llm zinciri
window_chain = (
    {
        "input": lambda x: x["input"],
        "history": lambda x: window_trimmer.invoke(x["history"])
    }
    | window_prompt
    | llm
)

window_konusma = RunnableWithMessageHistory(
    window_chain,
    session_getir,
    input_messages_key="input",
    history_messages_key="history"
)

print(f"\n=== Window Memory (son {WINDOW_SIZE} mesaj) ===")
for i in range(6):
    yanit = window_konusma.invoke(
        {"input": f"Bu {i+1}. mesajım."},
        config={"configurable": {"session_id": "window_session"}}
    )
    print(f"Mesaj {i+1}: {yanit.content[:60]}...")
