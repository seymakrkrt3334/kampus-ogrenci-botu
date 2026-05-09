# custom_tools.py
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.callbacks import BaseCallbackHandler
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()


# --- Araç 1: Hava Durumu ---
@tool
def hava_durumu_getir(sehir: str) -> str:
    """
    Verilen şehir için güncel hava durumu bilgisini getirir.
    Parametre: sehir - Türkçe şehir adı (örn: 'Ankara', 'İstanbul')
    """
    # Gerçek projede OpenWeatherMap API kullanılır
    # Demo için mock veri döndürüyoruz
    mock_veriler = {
        "ankara": {"sicaklik": 18, "durum": "Parçalı bulutlu", "nem": 45},
        "istanbul": {"sicaklik": 22, "durum": "Güneşli", "nem": 60},
        "izmir": {"sicaklik": 28, "durum": "Açık", "nem": 35}
    }
    sehir_lower = sehir.lower()
    if sehir_lower in mock_veriler:
        veri = mock_veriler[sehir_lower]
        return f"{sehir}: {veri['sicaklik']}°C, {veri['durum']}, Nem: %{veri['nem']}"
    return f"{sehir} için hava durumu verisi bulunamadı."

# --- Araç 2: Hesap Makinesi ---
@tool
def hesap_makinesi(ifade: str) -> str:
    """
    Matematiksel ifadeleri hesaplar.
    Parametre: ifade - matematiksel ifade (örn: '2 + 2', '15 * 7', '100 / 4')
    Güvenlik: Yalnızca sayısal işlemler kabul edilir.
    """
    try:
        # Güvenli eval - sadece temel operasyonlara izin ver
        izin_verilen = set('0123456789+-*/.() ')
        if not all(c in izin_verilen for c in ifade):
            return "Hata: Yalnızca temel matematiksel ifadeler kabul edilir."
        sonuc = eval(ifade)
        return f"{ifade} = {sonuc}"
    except Exception as e:
        return f"Hesaplama hatası: {str(e)}"

# --- Araç 3: Tarih/Saat ---
@tool
def bugunun_tarihi(format: str = "tam") -> str:
    """
    Bugünün tarih ve saat bilgisini döndürür.
    Parametre: format - 'tam' (tam tarih saat), 'tarih' (sadece tarih), 'saat' (sadece saat)
    """
    simdi = datetime.now()
    if format == "tarih":
        return simdi.strftime("%d %B %Y")
    elif format == "saat":
        return simdi.strftime("%H:%M:%S")
    else:
        return simdi.strftime("%d %B %Y, %H:%M:%S")

# --- Araç 4: Web Araması (DuckDuckGo) ---

web_arama = DuckDuckGoSearchRun()

# Tüm araçları listele
araclar = [hava_durumu_getir, hesap_makinesi, bugunun_tarihi, web_arama]

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

class KisaDebugCallback(BaseCallbackHandler):
    """Sadece tool çağrıları ve token kullanımını takip eder."""

    def __init__(self):
        self.tool_cagrilari = []
        self.tool_ciktilari = []
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def on_tool_start(self, serialized, input_str, **kwargs):
        arac_adi = serialized.get("name", "bilinmeyen_tool")
        self.tool_cagrilari.append(arac_adi)

    def on_tool_end(self, output, **kwargs):
        # Tool'dan dönen gerçek cevabı sakla; modelin sonradan değiştirip
        # değiştirmediğini debug çıktısında görmemizi sağlar.
        self.tool_ciktilari.append(str(output))

    def on_llm_end(self, response, **kwargs):
        # Farklı provider formatlarına uyumlu token okuma
        token_kullanimi = response.llm_output.get("token_usage", {}) if response.llm_output else {}
        self.prompt_tokens += token_kullanimi.get("prompt_tokens", 0)
        self.completion_tokens += token_kullanimi.get("completion_tokens", 0)
        self.total_tokens += token_kullanimi.get("total_tokens", 0)

        # Eğer llm_output boş gelirse AIMessage.usage_metadata'dan dene
        if self.total_tokens == 0 and response.generations:
            for nesil in response.generations:
                for uretim in nesil:
                    mesaj = getattr(uretim, "message", None)
                    kullanim = getattr(mesaj, "usage_metadata", None) if mesaj else None
                    if kullanim:
                        self.prompt_tokens += kullanim.get("input_tokens", 0)
                        self.completion_tokens += kullanim.get("output_tokens", 0)
                        self.total_tokens += kullanim.get("total_tokens", 0)

# Ajanı oluştur (LangChain v1 API)
ajan = create_agent(
    model=llm,
    tools=araclar,
    system_prompt=(
        "Sen yardımcı bir yapay zeka asistanısın. "
        "Gerektiğinde araçları kullanarak doğru ve kısa yanıt ver. "
        "tool çıktısını değiştirmeden aynen kullan. "
        "Bir tool çağrısı yaptıysan, tool çıktısındaki değerleri değiştirme; "
        "özellikle tarih/saat bilgilerini ve hesaplama sonucu gibi sayısal bilgileri aynen kullan."
    )
)

# Test sorguları
test_sorgulari = [
    "Ankara'nın hava durumu nedir? Ayrıca 15 * 23 + 7 hesapla.",
    "Bugünün tarihi ne?",
    "Python'ın en son sürümü nedir? Python'ın kendi orjinal web sitesinde bulunan son sürüm bilgisini ver."
]

for sorgu in test_sorgulari:
    debug_ozet = KisaDebugCallback()
    print(f"\n{'='*60}")
    print(f"SORGU: {sorgu}")
    print('='*60)
    sonuc = ajan.invoke(
        {"messages": [{"role": "user", "content": sorgu}]},
        config={"callbacks": [debug_ozet]}
    )
    son_mesaj = sonuc["messages"][-1].content
    print(f"\nSONUÇ: {son_mesaj}")
    print("\nDEBUG ÖZETİ")
    print(f"- Çağrılan tool'lar: {debug_ozet.tool_cagrilari if debug_ozet.tool_cagrilari else 'Yok'}")
    print(f"- Tool çıktıları: {debug_ozet.tool_ciktilari if debug_ozet.tool_ciktilari else 'Yok'}")
    print(
        f"- Token kullanımı: prompt={debug_ozet.prompt_tokens}, "
        f"completion={debug_ozet.completion_tokens}, total={debug_ozet.total_tokens}"
    )