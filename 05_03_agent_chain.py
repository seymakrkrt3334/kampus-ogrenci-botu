# 05_03_agent_chain.py — Tarif Asistanı (Agent + Chain Entegrasyonu)
#
# Mimari:
#   1. Pre-processing chain  → kullanıcı girdisini normalize eder
#   2. Agent (ReAct)         → araçları çağırır; 2 araç içinde chain barındırır
#   3. Post-processing chain → ham yanıtı biçimlendirir, şef notu ekler

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler
import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.5,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

parser = StrOutputParser()

# ---------------------------------------------------------------------------
# 1. PRE-PROCESSING CHAIN
#    Kullanıcı girdisini alır; malzemeleri çıkarır, niyeti netleştirir ve
#    temizlenmiş bir sorgu olarak agent'a aktarılacak metni üretir.
# ---------------------------------------------------------------------------
_normalize_prompt = ChatPromptTemplate.from_template(
    "Aşağıdaki kullanıcı sorgusunu analiz et ve şu formatta yanıt ver:\n\n"
    "NIYET: <tarif_iste | alisveris_listesi | besin_degeri | genel>\n"
    "MALZEMELER: <virgülle ayrılmış malzeme listesi veya 'belirtilmemiş'>\n"
    "TEMIZ_SORGU: <orijinal anlamı bozmadan düzeltilmiş, net Türkçe sorgu>\n\n"
    "Kullanıcı sorgusu:\n{ham_girdi}"
)

on_isleme_chain = _normalize_prompt | llm | parser


# ---------------------------------------------------------------------------
# 2. ARAÇLAR
# ---------------------------------------------------------------------------

# --- Araç 1: Tarif Oluştur (Chain-as-Tool) ---
@tool
def tarif_olustur(malzemeler: str) -> str:
    """
    Verilen malzemelere göre pratik bir yemek tarifi oluşturur.
    Parametre: malzemeler - virgülle ayrılmış malzeme listesi
               Örnek: 'patates, soğan, domates, yumurta'
    """
    _tarif_prompt = ChatPromptTemplate.from_template(
        "Elinde şu malzemeler var: {malzemeler}\n\n"
        "Bu malzemelerle yapılabilecek EN UYGUN bir Türk mutfağı yemeği tarifi yaz.\n"
        "Yanıt şu bölümleri içersin:\n"
        "- Yemek Adı\n"
        "- Kaç kişilik\n"
        "- Hazırlık + Pişirme süresi\n"
        "- Malzemeler (miktar bilgisiyle)\n"
        "- Yapılış adımları (numaralı)\n"
        "- Tahmini kalori (porsiyon başı)\n"
        "Türkçe yaz."
    )
    tarif_chain = _tarif_prompt | llm | parser
    return tarif_chain.invoke({"malzemeler": malzemeler})


# --- Araç 2: Besin Değeri Hesapla (Saf Hesap Aracı) ---
_BESIN_TABLOSU = {
    "tavuk": {"kalori": 165, "protein": 31, "yag": 3.6, "karbonhidrat": 0},
    "tavuk göğsü": {"kalori": 165, "protein": 31, "yag": 3.6, "karbonhidrat": 0},
    "et": {"kalori": 250, "protein": 26, "yag": 15, "karbonhidrat": 0},
    "kıyma": {"kalori": 250, "protein": 26, "yag": 15, "karbonhidrat": 0},
    "yumurta": {"kalori": 155, "protein": 13, "yag": 11, "karbonhidrat": 1.1},
    "süt": {"kalori": 61, "protein": 3.2, "yag": 3.3, "karbonhidrat": 4.8},
    "patates": {"kalori": 77, "protein": 2, "yag": 0.1, "karbonhidrat": 17},
    "domates": {"kalori": 18, "protein": 0.9, "yag": 0.2, "karbonhidrat": 3.9},
    "soğan": {"kalori": 40, "protein": 1.1, "yag": 0.1, "karbonhidrat": 9.3},
    "makarna": {"kalori": 371, "protein": 13, "yag": 1.5, "karbonhidrat": 74},
    "pirinç": {"kalori": 350, "protein": 6.5, "yag": 0.5, "karbonhidrat": 78},
    "ekmek": {"kalori": 265, "protein": 9, "yag": 3.2, "karbonhidrat": 49},
    "mercimek": {"kalori": 116, "protein": 9, "yag": 0.4, "karbonhidrat": 20},
    "zeytinyağı": {"kalori": 884, "protein": 0, "yag": 100, "karbonhidrat": 0},
    "peynir": {"kalori": 350, "protein": 25, "yag": 27, "karbonhidrat": 1.3},
    "salatalık": {"kalori": 15, "protein": 0.7, "yag": 0.1, "karbonhidrat": 3.6},
    "biber": {"kalori": 31, "protein": 1, "yag": 0.3, "karbonhidrat": 6},
    "havuç": {"kalori": 41, "protein": 0.9, "yag": 0.2, "karbonhidrat": 10},
    "ıspanak": {"kalori": 23, "protein": 2.9, "yag": 0.4, "karbonhidrat": 3.6},
    "sarımsak": {"kalori": 149, "protein": 6.4, "yag": 0.5, "karbonhidrat": 33},
}


@tool
def besin_degeri_hesapla(malzeme: str, miktar_gram: str) -> str:
    """
    Belirtilen malzemenin belirtilen gram miktarı için besin değerlerini hesaplar.
    Parametre: malzeme     - besin değeri hesaplanacak malzeme (örn: 'tavuk', 'makarna')
    Parametre: miktar_gram - gram cinsinden miktar (örn: '100', '250')
    """
    try:
        gram = float(str(miktar_gram).replace(",", "."))
    except ValueError:
        return f"Hata: '{miktar_gram}' geçerli bir gram değeri değil."

    anahtar = malzeme.lower().strip()
    veri = _BESIN_TABLOSU.get(anahtar)

    if veri is None:
        # Kısmi eşleşme dene
        for k, v in _BESIN_TABLOSU.items():
            if anahtar in k or k in anahtar:
                veri = v
                anahtar = k
                break

    if veri is None:
        return (
            f"'{malzeme}' için besin verisi bulunamadı. "
            f"Desteklenen malzemeler: {', '.join(_BESIN_TABLOSU.keys())}"
        )

    oran = gram / 100
    satirlar = [
        f"Besin Değerleri — {malzeme.title()} ({gram:.0f} g):",
        f"  Kalori      : {veri['kalori'] * oran:.1f} kcal",
        f"  Protein     : {veri['protein'] * oran:.1f} g",
        f"  Yağ         : {veri['yag'] * oran:.1f} g",
        f"  Karbonhidrat: {veri['karbonhidrat'] * oran:.1f} g",
        f"  (100 g baz alınarak hesaplanmıştır)",
    ]
    return "\n".join(satirlar)


# --- Araç 3: Alışveriş Listesi Oluştur (Chain-as-Tool) ---
@tool
def alisveris_listesi(tarif_adi: str, kisi_sayisi: str = "4") -> str:
    """
    Belirtilen tarif için kişi sayısına göre alışveriş listesi oluşturur.
    Parametre: tarif_adi   - alışveriş listesi istenilen yemeğin adı (örn: 'mercimek çorbası')
    Parametre: kisi_sayisi - kaç kişilik olacağı (varsayılan: '4')
    """
    _liste_prompt = ChatPromptTemplate.from_template(
        "{tarif_adi} tarifini {kisi_sayisi} kişilik yapmak için alışveriş listesi hazırla.\n\n"
        "Listeyi şu formatta ver:\n"
        "ALISVERIS LISTESI — {tarif_adi} ({kisi_sayisi} kişilik)\n\n"
        "Temel Malzemeler:\n"
        "- <malzeme>: <miktar ve birim>\n"
        "...\n\n"
        "Baharat ve Yardımcı Malzemeler:\n"
        "- <malzeme>: <miktar ve birim>\n"
        "...\n\n"
        "Tahmini Market Maliyeti: <TL aralığı>\n\n"
        "Türkçe yaz, net ve kısa ol."
    )
    liste_chain = _liste_prompt | llm | parser
    return liste_chain.invoke({"tarif_adi": tarif_adi, "kisi_sayisi": kisi_sayisi})


# Tüm araçlar
araclar = [tarif_olustur, besin_degeri_hesapla, alisveris_listesi]


# ---------------------------------------------------------------------------
# DEBUG CALLBACK
# ---------------------------------------------------------------------------
class KisaDebugCallback(BaseCallbackHandler):
    """Tool çağrıları ve token kullanımını takip eder."""

    def __init__(self):
        self.tool_cagrilari = []
        self.tool_ciktilari = []
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.tool_cagrilari.append(serialized.get("name", "bilinmeyen_tool"))

    def on_tool_end(self, output, **kwargs):
        self.tool_ciktilari.append(str(output)[:120] + "..." if len(str(output)) > 120 else str(output))

    def on_llm_end(self, response, **kwargs):
        token_kullanimi = response.llm_output.get("token_usage", {}) if response.llm_output else {}
        self.prompt_tokens += token_kullanimi.get("prompt_tokens", 0)
        self.completion_tokens += token_kullanimi.get("completion_tokens", 0)
        self.total_tokens += token_kullanimi.get("total_tokens", 0)

        if self.total_tokens == 0 and response.generations:
            for nesil in response.generations:
                for uretim in nesil:
                    mesaj = getattr(uretim, "message", None)
                    kullanim = getattr(mesaj, "usage_metadata", None) if mesaj else None
                    if kullanim:
                        self.prompt_tokens += kullanim.get("input_tokens", 0)
                        self.completion_tokens += kullanim.get("output_tokens", 0)
                        self.total_tokens += kullanim.get("total_tokens", 0)


# ---------------------------------------------------------------------------
# AGENT
# ---------------------------------------------------------------------------
ajan = create_agent(
    model=llm,
    tools=araclar,
    system_prompt=(
        "Sen Türk mutfağı konusunda uzmanlaşmış bir tarif asistanısın. "
        "Kullanıcının malzemelerine göre tarif öner, alışveriş listesi hazırla "
        "ve besin değerlerini hesapla. "
        "Araçları uygun şekilde kullan ve sonuçları olduğu gibi aktar. "
        "Her zaman Türkçe yanıt ver."
    ),
)


# ---------------------------------------------------------------------------
# 3. POST-PROCESSING CHAIN
#    Agent'ın ham yanıtını alır; yapılandırılmış başlıklar ekler,
#    pişirme zorluğu/süre tahmini yapar ve kısa bir şef notu ekler.
# ---------------------------------------------------------------------------
_format_prompt = ChatPromptTemplate.from_template(
    "Aşağıdaki tarif asistanı yanıtını kullanıcıya sunmak için düzenle.\n\n"
    "Şunları ekle veya belirt (eğer yanıtta yoksa tahmin et):\n"
    "1. Zorluk Seviyesi: Kolay / Orta / Zor\n"
    "2. Toplam Süre: <pişirme + hazırlık süresi>\n"
    "3. Şef Notu: Tarifi daha lezzetli yapacak 1-2 cümlelik kısa ipucu\n\n"
    "Yanıtın formatı:\n"
    "---\n"
    "[Orijinal yanıt burada — değiştirme, sadece aşağıya ekle]\n\n"
    "Zorluk Seviyesi : <Kolay/Orta/Zor>\n"
    "Toplam Sure     : <süre>\n"
    "Sef Notu        : <ipucu>\n"
    "---\n\n"
    "Ham yanıt:\n{ham_yanit}"
)

son_isleme_chain = _format_prompt | llm | parser


# ---------------------------------------------------------------------------
# TAM PIPELINE
#    pre_chain → agent.invoke → post_chain
# ---------------------------------------------------------------------------
def tam_pipeline(kullanici_girdisi: str) -> tuple[str, KisaDebugCallback]:
    """
    Kullanıcı girdisini 3 aşamadan geçirir ve son yanıtı döner.
    Ayrıca debug bilgisi için KisaDebugCallback nesnesini de döner.
    """
    # Aşama 1 — Ön-işleme
    normalize_sonuc = on_isleme_chain.invoke({"ham_girdi": kullanici_girdisi})

    # Normalize edilmiş sorguyu çıkar (TEMIZ_SORGU satırı)
    temiz_sorgu = kullanici_girdisi  # yedek: orijinal girdi
    for satir in normalize_sonuc.splitlines():
        if satir.startswith("TEMIZ_SORGU:"):
            temiz_sorgu = satir.replace("TEMIZ_SORGU:", "").strip()
            break

    # Aşama 2 — Agent
    debug = KisaDebugCallback()
    agent_sonuc = ajan.invoke(
        {"messages": [{"role": "user", "content": temiz_sorgu}]},
        config={"callbacks": [debug]},
    )
    ham_yanit = agent_sonuc["messages"][-1].content

    # Aşama 3 — Son-işleme
    son_yanit = son_isleme_chain.invoke({"ham_yanit": ham_yanit})

    return son_yanit, debug


# ---------------------------------------------------------------------------
# TEST SORGULARI
# ---------------------------------------------------------------------------
test_sorgulari = [
    "Evde patates, soğan, domates ve yumurta var. Ne pişirebilirim?",
    "Mercimek çorbası için alışveriş listesi lazım, 4 kişilik",
    "100 gram tavuk göğsünün besin değerleri nedir?",
    "Zeytinyağı, sarımsak ve makarna ile bir şey yap, tarifte kaç kalori var?",
]

for sorgu in test_sorgulari:
    print(f"\n{'='*65}")
    print(f"SORGU  : {sorgu}")
    print("="*65)

    sonuc, debug = tam_pipeline(sorgu)

    print(f"\nSONUC:\n{sonuc}")
    print("\nDEBUG OZETI")
    print(f"  Cagrilar  : {debug.tool_cagrilari if debug.tool_cagrilari else 'Yok'}")
    print(f"  Token     : prompt={debug.prompt_tokens}, "
          f"completion={debug.completion_tokens}, total={debug.total_tokens}")
