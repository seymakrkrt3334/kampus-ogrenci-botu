# 05_02_agent_uygulama.py — Kişisel Finans Asistanı Agent
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
import os
import requests
from dotenv import load_dotenv

load_dotenv()


_KUR_API_URL = "https://finans.truncgil.com/v4/today.json"
_KUR_HEADERS  = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    ),
    # Sıkıştırma devre dışı bırakılmazsa sunucu kısmi gzip yanıtı dönebilir
    # ve JSON parse hatası oluşur; identity zorlaması bunu önler.
    "Accept-Encoding": "identity",
    "Accept": "application/json",
}
_DESTEKLENEN = {"TL", "USD", "EUR", "GBP", "ALTIN"}


def _guncel_kurlar_getir() -> dict:
    """
    finans.truncgil.com API'sinden anlık USD, EUR, GBP alış kurlarını ve
    gram altın fiyatını TL cinsinden çeker.
    Hata durumunda boş dict döner.
    """
    try:
        resp = requests.get(_KUR_API_URL, timeout=5, headers=_KUR_HEADERS)
        resp.raise_for_status()
        data = resp.json()

        if not data or "USD" not in data:
            return {}

        # Gram altın için olası alan adlarını dene
        altin_tl = None
        for aday in ["Gram Altın", "Gram Altin", "GRA", "GA", "XAU", "GOLD"]:
            kayit = data.get(aday, {})
            fiyat = kayit.get("Buying") or kayit.get("Selling")
            if fiyat:
                altin_tl = float(fiyat)
                break

        guncelleme = data.get("Update_Date", "bilinmiyor")

        return {
            "TL":    1.0,
            "USD":   float(data["USD"]["Buying"]),
            "EUR":   float(data["EUR"]["Buying"]),
            "GBP":   float(data.get("GBP", {}).get("Buying", 0)) or None,
            "ALTIN": altin_tl,
            "tarih": guncelleme,
        }
    except Exception as e:
        return {"hata": str(e)}


# --- Araç 1: Döviz Çevirici ---
@tool
def doviz_cevirici(miktar: str, kaynak_para: str, hedef_para: str) -> str:
    """
    Verilen miktarı bir para biriminden diğerine anlık kurla çevirir.
    Parametre: miktar      - çevrilecek tutar (sayı, örn: '100' veya '250.5')
    Parametre: kaynak_para - kaynak para birimi (TL, USD, EUR, GBP, ALTIN)
    Parametre: hedef_para  - hedef para birimi  (TL, USD, EUR, GBP, ALTIN)
    Örnek: miktar='100', kaynak_para='USD', hedef_para='TL'
    Not: ALTIN = gram altın fiyatı (TL)
    """
    try:
        miktar_float = float(str(miktar).replace(",", "."))
    except ValueError:
        return f"Hata: '{miktar}' geçerli bir sayı değil."

    kaynak = kaynak_para.upper().strip()
    hedef  = hedef_para.upper().strip()

    if kaynak not in _DESTEKLENEN:
        return f"Hata: '{kaynak_para}' desteklenmiyor. Geçerli birimler: {', '.join(_DESTEKLENEN)}"
    if hedef not in _DESTEKLENEN:
        return f"Hata: '{hedef_para}' desteklenmiyor. Geçerli birimler: {', '.join(_DESTEKLENEN)}"

    kurlar = _guncel_kurlar_getir()

    if "hata" in kurlar:
        return f"API hatası: {kurlar['hata']}"
    if not kurlar:
        return "Kur verisi alınamadı. Lütfen tekrar deneyin."

    kaynak_kur = kurlar.get(kaynak)
    hedef_kur  = kurlar.get(hedef)

    if kaynak_kur is None:
        return f"'{kaynak}' için kur verisi bulunamadı."
    if hedef_kur is None:
        return f"'{hedef}' için kur verisi bulunamadı."

    tl_karsiligi = miktar_float * kaynak_kur
    sonuc        = tl_karsiligi / hedef_kur
    birim_kur    = kaynak_kur / hedef_kur

    return (
        f"{miktar_float} {kaynak} = {sonuc:,.4f} {hedef}\n"
        f"(Kur: 1 {kaynak} = {birim_kur:.4f} {hedef} | Güncelleme: {kurlar.get('tarih', '?')})"
    )


# --- Araç 2: Faiz Hesaplayıcı ---
@tool
def faiz_hesapla(anapara: str, yillik_faiz_orani: str, yil: str, tur: str = "bilesik") -> str:
    """
    Basit veya bileşik faiz hesabı yapar.
    Parametre: anapara          - başlangıç tutarı TL (örn: '15000')
    Parametre: yillik_faiz_orani - yıllık faiz oranı yüzde olarak (örn: '30' → %30)
    Parametre: yil              - yatırım süresi yıl cinsinden (örn: '5')
    Parametre: tur              - 'bilesik' (bileşik faiz) veya 'basit' (basit faiz)
    """
    try:
        anapara_f = float(str(anapara).replace(",", "."))
        oran_f    = float(str(yillik_faiz_orani).replace(",", "."))
        yil_i     = int(float(str(yil)))
    except ValueError as e:
        return f"Hata: Geçersiz sayısal değer — {e}"

    oran = oran_f / 100

    if tur.lower() == "basit":
        faiz   = anapara_f * oran * yil_i
        toplam = anapara_f + faiz
        return (
            f"Basit Faiz Hesabı:\n"
            f"  Anapara       : {anapara_f:,.2f} TL\n"
            f"  Faiz oranı    : %{oran_f} / yıl\n"
            f"  Süre          : {yil_i} yıl\n"
            f"  Kazanılan faiz: {faiz:,.2f} TL\n"
            f"  Toplam tutar  : {toplam:,.2f} TL"
        )
    else:
        toplam = anapara_f * ((1 + oran) ** yil_i)
        faiz   = toplam - anapara_f
        return (
            f"Bileşik Faiz Hesabı:\n"
            f"  Anapara       : {anapara_f:,.2f} TL\n"
            f"  Faiz oranı    : %{oran_f} / yıl\n"
            f"  Süre          : {yil_i} yıl\n"
            f"  Kazanılan faiz: {faiz:,.2f} TL\n"
            f"  Toplam tutar  : {toplam:,.2f} TL"
        )


# --- Araç 3: Bütçe Analizi ---
@tool
def butce_analiz(gelir: str, giderler: str) -> str:
    """
    Aylık gelir ve giderleri analiz eder; tasarruf oranını ve önerileri döndürür.
    Parametre: gelir    - aylık toplam gelir TL (örn: '25000')
    Parametre: giderler - 'kategori:tutar' çiftlerini virgülle ayırarak gir
               Örnek: 'kira:8000, market:4000, fatura:1500, ulasim:2000'
    """
    try:
        gelir_f = float(str(gelir).replace(",", "."))

        gider_dict = {}
        for parca in giderler.split(","):
            parca = parca.strip()
            if ":" in parca:
                kategori, tutar_str = parca.split(":", 1)
                gider_dict[kategori.strip()] = float(tutar_str.strip())

        toplam_gider = sum(gider_dict.values())
        tasarruf     = gelir_f - toplam_gider
        tasarruf_orani = (tasarruf / gelir_f * 100) if gelir_f > 0 else 0

        satirlar = [
            "Bütçe Analizi:",
            f"  Aylık gelir   : {gelir_f:,.2f} TL",
            "  Giderler:",
        ]
        for kat, tutar in gider_dict.items():
            oran = tutar / gelir_f * 100
            satirlar.append(f"    - {kat:<12}: {tutar:>8,.2f} TL  (%{oran:.1f})")

        satirlar += [
            f"  Toplam gider  : {toplam_gider:,.2f} TL",
            f"  Net tasarruf  : {tasarruf:,.2f} TL  (%{tasarruf_orani:.1f})",
        ]

        if tasarruf_orani >= 20:
            oneri = "Tasarruf oranınız iyi. Birikimi enstrümanlara yönlendirmeyi düşünün."
        elif tasarruf_orani >= 10:
            oneri = "Tasarruf oranınız orta düzeyde. Küçük harcama kısıntıları büyük fark yaratır."
        elif tasarruf_orani > 0:
            oneri = "Tasarruf oranınız düşük. Öncelikli giderleri gözden geçirin."
        else:
            oneri = "Dikkat: Giderleriniz gelirinizi aşıyor! Acil bütçe revizyonu gerekli."

        satirlar.append(f"  Öneri         : {oneri}")
        return "\n".join(satirlar)

    except Exception as e:
        return f"Bütçe analiz hatası: {str(e)}"


# Tüm araçları listele
araclar = [doviz_cevirici, faiz_hesapla, butce_analiz]


# --- LLM ---
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)


# --- Debug Callback ---
class KisaDebugCallback(BaseCallbackHandler):
    """Sadece tool çağrıları ve token kullanımını takip eder."""

    def __init__(self):
        self.tool_cagrilari = []
        self.tool_ciktilari = []
        self.prompt_tokens     = 0
        self.completion_tokens = 0
        self.total_tokens      = 0

    def on_tool_start(self, serialized, input_str, **kwargs):
        arac_adi = serialized.get("name", "bilinmeyen_tool")
        self.tool_cagrilari.append(arac_adi)

    def on_tool_end(self, output, **kwargs):
        self.tool_ciktilari.append(str(output))

    def on_llm_end(self, response, **kwargs):
        token_kullanimi = response.llm_output.get("token_usage", {}) if response.llm_output else {}
        self.prompt_tokens     += token_kullanimi.get("prompt_tokens", 0)
        self.completion_tokens += token_kullanimi.get("completion_tokens", 0)
        self.total_tokens      += token_kullanimi.get("total_tokens", 0)

        if self.total_tokens == 0 and response.generations:
            for nesil in response.generations:
                for uretim in nesil:
                    mesaj   = getattr(uretim, "message", None)
                    kullanim = getattr(mesaj, "usage_metadata", None) if mesaj else None
                    if kullanim:
                        self.prompt_tokens     += kullanim.get("input_tokens", 0)
                        self.completion_tokens += kullanim.get("output_tokens", 0)
                        self.total_tokens      += kullanim.get("total_tokens", 0)


# --- Agent ---
ajan = create_agent(
    model=llm,
    tools=araclar,
    system_prompt=(
        "Sen kişisel finans konusunda uzmanlaşmış yardımcı bir yapay zeka asistanısın. "
        "Kur bilgisi veren(Dolar, Euro, GBP, Altın), Döviz çevirme, faiz hesaplama, bütçe analizi ve güncel finansal bilgi sağlama "
        "konularında araçları kullanarak kesin ve kısa yanıtlar ver. "
        "Tool çıktısındaki sayısal değerleri değiştirmeden aynen kullan. "
        "Yanıtlarını Türkçe ver."
    )
)


# --- Test Sorguları ---
test_sorgulari = [
    "100 dolar kaç TL eder?",
    #"15000 TL'yi yıllık %30 bileşik faizle 5 yıl yatırsam ne kadar olur?",
    #"Aylık gelir 25000 TL, giderler: kira:8000, market:4000, fatura:1500, ulasim:2000. Bütçemi analiz et.",
    "Bugün 1 dolar kaç TL ediyor?"
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
    print(f"- Çağrılan tool'lar : {debug_ozet.tool_cagrilari if debug_ozet.tool_cagrilari else 'Yok'}")
    print(f"- Tool çıktıları    : {debug_ozet.tool_ciktilari if debug_ozet.tool_ciktilari else 'Yok'}")
    print(
        f"- Token kullanımı   : prompt={debug_ozet.prompt_tokens}, "
        f"completion={debug_ozet.completion_tokens}, total={debug_ozet.total_tokens}"
    )
