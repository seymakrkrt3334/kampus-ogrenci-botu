import os
from datetime import date
from io import StringIO
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq


def _decode_env_bytes(raw: bytes) -> str:
    if raw.startswith(b"\xff\xfe"):
        return raw.decode("utf-16-le")
    if raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16-be")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return raw.decode("utf-16-le")
        except UnicodeDecodeError:
            return raw.decode("latin-1")


def _load_dotenv_robust() -> None:
    """Windows Notepad bazen .env'i UTF-16 ile kaydeder; dotenv dosyayi utf-8 sanip cokebilir."""
    roots = [Path(__file__).resolve().parent, Path.cwd()]
    seen: set[Path] = set()
    for root in roots:
        env_path = (root / ".env").resolve()
        if env_path in seen:
            continue
        seen.add(env_path)
        if not env_path.is_file():
            continue
        raw = env_path.read_bytes()
        if not raw.strip():
            continue
        text = _decode_env_bytes(raw)
        load_dotenv(stream=StringIO(text), interpolate=False)
        return


_load_dotenv_robust()

app = FastAPI(title="Kampus Ogrenci Botu")


class ChatRequest(BaseModel):
    ogrenci_id: str
    mesaj: str
    kampus_adi: str = "Genel Kampus"
    bolum: str = "Belirtilmedi"
    koc_modu: bool = False
    yemekhane_bilgisi_ekle: bool = False
    menu_gunu: Optional[str] = None
    ajan_modu: bool = True
    post_islem: bool = True


class MenuResponse(BaseModel):
    gun: str
    tarih_notu: str
    ogle: Dict[str, str]
    aksam: Dict[str, str]
    not_: str = ""


class ChatResponse(BaseModel):
    cevap: str


_ORNEK_YEMEKHANE_MENU: Dict[str, Dict[str, Dict[str, str]]] = {
    "pazartesi": {
        "ogle": {
            "corba": "Ezogelin çorbası",
            "ana_yemek": "Tavuk sote + bulgur pilavı",
            "yardimci": "Mevsim salatası",
            "tatli": "Yoğurt",
            "icecek": "Ayran",
        },
        "aksam": {
            "corba": "Mercimek çorbası",
            "ana_yemek": "Kuru fasulye + pirinç pilavı",
            "yardimci": "Turşu",
            "tatli": "Helva",
            "icecek": "Çay",
        },
    },
    "sali": {
        "ogle": {
            "corba": "Tarhana çorbası",
            "ana_yemek": "İzgara köfte + patates püresi",
            "yardimci": "Çoban salatası",
            "tatli": "Sütlaç",
            "icecek": "Şalgam",
        },
        "aksam": {
            "corba": "Yoğurt çorbası",
            "ana_yemek": "Sebzeli makarna",
            "yardimci": "Roka-domates",
            "tatli": "Meyve",
            "icecek": "Ayran",
        },
    },
    "carsamba": {
        "ogle": {
            "corba": "İşkembe çorbası",
            "ana_yemek": "Et güveç + baharatlı pilav",
            "yardimci": "Yeşil salata",
            "tatli": "Hoşaf",
            "icecek": "Ayran",
        },
        "aksam": {
            "corba": "Domates çorbası",
            "ana_yemek": "Balık (ızgarada) + roka",
            "yardimci": "Enginar kalbi",
            "tatli": "Irımık helvası",
            "icecek": "Limonata",
        },
    },
    "persembe": {
        "ogle": {
            "corba": "Mercimek çorbası",
            "ana_yemek": "Türlü + bulgur pilavı",
            "yardimci": "Cacık",
            "tatli": "Yoğurt",
            "icecek": "Ayran",
        },
        "aksam": {
            "corba": "Şehriye çorbası",
            "ana_yemek": "Lahmacun menü (salata ile)",
            "yardimci": "Soğan salatası",
            "tatli": "Künefe (porsiyon)",
            "icecek": "Ayran",
        },
    },
    "cuma": {
        "ogle": {
            "corba": "Yayla çorbası",
            "ana_yemek": "Perde pilavı",
            "yardimci": "Turşu",
            "tatli": "Baklava",
            "icecek": "Çay",
        },
        "aksam": {
            "corba": "Tarhana çorbası",
            "ana_yemek": "Izgara tavuk + garnitür",
            "yardimci": "Akdeniz salatası",
            "tatli": "Sütlaç",
            "icecek": "Ayran",
        },
    },
    "cumartesi": {
        "ogle": {
            "corba": "Domates çorbası",
            "ana_yemek": "Mantı + yoğurt",
            "yardimci": "Salata",
            "tatli": "Revani",
            "icecek": "Ayran",
        },
        "aksam": {
            "corba": "Mercimek çorbası",
            "ana_yemek": "Pizza dilimi + çorba",
            "yardimci": "Yeşillik",
            "tatli": "Meyve",
            "icecek": "Kola / su",
        },
    },
    "pazar": {
        "ogle": {
            "corba": "Ezogelin çorbası",
            "ana_yemek": "Nohut yemeği + pilav",
            "yardimci": "Turşu",
            "tatli": "İrmik helvası",
            "icecek": "Ayran",
        },
        "aksam": {
            "corba": "Sebze çorbası",
            "ana_yemek": "Köri soslu tavuk + pilav",
            "yardimci": "Coleslaw",
            "tatli": "Magnolia",
            "icecek": "Çay",
        },
    },
}

_TR_GUN_TO_ANAHTAR = {
    "pazartesi": "pazartesi",
    "salı": "sali",
    "sali": "sali",
    "çarşamba": "carsamba",
    "carsamba": "carsamba",
    "perşembe": "persembe",
    "persembe": "persembe",
    "cuma": "cuma",
    "cumartesi": "cumartesi",
    "pazar": "pazar",
}

_KOC_MODU_METNI = (
    "KOÇ MODU AÇIK. Bu mesajda sınav dönemi ve çalışma koçu gibi davran. "
    "Öğrenciye küçük, ölçülebilir hedefler (ör. bugün 3 zaman blokları, her blokta tek konu) öner. "
    "Pomodoro, molalar, uyku düzeni ve odak için pratik ipuçları ver. "
    "Eleştirici olma; destekle ve ilerlemeyi görünür kıl. "
    "İstenirse 1 haftalık veya sınav haftası için örnek şablon sun."
)


def _normalize_gun(raw: str) -> str:
    key = raw.strip().lower()
    if key in _TR_GUN_TO_ANAHTAR:
        return _TR_GUN_TO_ANAHTAR[key]
    if key in _ORNEK_YEMEKHANE_MENU:
        return key
    raise HTTPException(
        status_code=400,
        detail=(
            "Gecersiz gun. Ornek: pazartesi, sali, carsamba, persembe, "
            "cuma, cumartesi, pazar"
        ),
    )


def _bugunun_gun_anahtari() -> str:
    wd = date.today().weekday()
    keys = [
        "pazartesi",
        "sali",
        "carsamba",
        "persembe",
        "cuma",
        "cumartesi",
        "pazar",
    ]
    return keys[wd]


def menu_verisi(gun_anahtari: str) -> MenuResponse:
    if gun_anahtari not in _ORNEK_YEMEKHANE_MENU:
        raise HTTPException(
            status_code=400,
            detail=(
                "Gecersiz gun. Ornek: pazartesi, sali, carsamba, persembe, "
                "cuma, cumartesi, pazar"
            ),
        )
    data = _ORNEK_YEMEKHANE_MENU[gun_anahtari]
    return MenuResponse(
        gun=gun_anahtari,
        tarih_notu="Ornek/demo menu; gercek liste kampuste duyurulan menudur.",
        ogle=data["ogle"],
        aksam=data["aksam"],
        not_="Resmi menü için kampus yemekhane / mobil uygulama duyurusunu takip edin.",
    )


def _menu_metni_format(menu: MenuResponse) -> str:
    satirlar = [
        f"YEMEKHANE MENÜSÜ ({menu.gun}) — örnek veri:",
        "Öğle: "
        + ", ".join(f"{k}: {v}" for k, v in menu.ogle.items()),
        "Akşam: "
        + ", ".join(f"{k}: {v}" for k, v in menu.aksam.items()),
        menu.not_,
    ]
    return "\n".join(satirlar)


def _ekstra_baglam(req: ChatRequest) -> str:
    parcalar = []
    if req.koc_modu:
        parcalar.append(_KOC_MODU_METNI)
    if req.yemekhane_bilgisi_ekle:
        mg = (req.menu_gunu or "").strip()
        gun_key = _normalize_gun(mg) if mg else _bugunun_gun_anahtari()
        m = menu_verisi(gun_key)
        parcalar.append(_menu_metni_format(m))
    if not parcalar:
        return "Ek ozel mod yok."
    return "\n\n".join(parcalar)


def _agent_user_content(req: ChatRequest, mesaj_override: Optional[str] = None) -> str:
    """Ajana gidecek tam kullanici paketi (baglam + modlar + mesaj)."""
    metin = (mesaj_override if mesaj_override is not None else req.mesaj).strip()
    parts = [
        f"Ogrenci baglami — Kampus: {req.kampus_adi}, Bolum: {req.bolum}.",
    ]
    if req.koc_modu:
        parts.append(_KOC_MODU_METNI)
    if req.yemekhane_bilgisi_ekle:
        mg = (req.menu_gunu or "").strip()
        gun_key = _normalize_gun(mg) if mg else _bugunun_gun_anahtari()
        parts.append(_menu_metni_format(menu_verisi(gun_key)))
    parts.append(f"Ogrenci mesaji:\n{metin}")
    return "\n\n".join(parts)


_histories: Dict[str, InMemoryChatMessageHistory] = {}


def _get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _histories:
        _histories[session_id] = InMemoryChatMessageHistory()
    return _histories[session_id]


groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    raise RuntimeError("GROQ_API_KEY bulunamadi. Lutfen .env dosyasina ekleyin.")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    groq_api_key=groq_key,
)

parser = StrOutputParser()

# --- Toolar (hocanın örnekleriyle aynı @tool deseni) ---


@tool
def yemekhane_menu_araci(gun: str = "") -> str:
    """Öğle ve akşam örnek yemekhane menüsünü döndürür. gun boşsa bugün; örn: carsamba, persembe."""
    try:
        g = (gun or "").strip()
        key = _bugunun_gun_anahtari() if not g else _normalize_gun(g)
        return _menu_metni_format(menu_verisi(key))
    except HTTPException as e:
        return f"Menü alınamadı: {e.detail}"


_calisma_plani_prompt = ChatPromptTemplate.from_template(
    "Konu: {konu}\n"
    "Toplam çalışma süresi (saat): {sure_saat}\n\n"
    "Bu konu için AYRINTILI pomodoro uyumlu çalışma planı yaz (Türkçe):\n"
    "- Oturum süreleri ve kaç pomodoro olduğu\n"
    "- Her oturumda net alt hedefler (ör. hangi konu/bölüm)\n"
    "- Kısa/mola uzun mola önerileri\n"
    "- Günün sonunda tekrar ve özet için 10–15 dk\n"
    "Numaralı liste ve alt maddeler kullan."
)


@tool
def calisma_plani_araci(konu: str, sure_saat: str = "3") -> str:
    """Verilen ders/konusu için saat bazlı çalışma planı üretir (içinde LCEL zinciri: prompt | llm | parser)."""
    zincir = _calisma_plani_prompt | llm | parser
    return zincir.invoke({"konu": konu.strip(), "sure_saat": sure_saat.strip() or "3"})


@tool
def kampuste_servis_saatleri() -> str:
    """Kütüphane, yemekhane, öğrenci işleri, danışmanlık için örnek saat ve yerler (demo veri)."""
    return (
        "=== ÖRNEK KAMPÜS SERVİS — gerçek saatler için okul duyurusu / OBS ===\n\n"
        "Merkez Kütüphane:\n"
        "  Hafta içi 08:30–22:00 | Cumartesi 09:00–18:00 | Pazar kapalı veya kısıtlı (kampüse göre değişir)\n"
        "  Zemin: ödünç / danışma | 1. kat: sessiz çalışma | Üst katlar: grup çalışma kabinleri (rezervasyonlu olabilir)\n\n"
        "Yemekhane:\n"
        "  Öğle ~11:30–14:00 | Akşam ~17:00–19:30 | Kahvalı bazı kampüslerde 07:30–10:00\n\n"
        "Öğrenci İşleri:\n"
        "  Genelde hafta içi 09:00–12:00 ve 13:00–16:30 | Yoğun dönemlerde ek sıra sistemi\n\n"
        "Psikolojik Danışmanlık / Rehberlik:\n"
        "  Randevu ile | Öğrenci portalından veya birim telefonundan\n\n"
        "BTK / Kampüs ağı:\n"
        "  Eduroam veya kurumsal Wi‑Fi | Şifre sıfırlama: bilgi işlem birimi\n"
    )


_ORNEK_DERS_TABLOSU: Dict[str, list] = {
    "pazartesi": [
        "09:00–09:50 | MAT 107 — Matematik | Amfi A | Prof.",
        "10:00–10:50 | FIZ 101 — Fizik | Lab-1 | Öğr. Gör.",
        "11:00–11:50 | BLM 101 — Programlamaya Giriş | D109 | Dr.",
        "13:30–15:10 | BLM 103 — Veri Yapıları | Bilgisayar Lab | Dr.",
        "15:20–17:00 | İNG 101 — İngilizce | D205 | Öğr. Gör.",
    ],
    "sali": [
        "09:00–09:50 | MAT 107 — Matematik | D301 | Prof.",
        "10:00–11:40 | BLM 105 — Ayrık Matematik | Amfi B | Dr.",
        "13:30–15:10 | SEÇMELİ — Sosyoloji | D120 | Dr.",
        "15:20–16:10 | ATA 101 — Atatürk İlkeleri | Amfi A | Öğr. Gör.",
    ],
    "carsamba": [
        "09:00–10:40 | FIZ 101 — Fizik | Amfi C | Prof.",
        "11:00–11:50 | MAT 107 — Matematik | D301 | Öğr. Gör.",
        "13:30–15:10 | BLM 107 — Nesne Yönelimli Programlama | Lab-2 | Dr.",
        "15:20–17:00 | SEÇMELİ — İş Sağlığı | D210 | Öğr. Gör.",
    ],
    "persembe": [
        "09:00–09:50 | BLM 101 — Programlamaya Giriş | Lab-1 | Dr.",
        "10:00–11:40 | MAT 107 — Matematik | Amfi A | Prof.",
        "13:30–14:20 | İNG 101 — İngilizce | D205 | Öğr. Gör.",
        "14:30–16:10 | BLM 109 — Mantık Devreleri | Lab-3 | Dr.",
    ],
    "cuma": [
        "09:00–10:40 | BLM 103 — Veri Yapıları | Bilgisayar Lab | Dr.",
        "11:00–11:50 | FIZ 101 — Fizik | Lab-1 | Öğr. Gör.",
        "13:30–15:10 | SEÇMELİ — Girişimcilik | Amfi D | Dr.",
        "15:20–16:10 | REHBERLİK — Bölüm danışmanlığı | Ofis saatleri | Randevulu",
    ],
    "cumartesi": [
        "(Örnek) Boş veya laboratuvar telafi — kampüs politikasına göre değişir.",
    ],
    "pazar": [
        "(Örnek) Genelde ders yok; etkinlik veya kulüp çalışması olabilir.",
    ],
}


@tool
def ornek_ders_programi(bolum: str, haftanin_gunu: str = "") -> str:
    """
    Örnek haftalık ders programı tablosu (DEMO — gerçek program OBS / danışmandır).
    Parametre bolum: örn. Bilgisayar Mühendisliği, Elektrik Mühendisliği.
    Parametre haftanin_gunu: boşsa tüm hafta; veya pazartesi, sali, carsamba...
    """
    bolum_aciklama = (bolum or "Belirtilmedi").strip()
    gun_raw = (haftanin_gunu or "").strip()
    satirlar = [
        f"=== ÖRNEK DERS PROGRAMI (demo) — Bölüm bağlamı: {bolum_aciklama} ===",
        "Uyarı: Saatler ve kodlar örnektir; kendi programınızı OBS / öğrenci bilgi sisteminden doğrulayın.",
        "",
    ]
    if "bilgisayar" in bolum_aciklama.lower():
        satirlar.append("Not: Mühendislik çekirdek + BLM kodlu ders örnekleri gösteriliyor.")
    elif "elektrik" in bolum_aciklama.lower():
        satirlar.append("Not: Örnek yapı mühendislik çekirdeğe benzer; ELE kodları için danışmanınıza bakın.")
    else:
        satirlar.append("Not: Genel mühendislik benzeri örnek slotlar; bölümünüze göre ders adları değişir.")
    satirlar.append("")

    if gun_raw:
        try:
            gun_key = _normalize_gun(gun_raw)
        except HTTPException:
            return (
                f"Geçersiz gün: '{gun_raw}'. Örnek: pazartesi, sali, carsamba, "
                "persembe, cuma — veya haftanin_gunu boş bırakarak tüm haftayı isteyin."
            )
        gunler = [gun_key]
    else:
        gunler = list(_ORNEK_DERS_TABLOSU.keys())

    for gun in gunler:
        satirlar.append(f"--- {gun.upper()} ---")
        for sat in _ORNEK_DERS_TABLOSU.get(gun, ["(Bu gün için örnek satır yok.)"]):
            satirlar.append(f"  • {sat}")
        satirlar.append("")
    satirlar.append(
        "Danışmanlık / program düzeltme: Bölüm sekreterliği veya OBS üzerinden ders kayıt ekranı."
    )
    return "\n".join(satirlar)


@tool
def kampu_yasam_rehberi(konu: str = "") -> str:
    """
    Kampüs ve öğrencilik için ayrıntılı ÖRNEK rehber (demo).
    konu: sınav | kulup | kayit | danisman | dijital | hepsi (boş veya hepsi = tamamı).
    """
    k = (konu or "hepsi").strip().lower()
    parcalar = []

    def blok_sinav() -> str:
        return (
            "SINAV DÖNEMİ:\n"
            "  • Takvim: OBS / akademik takvimden ara sınav ve final tarihlerini işleyin.\n"
            "  • Çalışma: Konu listesi → önce zayıf olduğunuz başlıklar → eski sınav örnekleri (varsa).\n"
            "  • Haklar: Mazeret ve sınav iptali kuralları öğrenci el kitabında.\n"
            "  • Mekân: Sınav salonu listesi duyurusu genelde sınavdan 1 hafta önce.\n"
        )

    def blok_kulup() -> str:
        return (
            "KULÜP VE ETKİNLİK:\n"
            "  • Örnek kulüpler: IEEE, Robotik, Müzik, Drama, Gönüllülük, Spor branşları.\n"
            "  • Üyelik: Kulüp tanıtım günleri, Instagram duyuruları veya öğrenci konseyi.\n"
            "  • Katılım belgesi bazı etkinliklerde transkripte işlenebilir (kampüse göre).\n"
        )

    def blok_kayit() -> str:
        return (
            "DERS KAYIT / OBS:\n"
            "  • Dönem başında danışman onayı ile ders seçimi; çakışma kontrolü yapın.\n"
            "  • Harç / yoklama: Mali işler ve devamsızlık limitleri için duyuruları takip edin.\n"
            "  • Yandal / çift ana dal varsa ek kredi üst sınırına dikkat.\n"
        )

    def blok_danisman() -> str:
        return (
            "DANIŞMANLIK:\n"
            "  • Akademik danışman: Program akışı, seçmeli seçimi, staj yönergesi.\n"
            "  • Kariyer merkezi: CV, staj ilanları, mock interview.\n"
            "  • Psikolojik destek: Randevu ile gizlilik içinde.\n"
        )

    def blok_dijital() -> str:
        return (
            "DİJİTAL ARAÇLAR:\n"
            "  • OBS / LMS: Notlar, duyurular, ödev teslimi.\n"
            "  • Kurumsal e-posta: Resmi yazışma ve sınav bildirimleri için kullanın.\n"
            "  • Kampüs Wi‑Fi ve VPN (varsa) kütüphane veri tabanlarına erişim için.\n"
        )

    hepsi = ("hepsi", "tümü", "tumu", "all", "")
    if k in hepsi or not k:
        parcalar.extend(
            [blok_sinav(), blok_kayit(), blok_kulup(), blok_danisman(), blok_dijital()]
        )
    else:
        if "sınav" in k or "sinav" in k:
            parcalar.append(blok_sinav())
        if "kulüp" in k or "kulup" in k or "etkinlik" in k:
            parcalar.append(blok_kulup())
        if "kayıt" in k or "kayit" in k or "obs" in k:
            parcalar.append(blok_kayit())
        if "danışman" in k or "danisman" in k or "rehber" in k:
            parcalar.append(blok_danisman())
        if "dijital" in k or "mail" in k or "lms" in k:
            parcalar.append(blok_dijital())
        if not parcalar:
            parcalar.extend(
                [
                    blok_sinav(),
                    blok_kayit(),
                    blok_kulup(),
                    blok_danisman(),
                    blok_dijital(),
                ]
            )

    return (
        "=== KAMPÜS YAŞAMI REHBERİ (örnek / demo) ===\n\n"
        + "\n".join(parcalar)
        + "\nResmi işlemler için mutlaka kurum web sitesi ve OBS bilgilerini kullanın."
    )


kampus_araclari = [
    yemekhane_menu_araci,
    calisma_plani_araci,
    kampuste_servis_saatleri,
    ornek_ders_programi,
    kampu_yasam_rehberi,
]

KAMPUS_TOOL_OZET_METNI = """
Bu projede kullanilan LangChain TOOLLAR (ajan modunda):
  1. yemekhane_menu_araci      — Ornek ogle/aksam menusu (gun secilebilir).
  2. calisma_plani_araci       — Konuya gore ayrintili pomodoro calisma plani (icinde LCEL zincir).
  3. kampuste_servis_saatleri  — Kutuphane, yemekhane, ogrenci isleri vb. ornek saatler.
  4. ornek_ders_programi       — DEMO haftalik ders tablosu (bolum + istege bagli gun).
  5. kampu_yasam_rehberi       — Sinav, kayit, kulup, danismanlik, dijital konularda ornek rehber.

Zincirler: agent_hazirlik_chain | prompt|llm (sohbet) | post_process_chain
""".strip()

# Ajan öncesi kısa netleştirme zinciri (LCEL — hocanın pipeline mantığına paralel)
_agent_hazirlik_prompt = ChatPromptTemplate.from_template(
    "Öğrenci ({kampus}, {bolum}) tek bir mesaj/soru yazmış:\n{mesaj}\n\n"
    "Ajanda kullanılacak TEK istemi çıkar (Türkçe):\n"
    "- Öğrencinin asıl SORUSUNU veya TALEBİNİ kelimesi kelimesine koru.\n"
    "- Ekstra konu veya varsayılan soru EKLEME.\n"
    "- Varsa tek paragrafa sığdır."
)
agent_hazirlik_chain = _agent_hazirlik_prompt | llm | parser

kampus_ajan = create_agent(
    model=llm,
    tools=kampus_araclari,
    system_prompt=(
        "Sen kampüsteki öğrenciye yardım eden bir ajansın. "
        "Öğrencinin SORUSUNU eksiksiz yanıtla: mümkünse alt başlıklar, madde işaretleri ve somut örnekler kullan. "
        "Tek araç yetmezse aynı soru için birden fazla aracı sırayla kullan (ör. menü + servis saatleri; program + rehber). "
        "Menü → yemekhane_menu_araci | Çalışma planı → calisma_plani_araci | Servis/yerler → kampuste_servis_saatleri | "
        "Ders programı örneği → ornek_ders_programi(bolum ve istenirse haftanin_gunu) | "
        "Sınav/kayıt/kulüp/danışman/dijital → kampu_yasam_rehberi(konu). "
        "Soru tam kampüs konusu değilse kibarca sınırla veya tek netleştirici soru sor. "
        "Araç çıktılarındaki sayı/saat/program DEMO ise bunu kullanıcıya kısaca hatırlat; içeriği uydurma. "
        "Yanıtını Türkçe ver."
    ),
)

# Post-processing (05_03_agent_chain.py'deki son_isleme_chain ile aynı mantık: kullanıcıya EN SON bunu gösterirsin)
_post_process_prompt = ChatPromptTemplate.from_template(
    "Aşağıdaki metin kampüs asistanının ham yanıtıdır (araç çıktıları dahil).\n\n"
    "Öğrenci bağlamı — Kampus: {kampus}, Bölüm: {bolum}\n\n"
    "HAM YANIT:\n{ham_yanit}\n\n"
    "Görevin bu metni ÖĞRENCİYE gösterilecek son sürüme çevirmek:\n"
    "- Sorunun tamamını karşıla; eksik başlık varsa ham metinden tamamla (yeni saat/ders UYDURMA).\n"
    "- İlgisiz tekrarları kısalt ama faydalı ayrıntıları koru; alt başlık satırları kullan (ör. BASLIK:).\n"
    "- İçeriği ve araçtan gelen sayıları UYDURMA; sadece düzenle.\n"
    "- İlk paragraf sorunun özünü net yanıtlasın.\n"
    "- Ana bilgiyi madde işaretleri veya numaralı liste ile okunur yap.\n"
    "- Uygunsa en alta tek satır 'Özet: ...' ekle.\n"
    "- Türkçe yaz.\n\n"
    "SON METİN:"
)
post_process_chain = _post_process_prompt | llm | parser

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Sen ogrenciye destek veren kampus asistanisin. "
            "Her mesajda ogrencinin SORDUGU konuya odaklan; yaniti mumkun oldugunca AYRINTILI ver "
            "(alt basliklar, maddeler, ornekler). Sorulmayan konuda gereksiz sapma. "
            "Kampus hayati, ders programi, kulup etkinlikleri, kutuphane, yemekhane, "
            "sinav donemi planlama ve motivasyon konularinda yardimci ol. "
            "Net ve uygulanabilir tavsiyeler ver; bilinmeyeni soyle. "
            "Bilmedigin bilgiyi uydurma, emin degilsen bunu acikca belirt. "
            "Her zaman Turkce yanit ver.",
        ),
        (
            "system",
            "Ogrenci baglami: kampus={kampus_adi}, bolum={bolum}. "
            "Yaniti bu baglama uygunlastir.",
        ),
        (
            "system",
            "Ek talimatlar / mod bilgisi:\n{ekstra_baglam}",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{mesaj}"),
    ]
)

chain = prompt | llm
chatbot = RunnableWithMessageHistory(
    chain,
    _get_history,
    input_messages_key="mesaj",
    history_messages_key="history",
)


def _kullaniciya_son_metin(req: ChatRequest, ham_yanit: str) -> str:
    """post_islem kapalıysa ham çıktı; açıksa post-processing zinciri (LCEL)."""
    if not req.post_islem:
        return ham_yanit if isinstance(ham_yanit, str) else str(ham_yanit)
    ham_str = (ham_yanit if isinstance(ham_yanit, str) else str(ham_yanit)).strip()
    return post_process_chain.invoke(
        {"ham_yanit": ham_str, "kampus": req.kampus_adi, "bolum": req.bolum}
    )


def chat_cevabi_uret(req: ChatRequest) -> str:
    """ajan_modu: hazırlık → ajan + tool → post-process; değilse bellekli zincir → post-process."""
    if req.ajan_modu:
        net_mesaj = agent_hazirlik_chain.invoke(
            {
                "kampus": req.kampus_adi,
                "bolum": req.bolum,
                "mesaj": req.mesaj,
            }
        )
        kullanici_paketi = _agent_user_content(req, mesaj_override=net_mesaj)
        sonuc = kampus_ajan.invoke(
            {"messages": [{"role": "user", "content": kullanici_paketi}]}
        )
        son_mesaj = sonuc["messages"][-1]
        ham = son_mesaj.content
        ham_str = ham if isinstance(ham, str) else str(ham)
        return _kullaniciya_son_metin(req, ham_str)
    response = chatbot.invoke(
        {
            "mesaj": req.mesaj,
            "kampus_adi": req.kampus_adi,
            "bolum": req.bolum,
            "ekstra_baglam": _ekstra_baglam(req),
        },
        config={"configurable": {"session_id": req.ogrenci_id}},
    )
    return _kullaniciya_son_metin(req, response.content)


@app.get("/")
def root():
    return {
        "message": "Kampus Ogrenci Botu calisiyor",
        "docs": "http://127.0.0.1:8000/docs",
        "terminal": "Ayni bot terminalde: py terminal_chat.py",
        "chat": "POST /chat — pipeline: hazirlik → agent+tools → post_islem zinciri; post_islem:false ham yanit",
        "yemekhane_menu": "GET /yemekhane/menu?gun=carsamba (gun opsiyonel)",
        "tools": KAMPUS_TOOL_OZET_METNI.split("\n"),
    }


@app.get("/yemekhane/menu", response_model=MenuResponse)
def yemekhane_menu_get(gun: Optional[str] = None):
    anahtar = (
        _bugunun_gun_anahtari()
        if not (gun or "").strip()
        else _normalize_gun(gun)
    )
    return menu_verisi(anahtar)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.ogrenci_id.strip():
        raise HTTPException(status_code=400, detail="ogrenci_id bos olamaz")
    if not req.mesaj.strip():
        raise HTTPException(status_code=400, detail="mesaj bos olamaz")

    try:
        metin = chat_cevabi_uret(req)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ajan/zincir hatasi: {e!s}") from e

    return ChatResponse(cevap=metin)