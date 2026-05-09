# Kampus Öğrenci Botu

Groq (Llama) + **LangChain** ile kampüs odaklı sohbet: **ajan**, **tool**’lar, **LCEL zincirleri** ve **FastAPI** API’si. Terminalden veya HTTP ile kullanılabilir.

## Özellikler

- **LLM:** Groq — `llama-3.3-70b-versatile` (`ChatGroq`)
- **Ajan:** `create_agent` + 5 adet `@tool` (menü, çalışma planı, servis bilgisi, örnek ders programı, kampüs yaşam rehberi)
- **Zincirler:** hazırlık (`prompt | llm | parser`), post-processing, sohbet modunda `prompt | llm` + mesaj geçmişi; çalışma planı aracı içinde ayrı zincir
- **API:** FastAPI — `POST /chat`, `GET /yemekhane/menu`
- **Terminal:** `terminal_chat.py` (komutlar: `/ajan`, `/post`, `/menu`, `/tools`, …)

> Menü ve ders programı **demo veridir**; gerçek bilgi için kurum duyurusu / OBS kullanılmalıdır.

## Kurulum

```bash
git clone https://github.com/seymakrkrt3334/kampus-ogrenci-botu.git
cd kampus-ogrenci-botu

python -m venv .venv
# Windows Git Bash:
source .venv/Scripts/activate
# veya: .venv\Scripts\activate

pip install -r requirements.txt
```

`.env.example` dosyasını `.env` olarak kopyalayın ve [Groq Console](https://console.groq.com/keys) üzerinden aldığınız anahtarı yazın:

```env
GROQ_API_KEY=gsk_...
```

`.env` **Git’e eklenmez** (`.gitignore`).

## Çalıştırma

**API (Swagger:** `http://127.0.0.1:8000/docs`)

```bash
uvicorn main:app --reload
```

**Terminal sohbet**

```bash
python terminal_chat.py
```

## LangChain tool’ları

| Tool | Açıklama |
|------|-----------|
| `yemekhane_menu_araci` | Örnek öğle/akşam menüsü |
| `calisma_plani_araci` | Konuya göre pomodoro çalışma planı (içinde `prompt \| llm \| parser`) |
| `kampuste_servis_saatleri` | Örnek kütüphane / yemekhane / öğrenci işleri bilgisi |
| `ornek_ders_programi` | Demo haftalık ders tablosu |
| `kampu_yasam_rehberi` | Sınav, kayıt, kulüp, danışmanlık vb. örnek rehber |

## API özeti

- `POST /chat` — gövde: `ogrenci_id`, `mesaj`, `kampus_adi`, `bolum`, isteğe bağlı `ajan_modu`, `post_islem`, `koc_modu`, `yemekhane_bilgisi_ekle`, `menu_gunu`
- `GET /yemekhane/menu?gun=pazartesi` — örnek menü JSON’u

## Proje yapısı

| Dosya | Rol |
|------|-----|
| `main.py` | FastAPI, LLM, ajan, zincirler, tool tanımları |
| `terminal_chat.py` | CLI; `chat_cevabi_uret` çağırır |
| `requirements.txt` | Bağımlılıklar |
| `01_*` … `05_*` | Ders / hocanın LangChain örnekleri |

## Lisans / katkı

Kişisel / eğitim projesi. İsterseniz repo ayarlarından lisans ekleyebilirsiniz.
