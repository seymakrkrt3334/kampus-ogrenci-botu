# -*- coding: utf-8 -*-
"""
Terminalden kampus ogrenci botu — her seferinde senin SORDUGUN soruya gore yanit verir.

Proje klasorunde:
  source .venv/Scripts/activate   # Git Bash
  py terminal_chat.py

Komutlar: /quit | /ajan | /post | /koc | /yemek | /gun | /menu | /tools | /yardim
/tools — kullanilan LangChain tool listesi (ajan modunda)
"""

from __future__ import annotations

import sys

from fastapi import HTTPException


def _menu_yazdir(m) -> None:
    print(f"\n--- Yemekhane (ornek) — {m.gun} ---")
    print(m.tarih_notu)
    print("Öğle:", ", ".join(f"{k}: {v}" for k, v in m.ogle.items()))
    print("Akşam:", ", ".join(f"{k}: {v}" for k, v in m.aksam.items()))
    print(m.not_)


def main() -> None:
    import main as app_main

    print("=== Kampus Ogrenci Botu (terminal) ===")
    print(app_main.KAMPUS_TOOL_OZET_METNI)
    print()
    print(
        "Komutlar: /quit | /ajan ac|kapa | /post ac|kapa | /koc ac|kapa | /yemek ac|kapa | "
        "/gun GUN | /menu [gun] | /tools | /yardim\n"
    )

    oid = input("Öğrenci ID: ").strip() or "terminal"
    kampus = input("Kampus [Enter=Genel Kampus]: ").strip() or "Genel Kampus"
    bolum = input("Bölüm [Enter=Belirtilmedi]: ").strip() or "Belirtilmedi"

    koc = False
    yemek = False
    menu_gunu: str | None = None
    ajan_modu = True
    post_islem = True

    while True:
        try:
            satir = input("\nSen: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGörüşürüz.")
            sys.exit(0)

        if not satir:
            continue

        low = satir.lower()
        if low in ("/quit", "quit", "exit", "/q", "/cik"):
            print("Görüşürüz.")
            break

        if satir.startswith("/"):
            parts = satir.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd == "/ajan":
                if arg.lower() in ("ac", "on", "1", "true", "a"):
                    ajan_modu = True
                    print("[Ajan modu: AÇIK — tool + chain hazırlık]")
                elif arg.lower() in ("kapa", "kapalı", "off", "0", "false", "k"):
                    ajan_modu = False
                    print("[Ajan modu: KAPALI — sadece bellekli sohbet zinciri]")
                else:
                    print("Kullanım: /ajan ac   veya   /ajan kapa")
                continue

            if cmd == "/post":
                if arg.lower() in ("ac", "on", "1", "true", "a"):
                    post_islem = True
                    print("[Post-process: AÇIK — kullanıcıya son gösterilen metin biçimlendirilir]")
                elif arg.lower() in ("kapa", "kapalı", "off", "0", "false", "k"):
                    post_islem = False
                    print("[Post-process: KAPALI — ham model/ajan çıktısı]")
                else:
                    print("Kullanım: /post ac   veya   /post kapa")
                continue

            if cmd == "/koc":
                if arg.lower() in ("ac", "on", "1", "true", "a"):
                    koc = True
                    print("[Koç modu: AÇIK]")
                elif arg.lower() in ("kapa", "kapalı", "off", "0", "false", "k"):
                    koc = False
                    print("[Koç modu: KAPALI]")
                else:
                    print("Kullanım: /koc ac   veya   /koc kapa")
                continue

            if cmd == "/yemek":
                if arg.lower() in ("ac", "on", "1", "true", "a"):
                    yemek = True
                    print("[Yemekhane menüsü prompt’a eklenecek: AÇIK]")
                elif arg.lower() in ("kapa", "kapalı", "off", "0", "false", "k"):
                    yemek = False
                    print("[Yemekhane bilgisi: KAPALI]")
                else:
                    print("Kullanım: /yemek ac   veya   /yemek kapa")
                continue

            if cmd == "/gun":
                if not arg:
                    menu_gunu = None
                    print("[Menü günü: bugün]")
                else:
                    menu_gunu = arg
                    print(f"[Menü günü: {menu_gunu}]")
                continue

            if cmd == "/menu":
                try:
                    if arg:
                        gun_key = app_main._normalize_gun(arg)
                    else:
                        gun_key = app_main._bugunun_gun_anahtari()
                    _menu_yazdir(app_main.menu_verisi(gun_key))
                except HTTPException as e:
                    print("Hata:", e.detail)
                continue

            if cmd == "/tools":
                print(app_main.KAMPUS_TOOL_OZET_METNI)
                continue

            if cmd in ("/help", "/yardim", "/?"):
                print(__doc__)
                continue

            print("Bilinmeyen komut. /yardim")
            continue

        mg = menu_gunu.strip() if menu_gunu and menu_gunu.strip() else None
        req = app_main.ChatRequest(
            ogrenci_id=oid,
            mesaj=satir,
            kampus_adi=kampus,
            bolum=bolum,
            koc_modu=koc,
            yemekhane_bilgisi_ekle=yemek,
            menu_gunu=mg,
            ajan_modu=ajan_modu,
            post_islem=post_islem,
        )

        try:
            cevap = app_main.chat_cevabi_uret(req)
        except Exception as e:
            print("\nHata:", e)
            continue
        print("\nBot:", cevap)


if __name__ == "__main__":
    main()
