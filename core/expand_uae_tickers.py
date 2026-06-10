"""
expand_uae_tickers.py
يضيف كل شركات DFM وADX لـ local_tickers.py تلقائياً
"""
import sqlite3

DB_PATH = "/home/ubuntu/investwise/core/investwise.db"
LOCAL_TICKERS = "/home/ubuntu/investwise/core/local_tickers.py"

# Mapping الأسماء → tickers (يدوي للشركات الكبيرة)
NAME_TO_TICKER = {
    # DFM
    "Air Arabia PJSC":                    "AIRARABIA.DU",
    "AJMAN BANK PJSC":                    "AJMANBANK.DU",
    "ARAMEX PJSC":                        "ARAMEX.DU",
    "Commercial Bank of Dubai":           "CBD.DU",
    "Deyaar Development":                 "DEYAAR.DU",
    "Dubai Financial Market":             "DFM.DU",
    "Dubai Investments":                  "DINV.DU",
    "Dubai Islamic Bank":                 "DIB.DU",
    "Dubai National Insurance":           "DNI.DU",
    "Drake&Scull Int":                    "DSI.DU",
    "Watania International Holding PJSC": "WATANIA.DU",
    "Emirate Integrated Telecom":         "DU.DU",
    "Emaar Properties":                   "EMAAR.DU",
    "Emirates NBD PJSC":                  "ENBD.DU",
    "Gulf Navigation Hld":                "GNAV.DU",
    "National Cement Co":                 "NCC.DU",
    "Islamic Arab Insurance":             "SALAMA.DU",
    "SHUAA Capital PSC":                  "SHUAA.DU",
    "National Central Cooling":           "TABREED.DU",
    "Takaful Emarat PSC":                 "TE.DU",
    "Union Properties":                   "UPP.DU",
    "Dubai Insurance Co PSC":             "DIC.DU",
    "Mashreqbank PSC":                    "MASQ.DU",
    "National General Insurance":         "NGI.DU",
    "Amlak Finance":                      "AMLAK.DU",
    "Sukoon Takaful PJSC":                "SUKOON.DU",
    "Emirates Refreshments Co":           "EMREF.DU",
    "Sukoon Insurance PJSC":              "SUKOON2.DU",
    "Amanat Holdings PJSC":               "AMANAT.DU",
    "Emaar Develop":                      "EMAARDEV.DU",
    "Dubai Electricity and Water":        "DEWA.DU",
    "Tecom PJSC":                         "TECOM.DU",
    "Salik Company PJSC":                 "SALIK.DU",
    "Emirates Central Cooling Systems":   "EMPOWER.DU",
    "Taaleem Holdings":                   "TAALEEM.DU",
    "Al Ansari Financial Services PJSC":  "ALANSARI.DU",
    "Dubai Taxi Company PJSC":            "DUBAITAXI.DU",
    "Parkin Company PJSC":                "PARKIN.DU",
    "Spinneys 1961 Holding":              "SPINNEYS.DU",
    "Talabat Holding":                    "TALABAT.DU",
    "ALEC Holdings PJSC":                 "ALEC.DU",

    # ADX
    "Abu Dhabi Aviation":                 "ADAVIATION.AE",
    "Finance House":                      "FH.AE",
    "Emirates Ins C":                     "EIC.AE",
    "United Arab Bk":                     "UAB.AE",
    "International Holding Company":      "IHC.AE",
    "HAYAH Insurance":                    "HAYAH.AE",
    "Al Qaiwain Cement":                  "QCC.AE",
    "Em Driving Co":                      "EDC.AE",
    "Insurance Hous":                     "IH.AE",
    "Ras Al Khaimah for White Cement and Construction M": "RAKCEC.AE",
    "Sharjah Cement AD":                  "SCRJ.AE",
    "Abu Dhabi National Takaful":         "ADNTC.AE",
    "Buhaira Nat In":                     "BNI.AE",
    "Abu Dhabi Islamic Bank PJSC":        "ADIB.AE",
    "Abu Dhabi National Insurance":       "ADNIC.AE",
    "Agthia Group":                       "AGTHIA.AE",
    "National Building Materials":        "NBM.AE",
    "Abu Dhabi Ship Building PJSC":       "ADSB.AE",
    "Emsteel Building Materials Pjsc":    "EMSTEEL.AE",
    "Bank Of Sharja":                     "BOS.AE",
    "Dana Gas":                           "DANA.AE",
    "Abu Dhabi Commercial Bank PJSC":     "ADCB.AE",
    "Aldar Properties":                   "ALDAR.AE",
    "Emirates Telecommunications":        "EAND.AE",
    "Gulf Cement Co":                     "GCC.AE",
    "Gulf Pharm Ind":                     "GPI.AE",
    "Methaq":                             "METHAQ.AE",
    "First Abu Dhabi Bank":               "FAB.AE",
    "Nat Bk Qaiwain":                     "NBQ.AE",
    "National Bank of Ras Al Khaimah":    "RAKBANK.AE",
    "Rak Ceramics":                       "RAKCEC2.AE",
    "Rak Properties":                     "RAKPROP.AE",
    "Apex Investment":                    "APEX.AE",
    "Sharjah Islami":                     "BANKSHJ.AE",
    "Abu Dhabi National Energy":          "TAQA.AE",
    "Waha Capital":                       "WAHA.AE",
    "Abu Dhabi National Hotels Co":       "ADNH.AE",
    "Commercial Bank International":      "CBI.AE",
    "Hily Holding PJSC":                  "HILY.AE",
    "National Corp Tourism Hotels":       "NCTH.AE",
    "NMDC PJSC":                          "NMDC.AE",
    "Ras Al Khaimah National Insurance":  "RAKNI.AE",
    "Gulf Medical Projects Co PSC":       "GMPC.AE",
    "Eshraq Investments PJSC":            "ESHRAQ.AE",
    "Manazel Real Estate":                "MANAZEL.AE",
    "Fujairah Building Industries":       "FBI.AE",
    "Oman Emirates Holding":              "OEH.AE",
    "Ooredoo QPSC":                       "OOREDOO.AE",
    "Sudatel Telecom Group":              "SUDATEL.AE",
    "Fujairah Cement Industries":         "FCI.AE",
    "RAPCO Investment":                   "RAPCO.AE",
    "Union Insurance":                    "UI.AE",
    "Al Khaleej Investment":              "AKI.AE",
    "National Bank of Fujairah":          "NBF.AE",
    "Al Ain Ahlia Insurance":             "AAIA.AE",
    "National Oil":                       "NOGA.AE",
    "Modon Holding":                      "MODON.AE",
    "Aram PJSC":                          "ARAM.AE",
    "Al Fujairah National Insurance Co PSC": "AFNIC.AE",
    "Al Wathba National Insurance Co":    "AWNIC.AE",
    "Al Dhafra Insurance Company PSC":    "ADIC.AE",
    "Sharjah Insurance Co PSC":           "SIC.AE",
    "Easy Lease Motor Cycle Rental PSC":  "EASYLEASE.AE",
    "Palms Sports PJSC":                  "PALMS.AE",
    "Ghitha Holding PJSC":               "GHITHA.AE",
    "Alpha Dhabi Holding PJSC":           "ALPHADHABI.AE",
    "Emirates Stallion PJSC":            "STALLION.AE",
    "ADNOC Drilling":                     "ADNOCDRILL.AE",
    "Fertiglobe":                         "FERTIGLOBE.AE",
    "ADNOC Distribution":                 "ADNOCDIST.AE",
}

def build_new_tickers_block():
    """يبني الـ UAE_TICKERS dictionary كامل"""
    conn = sqlite3.connect(DB_PATH)

    # جيب كل الشركات
    dfm_rows = conn.execute("SELECT name, market_cap, pe_ratio, beta FROM uae_fundamentals WHERE exchange='DFM'").fetchall()
    adx_rows = conn.execute("SELECT name, signal_daily, signal_weekly FROM uae_signals WHERE exchange='ADX'").fetchall()
    conn.close()

    all_companies = {}

    # DFM
    for row in dfm_rows:
        name = row[0]
        ticker = NAME_TO_TICKER.get(name, f"UNKNOWN_{name[:6].upper().replace(' ','')}.DU")
        all_companies[ticker] = {
            "name_en": name,
            "exchange": "DFM",
            "market_cap": row[1],
            "pe_ratio": row[2],
        }

    # ADX
    for row in adx_rows:
        name = row[0]
        ticker = NAME_TO_TICKER.get(name, f"UNKNOWN_{name[:6].upper().replace(' ','')}.AE")
        if ticker not in all_companies:
            all_companies[ticker] = {
                "name_en": name,
                "exchange": "ADX",
                "signal_daily": row[1],
                "signal_weekly": row[2],
            }

    return all_companies

def update_local_tickers():
    """يضيف الشركات الجديدة لـ local_tickers.py"""
    companies = build_new_tickers_block()

    # اقرأ الملف الحالي
    with open(LOCAL_TICKERS, "r", encoding="utf-8") as f:
        content = f.read()

    # شوف مين موجود بالفعل
    new_entries = []
    added = 0
    for ticker, info in companies.items():
        if ticker in content or "UNKNOWN" in ticker:
            continue
        name_en = info["name_en"]
        exchange = info["exchange"]
        entry = f'''
    "{ticker}": {{
        "name_en": "{name_en}",
        "name_ar": "",
        "aliases_ar": [],
        "aliases_en": ["{name_en.lower()}"],
        "sector": "General",
        "sector_ar": "عام",
        "currency": "AED",
        "exchange": "{exchange}",
    }},'''
        new_entries.append(entry)
        added += 1

    if not new_entries:
        print("✅ كل الشركات موجودة بالفعل في local_tickers.py")
        return

    # أضفهم قبل آخر } في UAE_TICKERS
    insert_before = '}\n\n\n# ═══════════════════════════════════════════════════════════════\n#  🌍 MARKET INDEX'
    new_block = "\n".join(new_entries) + "\n" + insert_before
    content = content.replace(insert_before, new_block)

    with open(LOCAL_TICKERS, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"✅ أضاف {added} شركة جديدة لـ UAE_TICKERS في local_tickers.py")
    print(f"📋 تجاهل {len(companies) - added} شركة (موجودة أو UNKNOWN)")

if __name__ == "__main__":
    update_local_tickers()