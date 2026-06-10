import cloudscraper
from bs4 import BeautifulSoup

scraper = cloudscraper.create_scraper(browser={'browser':'chrome','platform':'windows','mobile':False})
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0',
    'Referer': 'https://www.investing.com/',
    'X-Requested-With': 'XMLHttpRequest',
    'Content-Type': 'application/x-www-form-urlencoded',
}

# DIB ~7-9 AED, DEWA ~2.5 AED, DU ~6-7 AED, EAND ~18 AED
test_ids = {
    'DIB?':   ['1055162','1055163','1055164','1055170','1055180','941322','941323','941324'],
    'DEWA?':  ['941326','941327','941328','941329','941331','941332','941333'],
    'DU?':    ['941334','941335','941336','941337','941338','941340'],
    'EAND?':  ['1055165','1055166','1055167','1055168','1055169','1055171'],
}

for ticker, ids in test_ids.items():
    print(f'\n=== {ticker} ===')
    for id_ in ids:
        data = {
            'curr_id': id_, 'st_date': '01/01/2025', 'end_date': '01/10/2025',
            'interval_sec': 'Daily', 'sort_col': 'date', 'sort_ord': 'DESC',
            'action': 'historical_data',
        }
        r = scraper.post('https://www.investing.com/instruments/HistoricalDataAjax',
                        headers=headers, data=data, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        rows = soup.select('#curr_table tbody tr')
        if rows and 'No results' not in rows[0].text:
            first = [c.text.strip() for c in rows[0].find_all('td')[:3]]
            print(f'  ✅ id={id_} → {first}')