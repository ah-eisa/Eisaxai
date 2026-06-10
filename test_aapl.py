import urllib.request, json, re

req = urllib.request.Request(
    'http://localhost:8000/v1/chat',
    data=json.dumps({"message":"analyze AAPL","session_id":"test-aapl-agent","user_id":"test"}).encode(),
    headers={"Content-Type":"application/json","X-API-Key":"EisaX_2026_Secure"}
)
resp = urllib.request.urlopen(req, timeout=120)
d = json.loads(resp.read())
r = d.get('reply', '')
print('AGENT:', d.get('agent', '?'))
print('LEN:', len(r))

m = re.search(r'EisaX Score: \*\*(\d+)/100\*\*', r)
print('Score:', m.group(1) if m else 'N/A')

# Check dividend yield - should NOT be 104%
dm = re.search(r'Dividend.*?Yield.*?(\d+\.?\d*)%', r, re.IGNORECASE)
if dm:
    print(f'Dividend Yield: {dm.group(1)}%')
else:
    dm2 = re.search(r'(\d+\.?\d*)%.*dividend', r, re.IGNORECASE)
    print(f'Dividend mention: {dm2.group(0) if dm2 else "not found"}')

# Quality Score
qm = re.search(r'Quality Score.*?(\d+)%', r)
print(f'Quality Score: {qm.group(1)}% ' if qm else 'Quality Score: N/A')

# All sections
print('Has Score Card:', 'Score Card' in r)
print('Has FACT-CHECK:', 'FACT-CHECK' in r)
print('Has Positioning:', 'Positioning' in r)
print('Has Scenario:', 'Scenario' in r)

# Entry < Price
em = re.search(r'Entry.*?[\$]([\d,.]+)', r)
pm = re.search(r'Live Price.*?[\$]([\d,.]+)', r)
if em and pm:
    entry = float(em.group(1).replace(',',''))
    price = float(pm.group(1).replace(',',''))
    print(f'Entry({entry}) < Price({price}): {entry < price}')

print('---FIRST 600---')
print(r[:600])
