"""
Capital.com UAE Stocks — Playwright Interceptor
=================================================
بيدخل على Capital.com كأنك إنت، يضغط على UAE،
يعترض الـ API call الحقيقي، ويجيب كل الأسهم
"""

import asyncio
import json
import time
from playwright.async_api import async_playwright
from config import EMAIL, PASSWORD


# ========================================================
# 🔧 Settings
# ========================================================
CAPITAL_URL = "https://capital.com"
OUTPUT_FILE = "uae_stocks_raw.json"
HEADLESS    = False   # False = تشوف المتصفح


# ========================================================
# 📡 الـ calls اللي هنسمع عليها
# ========================================================
INTERCEPT_KEYWORDS = [
    "markets", "instruments", "watchlist",
    "securities", "stocks", "equity",
    "browse", "category", "country"
]

captured_responses = []


async def main():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=HEADLESS,
            slow_mo=300,
            args=["--start-maximized"]
        )
        context = await browser.new_context(
            viewport={"width": 1400, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )
        page = await context.new_page()

        # ── اعتراض الـ responses ──────────────────────────
        async def on_response(response):
            url = response.url
            if any(kw in url.lower() for kw in INTERCEPT_KEYWORDS):
                try:
                    body = await response.json()
                    captured_responses.append({
                        "url":    url,
                        "status": response.status,
                        "body":   body
                    })
                    # طباعة فورية للـ URLs المهمة
                    print(f"  📡 {response.status} | {url[:80]}")
                except Exception:
                    pass

        page.on("response", on_response)

        # ── الخطوة 1: افتح Capital.com ───────────────────
        print("\n🌐 بفتح Capital.com...")
        await page.goto(CAPITAL_URL, wait_until="domcontentloaded", timeout=30000)
        await asyncio.sleep(3)

        # ── الخطوة 2: Login ───────────────────────────────
        print("🔐 بعمل Login...")
        try:
            # دور على زرار Login
            login_btn = page.locator("text=Log in").first
            await login_btn.click()
            await asyncio.sleep(2)

            # email
            await page.fill('input[type="email"], input[name="email"], input[placeholder*="email" i]', EMAIL)
            await asyncio.sleep(0.5)

            # password
            await page.fill('input[type="password"]', PASSWORD)
            await asyncio.sleep(0.5)

            # submit
            await page.keyboard.press("Enter")
            await asyncio.sleep(5)

            print("  ✅ Login تم")
        except Exception as e:
            print(f"  ⚠️  Login manual: {e}")
            print("  → اعمل login يدوياً في المتصفح وبعدين اضغط Enter هنا")
            input("  ⏸️  اضغط Enter بعد ما تعمل login...")

        await page.screenshot(path="after_login.png")

        # ── الخطوة 3: روح على Trading / Markets ─────────
        print("\n📊 بروح على صفحة الأسواق...")
        try:
            # دور على رابط Trade أو Markets في الـ sidebar
            for selector in ["text=Trade", "text=Markets", "[href*='trade']", "[href*='market']"]:
                try:
                    await page.click(selector, timeout=3000)
                    await asyncio.sleep(2)
                    break
                except Exception:
                    continue
        except Exception:
            pass

        await asyncio.sleep(3)
        await page.screenshot(path="markets_page.png")

        # ── الخطوة 4: اضغط على UAE ───────────────────────
        print("🇦🇪 بضغط على UAE في الـ sidebar...")
        uae_clicked = False

        for selector in [
            "text=UAE",
            "text=United Arab Emirates",
            "[data-country='AE']",
            "[data-id*='UAE']",
        ]:
            try:
                await page.click(selector, timeout=5000)
                await asyncio.sleep(4)
                uae_clicked = True
                print(f"  ✅ ضغطت على UAE بـ selector: {selector}")
                break
            except Exception:
                continue

        if not uae_clicked:
            print("  ⚠️  مش قادر أضغط على UAE أوتوماتيكي")
            print("  → اضغط على UAE يدوياً في المتصفح")
            input("  ⏸️  اضغط Enter بعد ما تضغط على UAE وتشوف الأسهم...")

        await page.screenshot(path="uae_stocks.png")

        # ── الخطوة 5: scroll لتحميل كل الأسهم ──────────
        print("\n📜 بعمل scroll لتحميل كل الأسهم...")
        for i in range(10):
            await page.keyboard.press("End")
            await asyncio.sleep(1.5)

        await asyncio.sleep(3)

        # ── الخطوة 6: تحليل الـ responses ───────────────
        print(f"\n📊 تحليل {len(captured_responses)} response...")

        # حفظ كل الـ responses الأولاً
        with open("all_captured_responses.json", "w", encoding="utf-8") as f:
            json.dump(captured_responses, f, ensure_ascii=False, indent=2)

        # ابحث عن الـ response اللي فيه أسهم إماراتية
        uae_keywords = ["emaar", "adnoc", "aldar", "emirates nbd",
                        "dubai islamic", "salik", "dewa", "aramex",
                        "borouge", "air arabia"]

        best_response = None
        best_score    = 0

        for resp in captured_responses:
            body_str = json.dumps(resp.get("body", {})).lower()
            score    = sum(1 for kw in uae_keywords if kw in body_str)
            if score > best_score:
                best_score    = score
                best_response = resp

        if best_response and best_score > 0:
            print(f"\n  🎯 لقينا الـ endpoint! Score: {best_score}/10")
            print(f"  URL: {best_response['url']}")

            # استخراج الأسهم
            body = best_response["body"]
            print(f"  Body keys: {list(body.keys()) if isinstance(body, dict) else type(body)}")

            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(best_response, f, ensure_ascii=False, indent=2)

            print(f"  💾 محفوظ في {OUTPUT_FILE}")
        else:
            print("\n  ⚠️  مش لاقي UAE stocks في الـ responses")
            print("  → شوف all_captured_responses.json وابحث يدوياً")

            # طباعة كل الـ URLs اللي اتعترضت
            print("\n  كل الـ URLs اللي اتعترضت:")
            for r in captured_responses:
                print(f"    {r['url'][:100]}")

        await browser.close()
        print("\n✅ خلصنا!")


if __name__ == "__main__":
    asyncio.run(main())
