# EisaX Global Portfolio — Balanced Multi-Asset Mandate
**التاريخ:** May 18, 2026  |  **رأس المال:** $100,000  |  **المدة:** 5 سنوات  |  **الأسواق:** Global

---

## A. الملخص التنفيذي

| المؤشر | القيمة | التقييم |
|--------|--------|---------|
| العائد المتوقع (سنوي) | **~14.1%** | [STRONG] |
| التقلب المتوقع | **~12.7%** | [MODERATE] |
| Sharpe Ratio | **~0.76** | [ACCEPTABLE] |
| Beta (vs MSCI World) | **~0.65** | [LOW] |
| القيمة المتوقعة بعد 5 سنوات | **$193,387** | ربح متوقع **$93,387** |

**تصنيف نظام المحفظة:** **Multi-Asset Macro**
> Balanced cross-asset construction; no single regime dominates. Targets diversification across factor and macro drivers.
> **سلوك المحفظة مقابل المؤشر:** Designed for regime-balanced behavior; expect peer-like performance across most macro environments with reduced tail volatility.

**Confidence Calibration** · Score: **78%** · Evidence Breadth: **Broad** · Coverage Quality: **Full** · Reliability Tier: **Institutional-Lite** [MODERATE]

**Implementation Feasibility** · Deployability: **Moderate** [MODERATE] (71/100) · Rebalancing Complexity: **High** [HIGH] · Liquidity: **High** [HIGH] · Execution Friction: **Low** [LOW] · Est. Turnover ~22%/yr · Est. Slippage ~6 bp

**Benchmark Context** · Reference: **60/40 Balanced (Global Equity / Bonds, Gold overlay)** · Bench Return ~9.7% · Tracking Deviation: **High** [HIGH] (7.3% TE) · Active Share: **High** [HIGH] (75%) · Style Drift: **US-underweight · Duration-underweight · Crypto-tilted**

> *الأرقام تقريبية ومبنية على افتراضات تاريخية طويلة المدى. لا تُعدّ ضماناً للأداء المستقبلي.*

---

## B. تحليل جدوى التفويض

> تحقق من القيود المُفعَّلة قبل التحسين. كل قيد يُعرض مع القيمة الفعلية والحالة.

| القيد | الحد | الفعلي | الحالة |
|------------|-------|--------|--------|
| Region cap · US | 50.0% | 33.0% | [PASS] |
| Region cap · GCC | 30.0% | 30.0% | [AT CAP] |
| Region cap · Egypt | 10.0% | 8.5% | [NEAR CAP] |
| Region cap · Crypto | 10.0% | 5.2% | [PASS] |
| Region cap · Bonds | 35.0% | 5.0% | [PASS] |
| Region cap · Cash | 10.0% | 10.0% | [AT CAP] |
| Region cap · Commodities | 10.0% | 1.10 | [PASS] |
| Region cap · Diversification | 7.0% | 7.0% | [AT CAP] |
| Beta cap (vs MSCI World) | 1.00 | 0.65 | [PASS] |
| Volatility cap (annualized) | 18.0% | 12.7% | [PASS] |
| Minimum bonds + cash floor | 15.0% | 15.0% | [AT FLOOR] |
| Holdings count | ≥ 5 | 13.0% | [PASS] |

---

## C. Risk Diagnostics

### Adaptive Risk Disclosures

*Conditional on the constructed portfolio — only risks that materially apply are surfaced.*

| Severity | Topic | Note |
|----------|-------|------|
| [HIGH] | Crypto Liquidity Discontinuity | Crypto exposure of 5% subject to 24/7 trading, regulatory regime shifts, and liquidity discontinuities during stress events. Classify as satellite, not core. |
| [MODERATE] | Cross-Currency Exposure | 39% of portfolio in non-USD-denominated holdings. SAR/AED pegged to USD; EGP floats — currency translation risk applies during reporting. |

### Structural Asset-Class Risk Factors

| Asset Class | Risk Factor |
|-------------|-------------|
| GCC / KSA | Oil price sensitivity · Geopolitical risk · Currency peg risk |
| Egypt | Currency devaluation risk · Political risk · High inflation |
| EM Bonds | Default risk · FX risk · Liquidity risk |
| Gold | No yield · Storage cost · USD-sensitive |

> *Currency risk applies to non-USD holdings (SAR pegged, AED pegged, EGP floating).*

### Crypto Analytical Framework

> Crypto positions (5% of portfolio) are evaluated using a separate analytical lens:
> *network activity · ETF flows · realized volatility · liquidity regime · cycle positioning.* Equity valuation multiples and earnings-quality metrics are not applicable.

### Drawdown Modeling Note

> Modeled max drawdown is an estimate, not a guarantee. Crisis episodes (2008, 2020, 2022) frequently exceeded model predictions by 30–50%. Stress-test against −30%, −40%, and −50% scenarios before committing capital.

## D. Allocation Logic

**Mandate:** Balanced Multi-Asset Mandate · Moderate growth · Multi-asset · Diversified core allocation

### Regional Allocation

| Region | Weight | ~$ on $100k | Asset Classes |
|--------|--------|-------------|---------------|
| **US** | 33.0% | $33,000 | US Large Cap Tech, VLO (VLO) |
| **GCC** | 30.0% | $30,000 | 2222 (2222), 4190 (4190), FERTIGLB (FERTIGLB), ORDS (ORDS) |
| **Cash** | 10.0% | $10,000 | Cash / T-Bills |
| **Egypt** | 8.5% | $8,500 | EGAL (EGAL) |
| **Diversification** | 7.0% | $7,000 | US Healthcare |
| **Crypto** | 5.2% | $5,200 | Bitcoin |
| **Bonds** | 5.0% | $5,000 | US Treasuries (LT), EM Bonds |
| **Commodities** | 1.1% | $1,100 | Copper |
| **Gold** | 0.1% | $100 | — |

### Optimal Asset Weights

| Asset Class | Region | Weight | ~$ on $100k | Proxy | Role | Construction Rationale |
|-------------|--------|--------|-------------|-------|------|------------------------|
| **US Large Cap Tech** | US | 25.0% | $25,000 | `QQQ` | Strategic Core | Long-duration growth sleeve · captures secular AI/tech earnings |
| **Bitcoin** | Crypto | 5.2% | $5,234 | `BTC-USD` | Opportunistic Satellite | Asymmetric satellite · high-volatility return contributor (not a hedge) |
| **US Treasuries (LT)** | Bonds | 1.5% | $1,485 | `TLT` | Income / Diversification | Long-duration UST · negative correlation to equity in deflationary shocks |
| **EM Bonds** | Bonds | 3.5% | $3,515 | `EMB` | Income / Diversification | EM credit · spread carry with FX/default risk overlay |
| **Cash / T-Bills** | Cash | 10.0% | $10,000 | `BIL` | Income / Diversification | Dry powder · liquidity buffer and risk-free yield anchor |
| **US Healthcare** | Diversification | 7.0% | $7,000 | `XLV` | Income / Diversification | Short-duration anchor · capital preservation with low rate risk |
| **Copper** | Commodities | 1.1% | $1,101 | `CPER` | Real-Asset / Inflation Sleeve | Real-asset exposure · inflation pass-through and macro cyclicality |
| **2222 (2222)** | GCC | 8.5% | $8,500 | `2222` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |
| **4190 (4190)** | GCC | 8.5% | $8,500 | `4190` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |
| **FERTIGLB (FERTIGLB)** | GCC | 4.5% | $4,500 | `FERTIGLB` | Satellite / Diversifier | Regional exposure · GCC growth premium, low correlation to US equities |
| **ORDS (ORDS)** | GCC | 8.5% | $8,500 | `ORDS` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |
| **EGAL (EGAL)** | Egypt | 8.5% | $8,500 | `EGAL` | Tactical Allocation | Frontier-market tilt · structural reform exposure (higher idiosyncratic risk) |
| **VLO (VLO)** | US | 7.8% | $7,783 | `VLO` | Tactical Allocation | Energy cyclicality · inflation hedge and macro pro-cyclical exposure |

### Diversification Benefit

> **Diversification Ratio:** 1.83x — portfolio vol (12.6%) is 45% lower than weighted average of individual vols (23.1%)
> **vs Equal Weight:** Optimized vol 12.6% vs equal-weight 12.3%

### Benchmark-Relative Attribution

*Decomposition of excess return vs profile-mapped benchmark — distinguishes factor-driven from selection-driven performance.*

| Component | Value | Interpretation |
|-----------|-------|----------------|
| Excess Return vs Benchmark | +4.37% | Total active return |
| Beta Contribution | +0.59% | Factor-driven (market sensitivity differential) |
| Residual Contribution | +3.78% | Selection / concentration effects |

> *Excess return concentrated in residual/selection effects — review concentration risk before attributing to skill.*

> *Implementation note: Rounded portfolio failed institutional check (cap breach); raw optimizer weights preserved.*

---

## E. خطة إعادة التوازن

> إجراءات مُحدَّدة لتقليل التركز، مع الأثر الكمّي على بيتا، التقلب، ودوران المحفظة.

| الإجراء | بيتا قبل→بعد | تقلب قبل→بعد | تقليل التركز | الدوران | الصعوبة |
|-----------------|-------------------|------------------|-----------------|----------|------------|
| Reduce US Large Cap Tech (QQQ) 25.0% → 15.0% | 0.652 → 0.573 (-0.080) | 12.65% → 12.20% (-0.45pp) | −10.0pp | 10.0pp | [MODERATE] |

> *تقدير الأثر مبني على إعادة توزيع الوزن المُخفَّض على باقي الأصول بالتناسب.*

### خطوات التنفيذ المقترحة

1. افتح حساب وساطة مناسب للأسواق: Global
2. وزّع $100,000 حسب نسب الأصول في القسم D
3. أعد التوازن كل 6-12 شهر
4. تجنّب التصفية في تراجعات السوق قصيرة المدى — أفق 5 سنوات يستوعب دورات السوق


---


---

### تشخيصات كفاءة التنفيذ

| المقياس | القيمة | التصنيف |
| --- | --- | --- |
| معدل الدوران | 100.0% | (عالٍ) |
| فجوة التنفيذ | 19.4 bp | (متوسط) |
| أثر السوق | 3.0 bp | (منخفض) |
| الانزلاق المقدر | 17.4 bp | (متوسط) |
| تعقيد التنفيذ | high | — |
| ضغط السيولة | moderate | — |
| تكرار إعادة التوازن | quarterly | — |

*عقوبة معدل الدوران: غير مطبقة (linear λ=0.0010, quadratic λ=0.0005)*

*التنفيذ الواعي بالضرائب يتطلب بيانات الحصص على مستوى الحساب؛ مكان مخصص بانتظار التكامل مع تغذية الحصص الضريبية من الوسيط.*

*ملاحظة سيولة أسواق الخليج: GCC legs use a Sun-Thu trading calendar; execution windows should account for the four-day local trading week.*

## H. توزيع السيناريوهات المستقبلية

*الأفق: 5.0 سنوات · التضخم: 2.0% · البذرة: 42 · المسارات: 12000*

| السيناريو | الاحتمال | القيمة النهائية (P10) | القيمة النهائية (P50) | القيمة النهائية (P90) | أقصى انخفاض (P50) | مدة التعافي (أشهر، P50) |
| --- | --- | --- | --- | --- | --- | --- |
| soft landing | 30.0% | -11% | +38% | +110% | -20% | 5.0 |
| recession | 20.0% | -42% | +10% | +96% | -32% | 0.0 |
| stagflation | 10.0% | -37% | +19% | +128% | -32% | 0.0 |
| ai productivity boom | 10.0% | -15% | +48% | +149% | -23% | 4.0 |
| energy shock | 15.0% | -37% | +14% | +107% | -32% | 0.0 |
| liquidity crisis | 15.0% | -53% | +9% | +147% | -44% | 0.0 |

**الإجمالي المرجح بالاحتمالات**

| المقياس | القيمة |
| --- | --- |
| نطاق القيمة النهائية المتوقع (حقيقي) | $0.67 - $2.18 (P10-P90) |
| احتمال الخسارة (حقيقي) | 32% |
| احتمال بلوغ الهدف (>=4% حقيقي سنويا) | 53% |
| القيمة النهائية للعشر الأدنى | $0.67 |
| نطاق الانخفاض المتوقع | -28% إلى -52% |
| مدة التعافي (الوسيط) | 0 أشهر |

*إطار توزيعي فقط - تعكس النتائج افتراضات نموذجية وليست توقعا قطعيا.*

## G. ملحق المراجعة

| الحقل | القيمة |
|-------|--------|
| معرّف اللقطة (Snapshot ID) | `<HASH>` |
| هاش الكون الاستثماري | `<HASH>` |
| الـ Solver | CLARABEL (cvxpy QP) |
| حالة الـ Solver | optimal |
| عدد الأصول (الكون) | 31 |
| عدد الأصول (المختارة) | 13 |
| Max Beta | 1.0 |
| Max Volatility | 18.0% |
| Min Bonds + Cash | 15.0% |
| Max Drawdown (مطلوب) | غير محدد% |
| Risk Aversion | 4.0 |
| Risk-Free Rate | 4.5% |
| القيود المخصصة | — |

> *قابل للتكرار: نفس المدخلات → نفس Snapshot ID → نفس النتيجة. لا تعديلات صامتة.*

### قيود النموذج — الحدود الهيكلية لمحرك التحليل

- Historical simulation uses 252-day trailing window; structural breaks beyond that window are not captured.
- Correlation matrix is point-in-time; pairwise correlations rise toward 1.0 during liquidity events.
- Volatility is non-stationary; realized vol can diverge materially from in-sample estimates during regime shifts.
- Live-stock prices are cached at 15-minute intervals; intra-window movements not reflected.
- Beta estimates assume linear market sensitivity; convex behavior (gamma) ignored.
- Optimizer assumes frictionless rebalancing; transaction costs, slippage, and tax drag are out-of-scope.

> *ملاحظة شفافية: القيود أعلاه متأصلة في منهجية المحاكاة التاريخية لبناء المحافظ. عُرضت بشكل صريح لدعم المراجعة المؤسسية والحوكمة.*

### تشخيصات الأداء النسبي مقابل المؤشر

*المؤشر: S&P 500 (SPY) · الموثوقية: Indicative · النافذة: 3شهر*

| المقياس | القيمة | التصنيف |
| --- | --- | --- |
| العائد النسبي | -0.54% | (متوسط) |
| خطأ التتبع | 0.65% | (منخفض) |
| نسبة المعلومات | -0.84 | (منخفض) |
| ألفا المتجدد (12 شهر) | -0.07% | — |
| بيتا المتجدد (12 شهر) | 0.22 | — |
| الانخفاض النسبي | -0.26% | (منخفض) |
| احتواء الصعود | 0.20 | (منخفض) |
| احتواء الهبوط | 0.28 | (منخفض) |
| التذبذب النسبي | 0.23 | — |
| الحصة النشطة | 21.39% | (منخفض) |
| انحراف الأسلوب | material | — |

**تحليل العائد الفائض**

| المكون | المساهمة (نقطة مئوية) |
| --- | --- |
| أثر التخصيص | -0.14 |
| أثر الاختيار | -0.53 |
| أثر العوامل | -0.47 |
| أثر التركيز | +0.60 |

**سلوك المحفظة النسبي حسب النظام**

- البيئات التي يرجح أن تتفوق فيها المحفظة: سجل الأنظمة غير كاف
- البيئات التي يرجح أن تتأخر فيها المحفظة: سجل الأنظمة غير كاف

تشتت الأداء النسبي محدود؛ ومن المرجح أن تقود فروق التخصيص التدريجية النتائج النشطة أكثر من الابتعاد الهيكلي عن المؤشر.

### تحليل المخاطر بحسب العوامل

*النموذج: Carhart · R²: 0.00 · الثبات: 0.00 · الموثوقية: Indicative · النافذة: 0شهر*

| العامل | التحميل | إحصاء t | المساهمة (ret) | المساهمة (vol) |
| --- | --- | --- | --- | --- |
| MKT | 0.00 | 0.00 | +0.00% | 0.00% |
| SMB | 0.00 | 0.00 | +0.00% | 0.00% |
| HML | 0.00 | 0.00 | +0.00% | 0.00% |
| MOM | 0.00 | 0.00 | +0.00% | 0.00% |

*تمثل التحميلات انكشافات متجددة لمدة 36 شهرا؛ وتعدل إحصاءات t بطريقة Newey-West عند توفرها.*

**إعادة إنتاج المرحلة H**

| المقياس | القيمة |
| --- | --- |
| Phase H version | 0.1.0 |
| Seed | 42 |
| Engines ran | benchmark_relative, execution_diag, factor_decomp, forward_scenario |
| Flags | enabled=on, benchmark=on, tc_optimizer=on, forward_sim=on, factor_model=on, committee=on, tone_guard=on, deterministic_seed=42 |
| benchmark_relative hash | <HASH> |
| execution_diag hash | <HASH> |
| factor_decomp hash | <HASH> |
| forward_scenario hash | <HASH> |
| benchmark_relative version | 0.2.0 |
| execution_diag version | 0.2.0 |
| factor_decomp version | 0.2.0 |
| forward_scenario version | 0.2.0 |
| phase_h version | 0.1.0 |
