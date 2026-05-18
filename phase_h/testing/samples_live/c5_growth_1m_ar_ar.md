# EisaX Global Portfolio — Long-Horizon Growth Mandate
**التاريخ:** May 18, 2026  |  **رأس المال:** $1.0M  |  **المدة:** 8 سنوات  |  **الأسواق:** Global

---

## A. الملخص التنفيذي

| المؤشر | القيمة | التقييم |
|--------|--------|---------|
| العائد المتوقع (سنوي) | **~16.2%** | [STRONG] |
| التقلب المتوقع | **~16.5%** | [MODERATE] |
| Sharpe Ratio | **~0.71** | [ACCEPTABLE] |
| Beta (vs MSCI World) | **~0.74** | [MODERATE] |
| القيمة المتوقعة بعد 8 سنوات | **$3,319,334** | ربح متوقع **$2,319,334** |

**تصنيف نظام المحفظة:** **Multi-Asset Macro**
> Balanced cross-asset construction; no single regime dominates. Targets diversification across factor and macro drivers.
> **سلوك المحفظة مقابل المؤشر:** Designed for regime-balanced behavior; expect peer-like performance across most macro environments with reduced tail volatility.

**Confidence Calibration** · Score: **76%** · Evidence Breadth: **Broad** · Coverage Quality: **Full** · Reliability Tier: **Institutional-Lite** [MODERATE]

**Implementation Feasibility** · Deployability: **High** [STRONG] (88/100) · Rebalancing Complexity: **Moderate** [MODERATE] · Liquidity: **High** [HIGH] · Execution Friction: **Low** [LOW] · Est. Turnover ~18%/yr · Est. Slippage ~8 bp

**Benchmark Context** · Reference: **80/20 Growth (Global Equity / Bonds)** · Bench Return ~11.2% · Tracking Deviation: **High** [HIGH] (9.5% TE) · Active Share: **High** [HIGH] (61%) · Style Drift: **US-underweight · Crypto-tilted**

> *الأرقام تقريبية ومبنية على افتراضات تاريخية طويلة المدى. لا تُعدّ ضماناً للأداء المستقبلي.*

---

## B. تحليل جدوى التفويض

> تحقق من القيود المُفعَّلة قبل التحسين. كل قيد يُعرض مع القيمة الفعلية والحالة.

| القيد | الحد | الفعلي | الحالة |
|------------|-------|--------|--------|
| Region cap · US | 60.0% | 42.0% | [PASS] |
| Region cap · GCC | 35.0% | 34.0% | [NEAR CAP] |
| Region cap · Egypt | 15.0% | 8.5% | [PASS] |
| Region cap · Crypto | 10.0% | 10.0% | [AT CAP] |
| Region cap · Bonds | 20.0% | 5.5% | [PASS] |
| Beta cap (vs MSCI World) | 1.30 | 0.74 | [PASS] |
| Volatility cap (annualized) | 25.0% | 16.5% | [PASS] |
| Minimum bonds + cash floor | 5.0% | 5.5% | [AT FLOOR] |
| Holdings count | ≥ 5 | 10.0% | [PASS] |

---

## C. Risk Diagnostics

### Adaptive Risk Disclosures

*Conditional on the constructed portfolio — only risks that materially apply are surfaced.*

| Severity | Topic | Note |
|----------|-------|------|
| [HIGH] | Crypto Liquidity Discontinuity | Crypto exposure of 10% subject to 24/7 trading, regulatory regime shifts, and liquidity discontinuities during stress events. Classify as satellite, not core. |
| [MODERATE] | GCC Cyclical Concentration | GCC weighting of 34% creates oil-price and geopolitical concentration. Co-moves with commodity cycle and USD direction (peg-driven). |
| [MODERATE] | Cross-Currency Exposure | 43% of portfolio in non-USD-denominated holdings. SAR/AED pegged to USD; EGP floats — currency translation risk applies during reporting. |

### Structural Asset-Class Risk Factors

| Asset Class | Risk Factor |
|-------------|-------------|
| GCC / KSA | Oil price sensitivity · Geopolitical risk · Currency peg risk |
| Egypt | Currency devaluation risk · Political risk · High inflation |
| EM Bonds | Default risk · FX risk · Liquidity risk |

> *Currency risk applies to non-USD holdings (SAR pegged, AED pegged, EGP floating).*

### Crypto Analytical Framework

> Crypto positions (10% of portfolio) are evaluated using a separate analytical lens:
> *network activity · ETF flows · realized volatility · liquidity regime · cycle positioning.* Equity valuation multiples and earnings-quality metrics are not applicable.

### Drawdown Modeling Note

> Modeled max drawdown is an estimate, not a guarantee. Crisis episodes (2008, 2020, 2022) frequently exceeded model predictions by 30–50%. Stress-test against −30%, −40%, and −50% scenarios before committing capital.

## D. Allocation Logic

**Mandate:** Long-Horizon Growth Mandate · High return target · Diversified global growth · Long-horizon equity tilt

### Regional Allocation

| Region | Weight | ~$ on $100k | Asset Classes |
|--------|--------|-------------|---------------|
| **US** | 42.0% | $420,000 | US Large Cap Tech, GS (GS), B (B) |
| **GCC** | 34.0% | $340,000 | 2222 (2222), 4190 (4190), FERTIGLB (FERTIGLB), ORDS (ORDS) |
| **Crypto** | 10.0% | $100,000 | Bitcoin |
| **Egypt** | 8.5% | $85,000 | EGAL (EGAL) |
| **Bonds** | 5.5% | $55,000 | EM Bonds |

### Optimal Asset Weights

| Asset Class | Region | Weight | ~$ on $100k | Proxy | Role | Construction Rationale |
|-------------|--------|--------|-------------|-------|------|------------------------|
| **US Large Cap Tech** | US | 25.0% | $250,000 | `QQQ` | Strategic Core | Long-duration growth sleeve · captures secular AI/tech earnings |
| **Bitcoin** | Crypto | 10.0% | $100,000 | `BTC-USD` | Opportunistic Satellite | Asymmetric satellite · high-volatility return contributor (not a hedge) |
| **EM Bonds** | Bonds | 5.5% | $55,000 | `EMB` | Income / Diversification | EM credit · spread carry with FX/default risk overlay |
| **2222 (2222)** | GCC | 8.5% | $85,000 | `2222` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |
| **4190 (4190)** | GCC | 8.5% | $85,000 | `4190` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |
| **FERTIGLB (FERTIGLB)** | GCC | 8.5% | $85,000 | `FERTIGLB` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |
| **ORDS (ORDS)** | GCC | 8.5% | $85,000 | `ORDS` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |
| **EGAL (EGAL)** | Egypt | 8.5% | $85,000 | `EGAL` | Tactical Allocation | Frontier-market tilt · structural reform exposure (higher idiosyncratic risk) |
| **GS (GS)** | US | 8.5% | $85,000 | `GS` | Tactical Allocation | US equity core · liquid global benchmark proxy |
| **B (B)** | US | 8.5% | $85,000 | `B` | Tactical Allocation | Energy cyclicality · inflation hedge and macro pro-cyclical exposure |

### Diversification Benefit

> **Diversification Ratio:** 1.40x — portfolio vol (16.5%) is 29% lower than weighted average of individual vols (23.1%)
> **vs Equal Weight:** Optimized vol 16.5% vs equal-weight 12.3%

### Benchmark-Relative Attribution

*Decomposition of excess return vs profile-mapped benchmark — distinguishes factor-driven from selection-driven performance.*

| Component | Value | Interpretation |
|-----------|-------|----------------|
| Excess Return vs Benchmark | +5.01% | Total active return |
| Beta Contribution | +0.12% | Factor-driven (market sensitivity differential) |
| Residual Contribution | +4.90% | Selection / concentration effects |

> *Excess return concentrated in residual/selection effects — review concentration risk before attributing to skill.*

> *Implementation note: Institutional rounding applied (5%/2.5%/1% tiered grid). 21 sub-2.5% positions consolidated. Sharpe drift: -0.00.*

---

## E. خطة إعادة التوازن

> إجراءات مُحدَّدة لتقليل التركز، مع الأثر الكمّي على بيتا، التقلب، ودوران المحفظة.

| الإجراء | بيتا قبل→بعد | تقلب قبل→بعد | تقليل التركز | الدوران | الصعوبة |
|-----------------|-------------------|------------------|-----------------|----------|------------|
| Reduce US Large Cap Tech (QQQ) 25.0% → 15.0% | 0.745 → 0.677 (-0.067) | 16.46% → 16.54% (+0.08pp) | −10.0pp | 10.0pp | [HIGH] |

> *تقدير الأثر مبني على إعادة توزيع الوزن المُخفَّض على باقي الأصول بالتناسب.*

### خطوات التنفيذ المقترحة

1. افتح حساب وساطة مناسب للأسواق: Global
2. وزّع $1.0M حسب نسب الأصول في القسم D
3. أعد التوازن كل 6-12 شهر
4. تجنّب التصفية في تراجعات السوق قصيرة المدى — أفق 8 سنوات يستوعب دورات السوق


---


---

### تشخيصات كفاءة التنفيذ

| المقياس | القيمة | التصنيف |
| --- | --- | --- |
| معدل الدوران | 100.0% | (عالٍ) |
| فجوة التنفيذ | 21.2 bp | (متوسط) |
| أثر السوق | 3.7 bp | (منخفض) |
| الانزلاق المقدر | 19.2 bp | (متوسط) |
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
| soft landing | 30.0% | -12% | +44% | +133% | -24% | 4.0 |
| recession | 20.0% | -47% | +9% | +127% | -39% | 0.0 |
| stagflation | 10.0% | -46% | +19% | +161% | -40% | 0.0 |
| ai productivity boom | 10.0% | -18% | +55% | +184% | -28% | 4.0 |
| energy shock | 15.0% | -46% | +15% | +140% | -38% | 0.0 |
| liquidity crisis | 15.0% | -63% | +5% | +183% | -52% | 0.0 |

**الإجمالي المرجح بالاحتمالات**

| المقياس | القيمة |
| --- | --- |
| نطاق القيمة النهائية المتوقع (حقيقي) | $0.58 - $2.45 (P10-P90) |
| احتمال الخسارة (حقيقي) | 33% |
| احتمال بلوغ الهدف (>=4% حقيقي سنويا) | 54% |
| القيمة النهائية للعشر الأدنى | $0.58 |
| نطاق الانخفاض المتوقع | -34% إلى -61% |
| مدة التعافي (الوسيط) | 0 أشهر |

*إطار توزيعي فقط - تعكس النتائج افتراضات نموذجية وليست توقعا قطعيا.*

## G. ملحق المراجعة

| الحقل | القيمة |
|-------|--------|
| معرّف اللقطة (Snapshot ID) | `cb32951a8bc2` |
| هاش الكون الاستثماري | `36d2e7a59a86` |
| الـ Solver | CLARABEL (cvxpy QP) |
| حالة الـ Solver | optimal |
| عدد الأصول (الكون) | 31 |
| عدد الأصول (المختارة) | 10 |
| Max Beta | 1.3 |
| Max Volatility | 25.0% |
| Min Bonds + Cash | 5.0% |
| Max Drawdown (مطلوب) | غير محدد% |
| Risk Aversion | 1.5 |
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
| العائد النسبي | -0.93% | (متوسط) |
| خطأ التتبع | 0.67% | (منخفض) |
| نسبة المعلومات | -1.40 | (منخفض) |
| ألفا المتجدد (12 شهر) | +0.20% | — |
| بيتا المتجدد (12 شهر) | 0.03 | — |
| الانخفاض النسبي | -0.30% | (منخفض) |
| احتواء الصعود | 0.08 | (منخفض) |
| احتواء الهبوط | -0.89 | (منخفض) |
| التذبذب النسبي | 0.08 | — |
| الحصة النشطة | 15.50% | (منخفض) |
| انحراف الأسلوب | material | — |

**تحليل العائد الفائض**

| المكون | المساهمة (نقطة مئوية) |
| --- | --- |
| أثر التخصيص | -0.12 |
| أثر الاختيار | -0.89 |
| أثر العوامل | -1.13 |
| أثر التركيز | +1.20 |

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
| benchmark_relative hash | 968407ff4b70acce |
| execution_diag hash | 9b45f5f8faf2e6e6 |
| factor_decomp hash | 589884420dfffeae |
| forward_scenario hash | 42ab101e4f69a77c |
| benchmark_relative version | 0.2.0 |
| execution_diag version | 0.2.0 |
| factor_decomp version | 0.2.0 |
| forward_scenario version | 0.2.0 |
| phase_h version | 0.1.0 |
