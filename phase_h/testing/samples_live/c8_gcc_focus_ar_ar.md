# EisaX Global Portfolio — Balanced Multi-Asset Mandate
**التاريخ:** May 18, 2026  |  **رأس المال:** $300,000  |  **المدة:** 6 سنوات  |  **الأسواق:** GCC + Cash + Bonds

---

## A. الملخص التنفيذي

| المؤشر | القيمة | التقييم |
|--------|--------|---------|
| العائد المتوقع (سنوي) | **~9.5%** | [MODERATE] |
| التقلب المتوقع | **~8.3%** | [LOW] |
| Sharpe Ratio | **~0.60** | [ACCEPTABLE] |
| Beta (vs MSCI World) | **~0.20** | [LOW] |
| القيمة المتوقعة بعد 6 سنوات | **$517,137** | ربح متوقع **$217,137** |

**تصنيف نظام المحفظة:** **Defensive Income**
> Income-generating sleeves dominate. Lower sensitivity to equity drawdowns; primary risk vector is duration and credit spread widening.
> **سلوك المحفظة مقابل المؤشر:** Outperforms during equity drawdowns and disinflationary cycles; lags in strong risk-on rallies and steepening yield curves.

**Confidence Calibration** · Score: **85%** · Evidence Breadth: **Broad** · Coverage Quality: **Full** · Reliability Tier: **Institutional** [STRONG]

**Implementation Feasibility** · Deployability: **High** [STRONG] (100/100) · Rebalancing Complexity: **Moderate** [MODERATE] · Liquidity: **High** [HIGH] · Execution Friction: **Low** [LOW] · Est. Turnover ~12%/yr · Est. Slippage ~3 bp

**Benchmark Context** · Reference: **60/40 Balanced (Global Equity / Bonds, Gold overlay)** · Bench Return ~7.4% · Tracking Deviation: **Moderate** [MODERATE] (3.6% TE) · Active Share: **High** [HIGH] (44%) · Style Drift: **Duration-underweight**

> *الأرقام تقريبية ومبنية على افتراضات تاريخية طويلة المدى. لا تُعدّ ضماناً للأداء المستقبلي.*

---

## B. تحليل جدوى التفويض

> تحقق من القيود المُفعَّلة قبل التحسين. كل قيد يُعرض مع القيمة الفعلية والحالة.

| القيد | الحد | الفعلي | الحالة |
|------------|-------|--------|--------|
| Region cap · GCC | 40.0% | 40.0% | [AT CAP] |
| Region cap · Bonds | 45.0% | 40.0% | [PASS] |
| Region cap · Cash | 20.0% | 20.0% | [AT CAP] |
| Beta cap (vs MSCI World) | 1.00 | 0.20 | [PASS] |
| Volatility cap (annualized) | 18.0% | 8.3% | [PASS] |
| Minimum bonds + cash floor | 15.0% | 60.0% | [PASS] |
| Holdings count | ≥ 5 | 8.0% | [PASS] |

---

## C. Risk Diagnostics

### Adaptive Risk Disclosures

*Conditional on the constructed portfolio — only risks that materially apply are surfaced.*

| Severity | Topic | Note |
|----------|-------|------|
| [MODERATE] | GCC Cyclical Concentration | GCC weighting of 40% creates oil-price and geopolitical concentration. Co-moves with commodity cycle and USD direction (peg-driven). |
| [MODERATE] | Cross-Currency Exposure | 40% of portfolio in non-USD-denominated holdings. SAR/AED pegged to USD; EGP floats — currency translation risk applies during reporting. |

### Structural Asset-Class Risk Factors

| Asset Class | Risk Factor |
|-------------|-------------|
| GCC / KSA | Oil price sensitivity · Geopolitical risk · Currency peg risk |
| EM Bonds | Default risk · FX risk · Liquidity risk |

> *Currency risk applies to non-USD holdings (SAR pegged, AED pegged, EGP floating).*

### Correlation Cluster Risk

| Severity | Cluster | Combined Weight | Note |
|----------|---------|-----------------|------|
| [HIGH] | GCC + Commodities | 40.0% | Oil/commodity cycle co-movement |

### Drawdown Modeling Note

> Modeled max drawdown is an estimate, not a guarantee. Crisis episodes (2008, 2020, 2022) frequently exceeded model predictions by 30–50%. Stress-test against −30%, −40%, and −50% scenarios before committing capital.

## D. Allocation Logic

**Mandate:** Balanced Multi-Asset Mandate · Moderate growth · Multi-asset · Diversified core allocation

### Regional Allocation

| Region | Weight | ~$ on $100k | Asset Classes |
|--------|--------|-------------|---------------|
| **Bonds** | 40.0% | $120,000 | US Treasuries (LT), EM Bonds |
| **GCC** | 40.0% | $120,000 | 2222 (2222), 4190 (4190), 5110 (5110), FERTIGLB (FERTIGLB), ORDS (ORDS) |
| **Cash** | 20.0% | $60,000 | Cash / T-Bills |

### Optimal Asset Weights

| Asset Class | Region | Weight | ~$ on $100k | Proxy | Role | Construction Rationale |
|-------------|--------|--------|-------------|-------|------|------------------------|
| **US Treasuries (LT)** | Bonds | 15.0% | $45,000 | `TLT` | Income / Diversification | Long-duration UST · negative correlation to equity in deflationary shocks |
| **EM Bonds** | Bonds | 25.0% | $75,000 | `EMB` | Income / Diversification | EM credit · spread carry with FX/default risk overlay |
| **Cash / T-Bills** | Cash | 20.0% | $60,000 | `BIL` | Income / Diversification | Dry powder · liquidity buffer and risk-free yield anchor |
| **2222 (2222)** | GCC | 8.5% | $25,500 | `2222` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |
| **4190 (4190)** | GCC | 8.5% | $25,500 | `4190` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |
| **5110 (5110)** | GCC | 6.0% | $18,000 | `5110` | Satellite / Diversifier | Regional exposure · GCC growth premium, low correlation to US equities |
| **FERTIGLB (FERTIGLB)** | GCC | 8.5% | $25,500 | `FERTIGLB` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |
| **ORDS (ORDS)** | GCC | 8.5% | $25,500 | `ORDS` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |

### Diversification Benefit

> **Diversification Ratio:** 1.92x — portfolio vol (8.3%) is 48% lower than weighted average of individual vols (15.9%)
> **vs Equal Weight:** Optimized vol 8.3% vs equal-weight 11.7%

### Benchmark-Relative Attribution

*Decomposition of excess return vs profile-mapped benchmark — distinguishes factor-driven from selection-driven performance.*

| Component | Value | Interpretation |
|-----------|-------|----------------|
| Excess Return vs Benchmark | +2.11% | Total active return |
| Beta Contribution | +0.32% | Factor-driven (market sensitivity differential) |
| Residual Contribution | +1.78% | Selection / concentration effects |

> *Excess return concentrated in residual/selection effects — review concentration risk before attributing to skill.*

> *Implementation note: Institutional rounding applied (5%/2.5%/1% tiered grid). 3 sub-2.5% positions consolidated. Sharpe drift: +0.00.*

---

## E. خطة إعادة التوازن

> إجراءات مُحدَّدة لتقليل التركز، مع الأثر الكمّي على بيتا، التقلب، ودوران المحفظة.

| الإجراء | بيتا قبل→بعد | تقلب قبل→بعد | تقليل التركز | الدوران | الصعوبة |
|-----------------|-------------------|------------------|-----------------|----------|------------|
| Reduce EM Bonds (EMB) 25.0% → 15.0% | 0.197 → 0.196 (-0.000) | 8.28% → 8.65% (+0.37pp) | −10.0pp | 10.0pp | [HIGH] |
| Reduce US Treasuries (LT) (TLT) 15.0% → 9.0% | 0.197 → 0.232 (+0.035) | 8.28% → 8.44% (+0.16pp) | −6.0pp | 6.0pp | [MODERATE] |

> *تقدير الأثر مبني على إعادة توزيع الوزن المُخفَّض على باقي الأصول بالتناسب.*

### خطوات التنفيذ المقترحة

1. افتح حساب وساطة مناسب للأسواق: GCC + Cash + Bonds
2. وزّع $300,000 حسب نسب الأصول في القسم D
3. أعد التوازن كل 6-12 شهر
4. تجنّب التصفية في تراجعات السوق قصيرة المدى — أفق 6 سنوات يستوعب دورات السوق


---


---

### تشخيصات كفاءة التنفيذ

| المقياس | القيمة | التصنيف |
| --- | --- | --- |
| معدل الدوران | 100.0% | (عالٍ) |
| فجوة التنفيذ | 18.4 bp | (متوسط) |
| أثر السوق | 1.8 bp | (منخفض) |
| الانزلاق المقدر | 16.4 bp | (متوسط) |
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
| soft landing | 30.0% | -6% | +19% | +50% | -10% | 5.0 |
| recession | 20.0% | -21% | +11% | +57% | -18% | 0.0 |
| stagflation | 10.0% | -26% | +7% | +53% | -20% | 0.0 |
| ai productivity boom | 10.0% | -10% | +20% | +61% | -13% | 5.0 |
| energy shock | 15.0% | -23% | +10% | +53% | -18% | 0.0 |
| liquidity crisis | 15.0% | -37% | +3% | +74% | -29% | 0.0 |

**الإجمالي المرجح بالاحتمالات**

| المقياس | القيمة |
| --- | --- |
| نطاق القيمة النهائية المتوقع (حقيقي) | $0.79 - $1.55 (P10-P90) |
| احتمال الخسارة (حقيقي) | 31% |
| احتمال بلوغ الهدف (>=4% حقيقي سنويا) | 39% |
| القيمة النهائية للعشر الأدنى | $0.79 |
| نطاق الانخفاض المتوقع | -16% إلى -33% |
| مدة التعافي (الوسيط) | 3 أشهر |

*إطار توزيعي فقط - تعكس النتائج افتراضات نموذجية وليست توقعا قطعيا.*

## G. ملحق المراجعة

| الحقل | القيمة |
|-------|--------|
| معرّف اللقطة (Snapshot ID) | `ff02c16d069c` |
| هاش الكون الاستثماري | `a46c506fc1d7` |
| الـ Solver | CLARABEL (cvxpy QP) |
| حالة الـ Solver | optimal |
| عدد الأصول (الكون) | 11 |
| عدد الأصول (المختارة) | 8 |
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

*المؤشر: MSCI World (URTH) · الموثوقية: Indicative · النافذة: 3شهر*

| المقياس | القيمة | التصنيف |
| --- | --- | --- |
| العائد النسبي | +0.18% | (متوسط) |
| خطأ التتبع | 0.06% | (منخفض) |
| نسبة المعلومات | 3.06 | (مرتفع) |
| ألفا المتجدد (12 شهر) | 0.00% | — |
| بيتا المتجدد (12 شهر) | 1.00 | — |
| الانخفاض النسبي | 0.00% | (منخفض) |
| احتواء الصعود | 0.00 | (منخفض) |
| احتواء الهبوط | 0.00 | (منخفض) |
| التذبذب النسبي | 0.00 | — |
| الحصة النشطة | 100.00% | (مرتفع) |
| انحراف الأسلوب | severe | — |

**تحليل العائد الفائض**

| المكون | المساهمة (نقطة مئوية) |
| --- | --- |
| أثر التخصيص | +0.92 |
| أثر الاختيار | 0.00 |
| أثر العوامل | 0.00 |
| أثر التركيز | -0.75 |

**سلوك المحفظة النسبي حسب النظام**

- البيئات التي يرجح أن تتفوق فيها المحفظة: سجل الأنظمة غير كاف
- البيئات التي يرجح أن تتأخر فيها المحفظة: سجل الأنظمة غير كاف

خطأ التتبع مرتفع قياسا بتركيب المؤشر؛ وتشير الحصة النشطة فوق 60% إلى انحراف هيكلي ملموس قد يرفع تشتت الأداء المدفوع بالعوامل في أنظمة الضغط.

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
| benchmark_relative hash | 87d22c0514669ed0 |
| execution_diag hash | c80ac70cfe52ead6 |
| factor_decomp hash | d704dc5e0434423c |
| forward_scenario hash | d9126b3600231556 |
| benchmark_relative version | 0.2.0 |
| execution_diag version | 0.2.0 |
| factor_decomp version | 0.2.0 |
| forward_scenario version | 0.2.0 |
| phase_h version | 0.1.0 |
