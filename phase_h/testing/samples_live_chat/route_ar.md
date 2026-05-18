# EisaX Global Portfolio — Balanced Multi-Asset Mandate
**التاريخ:** May 18, 2026  |  **رأس المال:** $10,000  |  **المدة:** 5 سنوات  |  **الأسواق:** US + GCC + Gold

---

## A. الملخص التنفيذي

| المؤشر | القيمة | التقييم |
|--------|--------|---------|
| العائد المتوقع (سنوي) | **~13.5%** | [STRONG] |
| التقلب المتوقع | **~12.3%** | [MODERATE] |
| Sharpe Ratio | **~0.73** | [ACCEPTABLE] |
| Beta (vs MSCI World) | **~0.70** | [LOW] |
| القيمة المتوقعة بعد 5 سنوات | **$18,811** | ربح متوقع **$8,811** |

**تصنيف نظام المحفظة:** **Multi-Asset Macro**
> Balanced cross-asset construction; no single regime dominates. Targets diversification across factor and macro drivers.
> **سلوك المحفظة مقابل المؤشر:** Designed for regime-balanced behavior; expect peer-like performance across most macro environments with reduced tail volatility.

**Confidence Calibration** · Score: **85%** · Evidence Breadth: **Broad** · Coverage Quality: **Full** · Reliability Tier: **Institutional** [STRONG]

**Implementation Feasibility** · Deployability: **High** [STRONG] (97/100) · Rebalancing Complexity: **Moderate** [MODERATE] · Liquidity: **High** [HIGH] · Execution Friction: **Low** [LOW] · Est. Turnover ~14%/yr · Est. Slippage ~3 bp

**Benchmark Context** · Reference: **60/40 Balanced (Global Equity / Bonds, Gold overlay)** · Bench Return ~11.5% · Tracking Deviation: **Moderate** [MODERATE] (4.0% TE) · Active Share: **High** [HIGH] (45%) · Style Drift: **US-underweight · Hedge-overweight**

> *الأرقام تقريبية ومبنية على افتراضات تاريخية طويلة المدى. لا تُعدّ ضماناً للأداء المستقبلي.*

---

## B. تحليل جدوى التفويض

> تحقق من القيود المُفعَّلة قبل التحسين. كل قيد يُعرض مع القيمة الفعلية والحالة.

| القيد | الحد | الفعلي | الحالة |
|------------|-------|--------|--------|
| Region cap · US | 51.7% | 48.5% | [PASS] |
| Region cap · GCC | 31.7% | 31.5% | [AT CAP] |
| Region cap · Gold | 21.7% | 20.0% | [NEAR CAP] |
| Beta cap (vs MSCI World) | 1.00 | 0.70 | [PASS] |
| Volatility cap (annualized) | 18.0% | 12.3% | [PASS] |
| Minimum bonds + cash floor | 15.0% | 0.00 | [AUTO-RELAXED (bonds/cash region not included)] |
| Holdings count | ≥ 5 | 9.0% | [PASS] |

---

## C. Risk Diagnostics

### Adaptive Risk Disclosures

*Conditional on the constructed portfolio — only risks that materially apply are surfaced.*

| Severity | Topic | Note |
|----------|-------|------|
| [MODERATE] | GCC Cyclical Concentration | GCC weighting of 32% creates oil-price and geopolitical concentration. Co-moves with commodity cycle and USD direction (peg-driven). |
| [MODERATE] | Cross-Currency Exposure | 32% of portfolio in non-USD-denominated holdings. SAR/AED pegged to USD; EGP floats — currency translation risk applies during reporting. |

### Structural Asset-Class Risk Factors

| Asset Class | Risk Factor |
|-------------|-------------|
| GCC / KSA | Oil price sensitivity · Geopolitical risk · Currency peg risk |
| Gold | No yield · Storage cost · USD-sensitive |

> *Currency risk applies to non-USD holdings (SAR pegged, AED pegged, EGP floating).*

### Drawdown Modeling Note

> Modeled max drawdown is an estimate, not a guarantee. Crisis episodes (2008, 2020, 2022) frequently exceeded model predictions by 30–50%. Stress-test against −30%, −40%, and −50% scenarios before committing capital.

## D. Allocation Logic

**Mandate:** Balanced Multi-Asset Mandate · Moderate growth · Multi-asset · Diversified core allocation

### Regional Allocation

| Region | Weight | ~$ on $100k | Asset Classes |
|--------|--------|-------------|---------------|
| **US** | 48.5% | $4,850 | US Large Cap Tech, US Mid-Cap Equity, GS (GS), VLO (VLO) |
| **GCC** | 31.5% | $3,150 | 2222 (2222), 4190 (4190), FERTIGLB (FERTIGLB), ORDS (ORDS) |
| **Gold** | 20.0% | $2,000 | Gold |

### Optimal Asset Weights

| Asset Class | Region | Weight | ~$ on $100k | Proxy | Role | Construction Rationale |
|-------------|--------|--------|-------------|-------|------|------------------------|
| **US Large Cap Tech** | US | 25.0% | $2,500 | `QQQ` | Strategic Core | Long-duration growth sleeve · captures secular AI/tech earnings |
| **US Mid-Cap Equity** | US | 8.0% | $800 | `MDY` | Tactical Allocation | US equity core · liquid global benchmark proxy |
| **Gold** | Gold | 20.0% | $2,000 | `GLD` | Macro Hedge | Macro hedge · equity-duration compression, USD-weakening regimes |
| **2222 (2222)** | GCC | 8.5% | $850 | `2222` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |
| **4190 (4190)** | GCC | 8.5% | $850 | `4190` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |
| **FERTIGLB (FERTIGLB)** | GCC | 6.0% | $600 | `FERTIGLB` | Satellite / Diversifier | Regional exposure · GCC growth premium, low correlation to US equities |
| **ORDS (ORDS)** | GCC | 8.5% | $850 | `ORDS` | Tactical Allocation | Regional exposure · GCC growth premium, low correlation to US equities |
| **GS (GS)** | US | 7.5% | $750 | `GS` | Tactical Allocation | US equity core · liquid global benchmark proxy |
| **VLO (VLO)** | US | 8.0% | $800 | `VLO` | Tactical Allocation | Energy cyclicality · inflation hedge and macro pro-cyclical exposure |

### Diversification Benefit

> **Diversification Ratio:** 1.50x — portfolio vol (12.3%) is 34% lower than weighted average of individual vols (18.4%)
> **vs Equal Weight:** Optimized vol 12.3% vs equal-weight 12.2%

### Benchmark-Relative Attribution

*Decomposition of excess return vs profile-mapped benchmark — distinguishes factor-driven from selection-driven performance.*

| Component | Value | Interpretation |
|-----------|-------|----------------|
| Excess Return vs Benchmark | +1.94% | Total active return |
| Beta Contribution | -0.68% | Factor-driven (market sensitivity differential) |
| Residual Contribution | +2.61% | Selection / concentration effects |

> *Excess return concentrated in residual/selection effects — review concentration risk before attributing to skill.*

> *Implementation note: Institutional rounding applied (5%/2.5%/1% tiered grid). 7 sub-2.5% positions consolidated. Sharpe drift: -0.00.*

---

## E. خطة إعادة التوازن

> إجراءات مُحدَّدة لتقليل التركز، مع الأثر الكمّي على بيتا، التقلب، ودوران المحفظة.

| الإجراء | بيتا قبل→بعد | تقلب قبل→بعد | تقليل التركز | الدوران | الصعوبة |
|-----------------|-------------------|------------------|-----------------|----------|------------|
| Reduce US Large Cap Tech (QQQ) 25.0% → 15.0% | 0.696 → 0.622 (-0.074) | 12.26% → 11.60% (-0.66pp) | −10.0pp | 10.0pp | [HIGH] |
| Reduce Gold (GLD) 20.0% → 12.0% | 0.696 → 0.770 (+0.075) | 12.26% → 13.04% (+0.78pp) | −8.0pp | 8.0pp | [MODERATE] |

> *تقدير الأثر مبني على إعادة توزيع الوزن المُخفَّض على باقي الأصول بالتناسب.*

### خطوات التنفيذ المقترحة

1. افتح حساب وساطة مناسب للأسواق: US + GCC + Gold
2. وزّع $10,000 حسب نسب الأصول في القسم D
3. أعد التوازن كل 6-12 شهر
4. تجنّب التصفية في تراجعات السوق قصيرة المدى — أفق 5 سنوات يستوعب دورات السوق


---

### تشخيصات كفاءة التنفيذ

| المقياس | القيمة | التصنيف |
| --- | --- | --- |
| معدل الدوران | 100.0% | (عالٍ) |
| فجوة التنفيذ | 15.8 bp | (متوسط) |
| أثر السوق | 2.7 bp | (منخفض) |
| الانزلاق المقدر | 13.8 bp | (متوسط) |
| تعقيد التنفيذ | high | — |
| ضغط السيولة | moderate | — |
| تكرار إعادة التوازن | quarterly | — |

*عقوبة معدل الدوران: غير مطبقة (linear λ=0.0010, quadratic λ=0.0005)*

*التنفيذ الواعي بالضرائب يتطلب بيانات الحصص على مستوى الحساب؛ مكان مخصص بانتظار التكامل مع تغذية الحصص الضريبية من الوسيط.*

*ملاحظة سيولة أسواق الخليج: GCC legs use a Sun-Thu trading calendar; execution windows should account for the four-day local trading week.*

## F. طبقة التعليق بالذكاء الاصطناعي — نظرة مدير الاستثمار

*تعليق مولّد بالذكاء الاصطناعي. الأقسام A–E أعلاه قائمة على حسابات قابلة للتكرار.*

> **الاستثمار المنطقي**  
تستند هذه البنية إلى تعرض لعوامل السوق الأمريكية (25% تقنية كبرى، 8% أسهم متوسطة) مع تنويع إقليمي عبر الأسهم الخليجية (4190، 2222، ORDS) ووزن هامشي للذهب.  
يعكس العائد المتوقع ~13.47% وتقلب ~12.26% ملفاً متوازناً مناسباً لأفق 5 سنوات، مع اعتماد على عوامل النمو والتضخم ضمن نظام متعدد الأصول.

**متجه الخطر الرئيسي**  
يتمثل الخطر الأكثر جوهرية في تركز الحساسية تجاه قطاع التكنولوجيا الأميركي (25%)، والذي يظهر اعتماداً على سيناريو استمرار بيئة أسعار الفائدة المنخفضة ونمو الأرباح.


---

## H. توزيع السيناريوهات المستقبلية

*الأفق: 5.0 سنوات · التضخم: 2.0% · البذرة: 42 · المسارات: 12000*

| السيناريو | الاحتمال | القيمة النهائية (P10) | القيمة النهائية (P50) | القيمة النهائية (P90) | أقصى انخفاض (P50) | مدة التعافي (أشهر، P50) |
| --- | --- | --- | --- | --- | --- | --- |
| soft landing | 30.0% | -13% | +35% | +109% | -20% | 5.0 |
| recession | 20.0% | -44% | +7% | +87% | -34% | 0.0 |
| stagflation | 10.0% | -36% | +23% | +128% | -33% | 0.0 |
| ai productivity boom | 10.0% | -15% | +45% | +145% | -24% | 4.0 |
| energy shock | 15.0% | -34% | +20% | +119% | -32% | 0.0 |
| liquidity crisis | 15.0% | -56% | +6% | +139% | -45% | 0.0 |

**الإجمالي المرجح بالاحتمالات**

| المقياس | القيمة |
| --- | --- |
| نطاق القيمة النهائية المتوقع (حقيقي) | $0.65 - $2.17 (P10-P90) |
| احتمال الخسارة (حقيقي) | 32% |
| احتمال بلوغ الهدف (>=4% حقيقي سنويا) | 51% |
| القيمة النهائية للعشر الأدنى | $0.65 |
| نطاق الانخفاض المتوقع | -29% إلى -54% |
| مدة التعافي (الوسيط) | 1 أشهر |

*إطار توزيعي فقط - تعكس النتائج افتراضات نموذجية وليست توقعا قطعيا.*

## G. ملحق المراجعة

| الحقل | القيمة |
|-------|--------|
| معرّف اللقطة (Snapshot ID) | `18415711844b` |
| هاش الكون الاستثماري | `d7abbfd1bb0e` |
| الـ Solver | CLARABEL (cvxpy QP) |
| حالة الـ Solver | optimal |
| عدد الأصول (الكون) | 16 |
| عدد الأصول (المختارة) | 9 |
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

*المؤشر: S&P 500 (SPY) · الموثوقية: Institutional-Lite · النافذة: 3شهر*

| المقياس | القيمة | التصنيف |
| --- | --- | --- |
| العائد النسبي | -0.04% | (متوسط) |
| خطأ التتبع | 0.66% | (منخفض) |
| نسبة المعلومات | -0.06 | (منخفض) |
| ألفا المتجدد (12 شهر) | +0.31% | — |
| بيتا المتجدد (12 شهر) | 0.43 | — |
| الانخفاض النسبي | -0.20% | (منخفض) |
| احتواء الصعود | 0.41 | (منخفض) |
| احتواء الهبوط | -0.06 | (منخفض) |
| التذبذب النسبي | 0.69 | — |
| الحصة النشطة | 20.00% | (منخفض) |
| انحراف الأسلوب | material | — |

**تحليل العائد الفائض**

| المكون | المساهمة (نقطة مئوية) |
| --- | --- |
| أثر التخصيص | +0.27 |
| أثر الاختيار | -0.38 |
| أثر العوامل | -0.35 |
| أثر التركيز | +0.42 |

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
| benchmark_relative hash | 9b2a735e2b684abf |
| execution_diag hash | 089ff6c6e9d480c7 |
| factor_decomp hash | c642375d721c514d |
| forward_scenario hash | 3860b406242ce360 |
| benchmark_relative version | 0.2.0 |
| execution_diag version | 0.2.0 |
| factor_decomp version | 0.2.0 |
| forward_scenario version | 0.2.0 |
| phase_h version | 0.1.0 |
