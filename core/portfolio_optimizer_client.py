import httpx
import json
import logging
import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class PortfolioOptimizerClient:
    """عميل API للتواصل مع Portfolio Optimizer"""

    def __init__(self):
        self.base_url = "https://api.portfoliooptimizer.io/v1"
        self.timeout = httpx.Timeout(60.0)

    async def optimize_portfolio(
        self,
        assets: List[str],
        returns: List[float],
        covariance_matrix: List[List[float]],
        method: str = "max_sharpe"
    ) -> Dict:
        """
        تحسين المحفظة باستخدام Portfolio Optimizer مع معالجة البيانات
        """
        try:
            # 1. تنظيف البيانات من أي قيم غير صالحة وتحويلها لقوائم JSON
            clean_returns = np.nan_to_num(returns, nan=0.0).tolist()
            clean_matrix = np.nan_to_num(covariance_matrix, nan=0.0).tolist()

            # 2. بناء الـ Payload الصحيح (assetsReturns هو المفتاح المطلوب)
# بناء الـ Payload مع إضافة العائد خالي المخاطر (مثلاً 2% أو 0.02)
            payload = {
                "assets": len(assets),
                "assetsReturns": clean_returns,
                "assetsCovarianceMatrix": clean_matrix,
                "riskFreeRate": 0.02  # إضافة هذا السطر لحل مشكلة الـ 400
            }
            endpoint = "/portfolio/optimization/maximum-sharpe-ratio"
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}{endpoint}",
                    json=payload
                )
                
                if response.status_code != 200:
                    logger.error(f"❌ API Error {response.status_code}: {response.text}")
                    return {"success": False, "error": response.text}

                data = response.json()
                
                # 3. استخراج الأوزان
                weights_list = data.get("assetsWeights", [])
                
                # ربط الأوزان بأسماء الأصول في ديكشنري
                weights_dict = dict(zip(assets, weights_list))
                
                return {
                    "success": True,
                    "assets": assets,
                    "weights": weights_dict,
                    "expected_return": np.dot(weights_list, clean_returns),
                    "volatility": np.sqrt(np.dot(weights_list, np.dot(clean_matrix, weights_list))),
                    "sharpe_ratio": data.get("portfolioSharpeRatio", 0)
                }

        except Exception as e:
            logger.error(f"❌ Portfolio Optimizer Exception: {e}")
            return {"success": False, "error": str(e)}
