import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import json

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import state
from core.agents.finance import FinancialAgent

print("Running Export Logic Verification")

class TestExportLogic(unittest.TestCase):
    
    def setUp(self):
        # Mock dependencies in sys.modules to avoid heavy imports
        sys.modules["pypfopt"] = MagicMock()
        sys.modules["pypfopt.expected_returns"] = MagicMock()
        sys.modules["pypfopt.risk_models"] = MagicMock()
        sys.modules["pypfopt.efficient_frontier"] = MagicMock()
        sys.modules["empyrical"] = MagicMock()
        sys.modules["yfinance"] = MagicMock()
        
        # Reset state
        state.last_artifact = None
        self.agent = FinancialAgent()
        
    def test_artifact_state_optimize(self):
        """Test that optimization populates last_artifact"""
        print("\n--- Test: Optimization Artifact ---")
        
        # Mock portfolio manager to return dummy data
        with patch("core.portfolio_manager.optimize_and_get_data") as mock_opt:
            mock_opt.return_value = ({"AAPL": 1.0}, {"sharpe": 2.0, "expected_return": 0.1, "volatility": 0.05})
            
            # Mock generate_strategy_guide_llm to avoid LLM call
            with patch("core.portfolio_manager.generate_strategy_guide_llm") as mock_guide:
                mock_guide.return_value = "# Strategy Guide\n\nBuy AAPL."
                
                # Act
                res = self.agent._handle_optimize("test_id", {"tickers": ["AAPL"]}, "Optimize for AAPL", {})
                
                # Assert
                self.assertIsNotNone(state.last_artifact, "last_artifact should be set")
                self.assertIn(state.last_artifact["type"], ["portfolio", "strategy"])
                content = state.last_artifact["content"]
                self.assertTrue("Buy AAPL" in content or "AAPL:" in content)
                print("SUCCESS: Optimization artifact set correctly.")

    def test_export_pdf_flow(self):
        """Test that export uses the artifact"""
        print("\n--- Test: Export Flow ---")
        
        # Setup Artifact
        state.last_artifact = {
            "type": "custom_report",
            "content": "# My Custom Report\n\nContent content content.",
            "exportable": True,
            "source": "test"
        }
        
        # Act
        with patch("core.report_engine.ReportEngine.generate_pdf") as mock_pdf:
            mock_pdf.return_value = "static/reports/My_Custom_Report.pdf"
            
            res = self.agent._handle_export("test_id", {}, "Export to PDF")
            
            # Assert
            self.assertEqual(res["type"], "report.export")
            self.assertIn("My Custom Report", res["reply"] or "") # Check title usage
            self.assertIn("/static/reports/", res["data"]["url"])
            print("SUCCESS: Export used artifact and returned link.")
            
    def test_analytics_artifact(self):
        """Test that analytics populates last_artifact"""
        print("\n--- Test: Analytics Artifact ---")
        
        # Mock data
        msg = "Analyze AAPL"
        with patch("core.intent_classifier.IntentClassifier.extract_tickers", return_value=["AAPL"]):
            with patch("core.data.get_prices") as mock_prices:
                import pandas as pd
                mock_prices.return_value = pd.DataFrame({"AAPL": [100, 101, 102]})
                
                with patch("core.analytics.generate_technical_summary") as mock_summary:
                    mock_summary.return_value = {
                        "price": 102, "trend": "Bullish", "momentum": "Bullish", 
                        "condition": "Neutral", "rsi": 50, "sma_50": 90, "sma_200": 80, "macd": 1.0
                    }
                    
                    res = self.agent._handle_analytics("test_id", {}, msg)
                    
                    self.assertIsNotNone(state.last_artifact)
                    self.assertEqual(state.last_artifact["type"], "analysis")
                    self.assertIn("CIO Memorandum: AAPL", state.last_artifact["content"])
                    print("SUCCESS: Analytics artifact set correctly.")

    def test_arabic_export_flow(self):
        """Test that Arabic export command works with GeneralAgent bound artifact"""
        print("\n--- Test: Arabic Export Flow ---")
        
        # 1. Simulate GeneralAgent response that should bind an artifact
        from core.agents.general import GeneralAgent
        agent = GeneralAgent()
        
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="هذه محفظة مقترحة:\n| الأصل | النسبة |\n|---|---|\n| أسهم | 50% |"))]
        
        with patch.object(agent, "client_factory") as mock_factory:
            mock_factory.return_value.create_completion.return_value = mock_response
            
            # Context for investment
            context = {"memory": {"tickers": ["AAPL"]}, "history": []}
            # Hybrid classifier would say 'investment' for financial talk
            with patch("core.intent_classifier.IntentClassifier.classify_intent_hybrid", return_value="investment"):
                res = agent.think("عاوز محفظة", context, {})
                
                # Verify artifact bound
                self.assertIsNotNone(state.last_artifact)
                self.assertIn("محفظة", state.last_artifact["content"])
                print("SUCCESS: Arabic chat reply bound to state.last_artifact.")

                # 2. Simulate Export Command (Arabic)
                from core.intent_classifier import IntentClassifier
                intent = IntentClassifier.detect_primary_intent("صدرها لpdf")
                self.assertEqual(intent, "report_export")
                print("SUCCESS: Arabic 'صدرها لpdf' detected as report_export.")
                
                # 3. Test handle_export uses this artifact
                with patch("core.report_engine.ReportEngine.generate_pdf") as mock_pdf:
                    mock_pdf.return_value = "static/reports/report.pdf"
                    finance_agent = FinancialAgent()
                    export_res = finance_agent._handle_export("test_id", {}, "صدرها لpdf")
                    
                    self.assertEqual(export_res["type"], "report.export")
                    self.assertIn("/static/reports/", export_res["data"]["url"])
                    print("SUCCESS: Arabic export successfully delivered PDF from chat artifact.")

    def test_unicode_arabic_encoding(self):
        """Test that Arabic and em-dashes don't crash the PDF engine"""
        print("\n--- Test: Unicode & Arabic Encoding ---")
        from core.report_engine import ReportEngine
        engine = ReportEngine()
        
        arabic_content = "# تقرير الاستثمار\n\n- محفظة مخاطرة: 50%\n- أصول متنوعة — شاملة.\n\nاستثمار موفق."
        title = "تقرير تجريبي"
        
        try:
            path = engine.generate_pdf(title, arabic_content)
            self.assertTrue(os.path.exists(path))
            print(f"SUCCESS: PDF generated with Arabic content at {path}")
            # Optional: os.remove(path)
        except Exception as e:
            self.fail(f"PDF Generation failed with Unicode error: {e}")

    def test_table_rendering(self):
        """Test that tables are rendered correctly in PDF"""
        print("\n--- Test: Table Rendering ---")
        from core.report_engine import ReportEngine
        engine = ReportEngine()
        
        table_content = "# Portfolo Allocation\n\n| Asset | % | Rationale |\n|---|---|---|\n| US Equities | 50% | Growth |\n| Gold | 20% | Hedge |\n\nEnd of report."
        try:
            path = engine.generate_pdf("Table Test", table_content)
            self.assertTrue(os.path.exists(path))
            print(f"SUCCESS: PDF with table generated at {path}")
        except Exception as e:
            self.fail(f"PDF Table generation failed: {e}")

if __name__ == "__main__":
    unittest.main()
