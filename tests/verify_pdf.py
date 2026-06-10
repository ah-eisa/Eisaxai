import sys
import os
import unittest
import shutil

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.report_engine import ReportEngine

print("Running PDF Verification")

class TestPDF(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = "tests/output"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
        
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_pdf_generation(self):
        """Verify ReportEngine creates a valid PDF."""
        engine = ReportEngine(output_dir=self.test_dir)
        
        content = (
            "# Test Report\n"
            "## Section 1\n"
            "This is a test paragraph.\n"
            "- Bullet 1\n"
            "- Bullet 2\n"
        )
        
        filepath = engine.generate_pdf("Test Title", content, filename="test_report.pdf")
        
        print(f"Generated: {filepath}")
        
        self.assertTrue(os.path.exists(filepath))
        self.assertGreater(os.path.getsize(filepath), 1000) # Should be substantial size
        print("SUCCESS: PDF generated successfully.")

if __name__ == "__main__":
    unittest.main()
