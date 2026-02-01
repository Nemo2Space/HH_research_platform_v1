from src.analytics.options_flow import OptionsFlowAnalyzer

analyzer = OptionsFlowAnalyzer()

# Test with a single stock
print("Testing AAPL options flow...")
print(analyzer.get_flow_for_ai("AAPL"))