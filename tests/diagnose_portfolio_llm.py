"""
Diagnose Portfolio Builder LLM Issues
=====================================
Test what the LLM is returning for intent extraction.
"""

import sys

sys.path.insert(0, '..')

# Your test request
USER_REQUEST = """Create a concentrated, risk-controlled portfolio using only the tickers below (do not add any other tickers). Goal: maximize risk-adjusted expected return over the next 12 months while controlling drawdowns (high-vol biotech universe).
Universe (only these): ADPT, DYN, CERT, COGT, SRPT, MNKD, CDTX, BCRX, ARDX, TXG, VRDN, TWST, EWTX, AVDL, JANX, SDGR, ABCL, NVAX, STOK, AMLX, NTLA, PCRX, XERS, GPCR, NEOG, ELVN, BBNX, PGEN, QURE, WVE, OPK, DVAX, ORIC, MDXG, TRVI, ATAI, SPRY, CRMD, FTRE, ABUS, SANA, TSHA, PHAT, IOVA, GERN, AVXL, IMNM, GOSS, AKBA, SVRA, PROK, DAWN, TNGX, KURA, KALV, VIR, NRIX, RLAY, MRVI, MYGN, RZLT, TERN, CMPX, AQST, VSTM, ATYR, ESPR, PRME, PSNL, SRDX, XOMA, REPL, ERAS, CRVS, ATXS, LXRX, ALT, ALDX, ABEO, CTMX, OCGN, LRMR, RCKT, PACB, IMRX, AUTL, FULC, ABSI
Constraints: Select 12-18 stocks, max 10% position, max 35% per theme.
"""


def test_llm_intent():
    print("=" * 60)
    print("TESTING LLM INTENT EXTRACTION")
    print("=" * 60)

    # Test 1: Check if LLM client works
    print("\n1. Testing LLM Client...")
    try:
        from dashboard.portfolio_builder import get_llm_client
        client = get_llm_client()
        if client:
            model_name = getattr(client, 'model_name', 'Unknown')
            print(f"   ✓ LLM Client available: {model_name}")
        else:
            print("   ✗ LLM Client is None!")
            return
    except Exception as e:
        print(f"   ✗ Error getting LLM client: {e}")
        return

    # Test 2: Get the intent extraction prompt
    print("\n2. Building Intent Prompt...")
    try:
        from dashboard.portfolio_engine import get_intent_extraction_prompt

        # Minimal sectors/tickers for testing
        sectors = ["Healthcare", "Technology"]
        tickers = ["NVAX", "CERT", "ADPT", "BCRX"]

        prompt = get_intent_extraction_prompt(
            user_request=USER_REQUEST,
            sectors=sectors,
            tickers=tickers
        )
        print(f"   ✓ Prompt length: {len(prompt)} chars")
        print(f"\n   --- PROMPT PREVIEW (first 500 chars) ---")
        print(f"   {prompt[:500]}...")
    except Exception as e:
        print(f"   ✗ Error building prompt: {e}")
        return

    # Test 3: Call the LLM
    print("\n3. Calling LLM...")
    try:
        messages = [
            {"role": "system",
             "content": "You extract structured investment intent. Output ONLY a JSON object, no other text."},
            {"role": "user", "content": prompt}
        ]

        llm_output = client.chat(messages)

        print(f"   ✓ LLM responded, length: {len(llm_output)} chars")
        print(f"\n   --- RAW LLM OUTPUT ---")
        print(llm_output[:2000])
        if len(llm_output) > 2000:
            print(f"   ... (truncated, total {len(llm_output)} chars)")
    except Exception as e:
        print(f"   ✗ LLM call failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 4: Try to parse JSON
    print("\n4. Parsing JSON from output...")
    try:
        from dashboard.portfolio_engine import parse_llm_intent

        intent, errors = parse_llm_intent(llm_output, valid_tickers=tickers, valid_sectors=sectors)

        print(f"   Intent parsed:")
        print(f"     - objective: {intent.objective}")
        print(f"     - risk_level: {intent.risk_level}")
        print(f"     - max_holdings: {intent.max_holdings}")
        print(
            f"     - tickers_include: {intent.tickers_include[:5]}..." if intent.tickers_include else "     - tickers_include: []")
        print(f"     - restrict_to_tickers: {intent.restrict_to_tickers}")

        if errors:
            print(f"\n   ⚠️ Parse Errors:")
            for e in errors:
                print(f"     - {e}")
    except Exception as e:
        print(f"   ✗ Parse failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 5: Try manual JSON extraction
    print("\n5. Manual JSON extraction test...")
    import re
    import json

    # Try to find JSON in various formats
    patterns = [
        r'```json\s*([\s\S]*?)\s*```',  # ```json ... ```
        r'```\s*([\s\S]*?)\s*```',  # ``` ... ```
        r'\{[\s\S]*\}',  # Raw JSON object
    ]

    for i, pattern in enumerate(patterns):
        match = re.search(pattern, llm_output)
        if match:
            json_str = match.group(1) if '```' in pattern else match.group(0)
            try:
                parsed = json.loads(json_str)
                print(f"   ✓ Pattern {i + 1} found valid JSON!")
                print(f"     Keys: {list(parsed.keys())[:10]}")
                break
            except json.JSONDecodeError as e:
                print(f"   ✗ Pattern {i + 1} found text but invalid JSON: {e}")
    else:
        print("   ✗ No JSON pattern matched")


if __name__ == "__main__":
    test_llm_intent()