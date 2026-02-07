import streamlit as st
import time
import subprocess
import sys
import json
import logging
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.title("Yahoo Options Debug")

ticker = st.text_input("Ticker", value="PCAR")

if st.button("Test 1: subprocess.run (capture_output)"):
    st.write("Starting...")
    cmd = [sys.executable, "-m", "src.analytics.yahoo_options_subprocess", ticker, "4"]
    start = time.time()
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
        st.write(f"Done in {time.time()-start:.1f}s, rc={completed.returncode}")
        st.write(f"stdout length: {len(completed.stdout)}")
    except subprocess.TimeoutExpired:
        st.write(f"TIMEOUT after {time.time()-start:.1f}s")

if st.button("Test 2: Popen + communicate"):
    st.write("Starting...")
    cmd = [sys.executable, "-m", "src.analytics.yahoo_options_subprocess", ticker, "4"]
    start = time.time()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = proc.communicate(timeout=5)
        st.write(f"Done in {time.time()-start:.1f}s, rc={proc.returncode}")
        st.write(f"stdout length: {len(stdout)}")
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        st.write(f"TIMEOUT after {time.time()-start:.1f}s")

if st.button("Test 3: Popen + CREATE_NO_WINDOW"):
    st.write("Starting...")
    cmd = [sys.executable, "-m", "src.analytics.yahoo_options_subprocess", ticker, "4"]
    start = time.time()
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0,
        )
        stdout, stderr = proc.communicate(timeout=5)
        st.write(f"Done in {time.time()-start:.1f}s, rc={proc.returncode}")
        st.write(f"stdout length: {len(stdout)}")
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        st.write(f"TIMEOUT after {time.time()-start:.1f}s")

if st.button("Test 4: Thread + subprocess inside thread"):
    st.write("Starting...")
    result = [None]
    def _run():
        cmd = [sys.executable, "-m", "src.analytics.yahoo_options_subprocess", ticker, "4"]
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
        result[0] = completed.stdout
    
    start = time.time()
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=5)
    if t.is_alive():
        st.write(f"THREAD STILL ALIVE after {time.time()-start:.1f}s")
    elif result[0]:
        st.write(f"Done in {time.time()-start:.1f}s, stdout length: {len(result[0])}")
    else:
        st.write(f"Thread returned but no result after {time.time()-start:.1f}s")

if st.button("Test 5: Direct yfinance (NO subprocess)"):
    st.write("Starting direct yfinance call...")
    import yfinance as yf
    start = time.time()
    try:
        stock = yf.Ticker(ticker)
        st.write(f"Ticker created: {time.time()-start:.1f}s")
        info = stock.info
        st.write(f"Info fetched: {time.time()-start:.1f}s, price={info.get('currentPrice')}")
        exps = stock.options
        st.write(f"Expiries fetched: {time.time()-start:.1f}s, count={len(exps) if exps else 0}")
        if exps:
            chain = stock.option_chain(exps[0])
            st.write(f"First chain fetched: {time.time()-start:.1f}s, calls={len(chain.calls)}")
    except Exception as e:
        st.write(f"ERROR at {time.time()-start:.1f}s: {e}")

if st.button("Test 6: OptionsFlowAnalyzer.get_options_chain (EXACT scanner path)"):
    st.write("Starting exact scanner path...")
    from src.analytics.options_flow import OptionsFlowAnalyzer
    ofa = OptionsFlowAnalyzer()
    start = time.time()
    calls, puts, price, source = ofa.get_options_chain(ticker, skip_ibkr=True)
    elapsed = time.time() - start
    st.write(f"Done in {elapsed:.1f}s: calls={len(calls)}, puts={len(puts)}, price={price}, source={source}")

st.write("---")
st.write("Click each test to find which one hangs")
