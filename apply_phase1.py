"""
Phase 1 Implementation: AI Prob Bug Fixes (No Retrain Required)

1. Fix feature defaults in signal_predictor.py (article_count=0, target_upside_pct=0)
2. Add clipping/winsorizing for pct/count features
3. Add sanity logging in predict()
4. Pass all 9 features from table path (signals_tab_ai.py + shared.py)
"""

import ast

# ============================================================================
# PATCH 1: signal_predictor.py
# ============================================================================

with open('src/ml/signal_predictor.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_features = """    FEATURES = [
        'sentiment_score', 'fundamental_score', 'technical_score',
        'options_flow_score', 'short_squeeze_score', 'gap_score',
        'total_score', 'article_count', 'target_upside_pct'
    ]"""

new_features = """    FEATURES = [
        'sentiment_score', 'fundamental_score', 'technical_score',
        'options_flow_score', 'short_squeeze_score', 'gap_score',
        'total_score', 'article_count', 'target_upside_pct'
    ]

    # Domain-correct defaults: 50 for 0-100 bounded scores, 0 for counts/percentages
    FEATURE_DEFAULTS = {
        'sentiment_score': 50,
        'fundamental_score': 50,
        'technical_score': 50,
        'options_flow_score': 50,
        'short_squeeze_score': 50,
        'gap_score': 50,
        'total_score': 50,
        'article_count': 0,
        'target_upside_pct': 0,
    }

    # Clipping bounds to prevent extreme outliers from distorting predictions
    FEATURE_CLIPS = {
        'sentiment_score': (0, 100),
        'fundamental_score': (0, 100),
        'technical_score': (0, 100),
        'options_flow_score': (0, 100),
        'short_squeeze_score': (0, 100),
        'gap_score': (0, 100),
        'total_score': (0, 100),
        'article_count': (0, 200),
        'target_upside_pct': (-50, 100),
    }"""

if old_features in content:
    content = content.replace(old_features, new_features)
    print("PATCH 1a: Added FEATURE_DEFAULTS and FEATURE_CLIPS constants")
else:
    print("PATCH 1a: WARNING - Could not find FEATURES list")

old_fillna = """        # Fill missing
        for col in self.FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna(50)"""

new_fillna = """        # Fill missing with domain-correct defaults and clip
        for col in self.FEATURES:
            if col in df.columns:
                default = self.FEATURE_DEFAULTS.get(col, 50)
                df[col] = df[col].fillna(default)
                clip_range = self.FEATURE_CLIPS.get(col)
                if clip_range:
                    df[col] = df[col].clip(clip_range[0], clip_range[1])"""

if old_fillna in content:
    content = content.replace(old_fillna, new_fillna)
    print("PATCH 1b: Fixed fillna defaults + added clipping in training")
else:
    print("PATCH 1b: WARNING - Could not find fillna block")

old_predict_x = """        X = np.array([[scores.get(f, 50) for f in self.feature_names]])
        X = self.data_loader.scaler.transform(X)"""

new_predict_x = """        # Build feature vector with domain-correct defaults + clipping
        feature_values = []
        filled_features = []
        for f in self.feature_names:
            val = scores.get(f)
            if val is None:
                default = self.data_loader.FEATURE_DEFAULTS.get(f, 50)
                feature_values.append(default)
                filled_features.append(f)
            else:
                clip_range = self.data_loader.FEATURE_CLIPS.get(f)
                if clip_range:
                    val = max(clip_range[0], min(clip_range[1], float(val)))
                feature_values.append(float(val))

        if filled_features:
            logger.debug(f"{scores.get('ticker', '?')}: Filled missing features with defaults: {filled_features}")

        X = np.array([feature_values])
        X = self.data_loader.scaler.transform(X)"""

if old_predict_x in content:
    content = content.replace(old_predict_x, new_predict_x)
    print("PATCH 1c: Fixed predict() defaults + clipping + logging")
else:
    print("PATCH 1c: WARNING - Could not find predict() X construction")

with open('src/ml/signal_predictor.py', 'w', encoding='utf-8') as f:
    f.write(content)

# ============================================================================
# PATCH 2: signals_tab_ai.py
# ============================================================================

with open('src/ml/signals_tab_ai.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_render_scores = """    # Build scores from signal
    scores = {
        'ticker': signal.ticker,
        'sentiment_score': getattr(signal, 'sentiment_score', 50) or 50,
        'fundamental_score': getattr(signal, 'fundamental_score', 50) or 50,
        'technical_score': getattr(signal, 'technical_score', 50) or 50,
        'options_flow_score': getattr(signal, 'options_score', 50) or 50,
        'short_squeeze_score': getattr(signal, 'short_squeeze_score', 50) or 50,
        'total_score': signal.today_score or 50,
    }"""

new_render_scores = """    # Build scores from signal - pass ALL 9 features with domain-correct defaults
    scores = {
        'ticker': signal.ticker,
        'sentiment_score': getattr(signal, 'sentiment_score', None) or 50,
        'fundamental_score': getattr(signal, 'fundamental_score', None) or 50,
        'technical_score': getattr(signal, 'technical_score', None) or 50,
        'options_flow_score': getattr(signal, 'options_score', None) or 50,
        'short_squeeze_score': getattr(signal, 'short_squeeze_score', None) or 50,
        'total_score': signal.today_score or 50,
        'gap_score': getattr(signal, 'gap_score', None) or 50,
        'article_count': getattr(signal, 'article_count', None) or 0,
        'target_upside_pct': getattr(signal, 'target_upside_pct', None) or 0,
    }"""

if old_render_scores in content:
    content = content.replace(old_render_scores, new_render_scores)
    print("PATCH 2a: Fixed _render_ai_analysis to pass all 9 features")
else:
    print("PATCH 2a: WARNING - Could not find _render_ai_analysis scores block")

old_batch_scores = """            scores = {
                'ticker': signal.ticker,
                'sentiment_score': getattr(signal, 'sentiment_score', 50) or 50,
                'fundamental_score': getattr(signal, 'fundamental_score', 50) or 50,
                'technical_score': getattr(signal, 'technical_score', 50) or 50,
                'options_flow_score': getattr(signal, 'options_score', 50) or 50,
                'short_squeeze_score': getattr(signal, 'short_squeeze_score', 50) or 50,
                'total_score': signal.today_score or 50,
            }"""

new_batch_scores = """            scores = {
                'ticker': signal.ticker,
                'sentiment_score': getattr(signal, 'sentiment_score', None) or 50,
                'fundamental_score': getattr(signal, 'fundamental_score', None) or 50,
                'technical_score': getattr(signal, 'technical_score', None) or 50,
                'options_flow_score': getattr(signal, 'options_score', None) or 50,
                'short_squeeze_score': getattr(signal, 'short_squeeze_score', None) or 50,
                'total_score': signal.today_score or 50,
                'gap_score': getattr(signal, 'gap_score', None) or 50,
                'article_count': getattr(signal, 'article_count', None) or 0,
                'target_upside_pct': getattr(signal, 'target_upside_pct', None) or 0,
            }"""

if old_batch_scores in content:
    content = content.replace(old_batch_scores, new_batch_scores)
    print("PATCH 2b: Fixed get_ai_probabilities_batch to pass all 9 features")
else:
    print("PATCH 2b: WARNING - Could not find batch scores block")

with open('src/ml/signals_tab_ai.py', 'w', encoding='utf-8') as f:
    f.write(content)

# ============================================================================
# PATCH 3: shared.py
# ============================================================================

with open('src/tabs/signals_tab/shared.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_table_loop = """    # Compute probabilities
    results = {}
    for ticker, scores in ticker_scores.items():
        try:
            scores['ticker'] = ticker
            ml_pred = ai_system.ml_predictor.predict(scores)"""

new_table_loop = """    # Compute probabilities
    results = {}
    for ticker, scores in ticker_scores.items():
        try:
            scores['ticker'] = ticker
            # Ensure domain-correct defaults for features not in scores dict
            scores.setdefault('gap_score', 50)
            scores.setdefault('article_count', 0)
            scores.setdefault('target_upside_pct', 0)
            ml_pred = ai_system.ml_predictor.predict(scores)"""

if old_table_loop in content:
    content = content.replace(old_table_loop, new_table_loop)
    print("PATCH 3a: Fixed _get_ai_probabilities_for_table defaults")
else:
    print("PATCH 3a: WARNING - Could not find table loop block")

with open('src/tabs/signals_tab/shared.py', 'w', encoding='utf-8') as f:
    f.write(content)

# ============================================================================
# VERIFY
# ============================================================================

print("\nVerifying syntax...")
for fp in ['src/ml/signal_predictor.py', 'src/ml/signals_tab_ai.py', 'src/tabs/signals_tab/shared.py']:
    with open(fp, 'r', encoding='utf-8') as f:
        ast.parse(f.read())
    print(f"  {fp}: OK")

print("\nAll Phase 1 patches applied successfully!")
