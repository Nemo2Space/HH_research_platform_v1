-- Migration: Add Options Flow and Short Squeeze scores to screener_scores
-- Run this once to add the new columns

-- Add options_flow_score column (0-100, NULL if not analyzed)
ALTER TABLE screener_scores
ADD COLUMN IF NOT EXISTS options_flow_score FLOAT;

-- Add short_squeeze_score column (0-100, NULL if not analyzed)
ALTER TABLE screener_scores
ADD COLUMN IF NOT EXISTS short_squeeze_score FLOAT;

-- Add options_sentiment column for quick reference
ALTER TABLE screener_scores
ADD COLUMN IF NOT EXISTS options_sentiment VARCHAR(20);

-- Add squeeze_risk column for quick reference
ALTER TABLE screener_scores
ADD COLUMN IF NOT EXISTS squeeze_risk VARCHAR(20);

-- Create index for filtering by these scores
CREATE INDEX IF NOT EXISTS idx_screener_options_flow ON screener_scores(options_flow_score) WHERE options_flow_score IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_screener_squeeze ON screener_scores(short_squeeze_score) WHERE short_squeeze_score IS NOT NULL;

-- Update comment
COMMENT ON COLUMN screener_scores.options_flow_score IS 'Options flow bullishness score 0-100 (50=neutral, >70=bullish, <30=bearish)';
COMMENT ON COLUMN screener_scores.short_squeeze_score IS 'Short squeeze potential score 0-100 (>70=high risk, >50=moderate, <30=low)';
COMMENT ON COLUMN screener_scores.options_sentiment IS 'Options sentiment: BULLISH, BEARISH, NEUTRAL';
COMMENT ON COLUMN screener_scores.squeeze_risk IS 'Squeeze risk level: HIGH, MODERATE, LOW, MINIMAL';