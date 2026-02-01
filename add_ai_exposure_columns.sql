-- SQL Script: Add AI Exposure Columns for Theme-Based Filtering
-- Run this script to enable AI company filtering in portfolio builder
-- Author: HH Research Platform

-- Step 1: Add columns to screener table (or your main stock universe table)
ALTER TABLE screener
ADD COLUMN IF NOT EXISTS ai_exposure_score NUMERIC(5,2) DEFAULT NULL,
ADD COLUMN IF NOT EXISTS is_ai_company BOOLEAN DEFAULT NULL,
ADD COLUMN IF NOT EXISTS ai_category VARCHAR(50) DEFAULT NULL;

-- Add comment explaining the columns
COMMENT ON COLUMN screener.ai_exposure_score IS 'AI exposure score 0-100: how much of company revenue/focus is AI-related';
COMMENT ON COLUMN screener.is_ai_company IS 'True if company is primarily an AI builder/developer';
COMMENT ON COLUMN screener.ai_category IS 'AI category: chips, cloud, models, enterprise, tools, robotics, automotive';

-- Step 2: Create an index for faster filtering
CREATE INDEX IF NOT EXISTS idx_screener_ai_exposure ON screener(ai_exposure_score) WHERE ai_exposure_score IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_screener_is_ai_company ON screener(is_ai_company) WHERE is_ai_company = TRUE;

-- Step 3: Pre-populate known AI companies (core AI builders)
-- You can adjust scores based on your assessment

-- AI Infrastructure & Chips (highest exposure - their core business is AI)
UPDATE screener SET ai_exposure_score = 95, is_ai_company = TRUE, ai_category = 'chips'
WHERE ticker IN ('NVDA', 'AMD', 'ARM', 'MRVL');

UPDATE screener SET ai_exposure_score = 85, is_ai_company = TRUE, ai_category = 'chips'
WHERE ticker IN ('INTC', 'AVGO', 'QCOM', 'TSM', 'ASML', 'AMAT', 'LRCX', 'KLAC', 'MU');

-- AI Cloud & Platform (very high exposure - major AI investments)
UPDATE screener SET ai_exposure_score = 90, is_ai_company = TRUE, ai_category = 'cloud'
WHERE ticker IN ('MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META');

UPDATE screener SET ai_exposure_score = 80, is_ai_company = TRUE, ai_category = 'cloud'
WHERE ticker IN ('ORCL', 'IBM', 'SNOW', 'PLTR', 'DDOG', 'MDB', 'NET');

-- AI Software & Tools (high exposure)
UPDATE screener SET ai_exposure_score = 75, is_ai_company = TRUE, ai_category = 'enterprise'
WHERE ticker IN ('CRM', 'ADBE', 'NOW', 'WDAY', 'TEAM', 'HUBS', 'DOCU');

UPDATE screener SET ai_exposure_score = 80, is_ai_company = TRUE, ai_category = 'tools'
WHERE ticker IN ('SNPS', 'CDNS', 'ANSS');

-- AI Cybersecurity (significant AI component)
UPDATE screener SET ai_exposure_score = 70, is_ai_company = TRUE, ai_category = 'enterprise'
WHERE ticker IN ('PANW', 'CRWD', 'ZS', 'FTNT', 'S');

-- AI Pure-Play companies (highest exposure)
UPDATE screener SET ai_exposure_score = 100, is_ai_company = TRUE, ai_category = 'models'
WHERE ticker IN ('AI', 'PATH', 'SOUN', 'BBAI');

UPDATE screener SET ai_exposure_score = 85, is_ai_company = TRUE, ai_category = 'models'
WHERE ticker IN ('UPST', 'GFAI', 'PRCT');

-- AI Robotics & Automation
UPDATE screener SET ai_exposure_score = 75, is_ai_company = TRUE, ai_category = 'robotics'
WHERE ticker IN ('ISRG', 'ROK', 'TER', 'CGNX', 'IRBT');

-- AI in Automotive
UPDATE screener SET ai_exposure_score = 80, is_ai_company = TRUE, ai_category = 'automotive'
WHERE ticker IN ('TSLA', 'MBLY');

UPDATE screener SET ai_exposure_score = 70, is_ai_company = TRUE, ai_category = 'automotive'
WHERE ticker IN ('LAZR', 'INVZ', 'AEVA', 'OUST');

-- AI Infrastructure servers (moderate-high exposure)
UPDATE screener SET ai_exposure_score = 70, is_ai_company = TRUE, ai_category = 'chips'
WHERE ticker IN ('SMCI', 'DELL', 'HPE', 'ANET');

-- Step 4: Verify the updates
SELECT ticker, ai_exposure_score, is_ai_company, ai_category
FROM screener
WHERE is_ai_company = TRUE
ORDER BY ai_exposure_score DESC;

-- Step 5: Show summary
SELECT
    ai_category,
    COUNT(*) as count,
    ROUND(AVG(ai_exposure_score), 1) as avg_score
FROM screener
WHERE is_ai_company = TRUE
GROUP BY ai_category
ORDER BY avg_score DESC;