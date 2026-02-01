"""
Fix JSONB handling in retrieval.py
Run: python src/rag/fix_retrieval_jsonb.py
"""

import re

file_path = 'src/rag/retrieval.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Add Json import
if 'from psycopg2.extras import Json' not in content:
    content = content.replace(
        'from src.db.connection import get_connection',
        'from psycopg2.extras import Json\nfrom src.db.connection import get_connection'
    )

# Wrap filters with Json()
content = content.replace(
    'filters, retrieved_chunk_ids',
    'Json(filters), retrieved_chunk_ids'
)

# Wrap gating_details with Json()
content = content.replace(
    'result.gating_details,',
    'Json(result.gating_details),'
)

# Wrap the scores dict with Json() - need to find the pattern
# Look for the scores dict being passed to execute
old_scores = """                {
                    'top_similarity': result.top_similarity,
                    'chunks_above_threshold': result.chunks_above_threshold,
                    'chunk_scores': [
                        {
                            'chunk_id': c.chunk_id,
                            'vector': c.vector_score,
                            'lexical': c.lexical_score,
                            'rrf': c.rrf_score,
                        }
                        for c in result.chunks
                    ]
                },"""

new_scores = """                Json({
                    'top_similarity': result.top_similarity,
                    'chunks_above_threshold': result.chunks_above_threshold,
                    'chunk_scores': [
                        {
                            'chunk_id': c.chunk_id,
                            'vector': c.vector_score,
                            'lexical': c.lexical_score,
                            'rrf': c.rrf_score,
                        }
                        for c in result.chunks
                    ]
                }),"""

content = content.replace(old_scores, new_scores)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed retrieval.py JSONB handling!')