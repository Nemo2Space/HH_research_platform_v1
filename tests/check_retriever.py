"""Check RAGRetriever method signature."""
from src.rag.retrieval import RAGRetriever
import inspect

# Get retrieve method signature
sig = inspect.signature(RAGRetriever.retrieve)
print(f"RAGRetriever.retrieve signature:")
print(f"  {sig}")
print(f"\nParameters:")
for name, param in sig.parameters.items():
    print(f"  {name}: {param.default if param.default != inspect.Parameter.empty else 'required'}")