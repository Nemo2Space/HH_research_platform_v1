"""
Bond Chat Extension

Extends AlphaChat to support bond market context.
Can be monkey-patched or used as a mixin.

Location: src/ai/bond_chat_extension.py
"""


def extend_alpha_chat_for_bonds():
    """
    Extend AlphaChat class to support bond context.
    Call this once at startup.
    """
    try:
        from src.ai.chat import AlphaChat

        # Add bond context method if not present
        if not hasattr(AlphaChat, 'set_bond_context'):
            def set_bond_context(self, context: str):
                """Set bond market context for AI."""
                if not hasattr(self, '_bond_context'):
                    self._bond_context = ""
                self._bond_context = context

            AlphaChat.set_bond_context = set_bond_context

        if not hasattr(AlphaChat, 'get_bond_context'):
            def get_bond_context(self) -> str:
                """Get bond market context."""
                return getattr(self, '_bond_context', "")

            AlphaChat.get_bond_context = get_bond_context

        return True
    except ImportError:
        return False


# Auto-extend on import
extend_alpha_chat_for_bonds()