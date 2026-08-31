class ExpertBankCompatibilityError(RuntimeError):
    """Raised when a loaded model is incompatible with an expert bank."""


class EngineCompatibilityError(RuntimeError):
    """Raised when a model is incompatible with preemptLM engine"""
