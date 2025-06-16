from __future__ import annotations

"""Application bootstrap helper for Phase-1 infrastructure.

Importing this module wires up:
• ConfigurationService (already registered via core.config)  
• TranscriberFactory  
• LLMProviderFactory

After import you can fetch the container via :pydata:`container` and resolve
any registered singletons:

>>> from core.bootstrap import container
>>> cfg = container.resolve(IConfigurationService)
"""

from .container import global_container as container
from .factories.transcriber_factory import TranscriberFactory
from .factories.llm_factory import LLMProviderFactory

# Register singleton factory instances if not already present.
if TranscriberFactory not in container.registrations:
    container.register_instance(TranscriberFactory, TranscriberFactory())

if LLMProviderFactory not in container.registrations:
    container.register_instance(LLMProviderFactory, LLMProviderFactory()) 