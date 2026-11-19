"""Temporary model guidance at the in-memory failover identity seam."""
import logging
import re

from agent.system_prompt import model_guidance_parts

logger = logging.getLogger(__name__)


class _RoutedPrompt(str):
    # Keep ownership on the cache value. A rebuild or /model invalidation
    # drops it without a second slot to clear or a delimiter to parse.
    original: str
    original_route: tuple[str, str]
    base_length: int


def _identity(prompt, model, provider):
    for label, value in (("Model", model), ("Provider", provider)):
        if not value:
            continue
        matches = list(re.finditer(rf"(?m)^{label}: .*$", prompt))
        if matches:
            last = matches[-1]
            prompt = f"{prompt[:last.start()]}{label}: {value}{prompt[last.end():]}"
    return prompt


def routed_prompt(agent, prompt, model, provider, fallback_from):
    owned = isinstance(prompt, _RoutedPrompt)
    if owned and not fallback_from and prompt.original_route == (model, provider):
        return prompt.original
    base = _identity(prompt[:prompt.base_length] if owned else prompt, model, provider)
    if not fallback_from:
        return base
    if not getattr(agent, 'valid_tool_names', None) or not all(
        hasattr(agent, setting) for setting in ('_tool_use_enforcement', '_execution_guidance')
    ):
        return base
    missing = [text for text in model_guidance_parts(agent, model) if text not in base]
    if missing:
        headings = ', '.join(text.partition('\n')[0].lstrip('# ') for text in missing)
        logger.info('Fallback %s added prompt sections: %s', model, headings)
    result = _RoutedPrompt('\n\n'.join([base, *missing]))
    result.base_length = len(base)
    # Chain hops keep the first undo point.
    result.original = prompt.original if owned else prompt
    result.original_route = prompt.original_route if owned else fallback_from
    return result
