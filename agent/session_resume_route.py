"""Keep fallback accounting out of the primary route restored by UI clients."""
from persistence.gateway.routing import BARE_BILLING_PROVIDERS, GatewayRuntime
from persistence.gateway.routing import resume_route as stored_route


def resume_route(config, *, model, billing_provider, billing_base_url):
    from persistence.conversation.sessions import parse_model_config

    config = parse_model_config(config)
    nested = config.get("gateway_runtime")
    if isinstance(nested, dict) and nested.get("fallback_active") is True:
        if not config.get("model"):
            return "", GatewayRuntime()
        config = {key: config[key] for key in ("model", "provider", "api_mode", "base_url") if config.get(key)}
        return stored_route(config, model=config["model"])
    lock = config.get("browser_model_lock")
    if (model and billing_provider and billing_provider not in BARE_BILLING_PROVIDERS
            and not config.get("model") and not (isinstance(nested, dict) and nested.get("provider"))
            and not (isinstance(lock, dict) and lock.get("confirmed") and lock.get("model") == model)):
        from hermes_cli.config_effective import load_user_config_effective
        from hermes_cli.fallback_config import get_fallback_chain

        if ((config.get("provider") and config["provider"] != billing_provider)
                or any((r["model"], r["provider"]) == (model, billing_provider)
                       for r in get_fallback_chain(load_user_config_effective()))):
            return "", GatewayRuntime()
    return stored_route(config, model=model, billing_provider=billing_provider, billing_base_url=billing_base_url)
