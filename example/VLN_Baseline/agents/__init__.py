from agents.uni_navid_agent import UniNavidAgent
from agents.vint_agent import VintAgent
from agents.nomad_agent import NomadAgent
from agents.streamvln_agent import StreamVlnAgent


AGENT_REGISTRY = {
    UniNavidAgent.method_name: UniNavidAgent,
    VintAgent.method_name: VintAgent,
    NomadAgent.method_name: NomadAgent,
    StreamVlnAgent.method_name: StreamVlnAgent,
}


def available_methods():
    return sorted(AGENT_REGISTRY.keys())


def add_agent_cli_args(parser):
    for agent_cls in AGENT_REGISTRY.values():
        agent_cls.add_cli_args(parser)


def build_vln_agent(method, args):
    try:
        agent_cls = AGENT_REGISTRY[method]
    except KeyError as exc:
        methods = ", ".join(available_methods())
        raise ValueError(f"Unknown VLN method {method!r}. Available methods: {methods}") from exc
    return agent_cls(args)
