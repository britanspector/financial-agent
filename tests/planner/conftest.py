from financial_agent.knowledge.runtime import register_rag_tools
from financial_agent.market_data.runtime import register_market_tools
from financial_agent.tools.composite import merge_registries
from financial_agent.user_data.runtime import register_user_tools


class UnusedService:
    def execute(self, *args, **kwargs):
        raise AssertionError("Planner tests must not execute tools")


def planner_catalog():
    service = UnusedService()
    return merge_registries(
        register_user_tools(service),
        register_market_tools(service),
        register_rag_tools(service),
    )
