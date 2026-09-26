from app.ml.agent import expected_trace
from app.services.workflow import _AGENT_TRACE_STAGE_MAP


def test_runtime_agent_variant_maps_match_phase8_contracts():
    required_tools = [
        "case.read",
        "vision.analyze",
        "knowledge.search",
        "report.generate",
        "report.verify",
        "review.request",
    ]

    fixed = [step for step, _ in _AGENT_TRACE_STAGE_MAP["fixed_workflow"].values()]
    single = [
        "agent.route",
        *[step for step, _ in _AGENT_TRACE_STAGE_MAP["single_agent"].values()],
        "agent.verify",
        "human_review",
    ]
    supervisor = [
        "supervisor.route",
        *[step for step, _ in _AGENT_TRACE_STAGE_MAP["supervisor_multi_agent"].values() if step != "human_review"],
        "supervisor.aggregate",
        "human_review",
    ]

    assert fixed == expected_trace("fixed_workflow", {"required_tools": required_tools})
    assert single == expected_trace("single_agent", {"required_tools": required_tools})
    assert supervisor == expected_trace("supervisor_multi_agent", {"required_tools": required_tools})

