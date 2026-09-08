import pytest
from pydantic import ValidationError

from financial_agent.schemas import AgentState, Message, UserQuery


def test_state_json_round_trip():
    state = AgentState(request=UserQuery(query=" synthetic demo ", history=[Message(role="user", content="hello")]))
    assert state.request.query == "synthetic demo"
    assert AgentState.model_validate_json(state.model_dump_json()) == state


@pytest.mark.parametrize("query", ["", "  ", 123])
def test_invalid_query_rejected(query):
    with pytest.raises(ValidationError):
        UserQuery(query=query)


def test_history_and_ids_are_independent():
    first, second = UserQuery(query="first"), UserQuery(query="second")
    first.history.append(Message(role="assistant", content="synthetic reply"))
    assert second.history == []
    assert first.request_id != second.request_id


def test_unknown_field_and_invalid_role_rejected():
    with pytest.raises(ValidationError):
        UserQuery(query="demo", unrecognized=True)
    with pytest.raises(ValidationError):
        Message(role="invalid", content="demo")
