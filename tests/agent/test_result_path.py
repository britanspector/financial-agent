import pytest
from pydantic import RootModel

from financial_agent.agent.result_path import ResultPathError, resolve_result_path, result_path_type
from financial_agent.schemas import Schema


class Leaf(Schema):
    value: int


class Root(Schema):
    leaves: list[Leaf]


class RootList(RootModel[list[Leaf]]):
    pass


def test_static_and_runtime_result_paths_share_field_and_list_semantics():
    path = ["leaves", 0, "value"]
    assert result_path_type(Root, path) is int
    assert resolve_result_path({"leaves": [{"value": 7}]}, path) == 7


def test_root_list_output_is_transparent_below_tool_result_data():
    path = [0, "value"]
    assert result_path_type(RootList, path) is int
    assert resolve_result_path([{"value": 7}], path) == 7


@pytest.mark.parametrize("path", [["missing"], ["leaves", -1], ["leaves", "0"]])
def test_static_result_path_rejects_non_public_or_unsafe_segments(path):
    assert result_path_type(Root, path) is None


@pytest.mark.parametrize("path", [["missing"], ["leaves", -1], ["leaves", 2]])
def test_runtime_result_path_rejects_non_public_or_unavailable_values(path):
    with pytest.raises(ResultPathError):
        resolve_result_path({"leaves": [{"value": 7}]}, path)
