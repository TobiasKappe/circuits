import pytest

from ouca.circuits import Node
from ouca.circuits.logic import \
    LogicElement, AndGateElement, NorGateElement, OrGateElement, \
    NandGateElement, XorGateElement, XnorGateElement


class TestLogicElement:
    @pytest.mark.parametrize(
        'cls, inputs, resolvable',
        [
            [LogicElement, [None, None], False],
            [LogicElement, [True, None], True],
            [LogicElement, [None, True], True],
            [LogicElement, [False, None], True],
            [LogicElement, [None, False], True],
            [LogicElement, [True, True], True],
            [LogicElement, [None, None, False], True],
        ]
    )
    def test_is_resolvable(self, cls, inputs, resolvable):
        input_nodes = [Node(value=[i]) for i in inputs]
        output_node = Node()
        element = cls(input_nodes, output_node)
        assert element.is_resolvable() is resolvable

    @pytest.mark.parametrize(
        'cls, inputs, result',
        [
            [AndGateElement, [None, False], False],
            [AndGateElement, [False, None], False],
            [AndGateElement, [False, False], False],
            [AndGateElement, [None, True], True],
            [AndGateElement, [True, None], True],
            [AndGateElement, [False, True], False],
            [AndGateElement, [True, False], False],
            [AndGateElement, [True, True], True],

            [NorGateElement, [None, True], False],
            [NorGateElement, [True, None], False],
            [NorGateElement, [False, False], True],
            [NorGateElement, [None, False], True],
            [NorGateElement, [False, None], True],
            [NorGateElement, [False, True], False],
            [NorGateElement, [True, False], False],
            [NorGateElement, [True, True], False],
            [NorGateElement, [True, None], False],
            [NorGateElement, [None, True], False],

            [OrGateElement, [None, True], True],
            [OrGateElement, [True, None], True],
            [OrGateElement, [False, None], False],
            [OrGateElement, [None, False], False],
            [OrGateElement, [False, False], False],
            [OrGateElement, [False, True], True],
            [OrGateElement, [True, False], True],
            [OrGateElement, [True, True], True],
            [OrGateElement, [True, None], True],
            [OrGateElement, [None, True], True],

            [NandGateElement, [None, True], False],
            [NandGateElement, [True, None], False],
            [NandGateElement, [False, False], True],
            [NandGateElement, [False, None], True],
            [NandGateElement, [None, False], True],
            [NandGateElement, [False, True], False],
            [NandGateElement, [True, False], False],
            [NandGateElement, [True, True], False],
            [NandGateElement, [True, None], False],
            [NandGateElement, [None, True], False],

            [XorGateElement, [False, None], False],
            [XorGateElement, [None, False], False],
            [XorGateElement, [False, False], False],
            [XorGateElement, [False, True], True],
            [XorGateElement, [True, False], True],
            [XorGateElement, [True, True], False],
            [XorGateElement, [True, None], True],
            [XorGateElement, [None, True], True],

            [XnorGateElement, [False, False], True],
            [XnorGateElement, [False, None], True],
            [XnorGateElement, [None, False], True],
            [XnorGateElement, [False, True], False],
            [XnorGateElement, [True, False], False],
            [XnorGateElement, [True, True], True],
            [XnorGateElement, [True, None], False],
            [XnorGateElement, [None, True], False],
        ]
    )
    def test_resolve(self, cls, inputs, result):
        input_nodes = [Node(value=[i]) for i in inputs]
        output_node = Node()
        element = cls(input_nodes, output_node)

        nodes = list(element.resolve())
        assert nodes == [output_node]
        assert output_node.value == [result]
