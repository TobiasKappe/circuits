from ouca.circuits import sim

import pytest


class TestCombinatorialElement:
    @pytest.mark.parametrize(
        'cls, inputs, resolvable',
        [
            [sim.AndGateElement, [None, None], False],
            [sim.AndGateElement, [True, None], False],
            [sim.AndGateElement, [None, True], False],
            [sim.AndGateElement, [False, None], True],
            [sim.AndGateElement, [None, False], True],
            [sim.AndGateElement, [True, True], True],
            [sim.AndGateElement, [None, None, False], True],
            [sim.NorGateElement, [None, None], False],
            [sim.NorGateElement, [False, None], False],
            [sim.NorGateElement, [None, False], False],
            [sim.NorGateElement, [False, None], False],
            [sim.NorGateElement, [True, None], True],
            [sim.NorGateElement, [None, True], True],
            [sim.NorGateElement, [False, False], True],
            [sim.NorGateElement, [None, None, True], True],
            [sim.OrGateElement, [None, None], False],
            [sim.OrGateElement, [True, None], True],
            [sim.OrGateElement, [None, True], True],
            [sim.OrGateElement, [False, None], False],
            [sim.OrGateElement, [None, False], False],
            [sim.OrGateElement, [False, False], True],
            [sim.OrGateElement, [None, None, True], True],
            [sim.NandGateElement, [None, None], False],
            [sim.NandGateElement, [None, False], False],
            [sim.NandGateElement, [False, None], False],
            [sim.NandGateElement, [None, True], True],
            [sim.NandGateElement, [True, None], True],
            [sim.NandGateElement, [False, False], True],
            [sim.NandGateElement, [None, None, True], True],
            [sim.XorGateElement, [None, None], False],
            [sim.XorGateElement, [None, False], False],
            [sim.XorGateElement, [False, None], False],
            [sim.XorGateElement, [None, True], False],
            [sim.XorGateElement, [True, None], False],
            [sim.XorGateElement, [False, False], True],
            [sim.XorGateElement, [False, True], True],
            [sim.XorGateElement, [True, False], True],
            [sim.XorGateElement, [True, True], True],
            [sim.XorGateElement, [None, None, True], False],
            [sim.XnorGateElement, [None, None], False],
            [sim.XnorGateElement, [None, False], False],
            [sim.XnorGateElement, [False, None], False],
            [sim.XnorGateElement, [None, True], False],
            [sim.XnorGateElement, [True, None], False],
            [sim.XnorGateElement, [False, False], True],
            [sim.XnorGateElement, [False, True], True],
            [sim.XnorGateElement, [True, False], True],
            [sim.XnorGateElement, [True, True], True],
            [sim.XnorGateElement, [None, None, True], False],
        ]
    )
    def test_is_resolvable(self, cls, inputs, resolvable):
        input_nodes = [sim.Node(value=[i]) for i in inputs]
        output_node = sim.Node()
        element = cls(input_nodes, output_node)
        assert element.is_resolvable() is resolvable

    @pytest.mark.parametrize(
        'cls, inputs, result',
        [
            [sim.AndGateElement, [None, False], False],
            [sim.AndGateElement, [False, None], False],
            [sim.AndGateElement, [False, False], False],
            [sim.AndGateElement, [False, True], False],
            [sim.AndGateElement, [True, False], False],
            [sim.AndGateElement, [True, True], True],
            [sim.NorGateElement, [None, True], False],
            [sim.NorGateElement, [True, None], False],
            [sim.NorGateElement, [False, False], True],
            [sim.NorGateElement, [False, True], False],
            [sim.NorGateElement, [True, False], False],
            [sim.NorGateElement, [True, True], False],
            [sim.OrGateElement, [None, True], True],
            [sim.OrGateElement, [True, None], True],
            [sim.OrGateElement, [False, False], False],
            [sim.OrGateElement, [False, True], True],
            [sim.OrGateElement, [True, False], True],
            [sim.OrGateElement, [True, True], True],
            [sim.NandGateElement, [None, True], False],
            [sim.NandGateElement, [True, None], False],
            [sim.NandGateElement, [False, False], True],
            [sim.NandGateElement, [False, True], False],
            [sim.NandGateElement, [True, False], False],
            [sim.NandGateElement, [True, True], False],
            [sim.XorGateElement, [False, False], False],
            [sim.XorGateElement, [False, True], True],
            [sim.XorGateElement, [True, False], True],
            [sim.XorGateElement, [True, True], False],
            [sim.XnorGateElement, [False, False], True],
            [sim.XnorGateElement, [False, True], False],
            [sim.XnorGateElement, [True, False], False],
            [sim.XnorGateElement, [True, True], True],
        ]
    )
    def test_resolve(self, cls, inputs, result):
        input_nodes = [sim.Node(value=[i]) for i in inputs]
        output_node = sim.Node()
        element = cls(input_nodes, output_node)

        nodes = list(element.resolve())
        assert nodes == [output_node]
        assert output_node.value == [result]
