from functools import singledispatchmethod
from typing import List

from circuits.element import Element
from circuits.node import Node
from circuits.circuit import Circuit
from circuits.exceptions import CircuitSubscopeException
from circuits.registry import ElementRegistry


@ElementRegistry.add_impl('SubCircuit')
class SubCircuitElement(Element):
    @singledispatchmethod
    def __init__(
        self,
        circuit: Circuit,
        inputs: List[Node],
        outputs: List[Node],
        **kwargs
    ):
        self.circuit = circuit
        self.inputs = inputs
        self.outputs = outputs
        self.subcircuit_id = None
        super().__init__(inputs + outputs, **kwargs)

    @__init__.register
    def load(self, raw_element: dict, nodes: List[Node], context: dict):
        self.subcircuit_id = int(raw_element['id'])
        if self.subcircuit_id in context:
            self.circuit = Circuit.load(context[self.subcircuit_id], context)
        else:
            raise CircuitSubscopeException(
                f'Could not find subscope with id {self.subcircuit_id}'
            )

        self.inputs = []
        for raw_input_node in raw_element['inputNodes']:
            self.inputs.append(nodes[raw_input_node])

        self.outputs = []
        for raw_output_node in raw_element['outputNodes']:
            self.outputs.append(nodes[raw_output_node])

        self.nodes = set(self.inputs + self.outputs)

    def is_resolvable(self):
        return True

    def resolve(self):
        for i, inp in enumerate(self.circuit.inputs):
            inp.state = self.inputs[i].value
        old_values = [o.value for o in self.circuit.outputs]

        self.circuit.simulate()
        for i, output in enumerate(self.circuit.outputs):
            if old_values[i] != output.value:
                self.outputs[i].value = output.value
                yield self.outputs[i]
