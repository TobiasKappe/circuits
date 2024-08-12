from functools import singledispatchmethod
from typing import List

from circuits.element import Element
from circuits.node import Node
from circuits.registry import ElementRegistry


@ElementRegistry.add_impl('Splitter')
class SplitterElement(Element):
    @singledispatchmethod
    def __init__(
        self,
        inp1: Node,
        outputs: Node,
        bitwidth_in: int,
        bitwidths_out: List[int],
        **kwargs
    ):
        assert bitwidth_in == sum(bitwidths_out)

        super().__init__([inp1] + outputs, **kwargs)
        self.inp1 = inp1
        self.outputs = outputs
        self.bitwidths_out = bitwidths_out
        self.prev_inp = None
        self.prev_outputs = [None for length in bitwidths_out]

    @__init__.register
    def load(self, raw_element: dict, **kwargs):
        super().__init__(raw_element, **kwargs)
        self.bitwidths_out = self.params[2]
        self.prev_inp = None
        self.prev_outputs = [None for length in self.bitwidths_out]

        # The bitwidths on output nodes of splitters are incorrect as exported
        # from CircuitVerse; we patch them up as best we can.
        for i, bitwidth in enumerate(self.bitwidths_out):
            self.outputs[i].bitwidth = bitwidth

    def is_resolvable(self):
        if self.prev_inp != self.inp1.value:
            if self.inp1.value is None:
                return False
            return None not in self.inp1.value
        elif self.prev_outputs != [o.value for o in self.outputs]:
            for output in self.outputs:
                if output.value is None:
                    return False
                if None in output.value:
                    return False
            return True
        return False

    def resolve(self):
        if self.prev_inp != self.inp1.value:
            pos = 0
            for i, width in enumerate(self.bitwidths_out):
                self.outputs[i].value = self.inp1.value[pos:pos+width]
                yield self.outputs[i]
                pos = pos + width
        elif self.prev_outputs != [o.value for o in self.outputs]:
            combined = []
            for output in self.outputs:
                combined += output.value
            self.inp1.value = combined
            yield self.inp1

        self.prev_inp = self.inp1.value
        self.prev_outputs = [o.value for o in self.outputs]
