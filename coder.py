from functools import singledispatchmethod
from typing import List

from ouca.circuits.element import Element
from ouca.circuits.node import Node
from ouca.circuits.registry import ElementRegistry


@ElementRegistry.add_impl('Decoder')
class DecoderElement(Element):
    @singledispatchmethod
    def __init__(
        self,
        signal_size: int,
        input: Node,
        output1: List[Node],
        **kwargs
    ):
        self.signal_size = signal_size
        self.input = input
        self.output1 = output1
        super().__init__([self.input] + self.output1, **kwargs)

    @__init__.register
    def load(self, raw_element: dict, **kwargs):
        super().__init__(raw_element, **kwargs)
        self.signal_size = self.params[1]

    def is_resolvable(self):
        if self.input.value is None:
            return False
        if None in self.input.value:
            return False

        return True

    def resolve(self):
        index = 0
        for val in reversed(self.input.value):
            index = index << 1
            if val is True:
                index |= 1

        for i in range(2**self.signal_size):
            self.output1[i].value = [i == index]
            yield self.output1[i]


@ElementRegistry.add_impl('PriorityEncoder')
class PriorityEncoderElement(Element):
    @singledispatchmethod
    def __init__(
        self,
        inp1: List[Node],
        output1: List[Node],
        enable: Node,
        **kwargs
    ):
        super().__init__(inp1 + output1 + [enable], **kwargs)
        self.inp1 = inp1
        self.output1 = output1
        self.enable = enable
        self.sanity_check()

    @__init__.register
    def load(self, raw_element: dict, **kwargs):
        super().__init__(raw_element, **kwargs)
        self.bitwidth = self.params[1]
        self.sanity_check()

    def sanity_check(self):
        assert len(self.output1) == self.bitwidth
        assert len(self.inp1) == 2**self.bitwidth

        for node in self.nodes:
            assert node.bitwidth == 1

    def is_resolvable(self):
        for inp in self.inp1:
            if inp.value is None:
                return False
        return True

    def resolve(self):
        index = -1
        for i, inp in enumerate(self.inp1):
            if inp.value == [True]:
                index = i

        if index < 0:
            self.enable.value = [False]
        else:
            for outp in self.output1:
                outp.value = [index & 1 > 0]
                yield outp
                index = index >> 1
            self.enable.value = [True]

        yield self.enable
