from functools import singledispatchmethod
from typing import List

from ouca.circuits.element import Element
from ouca.circuits.node import Node
from ouca.circuits.registry import ElementRegistry


class SBElement(Element):
    @singledispatchmethod
    def __init__(
        self,
        input: Node,
        output1: List[Node],
        enable: Node,
        **kwargs
    ):
        self.inp1 = input
        self.output1 = output1
        self.enable = enable
        super().__init__([self.inp1, self.output1, self.enable], **kwargs)

    @__init__.register
    def load(self, raw_element: dict, **kwargs):
        super().__init__(raw_element, **kwargs)
        self.bitwidth = self.params[1]

    def output_index(self, index):
        if index == -1:
            self.output1.value = [False] * self.bitwidth
            self.enable.value = [False]
        else:
            bits = []
            while index:
                bits.append(index % 2 == 1)
                index = index >> 1

            self.output1.value = bits + [False] * (self.bitwidth - len(bits))
            self.enable.value = [True]

        yield self.enable
        yield self.output1


@ElementRegistry.add_impl('MSB')
class MSBElement(SBElement):
    def is_resolvable(self):
        if self.inp1.value is None:
            return False
        for val in reversed(self.inp1.value):
            if val is True:
                return True
            if val is None:
                return False

        return True

    def resolve(self):
        index = -1
        for i, val in enumerate(self.inp1.value):
            if val:
                index = i

        return self.output_index(index)


@ElementRegistry.add_impl('LSB')
class LSBElement(SBElement):
    def is_resolvable(self):
        if self.inp1.value is None:
            return False
        for val in self.inp1.value:
            if val is True:
                return True
            if val is None:
                return False

        return True

    def resolve(self):
        index = -1
        for i, val in enumerate(self.inp1.value):
            if val:
                index = i
                break

        return self.output_index(index)
