from functools import singledispatchmethod
from typing import List

from ouca.circuits.element import Element
from ouca.circuits.node import Node
from ouca.circuits.registry import ElementRegistry


@ElementRegistry.add_impl('Multiplexer')
class MultiplexerElement(Element):
    @singledispatchmethod
    def __init__(
        self,
        signal_size: int,
        controlSignalInput: Node,
        inp: List[Node],
        output1: Node,
        **kwargs
    ):
        self.controlSignalInput = controlSignalInput
        self.signal_size = signal_size
        self.inp = inp
        self.output1 = output1
        super().__init__([self.controlSignalInput] + inp + [output1], **kwargs)

    @__init__.register
    def load(self, raw_element: dict, **kwargs):
        super().__init__(raw_element, **kwargs)
        self.signal_size = self.params[2]
        self.bitwidth = self.params[1]

    def control_index(self):
        index = 0
        for val in reversed(self.controlSignalInput.value):
            index = index << 1
            if val is True:
                index |= 1
        return index

    def is_resolvable(self):
        if self.controlSignalInput.value is None:
            return False
        if None in self.controlSignalInput.value:
            return False

        index = self.control_index()
        return self.inp[index].value is not None

    def resolve(self):
        index = self.control_index()
        self.output1.value = self.inp[index].value
        yield self.output1


@ElementRegistry.add_impl('Demultiplexer')
class DemultiplexerElement(Element):
    @singledispatchmethod
    def __init__(
        self,
        signal_size: int,
        controlSignalInput: Node,
        input: Node,
        output1: List[Node],
        **kwargs
    ):
        self.signal_size = signal_size
        self.controlSignalInput = controlSignalInput
        self.input = input
        self.output1 = output1
        super().__init__(
            [self.controlSignalInput, self.input] + self.output1,
            **kwargs
        )

    @__init__.register
    def load(self, raw_element: dict, **kwargs):
        super().__init__(raw_element, **kwargs)
        self.signal_size = self.params[2]
        self.bitwidth = self.params[1]

    def control_index(self):
        index = 0
        for val in reversed(self.controlSignalInput.value):
            index = index << 1
            if val is True:
                index |= 1
        return index

    def is_resolvable(self):
        if self.controlSignalInput.value is None:
            return False
        if None in self.controlSignalInput.value:
            return False
        if self.input.value is None:
            return False

        return True

    def resolve(self):
        index = self.control_index()

        for i, output in enumerate(self.output1):
            if i == index:
                output.value = self.input.value
            else:
                output.value = [False] * self.bitwidth

            yield output
