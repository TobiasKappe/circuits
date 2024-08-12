from functools import singledispatchmethod
from typing import List, Union

from riscv.data import RiscInteger

from circuits.element import Element
from circuits.node import Node
from circuits.registry import ElementRegistry


@ElementRegistry.add_impl('RAM')
class RAMElement(Element):
    @singledispatchmethod
    def __init__(
        self,
        clock: Node,
        dataIn: Node,
        dataOut1: Node,
        dataOut2: Node,
        en: Node,
        memAddr1: Node,
        memAddr2: Node,
        values: Union[List[RiscInteger], None] = None,
        **kwargs
    ):
        super().__init__(
            [clock, dataIn, dataOut1, dataOut2, en, memAddr1, memAddr2],
            **kwargs
        )
        self.clock = clock
        self.dataIn = dataIn
        self.dataOut1 = dataOut1
        self.dataOut2 = dataOut2
        self.en = en
        self.memAddr1 = memAddr1
        self.memAddr2 = memAddr2
        self.values = values or [RiscInteger(0) for _ in range(16)]
        self.prev_clock = None
        self.validate()

    @__init__.register
    def load(self, raw_element: dict, **kwargs):
        super().__init__(raw_element, **kwargs)
        self.values = [
            RiscInteger(v, signed=False)
            for v in raw_element['customData']['constructorParamaters'][0]
        ]
        self.prev_clock = None
        self.validate()

    def validate(self):
        assert len(self.values) == 16
        assert all(isinstance(v, RiscInteger) for v in self.values)

        assert self.clock.bitwidth == 1
        assert self.dataIn.bitwidth == 32
        assert self.dataOut1.bitwidth == 32
        assert self.dataOut2.bitwidth == 32
        assert self.en.bitwidth == 1
        assert self.memAddr1.bitwidth == 32
        assert self.memAddr2.bitwidth == 32

    def is_resolvable(self):
        return True

    def resolve(self):
        if self.memAddr1.value is not None:
            addr = (RiscInteger(self.memAddr1.value) & RiscInteger(0x3F)) >> 2
            if (self.prev_clock is False and self.clock.value[0] is True and
               self.en.value is not None and self.en.value[0] is True and
               self.dataIn.value is not None):
                self.values[addr.to_int()] = RiscInteger(self.dataIn.value)
            self.dataOut1.value = self.values[addr.to_int()].bits
            yield self.dataOut1

        if self.memAddr2.value is not None:
            addr = (RiscInteger(self.memAddr2.value) & RiscInteger(0x3F)) >> 2
            self.dataOut2.value = self.values[addr.to_int()].bits
            yield self.dataOut2

        if self.clock.value is not None:
            self.prev_clock = self.clock.value[0]
