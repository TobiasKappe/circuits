from typing import List, Union

from ouca.circuits.element import Element
from ouca.circuits.exceptions import ContentionException


class Node:
    def __init__(
        self,
        value: Union[List[bool], None] = None,
        connections=None,
        bitwidth=1,
        index=None,
    ):
        self.value = value
        self.upstream = None
        self.connections = connections or []
        self.bitwidth = bitwidth
        self.index = index

    @classmethod
    def load(cls, raw_element: dict, index=None):
        return cls(bitwidth=raw_element['bitWidth'], index=index)

    def describe(self, thing):
        if isinstance(thing, Node):
            return f'node {thing.index}'
        elif isinstance(thing, Element):
            return f'an element of type {thing.__class__.__name__}'
        else:
            raise Exception(f'Cannot describe {thing}')

    def propagate(self, seen):
        for connection in self.connections:
            if connection in seen:
                continue

            if connection.value != self.value:
                if connection.upstream is None:
                    connection.upstream = self
                elif connection.upstream is not self:
                    raise ContentionException(
                        f'Node {self.index} and '
                        f'{self.describe(connection.upstream)} '
                        f'are competing for node {connection.index}'
                    )

                connection.value = self.value
                yield connection
                yield from connection.propagate(seen | {self})
