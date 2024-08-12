class ElementRegistry:
    IMPL_MAP = {}

    @classmethod
    def add_impl(cls, name):
        def wrapper(impl):
            cls.IMPL_MAP[name] = impl
            return impl

        return wrapper

    @classmethod
    def get_impl(cls, name):
        if name not in cls.IMPL_MAP:
            raise Exception(f'Unknown element: "{name}"')

        return cls.IMPL_MAP[name]
