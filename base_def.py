"""
Base definitions

"""



class BaseEnum(object):
    """Base class for enums.
    From oceanobservatories/mi-instrument/mi/core/common.py
    @author Steve Foley
    @author Edward Hunter
    @license Apache 2.0
    Used as a base class here without modification

    Used to code agent and instrument states, events, commands and errors.
    To use, derive a class from this subclass and set values equal to it
    such as:
    @code
    class FooEnum(BaseEnum):
       VALUE1 = "Value 1"
       VALUE2 = "Value 2"
    @endcode
    and address the values as FooEnum.VALUE1 after you import the
    class/package.

    Enumerations are part of the code in the MI modules since they are tightly
    coupled with what the drivers can do. By putting the values here, they
    are quicker to execute and more compartmentalized so that code can be
    re-used more easily outside of a capability container as needed.
    """

    @classmethod
    def list(cls):
        """List the values of this enum."""
        return [getattr(cls,attr) for attr in dir(cls) if\
                not callable(getattr(cls,attr)) and not attr.startswith('__')]

    @classmethod
    def dict(cls):
        """Return a dict representation of this enum."""
        result = {}
        for attr in dir(cls):
            if not callable(getattr(cls,attr)) and not attr.startswith('__'):
                result[attr] = getattr(cls,attr)
        return result

    @classmethod
    def has(cls, item):
        """Is the object defined in the class?

        Use this function to test
        a variable for enum membership. For example,
        @code
        if not FooEnum.has(possible_value)
        @endcode
        @param item The attribute value to test for.
        @retval True if one of the class attributes has value item, false
        otherwise.
        """
        return item in cls.list()
