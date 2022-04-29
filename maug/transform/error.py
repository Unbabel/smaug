import enum


class ErrorType(enum.Enum):
    """Defines error types that a specific transform produces."""

    UNDEFINED = 0
    """The error type is not defined."""

    NOT_CRITICAL = 1
    """The transform does not induce any critical error in the translation. 
    Nevertheless, if induced on a critical error example, the example should 
    still be classified as critical."""

    MISTRANSLATION = 2
    """The transform creates a mistranslation error. The content of the 
    translation has a different meaning when compared to the source, is not 
    translated (remains in the source language), or is translated into 
    gibberish."""

    HALLUCINATION = 3
    """The translation creates an hallucination, where new content is added to
    the translation."""

    DELETION = 4
    """The translation creates a deletion error, where critical content is
    removed from the translation."""
