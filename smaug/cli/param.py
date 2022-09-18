import click

from typing import Any, Optional


class IntOrFloatParamType(click.ParamType):

    name = "int-or-float"

    def convert(
        self, value: Any, param: Optional[click.Parameter], ctx: Optional[click.Context]
    ) -> Any:
        if isinstance(value, (int, float)):
            return value

        if "." in value:
            try:
                return float(value)
            except ValueError:
                self.fail(f"{value!r} is not a valid float", param, ctx)
        else:
            try:
                return int(value)
            except ValueError:
                self.fail(f"{value!r} is not a valid int", param, ctx)


INT_OR_FLOAT = IntOrFloatParamType()
