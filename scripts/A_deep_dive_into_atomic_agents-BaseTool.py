# For a detailed explanation of this code checkout the notebook:
# /notebooks/A_deep_dive_into_atomic_agents-BaseTool.ipynb

from pydantic import Field
from rich.console import Console
from sympy import sympify

from atomic_agents.agents.base_agent import BaseAgentIO
from atomic_agents.lib.tools.base import BaseTool, BaseToolConfig


class CalculatorToolSchema(BaseAgentIO):
    expression: str = Field(
        ..., description="Mathematical expression to evaluate. For example, '2 + 2'."
    )

    class Config:
        title = "CalculatorTool"
        description = (
            "Tool for performing calculations. Supports basic arithmetic operations "
            "like addition, subtraction, multiplication, and division, but also more "
            "complex operations like exponentiation and trigonometric functions. "
            "Use this tool to evaluate mathematical expressions."
        )
        json_schema_extra = {"title": title, "description": description}


class CalculatorToolOutputSchema(BaseAgentIO):
    result: str = Field(..., description="Result of the calculation.")


class CalculatorToolConfig(BaseToolConfig):
    pass


class CalculatorTool(BaseTool):
    input_schema = CalculatorToolSchema
    output_schema = CalculatorToolOutputSchema

    def __init__(self, config: CalculatorToolConfig = CalculatorToolConfig()):
        super().__init__(config)

    def run(self, params: CalculatorToolSchema) -> CalculatorToolOutputSchema:
        # Explicitly convert the string form of the expression
        parsed_expr = sympify(str(params.expression))
        # Evaluate the expression numerically
        result = parsed_expr.evalf()
        return CalculatorToolOutputSchema(result=str(result))


if __name__ == "__main__":
    rich_console = Console()
    rich_console.print(CalculatorTool().run(CalculatorToolSchema(expression="2 + 2")))
