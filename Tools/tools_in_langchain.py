# -*- coding: utf-8 -*-
"""Tools-in-langchain.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1lCBnoSqkPlgq1qANxk6L65M9k0nuEZYg

# Install Libraries
"""

!pip install langchain langchain-core langchain-community pydantic duckduckgo-search langchain_experimental

"""## Build in Tool: DuckDuckGo Search"""

from langchain_community.tools import DuckDuckGoSearchRun
from IPython.display import display, Markdown
search_tool = DuckDuckGoSearchRun()

results = search_tool.invoke('Top new in India today')

display(Markdown(f"### Search Results\n{results}"))

"""## Build in Tool: Shell Tool"""

from langchain_community.tools import ShellTool

shell_tool = ShellTool()

result = shell_tool.invoke('ls')

display(Markdown(f"### Search Results\n{result}"))

"""## Custom Tools"""

from langchain_core.tools import tool

# Step 1 - create a function

def multiply(a: int, b: int) -> int:
  "Multiply two numbers"
  return a*b

# Step 2 - add type hinting
def multiply(a: int, b: int) -> int:
  "Multiply two numbers"
  return a*b

# Step 3 - add tool decorator
@tool
def multiply(a: int, b: int) -> int:
  "Multiply two numbers"
  return a*b

result = multiply.invoke({'a':3, 'b':5})

display(Markdown(f"### Answer\n{result}"))

print(multiply.name)
print(multiply.description)
print(multiply.args)

print(multiply.args_schema.model_json_schema())

"""## Method 2: Using Structured Tool"""

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
  a: int = Field(description='The first number')
  b: int = Field(description='The second number')

def multiply(a: int, b: int) -> int:
  return a*b

multiply_tool = StructuredTool.from_function(
    func=multiply,
    name='multiply',
    description = 'Multiply two numbers',
    args_schema=MultiplyInput
)

result = multiply_tool.invoke({'a':2, 'b':10})
display(Markdown(f"### Answer\n{result}"))

multiply_tool.args_schema.model_json_schema()

"""# Method 3 - Using BaseTool Class"""

from langchain.tools import BaseTool
from typing import Type

class MultiplyInput(BaseModel):
  a: int = Field(required=True, description='The first number')
  b: int = Field(required=True, description='The second number')

class MultiplyTool(BaseTool):
  name: str = 'multiply'
  description : str = 'Multiply two numbers'
  args_schema: Type[BaseModel] = MultiplyInput

  def _run(self, a : int, b: int) -> int:
    return a * b

multiply_tool = MultiplyTool()

result = multiply_tool.invoke({'a':3, 'b':7})
display(Markdown(f"### Answer\n{result}"))

"""# Toolkit"""

from langchain_core.tools import tool

#Custom tools
@tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

class MathToolkit:
  def get_tools(self):
    return [add, multiply]

toolkit = MathToolkit()

tools = toolkit.get_tools()

for tool in tools:
  print(tool.name, "=>", tool.description)