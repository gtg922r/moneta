"""Recursive descent parser for query expressions.

Grammar:
    expr     -> compare
    compare  -> additive (('>' | '<' | '>=' | '<=') additive)?
    additive -> term (('+' | '-') term)*
    term     -> factor (('*' | '/') factor)*
    factor   -> NUMBER | ASSET_NAME | '(' expr ')'

The parser tokenizes the expression string, parses it into an AST of
dataclass nodes, and provides an evaluate() function that operates on
NumPy arrays (one value per simulation run).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Union

import numpy as np


# ---------------------------------------------------------------------------
# AST node types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NumberNode:
    """A numeric literal."""

    value: float


@dataclass(frozen=True)
class AssetNode:
    """A reference to an asset name (resolved at evaluation time)."""

    name: str


@dataclass(frozen=True)
class BinOpNode:
    """A binary arithmetic operation (+, -, *, /)."""

    left: Node
    op: str  # '+', '-', '*', '/'
    right: Node


@dataclass(frozen=True)
class CompareNode:
    """A comparison operation (>, <, >=, <=)."""

    left: Node
    op: str  # '>', '<', '>=', '<='
    right: Node


Node = Union[NumberNode, AssetNode, BinOpNode, CompareNode]


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

# Token pattern: numbers (int or float), comparison operators (>=, <=, >, <),
# arithmetic operators (+, -, *, /), parentheses, or identifiers (asset names).
_TOKEN_PATTERN = re.compile(
    r"""
    \s*                       # skip leading whitespace
    (?:
        (\d+(?:\.\d+)?)       # group 1: number
      | (>=|<=|>|<)           # group 2: comparison operator
      | ([+\-*/])             # group 3: arithmetic operator
      | ([()]  )              # group 4: parenthesis
      | ([a-zA-Z_]\w*)        # group 5: identifier (asset name)
    )
    """,
    re.VERBOSE,
)


@dataclass
class Token:
    """A single token from the expression string."""

    kind: str  # 'number', 'compare_op', 'arith_op', 'paren', 'ident'
    value: str
    pos: int  # character position in original string


def tokenize(expression: str) -> list[Token]:
    """Tokenize an expression string into a list of tokens.

    Raises ExpressionError on unrecognized characters.
    """
    tokens: list[Token] = []
    pos = 0
    remaining = expression

    while pos < len(expression):
        # Skip whitespace
        if expression[pos].isspace():
            pos += 1
            continue

        m = _TOKEN_PATTERN.match(expression, pos)
        if m is None or m.start() != pos:
            raise ExpressionError(
                f"Unexpected character '{expression[pos]}' at position {pos}",
                expression=expression,
                position=pos,
            )

        if m.group(1) is not None:
            tokens.append(Token("number", m.group(1), pos))
        elif m.group(2) is not None:
            tokens.append(Token("compare_op", m.group(2), pos))
        elif m.group(3) is not None:
            tokens.append(Token("arith_op", m.group(3), pos))
        elif m.group(4) is not None:
            tokens.append(Token("paren", m.group(4), pos))
        elif m.group(5) is not None:
            tokens.append(Token("ident", m.group(5), pos))

        pos = m.end()

    return tokens


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class ExpressionError(Exception):
    """Error raised when parsing or evaluating a query expression."""

    def __init__(
        self,
        message: str,
        *,
        expression: str | None = None,
        position: int | None = None,
    ):
        self.expression = expression
        self.position = position
        super().__init__(message)


class _Parser:
    """Recursive descent parser that converts tokens into an AST."""

    def __init__(self, tokens: list[Token], expression: str):
        self.tokens = tokens
        self.expression = expression
        self.pos = 0

    def _peek(self) -> Token | None:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def _advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _expect(self, kind: str, value: str | None = None) -> Token:
        tok = self._peek()
        if tok is None:
            raise ExpressionError(
                f"Unexpected end of expression, expected {kind}"
                + (f" '{value}'" if value else ""),
                expression=self.expression,
            )
        if tok.kind != kind or (value is not None and tok.value != value):
            raise ExpressionError(
                f"Expected {kind}"
                + (f" '{value}'" if value else "")
                + f" but got '{tok.value}' at position {tok.pos}",
                expression=self.expression,
                position=tok.pos,
            )
        return self._advance()

    def parse(self) -> Node:
        """Parse the full expression and verify all tokens are consumed."""
        if not self.tokens:
            raise ExpressionError(
                "Empty expression",
                expression=self.expression,
            )
        node = self._expr()
        if self.pos < len(self.tokens):
            tok = self.tokens[self.pos]
            raise ExpressionError(
                f"Unexpected token '{tok.value}' at position {tok.pos}",
                expression=self.expression,
                position=tok.pos,
            )
        return node

    def _expr(self) -> Node:
        """expr -> compare"""
        return self._compare()

    def _compare(self) -> Node:
        """compare -> additive (('>' | '<' | '>=' | '<=') additive)?"""
        left = self._additive()
        tok = self._peek()
        if tok is not None and tok.kind == "compare_op":
            self._advance()
            right = self._additive()
            return CompareNode(left, tok.value, right)
        return left

    def _additive(self) -> Node:
        """additive -> term (('+' | '-') term)*"""
        left = self._term()
        while True:
            tok = self._peek()
            if tok is not None and tok.kind == "arith_op" and tok.value in ("+", "-"):
                self._advance()
                right = self._term()
                left = BinOpNode(left, tok.value, right)
            else:
                break
        return left

    def _term(self) -> Node:
        """term -> factor (('*' | '/') factor)*"""
        left = self._factor()
        while True:
            tok = self._peek()
            if tok is not None and tok.kind == "arith_op" and tok.value in ("*", "/"):
                self._advance()
                right = self._factor()
                left = BinOpNode(left, tok.value, right)
            else:
                break
        return left

    def _factor(self) -> Node:
        """factor -> NUMBER | ASSET_NAME | '(' expr ')'"""
        tok = self._peek()
        if tok is None:
            raise ExpressionError(
                "Unexpected end of expression, expected a value",
                expression=self.expression,
            )

        if tok.kind == "number":
            self._advance()
            return NumberNode(float(tok.value))

        if tok.kind == "ident":
            self._advance()
            return AssetNode(tok.value)

        if tok.kind == "paren" and tok.value == "(":
            self._advance()
            node = self._expr()
            self._expect("paren", ")")
            return node

        raise ExpressionError(
            f"Unexpected token '{tok.value}' at position {tok.pos}, "
            "expected a number, asset name, or '('",
            expression=self.expression,
            position=tok.pos,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_expression(expression: str) -> Node:
    """Parse a query expression string into an AST.

    Args:
        expression: The expression string, e.g. "portfolio + equity > 2000000"

    Returns:
        The root AST node.

    Raises:
        ExpressionError: If the expression is malformed.
    """
    tokens = tokenize(expression)
    parser = _Parser(tokens, expression)
    return parser.parse()


def evaluate(
    node: Node,
    values: dict[str, np.ndarray],
) -> np.ndarray:
    """Evaluate an AST node against a dict of asset name -> value arrays.

    Args:
        node: The AST node to evaluate.
        values: Mapping from asset name to float64[n_runs] arrays.
            All arrays must have the same length.

    Returns:
        float64[n_runs] for arithmetic expressions, or
        bool[n_runs] for comparison expressions.

    Raises:
        ExpressionError: If an asset name is not found or division by zero occurs.
    """
    if isinstance(node, NumberNode):
        # Broadcast the scalar to match the size of other arrays.
        # Determine size from any entry in values_dict.
        if values:
            size = next(iter(values.values())).shape[0]
        else:
            size = 1
        return np.full(size, node.value, dtype=np.float64)

    if isinstance(node, AssetNode):
        if node.name not in values:
            available = ", ".join(sorted(values.keys())) if values else "(none)"
            raise ExpressionError(
                f"Unknown asset name '{node.name}'. "
                f"Available: {available}"
            )
        return values[node.name].astype(np.float64)

    if isinstance(node, BinOpNode):
        left = evaluate(node.left, values)
        right = evaluate(node.right, values)
        if node.op == "+":
            return left + right
        elif node.op == "-":
            return left - right
        elif node.op == "*":
            return left * right
        elif node.op == "/":
            if np.any(right == 0.0):
                raise ExpressionError("Division by zero in expression")
            return left / right
        else:
            raise ExpressionError(f"Unknown operator '{node.op}'")

    if isinstance(node, CompareNode):
        left = evaluate(node.left, values)
        right = evaluate(node.right, values)
        if node.op == ">":
            return left > right
        elif node.op == "<":
            return left < right
        elif node.op == ">=":
            return left >= right
        elif node.op == "<=":
            return left <= right
        else:
            raise ExpressionError(f"Unknown comparison operator '{node.op}'")

    raise ExpressionError(f"Unknown node type: {type(node)}")
