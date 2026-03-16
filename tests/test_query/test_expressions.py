"""Tests for the query expression parser and evaluator."""

import numpy as np
import pytest

from moneta.query.expressions import (
    AssetNode,
    BinOpNode,
    CompareNode,
    ExpressionError,
    NumberNode,
    evaluate,
    parse_expression,
    tokenize,
)


# ===================================================================
# Tokenizer tests
# ===================================================================


class TestTokenize:
    def test_simple_comparison(self):
        tokens = tokenize("portfolio > 200000")
        assert len(tokens) == 3
        assert tokens[0].kind == "ident"
        assert tokens[0].value == "portfolio"
        assert tokens[1].kind == "compare_op"
        assert tokens[1].value == ">"
        assert tokens[2].kind == "number"
        assert tokens[2].value == "200000"

    def test_arithmetic_expression(self):
        tokens = tokenize("a + b * c")
        assert len(tokens) == 5
        kinds = [t.kind for t in tokens]
        assert kinds == ["ident", "arith_op", "ident", "arith_op", "ident"]

    def test_parentheses(self):
        tokens = tokenize("(a + b) * c")
        assert len(tokens) == 7
        assert tokens[0].kind == "paren"
        assert tokens[0].value == "("
        assert tokens[4].kind == "paren"
        assert tokens[4].value == ")"

    def test_float_number(self):
        tokens = tokenize("3.14")
        assert len(tokens) == 1
        assert tokens[0].kind == "number"
        assert tokens[0].value == "3.14"

    def test_asset_with_underscores(self):
        tokens = tokenize("investment_portfolio")
        assert len(tokens) == 1
        assert tokens[0].kind == "ident"
        assert tokens[0].value == "investment_portfolio"

    def test_comparison_operators(self):
        for op in [">", "<", ">=", "<="]:
            tokens = tokenize(f"a {op} b")
            assert tokens[1].kind == "compare_op"
            assert tokens[1].value == op

    def test_invalid_character(self):
        with pytest.raises(ExpressionError, match="Unexpected character"):
            tokenize("a @ b")

    def test_empty_string(self):
        tokens = tokenize("")
        assert tokens == []


# ===================================================================
# Parser tests
# ===================================================================


class TestParse:
    def test_simple_comparison(self):
        """Parse 'portfolio > 200000' -> CompareNode."""
        node = parse_expression("portfolio > 200000")
        assert isinstance(node, CompareNode)
        assert isinstance(node.left, AssetNode)
        assert node.left.name == "portfolio"
        assert node.op == ">"
        assert isinstance(node.right, NumberNode)
        assert node.right.value == 200000.0

    def test_addition_comparison(self):
        """Parse 'a + b > 100' -> CompareNode(BinOpNode(a, +, b), '>', 100)."""
        node = parse_expression("a + b > 100")
        assert isinstance(node, CompareNode)
        assert isinstance(node.left, BinOpNode)
        assert node.left.op == "+"
        assert isinstance(node.left.left, AssetNode)
        assert node.left.left.name == "a"
        assert isinstance(node.left.right, AssetNode)
        assert node.left.right.name == "b"
        assert node.op == ">"
        assert isinstance(node.right, NumberNode)
        assert node.right.value == 100.0

    def test_multiplication_precedence(self):
        """Parse 'a + b * c' -> BinOp(a, +, BinOp(b, *, c))."""
        node = parse_expression("a + b * c")
        assert isinstance(node, BinOpNode)
        assert node.op == "+"
        assert isinstance(node.left, AssetNode)
        assert node.left.name == "a"
        assert isinstance(node.right, BinOpNode)
        assert node.right.op == "*"
        assert node.right.left.name == "b"
        assert node.right.right.name == "c"

    def test_division_precedence(self):
        """Parse 'a - b / c' -> BinOp(a, -, BinOp(b, /, c))."""
        node = parse_expression("a - b / c")
        assert isinstance(node, BinOpNode)
        assert node.op == "-"
        assert isinstance(node.right, BinOpNode)
        assert node.right.op == "/"

    def test_left_associativity_addition(self):
        """Parse 'a + b + c' -> BinOp(BinOp(a, +, b), +, c)."""
        node = parse_expression("a + b + c")
        assert isinstance(node, BinOpNode)
        assert node.op == "+"
        assert isinstance(node.left, BinOpNode)
        assert node.left.op == "+"
        assert node.left.left.name == "a"
        assert node.left.right.name == "b"
        assert node.right.name == "c"

    def test_parentheses_override_precedence(self):
        """Parse '(a + b) * c' -> BinOp(BinOp(a, +, b), *, c)."""
        node = parse_expression("(a + b) * c")
        assert isinstance(node, BinOpNode)
        assert node.op == "*"
        assert isinstance(node.left, BinOpNode)
        assert node.left.op == "+"

    def test_all_comparison_operators(self):
        for op in [">", "<", ">=", "<="]:
            node = parse_expression(f"a {op} b")
            assert isinstance(node, CompareNode)
            assert node.op == op

    def test_all_arithmetic_operators(self):
        for op in ["+", "-", "*", "/"]:
            node = parse_expression(f"a {op} b")
            assert isinstance(node, BinOpNode)
            assert node.op == op

    def test_asset_name_with_underscores(self):
        """Asset names with underscores should parse correctly."""
        node = parse_expression("investment_portfolio")
        assert isinstance(node, AssetNode)
        assert node.name == "investment_portfolio"

    def test_number_only(self):
        node = parse_expression("42")
        assert isinstance(node, NumberNode)
        assert node.value == 42.0

    def test_float_number(self):
        node = parse_expression("3.14")
        assert isinstance(node, NumberNode)
        assert node.value == 3.14

    def test_nested_parentheses(self):
        node = parse_expression("((a))")
        assert isinstance(node, AssetNode)
        assert node.name == "a"

    def test_complex_expression(self):
        """Parse 'a + b * c - d / e >= 1000'."""
        node = parse_expression("a + b * c - d / e >= 1000")
        assert isinstance(node, CompareNode)
        assert node.op == ">="

    def test_of_expression(self):
        """The 'of' field uses the same parser — just no comparison."""
        node = parse_expression("portfolio + equity")
        assert isinstance(node, BinOpNode)
        assert node.op == "+"

    # Error cases

    def test_empty_expression_error(self):
        with pytest.raises(ExpressionError, match="Empty expression"):
            parse_expression("")

    def test_dangling_operator_error(self):
        with pytest.raises(ExpressionError):
            parse_expression("a +")

    def test_double_operator_error(self):
        with pytest.raises(ExpressionError):
            parse_expression("a + + b")

    def test_unclosed_paren_error(self):
        with pytest.raises(ExpressionError):
            parse_expression("(a + b")

    def test_extra_close_paren_error(self):
        with pytest.raises(ExpressionError):
            parse_expression("a + b)")

    def test_leading_operator_error(self):
        with pytest.raises(ExpressionError):
            parse_expression("* a")

    def test_whitespace_only_error(self):
        with pytest.raises(ExpressionError, match="Empty expression"):
            parse_expression("   ")


# ===================================================================
# Evaluator tests
# ===================================================================


class TestEvaluate:
    def test_number_node(self):
        """NumberNode(5) -> array of 5s."""
        values = {"a": np.array([1.0, 2.0, 3.0])}
        node = NumberNode(5.0)
        result = evaluate(node, values)
        np.testing.assert_array_equal(result, [5.0, 5.0, 5.0])

    def test_asset_node(self):
        """AssetNode resolves correctly."""
        values = {"portfolio": np.array([100.0, 200.0, 300.0])}
        node = AssetNode("portfolio")
        result = evaluate(node, values)
        np.testing.assert_array_equal(result, [100.0, 200.0, 300.0])

    def test_addition(self):
        values = {
            "a": np.array([10.0, 20.0]),
            "b": np.array([1.0, 2.0]),
        }
        node = BinOpNode(AssetNode("a"), "+", AssetNode("b"))
        result = evaluate(node, values)
        np.testing.assert_array_equal(result, [11.0, 22.0])

    def test_subtraction(self):
        values = {
            "a": np.array([10.0, 20.0]),
            "b": np.array([1.0, 2.0]),
        }
        node = BinOpNode(AssetNode("a"), "-", AssetNode("b"))
        result = evaluate(node, values)
        np.testing.assert_array_equal(result, [9.0, 18.0])

    def test_multiplication(self):
        values = {"a": np.array([3.0, 4.0])}
        node = BinOpNode(AssetNode("a"), "*", NumberNode(2.0))
        result = evaluate(node, values)
        np.testing.assert_array_equal(result, [6.0, 8.0])

    def test_division(self):
        values = {"a": np.array([10.0, 20.0])}
        node = BinOpNode(AssetNode("a"), "/", NumberNode(5.0))
        result = evaluate(node, values)
        np.testing.assert_array_equal(result, [2.0, 4.0])

    def test_comparison_gt(self):
        """Comparison: half True, half False -> correct boolean array."""
        values = {"a": np.array([50.0, 100.0, 150.0, 200.0])}
        node = CompareNode(AssetNode("a"), ">", NumberNode(100.0))
        result = evaluate(node, values)
        np.testing.assert_array_equal(result, [False, False, True, True])

    def test_comparison_lt(self):
        values = {"a": np.array([50.0, 100.0, 150.0])}
        node = CompareNode(AssetNode("a"), "<", NumberNode(100.0))
        result = evaluate(node, values)
        np.testing.assert_array_equal(result, [True, False, False])

    def test_comparison_gte(self):
        values = {"a": np.array([50.0, 100.0, 150.0])}
        node = CompareNode(AssetNode("a"), ">=", NumberNode(100.0))
        result = evaluate(node, values)
        np.testing.assert_array_equal(result, [False, True, True])

    def test_comparison_lte(self):
        values = {"a": np.array([50.0, 100.0, 150.0])}
        node = CompareNode(AssetNode("a"), "<=", NumberNode(100.0))
        result = evaluate(node, values)
        np.testing.assert_array_equal(result, [True, True, False])

    def test_combined_arithmetic_and_comparison(self):
        """Evaluate 'a + b > 100' with known arrays."""
        values = {
            "a": np.array([40.0, 60.0, 80.0, 100.0]),
            "b": np.array([20.0, 30.0, 40.0, 50.0]),
        }
        # a + b = [60, 90, 120, 150]
        node = parse_expression("a + b > 100")
        result = evaluate(node, values)
        np.testing.assert_array_equal(result, [False, False, True, True])

    def test_precedence_evaluation(self):
        """a + b * c should compute b*c first, then add a."""
        values = {
            "a": np.array([10.0]),
            "b": np.array([3.0]),
            "c": np.array([4.0]),
        }
        node = parse_expression("a + b * c")
        result = evaluate(node, values)
        # 10 + 3*4 = 22, not (10+3)*4 = 52
        np.testing.assert_array_equal(result, [22.0])

    def test_unknown_asset_error(self):
        """Unknown asset name -> clear error."""
        values = {"portfolio": np.array([100.0])}
        node = AssetNode("savings")
        with pytest.raises(ExpressionError, match="Unknown asset name 'savings'"):
            evaluate(node, values)

    def test_unknown_asset_lists_available(self):
        """Error message should list available asset names."""
        values = {"portfolio": np.array([100.0]), "equity": np.array([200.0])}
        node = AssetNode("savings")
        with pytest.raises(ExpressionError, match="Available: equity, portfolio"):
            evaluate(node, values)

    def test_division_by_zero_error(self):
        values = {"a": np.array([10.0, 20.0])}
        node = BinOpNode(AssetNode("a"), "/", NumberNode(0.0))
        with pytest.raises(ExpressionError, match="Division by zero"):
            evaluate(node, values)

    def test_division_by_zero_in_array(self):
        """Division by zero when some elements are zero."""
        values = {
            "a": np.array([10.0, 20.0]),
            "b": np.array([5.0, 0.0]),
        }
        node = BinOpNode(AssetNode("a"), "/", AssetNode("b"))
        with pytest.raises(ExpressionError, match="Division by zero"):
            evaluate(node, values)

    def test_evaluate_parsed_of_expression(self):
        """Parse and evaluate an 'of' expression (no comparison)."""
        values = {
            "portfolio": np.array([1000.0, 2000.0]),
            "equity": np.array([500.0, 800.0]),
        }
        node = parse_expression("portfolio + equity")
        result = evaluate(node, values)
        np.testing.assert_array_equal(result, [1500.0, 2800.0])
