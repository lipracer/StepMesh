import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fserver import send_d


class TestSendDDecorator:
    """Test suite for send_d decorator function."""

    def test_send_d_with_no_args_function(self):
        """Test send_d decorates a function with no arguments."""
        def original_func():
            return "no_args_result"

        decorated = send_d(original_func)
        result = decorated()
        assert result == "no_args_result"

    def test_send_d_with_single_arg(self):
        """Test send_d decorates a function with a single argument."""
        def original_func(x):
            return x * 2

        decorated = send_d(original_func)
        result = decorated(5)
        assert result == 10

    def test_send_d_with_multiple_args(self):
        """Test send_d decorates a function with multiple arguments."""
        def original_func(a, b, c):
            return a + b + c

        decorated = send_d(original_func)
        result = decorated(1, 2, 3)
        assert result == 6

    def test_send_d_with_mixed_arg_types(self):
        """Test send_d decorates a function with mixed argument types."""
        def original_func(a, b, c):
            return f"{a}-{b}-{c}"

        decorated = send_d(original_func)
        result = decorated("hello", 123, [1, 2, 3])
        assert result == "hello-123-[1, 2, 3]"

    def test_send_d_with_keyword_args(self):
        """Test send_d decorates a function with keyword arguments."""
        def original_func(a, b=10, c=20):
            return a + b + c

        decorated = send_d(original_func)
        result = decorated(5)
        assert result == 35

        result = decorated(5, b=15)
        assert result == 40

        result = decorated(5, b=15, c=25)
        assert result == 45

    def test_send_d_with_args_and_kwargs(self):
        """Test send_d decorates a function with both *args and **kwargs."""
        def original_func(*args, **kwargs):
            return (args, kwargs)

        decorated = send_d(original_func)

        result = decorated()
        assert result == ((), {})

        result = decorated(1, 2, 3)
        assert result == ((1, 2, 3), {})

        result = decorated(a=1, b=2)
        assert result == ((), {"a": 1, "b": 2})

        result = decorated(1, 2, 3, a=4, b=5)
        assert result == ((1, 2, 3), {"a": 4, "b": 5})

    def test_send_d_preserves_function_return_value(self):
        """Test that send_d preserves the exact return value of the original function."""
        def original_func():
            return {"key": [1, 2, 3], "nested": {"a": "b"}}

        decorated = send_d(original_func)
        result = decorated()
        assert result == {"key": [1, 2, 3], "nested": {"a": "b"}}

    def test_send_d_with_none_return(self):
        """Test send_d with function that returns None."""
        results = []

        def original_func(x):
            results.append(x)

        decorated = send_d(original_func)
        result = decorated("test")
        assert result is None
        assert results == ["test"]

    def test_send_d_with_empty_args(self):
        """Test send_d when called with no arguments (after decoration)."""
        def original_func():
            return "empty_call"

        decorated = send_d(original_func)
        result = decorated()
        assert result == "empty_call"

    def test_send_d_passes_exception(self):
        """Test send_d passes through exceptions from the wrapped function."""
        def original_func():
            raise ValueError("test error")

        decorated = send_d(original_func)
        with pytest.raises(ValueError, match="test error"):
            decorated()

    def test_send_d_passes_exception_with_args(self):
        """Test send_d passes through exceptions with arguments."""
        def original_func(x, y):
            raise TypeError(f"Invalid types: {x}, {y}")

        decorated = send_d(original_func)
        with pytest.raises(TypeError, match="Invalid types: int, str"):
            decorated("int", "str")

    def test_send_d_with_various_return_types(self):
        """Test send_d with various return types from wrapped function."""
        # Return integer
        def return_int():
            return 42

        decorated = send_d(return_int)
        assert decorated() == 42

        # Return float
        def return_float():
            return 3.14159

        decorated = send_d(return_float)
        assert decorated() == 3.14159

        # Return boolean
        def return_bool():
            return True

        decorated = send_d(return_bool)
        assert decorated() is True

        # Return list
        def return_list():
            return [1, 2, 3]

        decorated = send_d(return_list)
        assert decorated() == [1, 2, 3]

        # Return tuple
        def return_tuple():
            return (1, "two", 3.0)

        decorated = send_d(return_tuple)
        assert decorated() == (1, "two", 3.0)

        # Return set
        def return_set():
            return {1, 2, 3}

        decorated = send_d(return_set)
        assert decorated() == {1, 2, 3}

    def test_send_d_multiple_calls(self):
        """Test send_d works correctly with multiple calls."""
        call_count = [0]

        def original_func(x):
            call_count[0] += 1
            return x * call_count[0]

        decorated = send_d(original_func)

        assert decorated(5) == 5
        assert decorated(5) == 10
        assert decorated(5) == 15

    def test_send_d_decorator_can_be_reused(self):
        """Test send_d decorator can be applied to different functions."""
        def func1():
            return "func1"

        def func2():
            return "func2"

        def func3(x):
            return x * 3

        decorated1 = send_d(func1)
        decorated2 = send_d(func2)
        decorated3 = send_d(func3)

        assert decorated1() == "func1"
        assert decorated2() == "func2"
        assert decorated3(7) == 21


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
