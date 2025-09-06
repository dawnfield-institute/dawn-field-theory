"""
Fracton DSL Compiler - Optional compilation for Fracton domain-specific syntax

This module provides compilation capabilities for Fracton-specific syntax,
allowing for more natural expression of recursive algorithms and entropy dynamics.
"""

import ast
import re
from typing import Any, Dict, List, Optional, Tuple


def compile_fracton_dsl(source_code: str, optimize: bool = True) -> str:
    """
    Compile Fracton DSL syntax to standard Python code.
    
    The Fracton DSL provides syntactic sugar for common recursive patterns,
    entropy expressions, and field operations.
    
    Args:
        source_code: Fracton DSL source code
        optimize: Whether to apply optimization passes
        
    Returns:
        Compiled Python code
        
    Example:
        fracton_code = '''
        recursive fibonacci(memory, context):
            entropy_gate(0.3, 0.8)
            
            if context.depth <= 1:
                return 1
            
            field_transform entropy > 0.5:
                a = recurse fibonacci(memory, context.deeper(1))
                b = recurse fibonacci(memory, context.deeper(2))
                return a + b
            else:
                return crystallize(memory.get("cached_result", 1))
        '''
        
        python_code = compile_fracton_dsl(fracton_code)
    """
    # This is a placeholder implementation
    # A full DSL compiler would need proper parsing and AST transformation
    
    # For now, provide basic syntax transformations
    compiled_code = source_code
    
    # Transform recursive function declarations
    compiled_code = _transform_recursive_declarations(compiled_code)
    
    # Transform entropy gates
    compiled_code = _transform_entropy_gates(compiled_code)
    
    # Transform field operations
    compiled_code = _transform_field_operations(compiled_code)
    
    # Transform recurse calls
    compiled_code = _transform_recurse_calls(compiled_code)
    
    if optimize:
        compiled_code = _optimize_compiled_code(compiled_code)
    
    return compiled_code


def _transform_recursive_declarations(code: str) -> str:
    """Transform 'recursive funcname' to '@fracton.recursive def funcname'."""
    pattern = r'recursive\s+(\w+)\s*\('
    replacement = r'@fracton.recursive\ndef \1('
    return re.sub(pattern, replacement, code)


def _transform_entropy_gates(code: str) -> str:
    """Transform 'entropy_gate(min, max)' to '@fracton.entropy_gate(min, max)'."""
    # Look for entropy_gate calls that appear to be decorators
    lines = code.split('\n')
    transformed_lines = []
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('entropy_gate(') and stripped.endswith(')'):
            # Transform to decorator syntax
            indent = len(line) - len(line.lstrip())
            transformed_line = ' ' * indent + '@fracton.' + stripped
            transformed_lines.append(transformed_line)
        else:
            transformed_lines.append(line)
    
    return '\n'.join(transformed_lines)


def _transform_field_operations(code: str) -> str:
    """Transform field_transform blocks to conditional branches."""
    # This is a simplified transformation
    # A full implementation would need proper parsing
    
    # Transform "field_transform entropy > 0.5:" to "if context.entropy > 0.5:"
    pattern = r'field_transform\s+entropy\s*([><=]+)\s*([0-9.]+)\s*:'
    replacement = r'if context.entropy \1 \2:'
    
    return re.sub(pattern, replacement, code)


def _transform_recurse_calls(code: str) -> str:
    """Transform 'recurse funcname(args)' to 'fracton.recurse(funcname, args)'."""
    # Handle both simple and complex recurse calls
    # Pattern: recurse function_name(args) -> fracton.recurse(function_name, args)
    
    def replacement_func(match):
        func_name = match.group(1)
        args = match.group(2).strip()
        if args:
            return f'fracton.recurse({func_name}, {args})'
        else:
            return f'fracton.recurse({func_name})'
    
    # Match patterns like "recurse fibonacci(memory, context.deeper())"
    pattern = r'recurse\s+(\w+)\s*\(([^)]*)\)'
    return re.sub(pattern, replacement_func, code)


def _optimize_compiled_code(code: str) -> str:
    """Apply optimization passes to compiled code."""
    # Simple optimizations - a full compiler would do much more
    
    # Remove redundant imports
    lines = code.split('\n')
    import_lines = set()
    other_lines = []
    
    for line in lines:
        if line.strip().startswith('import ') or line.strip().startswith('from '):
            import_lines.add(line.strip())
        else:
            other_lines.append(line)
    
    # Reconstruct with deduplicated imports
    optimized_lines = list(import_lines) + [''] + other_lines
    
    return '\n'.join(optimized_lines)


class FractonDSLParser:
    """
    Advanced parser for Fracton DSL syntax.
    
    This class provides more sophisticated parsing capabilities for complex
    Fracton DSL constructs beyond simple regex transformations.
    """
    
    def __init__(self):
        self.tokens = []
        self.current_token = 0
    
    def parse(self, source_code: str) -> ast.AST:
        """
        Parse Fracton DSL source code into an AST.
        
        Args:
            source_code: Fracton DSL source
            
        Returns:
            Python AST representing the parsed code
        """
        # Tokenize the source
        self.tokens = self._tokenize(source_code)
        self.current_token = 0
        
        # Parse into AST
        return self._parse_module()
    
    def _tokenize(self, source_code: str) -> List[Dict[str, Any]]:
        """Tokenize Fracton DSL source code."""
        # This is a placeholder - a real tokenizer would be more sophisticated
        tokens = []
        
        # Simple line-by-line tokenization
        for line_num, line in enumerate(source_code.split('\n'), 1):
            stripped = line.strip()
            if stripped:
                tokens.append({
                    'type': 'line',
                    'value': line,
                    'line_number': line_num
                })
        
        return tokens
    
    def _parse_module(self) -> ast.Module:
        """Parse a module (top-level construct)."""
        # Placeholder implementation
        return ast.Module(body=[], type_ignores=[])


def validate_fracton_syntax(source_code: str) -> Tuple[bool, List[str]]:
    """
    Validate Fracton DSL syntax and return errors if any.
    
    Args:
        source_code: Fracton DSL source to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
        
    Example:
        is_valid, errors = validate_fracton_syntax(my_fracton_code)
        if not is_valid:
            for error in errors:
                print(f"Syntax error: {error}")
    """
    errors = []
    
    # Check for basic syntax requirements
    lines = source_code.split('\n')
    
    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        
        # Check for invalid entropy values
        entropy_matches = re.findall(r'entropy\s*[><=]+\s*([0-9.]+)', stripped)
        for entropy_val in entropy_matches:
            try:
                val = float(entropy_val)
                if not (0.0 <= val <= 1.0):
                    errors.append(f"Line {line_num}: Entropy value {val} outside valid range [0.0, 1.0]")
            except ValueError:
                errors.append(f"Line {line_num}: Invalid entropy value '{entropy_val}'")
        
        # Check for unmatched braces/parentheses
        open_parens = stripped.count('(')
        close_parens = stripped.count(')')
        if open_parens != close_parens:
            errors.append(f"Line {line_num}: Unmatched parentheses")
        
        # Check for invalid recurse syntax
        if 'recurse' in stripped and not re.search(r'recurse\s+\w+\s*\(', stripped):
            if 'fracton.recurse(' not in stripped:
                errors.append(f"Line {line_num}: Invalid recurse syntax")
    
    return len(errors) == 0, errors


def fracton_to_python(fracton_file: str, python_file: str, 
                     optimize: bool = True) -> None:
    """
    Compile a Fracton DSL file to Python.
    
    Args:
        fracton_file: Path to input Fracton DSL file
        python_file: Path to output Python file
        optimize: Whether to apply optimizations
        
    Example:
        fracton_to_python("algorithm.fract", "algorithm.py")
    """
    try:
        with open(fracton_file, 'r', encoding='utf-8') as f:
            fracton_code = f.read()
        
        # Validate syntax
        is_valid, errors = validate_fracton_syntax(fracton_code)
        if not is_valid:
            error_msg = "Fracton DSL syntax errors:\n" + "\n".join(errors)
            raise SyntaxError(error_msg)
        
        # Compile to Python
        python_code = compile_fracton_dsl(fracton_code, optimize)
        
        # Add necessary imports
        imports = [
            "import fracton",
            "from fracton import recursive, entropy_gate, recurse, crystallize",
            ""
        ]
        
        full_python_code = "\n".join(imports) + "\n" + python_code
        
        # Write output
        with open(python_file, 'w', encoding='utf-8') as f:
            f.write(full_python_code)
        
        print(f"Successfully compiled {fracton_file} to {python_file}")
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not find Fracton file: {fracton_file}")
    except Exception as e:
        raise RuntimeError(f"Compilation failed: {e}")


# DSL syntax examples and documentation

FRACTON_DSL_EXAMPLES = {
    "fibonacci": '''
recursive fibonacci(memory, context):
    entropy_gate(0.3, 0.8)
    
    if context.depth <= 1:
        return 1
    
    field_transform entropy > 0.5:
        a = recurse fibonacci(memory, context.deeper(1))
        b = recurse fibonacci(memory, context.deeper(2))
        return a + b
    else:
        return crystallize(memory.get("cached_result", 1))
''',
    
    "pattern_analysis": '''
recursive analyze_patterns(memory, context):
    entropy_gate(0.5, 0.9)
    
    patterns = memory.get("patterns", [])
    
    field_transform entropy > 0.7:
        # High entropy: explore new patterns
        new_pattern = recurse discover_pattern(memory, context.deeper(1))
        patterns.append(new_pattern)
        memory.set("patterns", patterns)
    else:
        # Low entropy: crystallize existing patterns
        crystalized = crystallize(patterns)
        memory.set("stable_patterns", crystalized)
    
    return patterns
''',
    
    "adaptive_recursion": '''
recursive adaptive_process(memory, context):
    entropy_gate(0.1, 1.0)  # Accept any entropy
    
    # Adaptive entropy adjustment
    if context.entropy < 0.3:
        new_context = context.with_entropy(0.6)
        return recurse explore_options(memory, new_context)
    elif context.entropy > 0.8:
        new_context = context.with_entropy(0.4)
        return recurse consolidate_results(memory, new_context)
    else:
        return recurse standard_process(memory, context)
'''
}


def get_dsl_example(example_name: str) -> str:
    """
    Get a Fracton DSL example by name.
    
    Args:
        example_name: Name of the example to retrieve
        
    Returns:
        Fracton DSL example code
        
    Available examples: fibonacci, pattern_analysis, adaptive_recursion
    """
    if example_name not in FRACTON_DSL_EXAMPLES:
        available = ", ".join(FRACTON_DSL_EXAMPLES.keys())
        raise ValueError(f"Unknown example '{example_name}'. Available: {available}")
    
    return FRACTON_DSL_EXAMPLES[example_name]


def list_dsl_examples() -> List[str]:
    """Get list of available Fracton DSL examples."""
    return list(FRACTON_DSL_EXAMPLES.keys())
