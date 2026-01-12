# SequenceGenerator Documentation

This document describes the SequenceGenerator DSL for programmatically generating integer sequences.

## Overview

The **SequenceGenerator** is a deterministic domain-specific language (DSL) for generating integer sequences based on Kolmogorov complexity principles.

**Key characteristics:**
- **Deterministic**: Same program + seq_len produces identical output
- **Stateful**: Maintains runtime context (index, last value, seed, counters, macros)
- **Self-contained**: Single importable module

**Location**: `charly_01/SequenceGenerator.py`

---

## Public API

```python
class SequenceGenerator:
    def __init__(self, seq_len: int, seq_prog: str):
        """
        Args:
            seq_len: Total length of sequence (must be > 0)
            seq_prog: DSL program string
        """

    def val(self, idx: int) -> float:
        """Get value at index (0 to seq_len-1)"""

    def to_numpy(self) -> np.ndarray:
        """Generate entire sequence as numpy array (dtype int32)"""

    def reset(self) -> None:
        """Reset runtime context (clears cache, reinitializes state)"""
```

---

## Runtime Context

```python
@dataclass
class SeqContext:
    i: int = 0              # Current index (IDX variable)
    last: int = 0           # Previously emitted value (LAST variable)
    seed: int = 1           # Deterministic RNG seed
    counters: dict = None   # Named integer counters
    macros: dict = None     # Macro variables defined by LET
```

**Predefined macro**: `MAXIDX` equals `seq_len` automatically.

---

## DSL Syntax

### Atomic Constructs

| Construct | Syntax | Effect |
|-----------|--------|--------|
| Digit Literal | `0`-`9` | Emit digit, IDX++, LAST=digit |
| Grouping | `(program)` | Execute nested program |
| Expression Emit | `{expr}` | Emit arithmetic result, IDX++, LAST=result |
| Conditional | `[cond?A:B]` | Branch based on condition |
| Infinite Repeat | `X*` | Repeat atom X infinitely |
| Finite Repeat | `X*{k}` | Repeat atom X exactly k times |

### Statements

| Statement | Syntax | Purpose |
|-----------|--------|---------|
| SEED | `SEED(expr)` | Set RNG seed from expression |
| LET | `LET(NAME, expr)` | Define macro variable |
| RESET | `RESET(NAME)` | Zero a named counter |
| DEF | `DEF(NAME, (program))` | Define reusable sub-sequence |
| CALL | `CALL(NAME)` | Inline a defined sub-sequence |

### Built-in Variables

```
IDX      Current sequence index
LAST     Most recently emitted value
MAXIDX   Sequence length (predefined)
<NAME>   Any macro defined via LET
<NAME>   Any counter defined via INC/COUNT/RESET
```

### Operators

**Arithmetic**: `+`, `-`, `*`, `//` (integer division), `%` (modulo)

**Comparison**: `==`, `!=`, `<`, `<=`, `>`, `>=`

**Boolean**: `&` (AND), `|` (OR), `!` (NOT)

**Unary**: `-` (negation)

### Functions

```
ABS(x)                    Absolute value
MIN(a, b)                 Minimum
MAX(a, b)                 Maximum
CLAMP(x, lo, hi)          Clamp x to [lo, hi]
RAND(a, b)                Random int in [a, b] (deterministic)
COUNT(NAME)               Get counter value
INC(NAME)                 Increment counter, return new value
LIMIT(NAME, n)            If COUNT(NAME) < n: increment and return 1, else 0
CHOICE(v1:w1, v2:w2, ...) Weighted random choice
RESET(NAME)               Zero counter
```

### Comments and Separators

- **Comments**: `/* ... */` (non-nested, can span lines)
- **Separators**: Whitespace and semicolon `;`

---

## Example Programs

### Simple Cycling Sequence
```
(0 1 2)*
```
Output: 0, 1, 2, 0, 1, 2, 0, 1, 2, ...

### Macro-based Sequence
```
LET(STEP, 10);
{ IDX // STEP }*
```
Output: 0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1, 2,2,...

### Parabolic Hump
```
LET(S, 1000);
{ CLAMP(9 - (IDX - 500) * (IDX - 500) // S, 0, 9) }*
```
Generates a parabolic curve centered at index 500.

### Conditional Branching
```
[IDX < 100 ? {5} : {0}]*
```
Emits 5 for first 100 indices, then 0.

### Random Weighted Choice
```
SEED(42);
{ CHOICE(1:30, 2:50, 3:20) }*{1000}
```
1000 values with weighted distribution (1→30%, 2→50%, 3→20%).

### Multiple Humps with DEF/CALL
```
SEED(7);
LET(S, 120000);
DEF(HUMP, ( {CLAMP(9-(IDX-C)*(IDX-C)//S, 0, 9)} ));
LET(C, 1000); CALL(HUMP);
DEF(NHUMP, ( { -CLAMP(9-(IDX-C)*(IDX-C)//S, 0, 9)} ));
LET(C, 3000); CALL(NHUMP);
LET(C, 4000); CALL(HUMP);
{CLAMP(LAST - CLAMP(9-(IDX-(MAXIDX-200))*(IDX-(MAXIDX-200))//8000, 0, 9), -9, 9)}*
```
Three reusable humps with fade-out at sequence end.

### Counter-based Pattern
```
{ [INC(CTR) == 3 ? {0} : {LAST + 1}] }*
```
Counts up and resets every 3rd element.

---

## Usage Examples

### Basic Usage
```python
from SequenceGenerator import SequenceGenerator

# Create generator
gen = SequenceGenerator(seq_len=1000, seq_prog="(0 1 2)*")

# Get individual values
print(gen.val(0))      # 0
print(gen.val(1))      # 1
print(gen.val(2))      # 2
print(gen.val(3))      # 0 (cycles)

# Get entire sequence
arr = gen.to_numpy()   # np.ndarray shape (1000,)
```

### Deterministic Random Sequence
```python
prog = """
SEED(12345);
{ RAND(1, 6) }*
"""
gen = SequenceGenerator(seq_len=100, seq_prog=prog)
# Same seed always produces same "dice rolls"
```

### Integration in Charly
```yaml
# config.yaml
neuron_field:
  threshold:
    type: float
    sequence: "{ 5 + IDX // 100 }*"
```

```python
# In simulation code
gen = SequenceGenerator(seq_len=neuron_count, seq_prog=threshold_seq)
for idx in range(neuron_count):
    threshold = int(gen.val(idx))
```

---

## GUI Playground

**Class**: `SeqPlayground(tk.Tk)`

### Features

- **N slider**: Sequence length (50 to 50000)
- **View toggle**: Color map or bar chart
- **Render**: Ctrl+Enter to compile and visualize
- **Dual plots**: Full sequence + zoom view

### Visualization Modes

**Color mode**:
- Red = negative values (intensity by |value|)
- Green = positive values (intensity by value)

**Bar mode**:
- Green bars = positive
- Red bars = negative
- Height = |value|

### Running the Playground
```bash
python charly_01/SequenceGenerator.py
```

---

## AST Node Types

### Statement Nodes

| Node | Purpose |
|------|---------|
| `Literal` | Emit single digit |
| `Sequence` | Cycle through children |
| `Repeat` | Repeat node N times or infinitely |
| `Condition` | [cond ? then : else] branching |
| `ExprEmit` | Evaluate and emit expression |
| `StatementSeed` | Set RNG seed |
| `StatementLet` | Define macro |
| `StatementReset` | Zero counter |

### Expression Nodes

| Node | Purpose |
|------|---------|
| `Num` | Number literal |
| `Var` | Variable reference |
| `Neg` | Unary negation |
| `BinOp` | Binary operation |
| `FuncCall` | Function call |

---

## Deterministic RNG

Uses SplitMix64 algorithm:

```python
def rand_int(seed: int, idx: int, a: int, b: int, salt: int = 0) -> int:
    x = seed ^ (idx * 0xD1B54A32D192ED03) ^ (salt * 0x9E3779B97F4A7C15)
    return a + (splitmix64(x) % (b - a + 1))
```

Same seed + index always produces same random value.

---

## DEF/CALL Semantics

Each `CALL(NAME)` gets a **deep copy** of the AST:

```python
# DEF stores AST
self.funcs[name] = body

# CALL creates fresh instance
node = copy.deepcopy(self.funcs[name])
node.reset_stream()
```

This ensures:
- No state pollution between calls
- Fresh stream state per invocation
- LAST/IDX/counters remain shared (global state)

---

## Caching Strategy

```python
def _ensure(self, idx: int) -> None:
    while len(self._cache) <= idx:
        v = self.ast.emit(self.ctx)
        self._cache.append(int(v))
```

- Values generated on-demand
- Stored in `_cache` list
- `reset()` clears cache and reinitializes context

---

## Error Handling

| Error | Cause |
|-------|-------|
| `SyntaxError` | Invalid DSL syntax |
| `NameError` | Undefined variable or function |
| `ValueError` | Invalid operation (bad seed, seq_len <= 0) |
| `IndexError` | Out-of-bounds access (idx >= seq_len) |
