# Pattern Analysis from Initial Runs

## Discoveries Found

### Promising Candidates (Error < 10^-20)
1. **mp.tanh(mp.sqrt(79)*mp.pi)** - Error: 10^-23
   - Very close to 1 (0.999999999999999999999998...)
   - Uses discriminant 79 (not a Heegner number!)
   - This is potentially novel!

2. **1/mp.tanh(mp.sqrt(79)*mp.pi)** - Error: 10^-23
   - Reciprocal also near-integer (1.00000000000000000000001...)
   - Consistent pattern

3. **mp.tanh(mp.sqrt(427)*mp.pi/3)** - Error: 10^-18
   - Another non-Heegner discriminant
   - Different structure with /3

## Rejection Patterns

### Major Issues Identified:
1. **High Rejection Rate**: ~98% of expressions rejected
   - 168/240 rejected for "high error" in iteration 1
   - Many duplicates (308/480 in iteration 2)

2. **Stuck Patterns**:
   - Gauss multiplication formula keeps appearing
   - Still seeing Heegner numbers (163) despite filters
   - Many expressions evaluating to exact 2 (sqrt(2)^2)

3. **Trivial Identities Still Appearing**:
   - Gamma product patterns (Gauss multiplication)
   - Powers of sqrt(2)
   - Simple algebraic identities simplifying to 1

## Key Insights

### What's Working:
- **Hyperbolic functions with novel discriminants** (79, 427) are producing near-integers
- **tanh patterns** seem particularly promising
- Non-standard discriminants are yielding results

### What's Not Working:
- Too many expressions still following known patterns
- Polylogarithm expressions not yielding good results yet
- AGM expressions not producing near-integers
- High duplicate rate suggests agents converging too quickly

## Breakthrough Opportunities

### Unexplored Areas:
1. **Discriminants to explore**: 11, 13, 17, 21, 23, 29, 31, 33, 37, 41
2. **Function combinations**:
   - tanh with other discriminants
   - Mixed tanh and gamma
   - Polylog at special algebraic points

### Specific Patterns to Investigate:
Based on mp.tanh(mp.sqrt(79)*mp.pi) success:
- mp.tanh(mp.sqrt(D)*mp.pi/n) for various D and n
- mp.coth(mp.sqrt(D)*mp.pi)
- Products/ratios of tanh at different discriminants

## Recommendations for Improvement

### Immediate Actions:
1. **Focus on hyperbolic functions** with discriminants 11-100 (excluding Heegner)
2. **Reduce temperature** in exploitation mode to refine good patterns
3. **Inject specific variations** of the tanh(sqrt(79)*pi) discovery

### Strategic Changes:
1. **Pattern Memory**: Track which discriminants have been tested
2. **Diversity Enforcement**: Penalize expressions too similar to recent ones
3. **Guided Exploration**: Use the tanh success to guide similar explorations

### Expression Templates to Try:
```python
# Based on successful pattern
"mp.tanh(mp.sqrt(11)*mp.pi)"
"mp.tanh(mp.sqrt(13)*mp.pi)"
"mp.tanh(mp.sqrt(17)*mp.pi)"
"mp.tanh(mp.sqrt(21)*mp.pi)"
"mp.tanh(mp.sqrt(23)*mp.pi)"

# Variations
"mp.tanh(mp.sqrt(79)*mp.pi/2)"
"mp.tanh(mp.sqrt(79)*mp.pi*2)"
"mp.coth(mp.sqrt(79)*mp.pi)"

# Mixed
"mp.tanh(mp.sqrt(79)*mp.pi) * mp.gamma(1/7)"
"mp.tanh(mp.sqrt(79)*mp.pi) + mp.tanh(mp.sqrt(11)*mp.pi)"
```

## Conclusion

The system IS finding potentially novel results! The mp.tanh(mp.sqrt(79)*mp.pi) discovery with 10^-23 error is exactly the type of finding we're looking for:
- Uses a non-Heegner discriminant
- Not following known patterns
- Extremely close to an integer
- Simple and elegant expression

The key is to:
1. Exploit this success pattern
2. Reduce noise from known patterns
3. Focus exploration on similar structures
4. Use AI guidance to break out of local optima