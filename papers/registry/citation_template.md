# Hardware Configuration Citation Template

For consistent hardware documentation across Dawn Field Theory preprints and experiments.

## Standard Citation Format

```markdown
**Hardware Configuration**: All computational results obtained on RTX 3070ti + i9-12th gen laptop platform (see [resources/specs/hardware_timeline.yaml](hardware_timeline.yaml) for complete specifications).
```

## Performance Context Note

```markdown
**Performance Context**: Timing results and computational metrics specific to documented hardware configuration. Relative performance ratios should remain consistent across platforms, while absolute timing will vary with hardware specifications.
```

## Reproducibility Statement

```markdown
**Reproducibility**: Results reproducible with equivalent hardware specifications. CUDA acceleration required for optimal performance. Different platforms may show absolute timing variations while maintaining relative performance characteristics.
```

## Complete Integration Example

```markdown
### Computational Platform and Performance Context

**Hardware Configuration**: All computational results obtained on RTX 3070ti + i9-12th gen laptop platform (see [resources/specs/hardware_timeline.yaml](hardware_timeline.yaml) for complete specifications).

**Performance Context**: Timing results and computational metrics specific to documented hardware configuration. Relative performance ratios should remain consistent across platforms, while absolute timing will vary with hardware specifications.

**Reproducibility**: Results reproducible with equivalent hardware specifications. CUDA acceleration required for optimal performance. Different platforms may show absolute timing variations while maintaining relative performance characteristics.
```

## Usage Guidelines

1. **Include at beginning of computational sections**: Add hardware context before presenting performance results
2. **Reference centralized specs**: Always link to hardware_timeline.yaml rather than duplicating specifications
3. **Distinguish absolute vs relative metrics**: Clarify which results are hardware-dependent
4. **Note computational constraints**: Document VRAM limits, grid resolution constraints, etc.

## Integration Points

- **Preprint drafts**: Add to computational implementation or results sections
- **Experiment documentation**: Include in methodology sections
- **Performance benchmarks**: Essential for comparative analysis
- **Validation studies**: Required for reproducibility

---

*This template ensures consistent hardware documentation while maintaining centralized specification management.*
