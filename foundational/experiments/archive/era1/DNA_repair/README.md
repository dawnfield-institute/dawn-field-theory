# DNA Repair

## Hypothesis
Shannon entropy profiling, combined with QBE-inspired scoring, can detect and correct mutation-induced disorder in DNA and protein sequences without the need for sequence alignment or biological annotation.

## Status
archived

## Key Results
- **BRCA1 Frameshift Mutation (5382insC)**: Entropy-guided repair successfully detected and corrected the pathogenic frameshift mutation. The entropy profile of the repaired sequence realigned with the original healthy baseline.
- **Protein Sequence Repair (GDF15_HUMAN)**: Random mutations were introduced and repaired using entropy-based detection and QBE scoring, restoring both sequence identity and local entropy alignment.
- Confirms that entropy metrics can guide symbolic correction in biological sequences.
- Validates QBE scoring as a practical tool for mutation repair.
- Supports the Dawn Field Theory hypothesis that information-theoretic balance underpins structural integrity.

## FDO Links

## Scripts
| Script | Purpose |
|--------|---------|
| DNA_repairer.py | Entropy-based mutation detection and repair using Shannon entropy profiling and QBE-enhanced symbolic correction |

## References
- [Entropy-Based Repair of BRCA1 Frameshift Mutation (5382insC)](reference_material/Entropy-Based%20Repair%20of%20BRCA1%20Frameshift%20Mutation%20_5382insC__1_.md)
- [Entropy-Based Mutation Detection and Repair in Protein Sequences](reference_material/Entropy-Based%20Mutation%20Detection%20and%20Repair%20in%20Protein%20Sequences.md)
