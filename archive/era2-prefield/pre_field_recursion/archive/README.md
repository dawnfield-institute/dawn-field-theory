# Archive Index

This directory contains historical files from the Pre-Field Recursion project evolution (v1.0 → v2.2).

**Purpose:** Preserve development history while keeping the main project clean.

---

## Documentation (7 files)

### Session Notes (Chronological)
1. **PHASE1_COMPLETE.md** (Sep 30, 2025)
   - v2.0 formal framework implementation
   - Initial test suite (5/5 passing)
   - Baseline performance established

2. **PHASE2_SESSION_SUMMARY.md** (Oct 1, 2025)
   - v2.1 adaptive acceleration attempt
   - Over-damping issues discovered
   - Oscillatory behavior revealed

3. **PHASE3_SESSION_SUMMARY.md** (Oct 1, 2025)
   - v2.2 resonance-aware breakthrough
   - FFT-based frequency detection
   - 5.11x speedup achieved

### Technical Specifications
4. **UPGRADE_PLAN.md** (31,499 bytes)
   - Detailed v2.1 and v2.2 specifications
   - Implementation roadmap
   - Parameter tuning strategies

5. **IMPLEMENTATION_PROGRESS.md**
   - Phase-by-phase development tracking
   - Component completion status
   - Issue log

6. **V22_COMPLETE.md**
   - Executive summary of v2.2 achievements
   - Results visualization
   - Next steps

7. **PROJECT_CLEANUP.md**
   - Documentation of cleanup process (Oct 1)
   - Before/after structure comparison
   - File inventory

---

## Legacy READMEs (3 files)

1. **README_old.md** (Original)
   - v1.0 documentation
   - Initial experimental design
   - PAC conservation focus

2. **README_v1_backup.md** (Backup)
   - Identical to README_old.md
   - Preserved for safety

3. **README_v2.md** (Sep 30, 2025)
   - v2.0 formal framework documentation
   - Mathematical foundations
   - API reference

---

## Test Files (3 files)

1. **test_v2_alpha.py** (8,741 bytes)
   - v2.0 initial test suite
   - 5 comprehensive tests
   - All passing ✅

2. **test_convergence_v21.py** (9,431 bytes)
   - v2.0 vs v2.1 comparison
   - 6-panel visualization
   - Revealed over-damping issue

3. **test_convergence_v22.py** (14,059 bytes)
   - v2.0 vs v2.1 vs v2.2 comparison
   - Resonance detection validation
   - 5.11x speedup demonstration

**Note:** Functionality now consolidated in `../test_suite.py`

---

## Legacy Code (1 file)

1. **pre_field_recursion_unified.py** (23,346 bytes)
   - Original monolithic implementation
   - Pre-v2.0 experimental framework
   - Historical reference only

---

## Why These Files Were Archived

### Purpose
- **Preserve History:** Complete record of project evolution
- **Clean Main Directory:** Only active files in root
- **Enable Reverting:** Can recover old implementations if needed
- **Document Lessons:** Failed approaches inform future work

### Usage
- **For Reference:** Understanding design decisions
- **For Comparison:** See how implementations evolved
- **For Learning:** Study what worked and what didn't
- **For Recovery:** Restore old versions if needed

### Not Deprecated
These files aren't "bad" - they represent the natural evolution of research code:
- v2.0: Solid foundation ✅
- v2.1: Valuable learning ⚠️  
- v2.2: Built on lessons from both 🎯

---

## Active Files (Use These Instead)

### In Main Directory
- `../main.py` - Primary entry point (replaces pre_field_recursion_unified.py)
- `../test_suite.py` - Unified testing (replaces test_v2_*.py files)
- `../README.md` - Current documentation (replaces README_*.md)
- `../CHANGELOG.md` - Version history summary

### For Detailed Information
- **Technical Details:** Refer to archived PHASE*.md files
- **Specifications:** See UPGRADE_PLAN.md
- **Achievements:** Read V22_COMPLETE.md

---

## File Sizes

```
Total: ~165 KB across 14 files

Documentation:
├── UPGRADE_PLAN.md              31,499 bytes (largest)
├── IMPLEMENTATION_PROGRESS.md    4,995 bytes
├── PHASE1_COMPLETE.md            9,048 bytes
├── PHASE2_SESSION_SUMMARY.md     7,476 bytes
├── PHASE3_SESSION_SUMMARY.md     8,471 bytes
├── PROJECT_CLEANUP.md            8,155 bytes
└── V22_COMPLETE.md               6,592 bytes

Test Files:
├── test_convergence_v22.py      14,059 bytes
├── test_convergence_v21.py       9,431 bytes
└── test_v2_alpha.py              8,741 bytes

Legacy Code:
└── pre_field_recursion_unified.py 23,346 bytes

Legacy READMEs:
├── README_v2.md                  6,655 bytes
├── README_old.md                 6,182 bytes
└── README_v1_backup.md           6,182 bytes
```

---

## Restoration Instructions

If you need to restore any archived file:

```bash
# Copy file back to main directory
cp archive/FILENAME.py ../

# Or view without restoring
cat archive/FILENAME.py

# Or compare with current version
diff archive/old_file.py ../new_file.py
```

---

## Archive Maintenance

### When to Add Files
- Old test files after consolidation
- Superseded documentation
- Deprecated implementations
- Historical benchmarks

### When NOT to Archive
- Active development code
- Current documentation
- Latest test suites
- Required dependencies

### Periodic Review
Annually, review archive for:
- Files that can be permanently removed (>2 years old)
- Important insights to extract into main docs
- Lessons learned to document

---

**Dawn Field Institute**  
*Archive Created: October 1, 2025*  
*Purpose: Preserve history, maintain cleanliness*
