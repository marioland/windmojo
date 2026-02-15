#!/bin/bash
# Run Phase 2 training for all 24 feature selections
# This will take approximately 2-3 hours to complete

set -e  # Exit on error

echo "================================================================================================"
echo "PHASE 2: ALL 24 FEATURE SELECTIONS"
echo "================================================================================================"
echo ""
echo "This script will train 24 different feature selection strategies and compare results."
echo "Estimated time: 2-3 hours"
echo ""

# Configuration
CONFIG="config/lake_como/dervio.yaml"
HOURS="08:00,13:00,14:00,15:00,16:00"

# Train all selections
echo "Starting Phase 2 training..."
echo ""

uv run python windmojo/train_phase2.py \
    --config "$CONFIG" \
    --selection all \
    --hours "$HOURS"

# Generate comparison report
echo ""
echo "Generating comparison report..."
uv run python scripts/generate_phase2_comparison_report.py > data/lake_como/dervio/models/phase2_comparison/report.txt

echo ""
echo "================================================================================================"
echo "✓ PHASE 2 COMPLETE"
echo "================================================================================================"
echo ""
echo "Results saved to: data/lake_como/dervio/models/phase2_results/"
echo "Comparison report: data/lake_como/dervio/models/phase2_comparison/report.txt"
echo ""
echo "Next steps:"
echo "  1. Review comparison report to identify best selection"
echo "  2. Analyze feature importance for top selections"
echo "  3. Proceed to production model training with best selection"
echo ""
