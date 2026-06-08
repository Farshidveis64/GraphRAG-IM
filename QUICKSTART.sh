#!/bin/bash

echo "GraphRAG-IM - Quick Commands Reference"
echo "======================================"
echo ""

echo "1. INSTALLATION"
echo "   pip install -r requirements.txt        # core (offline) only"
echo "   pip install -r requirements.txt torch torch-geometric sentence-transformers  # neural"
echo ""

echo "2. DATA PREPROCESSING"
echo "   Build an example on-disk dataset:"
echo "   python scripts/preprocess.py --dataset example --output_dir data/processed"
echo ""

echo "3. INFERENCE + EVALUATION (offline, no GPU / API key)"
echo "   Synthetic, DPP vs top-k:"
echo "   python scripts/evaluate.py --config configs/default.yaml --dataset synthetic --k 20 --compare_selection --output outputs"
echo ""
echo "   On the disk dataset:"
echo "   python scripts/evaluate.py --dataset example --k 20 --compare_selection --output outputs"
echo ""

echo "4. TRAINING (neural scorer; requires torch + torch-geometric)"
echo "   python scripts/train.py --config configs/default.yaml --dataset synthetic --output_dir outputs/model"
echo ""

echo "5. FIGURES"
echo "   python scripts/visualize.py --output_dir outputs/figures"
echo ""

echo "6. TESTS"
echo "   python tests/run_tests.py        # no pytest needed"
echo "   pytest tests/                    # if pytest is installed"
echo ""

echo "7. KEY CONFIGURATION PARAMETERS (configs/default.yaml)"
echo "   Diffusion:"
echo "   - p0: 0.1            base activation probability"
echo "   - rho_mode: cosine   content-conditioned engagement"
echo "   Selection:"
echo "   - k: 50              seed budget"
echo "   - method: dpp        content-conditioned DPP"
echo "   - theta: 0.1         content-dissimilarity strength"
echo ""

echo "For more details, see README.md"
