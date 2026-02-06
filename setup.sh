#!/bin/bash
# Setup script for OpenVLA-OFT Franka deployment

set -e  # Exit on error

echo "=========================================="
echo "OpenVLA-OFT Franka Setup"
echo "=========================================="

# Check if conda environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Error: No conda environment activated!"
    echo "Please activate your OpenVLA-OFT environment first:"
    echo "  conda activate openvla-oft"
    exit 1
fi

echo ""
echo "Active conda environment: $CONDA_DEFAULT_ENV"
echo ""

# Install Franka-specific dependencies
echo "Installing Franka-specific dependencies..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. List cameras:        python camera_utils.py"
echo "  2. Test cameras:        python test_cameras.py --external-camera XXX --wrist-camera YYY"
echo "  3. Test robot:          python test_robot.py --nuc-ip 192.168.1.143"
echo "  4. Test policy:         python test_policy.py"
echo "  5. Run deployment:      python main.py --instruction 'pick up the red block'"
echo ""
echo "For mock testing (no hardware):"
echo "  python main.py --use-mock-cameras --use-mock-robot --instruction 'test task'"
echo ""
