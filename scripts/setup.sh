#!/bin/bash
# Setup script for NeMo verification environment
#
# Usage: ./setup.sh
#
# This script creates a Python 3.12 virtual environment and installs
# all dependencies needed to run the NeMo verification scripts.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== NeMo Verification Environment Setup ==="
echo ""

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' is not installed."
    echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Create venv with Python 3.12
echo "Creating Python 3.12 virtual environment..."
uv venv --python 3.12

# Activate venv
source .venv/bin/activate

# Install core dependencies first (to avoid resolver conflicts)
echo ""
echo "Installing core dependencies..."
uv pip install torch numpy soundfile numba

# Install NeMo without deps to avoid numba version conflicts
echo ""
echo "Installing NeMo toolkit..."
uv pip install nemo_toolkit --no-deps

# Install remaining dependencies
echo ""
echo "Installing NeMo runtime dependencies..."
uv pip install -r requirements.txt

# Create mock nv_one_logger package (NVIDIA internal, not on PyPI)
echo ""
echo "Creating mock for nv_one_logger (NVIDIA internal package)..."
SITE_PACKAGES=".venv/lib/python3.12/site-packages"
mkdir -p "$SITE_PACKAGES/nv_one_logger/api"
mkdir -p "$SITE_PACKAGES/nv_one_logger/training_telemetry/api"
mkdir -p "$SITE_PACKAGES/nv_one_logger/training_telemetry/integration"

cat > "$SITE_PACKAGES/nv_one_logger/__init__.py" << 'EOF'
# Mock nv_one_logger package for NeMo compatibility
EOF

cat > "$SITE_PACKAGES/nv_one_logger/api/__init__.py" << 'EOF'
# Mock module
EOF

cat > "$SITE_PACKAGES/nv_one_logger/api/config.py" << 'EOF'
class OneLoggerConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
EOF

cat > "$SITE_PACKAGES/nv_one_logger/training_telemetry/__init__.py" << 'EOF'
# Mock module
EOF

cat > "$SITE_PACKAGES/nv_one_logger/training_telemetry/api/__init__.py" << 'EOF'
# Mock module
EOF

cat > "$SITE_PACKAGES/nv_one_logger/training_telemetry/api/callbacks.py" << 'EOF'
def on_app_start():
    pass
EOF

cat > "$SITE_PACKAGES/nv_one_logger/training_telemetry/api/config.py" << 'EOF'
class TrainingTelemetryConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
EOF

cat > "$SITE_PACKAGES/nv_one_logger/training_telemetry/api/training_telemetry_provider.py" << 'EOF'
class _MockConfig:
    telemetry_config = None

class TrainingTelemetryProvider:
    _instance = None
    config = _MockConfig()
    
    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def with_base_config(self, config):
        return self
    
    def with_export_config(self):
        return self
    
    def configure_provider(self):
        pass
    
    def set_training_telemetry_config(self, config):
        self.config.telemetry_config = config
EOF

cat > "$SITE_PACKAGES/nv_one_logger/training_telemetry/integration/__init__.py" << 'EOF'
# Mock module
EOF

cat > "$SITE_PACKAGES/nv_one_logger/training_telemetry/integration/pytorch_lightning.py" << 'EOF'
from lightning.pytorch.callbacks import Callback

class TimeEventCallback(Callback):
    """Mock TimeEventCallback that does nothing."""
    def __init__(self, provider=None, call_on_app_start=True):
        super().__init__()
EOF

# Verify installation
echo ""
echo "Verifying NeMo installation..."
python -c "import nemo.collections.asr as nemo_asr; print('NeMo ASR imported successfully!')" 2>&1 | grep -v "Warning\|warning\|W0"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate the environment:"
echo "  source scripts/.venv/bin/activate"
echo ""
echo "To run the verification script:"
echo "  python scripts/verify_nemo.py \\"
echo "    --model models/nvidia/parakeet-ctc-0.6b/parakeet-ctc-0.6b.nemo \\"
echo "    --audio samples/2086-149220-0033.wav"
