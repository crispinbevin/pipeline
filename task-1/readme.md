# PDF Extraction Pipeline

## Setup Instructions

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git

# Navigate to the project directory
cd pipeline

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate   # On Windows
# source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the script
python doc_extract_preprocess.py
