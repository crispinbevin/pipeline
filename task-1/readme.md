# PDF Extraction Pipeline

## Setup Instructions

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git

# Navigate to the project directory
cd pipeline/task-1

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate   # On Windows
# source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt

#Create folder to place documents OR place documents in the same directory and parse it by speicifying in command line
mkdir docs



# Run the script
# Process a single file
python script.py ./path/to/document.pdf

# Process all PDFs in a directory
python script.py ./path/to/pdfs

# Default to "./input-docs" if nothing provided
python script.py
