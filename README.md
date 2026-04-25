# Assignment 1 - Membership Inference Attack (MIA)

This repository contains starter files for Assignment 1 (MIA).

## Repository Structure

- `task_template.py`: starter script for loading data/model, generating a submission CSV, and submitting it.
- `env.example`: example environment variable file.
- `data/pub.pt`: public dataset.
- `data/priv.pt`: private dataset used for scoring.
- `data/model.pt`: target model weights.
- `data/submission.csv`: generated submission file.
- `Tutorial 1 - Assignment 1 Walkthrough.html`: walkthrough/reference material.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install torch torchvision pandas requests python-dotenv
```

3. Create a `.env` file in the project root and add your API key:

```env
API_KEY=your_api_key_here
```

You can copy `env.example` and replace the placeholder value.

## Run

From the project root:

```bash
python task_template.py
```

The script will:

1. Load datasets and model.
2. Create a `data/submission.csv` file.
3. Submit the CSV to the remote server.

## Important Note

`task_template.py` currently creates a random submission score for each sample as a placeholder. Replace this logic with your actual attack/scoring method before final submission.

## Troubleshooting

- If submission fails, verify `API_KEY` is present in `.env`.
- Ensure all files in `data/` exist and are not moved/renamed.
- If Python cannot import packages, re-check that your virtual environment is active.
