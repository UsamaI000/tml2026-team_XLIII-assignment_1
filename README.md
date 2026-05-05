# Assignment 1 - Membership Inference Attack (MIA)

Best Results: Multi-Attack with 70 Shadow Models (25 Epochs, MLP Classifier)

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install torch torchvision pandas numpy scikit-learn requests python-dotenv
```

3. Create a `.env` file in the project root and add your API key:

```env
API_KEY=your_api_key_here
```

You can copy `env.example` and replace the placeholder value.

## Recreate Best Results

Run the multi-attack approach with shadow models:

```bash
python shadow_model_updated_arch_multi_attack.py
```

### Configuration (in `shadow_model.py`)

The best results use these settings:
- **N_SHADOW**: 70 shadow models
- **EPOCHS**: 25 epochs per model
- **USE_MLP**: True (neural network attack classifier)
- **FEATURE_MODE**: "full" (all features including logits, probs, loss, entropy, margin)
- **RANDOM_SEED**: 12

### Output

The script generates `multi_attack_70m_25e.csv` containing:
- `id`: sample identifier
- `score`: membership probability (0 = non-member, 1 = member)

### Submission

To submit the generated CSV:

```bash
python task_template.py  # modify to use the generated CSV
```

