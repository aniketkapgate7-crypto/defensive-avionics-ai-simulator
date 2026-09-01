# Complete Project Structure

```text
defensive-avionics-ai-simulator/
├── .github/workflows/ci.yml
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
├── requirements.txt
├── requirements/
│   ├── base.txt
│   ├── dev.txt
│   ├── policy.txt
│   ├── signal.txt
│   ├── ui.txt
│   └── vision.txt
├── configs/
│   ├── default.yaml
│   ├── policy.yaml
│   ├── signal.yaml
│   ├── ui.yaml
│   └── vision.yaml
├── data/
│   ├── README.md
│   ├── raw/radioml/
│   ├── raw/vision/images/
│   ├── raw/vision/labels/
│   ├── interim/
│   └── processed/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DATASET_SETUP.md
│   ├── METRICS.md
│   ├── PRESENTATION_OUTLINE.md
│   ├── PROJECT_PLAN.md
│   ├── PROJECT_STRUCTURE.md
│   └── SAFETY_AND_SCOPE.md
├── assets/
│   ├── README.md
│   └── ui/
├── models/
│   ├── README.md
│   ├── policy/
│   ├── signal/
│   └── vision/
├── notebooks/
│   └── README.md
├── outputs/
│   ├── README.md
│   ├── figures/
│   ├── logs/
│   ├── reports/
│   └── videos/
├── scripts/
│   └── check_setup.py
├── templates/
│   ├── DATASET_CARD_TEMPLATE.md
│   ├── EXPERIMENT_RECORD_TEMPLATE.md
│   └── MODEL_CARD_TEMPLATE.md
├── src/defensive_avionics/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── config.py
│   ├── common/
│   ├── integration/
│   ├── policy/
│   ├── signal/
│   ├── ui/
│   └── vision/
└── tests/
    ├── test_config.py
    ├── test_orchestrator.py
    └── test_urgency.py
```

## Ownership rule

Each model module owns its data adapter, preprocessing, training, evaluation,
and inference code. Only typed predictions may cross into the integration
layer. The UI reads the integrated state and never calls a training function.
