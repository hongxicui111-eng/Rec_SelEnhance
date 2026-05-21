# Rec_SelEnhance

**Self-Evolving Recommendation System with Selective Enhancement**

An LLM-driven agent that iteratively improves SASRec recommendation models through selective enhancement — automatically identifying model weaknesses, proposing improvements, and validating results in a continuous self-evolution loop.

## Architecture

The system consists of three core components:

- **Recmodel/** — SASRec recommendation model with fine-tuning, error case extraction, and surprise evaluation
- **agent/** — Self-evolving agent with LLM analysis, iterative memory, fault-tolerant training, and auto-correction loops
- **run_evolve.py** — Main entry point for running the evolution process

## Quick Start

```bash
# Set up LLM service (e.g., vLLM with Qwen2.5-72B-Instruct)
python run_evolve.py --data Beauty --backbone SASRec --iterations 30
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `LLM_API_URL` | LLM service URL |
| `LLM_API_KEY` | API Key |
| `LLM_MODEL` | Model name |
| `PROJECT_ROOT` | Recommendation model project root directory |

## Evolution Process

The agent follows a 7-phase iterative loop:

1. **Phase 0** — Baseline training and evaluation
2. **Phase 1** — Error case extraction from training data
3. **Phase 2** — LLM-based analysis of model weaknesses
4. **Phase 3** — Improvement proposal generation
5. **Phase 4** — Code/structure modification
6. **Phase 5** — Retraining with improvements
7. **Phase 6** — Results validation and journaling

Each iteration feeds back into Phase 0, creating a continuous self-improvement cycle.

## Supported Datasets

- Amazon Beauty
- Amazon Toys and Games
- Amazon Yelp
- Amazon Video Games
- Amazon Sports and Outdoors

## License

MIT