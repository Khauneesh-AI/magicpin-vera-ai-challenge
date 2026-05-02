#!/bin/bash
# Run judge simulator against deployed Railway bot
# Usage: ./run_judge.sh [scenario]
# Examples:
#   ./run_judge.sh              # runs full_evaluation
#   ./run_judge.sh all          # runs all scenarios (warmup + auto-reply + intent + hostile)
#   ./run_judge.sh phase2_short # quick test

cd "$(dirname "$0")"

export BOT_URL=https://magicpin-vera-ai-challenge-production-66a3.up.railway.app
export LLM_PROVIDER=openai
export LLM_API_KEY=$(sed -n 's/^OPENAI_API_KEY=//p' .env | tr -d '"')
export LLM_MODEL=gpt-5.4
export TEST_SCENARIO=${1:-full_evaluation}

.venv/bin/python judge_simulator.py
