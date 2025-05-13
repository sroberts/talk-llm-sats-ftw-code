# LLM SATs FTW

A collection of experiments exploring Large Language Models (LLMs) and their performance on SAT-style tasks, powered by Streamlit.

## Features

- Multiple LLM-based experiments
- Interactive Streamlit UI
- Easy to extend and customize

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/talk-llm-sats-ftw-code.git
    cd talk-llm-sats-ftw-code
    ```

2. **Create a virtual environment (optional but recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## OpenAI API Key Required

To use these experiments, you must provide your own OpenAI API key. You can obtain an API key by signing up at [OpenAI](https://platform.openai.com/account/api-keys). The app will prompt you to enter your key when you run an experiment.

## Running Experiments

Each experiment is implemented as a separate Streamlit app in the main directory, named `experiment-<number>.py`.

To run an experiment, use:

```bash
streamlit run *.py
```

For example:

```bash
streamlit run experiment-1-starburst.py
```

## Available Experiments

- `experiment-1-starburst.py`: Starburst
- `experiment-2-ach.py`: Analysis of Competing Hypotheses (ACH)
- `experiment-3-kac.py`: Key Assumptions Check (KAC)

## License

MIT License
