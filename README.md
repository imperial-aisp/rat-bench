# 📚 RAT-Bench anonymous submission

Benchmark for evaluating PII removal and text anonymization tools, with a focus on **re-identification risk**.

# ⚙️ Installation

This project uses the [`uv`](https://github.com/astral-sh/uv) package manager.

### 1. Set up the virtual environment

Run the following inside the project root:

```bash
uv sync
```

This will create a virtual environment inside the `.venv` folder. Run all next commands within this virtual environment.

Make sure to then activate the environment, by running `source .venv/bin/activate`.

To install the [correctmatch](https://github.com/imperial-aisp/pycorrectmatch/tree/master) package from Rocher et al. that is used to compute re-identification risk, run the following:

```bash
python install_correctmatch.py
```

You'll need to install the `spacy` transformers by running

```
uv pip install pip
python -m spacy download en_core_web_lg
```

If you would need anonymizers also for other languages, repeat the same process to download `es_core_news_lg`(Spanish) and `zh_core_web_lg`(Chinese).

Copy ``pii_benchmark/credentials_example.py`` → ``pii_benchmark/credentials.py``.

Add your API keys inside ``credentials.py``.

(Note: ``credentials.py`` is already in ``.gitignore`` and won’t be committed.)

# ⚙️ Data

We have included the processed datasets needed to run our experiments in this repo:

`data/100_profiles.csv` contains the profiles, including direct and indirect identifiers, used to generate benchmark entries.
`data/population_sample.csv` contains the 3M sample of [US census data](https://www.census.gov/programs-surveys/acs/microdata/access/2010.html), preprocessed to account for weighting of each row.
