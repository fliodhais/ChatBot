# ChartBot

## Setup

### Virtual Environment (optional, recommended)

1. Create virtual env `py -3 -m venv VENVNAME`
2. Enter venv `VENVNAME\Scripts\activate`

### Next steps

3. install dependencies `pip install -r requirements.txt`
4. `set FLASK_APP = api` (does not work for now it seems), tried adding .env
5. `py -m flask run`

## Build/Deploy

1. build `docker build --tag python-docker .`

## Pre-deploy maintanence

1. `docker scan python-docker`

## Run

1. `docker run`

## Commands

leave venv `venv\Scripts\deactivate`

uninstall everything (except pip/setuptools/wheel) `pip uninstall -r requirementst.txt -y`

## Versions

Python = 3.9.12 (stable) | 3.10.5 (seems ok)
