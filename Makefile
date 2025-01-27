.PHONY: setup
setup:
	python3.11 -m venv Fetch-venv
	source Fetch-venv/bin/activate && \
	export PYTHONPATH=$$(pwd) && \
	python3.11 -m pip install --upgrade pip && \
	pip install -r requirements.txt
