install:
	pip install --upgrade pip

installReq:
	python --version
	pip --version
	python -m pip install black
	python3 -m pip install -r requirements.txt --user

format:
	black *.py

train:
	python train.py

eval:
	echo "## Model Metrics" > report.md
	cat ./Results/metrics.txt >> report.md

	echo '\n## Confusion Matrix Plot' >> report.md
	echo '![Confusion Matrix](./Results/model_results.png)' >> report.md

	cml comment create report.md