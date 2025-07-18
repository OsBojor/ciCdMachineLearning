install:
	pip install --upgrade pip

installReq:
	python --version
	pip --version
	python -m pip install black
	python -m pip install pandas
	python -m pip install scikit-learn
	python -m pip install skops
	python -m pip install matplotlib
	python -m pip install -r requirements.txt

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