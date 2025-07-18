install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	autopep8 *.py

train:
	python train.py

eval:
	echo '## Model Metrics' report.md
	cat ./Results/metrics.txt report.md

	echo '\n Confusion Matrix Plot' >> report.md
	echo '![Confusion Matrix](./Results/modelResults.png)' >> report.md