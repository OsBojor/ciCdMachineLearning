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
	python -m pip install gradio
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

update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)x
	git commit -am "Update with new results"
	git push --force origin HEAD:update

hf-login:
	git pull origin update
	git switch update
	python -m pip install -U "huggingface_hub[cli]"
	echo $(HUGGING_FACE)
	huggingface-cli login --token $(HUGGING_FACE) --add-to-git-credential

push-hub:
	huggingface-cli upload osBojor/studentsAcademicSucessPrediction ./App --repo-type=space --commit-message="Sync App files"
	huggingface-cli upload osBojor/studentsAcademicSucessPrediction ./Model --repo-type=space --commit-message="Sync Model files"
	huggingface-cli upload osBojor/studentsAcademicSucessPrediction ./Results --repo-type=space --commit-message="Sync Results files"

deploy: hf-login push-hub