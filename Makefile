## Set up python interpreter environment
create_env:
	conda env create -f environment.yml

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Get dataset using Kaggle API
get_data:
	kaggle competitions download -c digit-recognizer
	mv digit-recognizer.zip data/raw