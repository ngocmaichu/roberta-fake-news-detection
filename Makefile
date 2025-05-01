# Variables
NOTEBOOK = roberta_fakenews_final.ipynb
MODEL_DIR = ./results
HUB_REPO = ngocmaichu/roberta-fake-news-detection  # Change this to your HF repo name
PYTHON = python

# Default target
all: convert run push

# Convert notebook to script (for CLI execution or versioning)
convert:
	jupyter nbconvert --to script $(NOTEBOOK)

# Execute notebook directly (no .py file required)
run:
	jupyter nbconvert --to notebook --execute $(NOTEBOOK) --inplace

# Push model and tokenizer to Hugging Face Hub
push:
	$(PYTHON) -c "from transformers import AutoModelForSequenceClassification; \
AutoModelForSequenceClassification.from_pretrained('$(MODEL_DIR)').push_to_hub('$(HUB_REPO)')"
	$(PYTHON) -c "from transformers import AutoTokenizer; \
AutoTokenizer.from_pretrained('$(MODEL_DIR)').push_to_hub('$(HUB_REPO)')"

# Clean artifacts
clean:
	rm -rf $(MODEL_DIR)
	rm -f roberta_fakenews_final.py
