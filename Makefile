# Makefile for fine-tuning RoBERTa with custom weights and training efficiency

PYTHON = python
NOTEBOOK = roberta_fakenews_final.ipynb
SCRIPT = roberta_fakenews_final.py
MODEL_DIR = ./results
HUB_REPO = ngocmaichu/roberta-fake-news-detection

# Default pipeline
all: convert train push

# Step 1: Convert notebook to script
convert:
	jupyter nbconvert --to script $(NOTEBOOK) --output $(SCRIPT)

# Step 2: Train the model with class weights
train: convert
	$(PYTHON) $(SCRIPT)

# Step 3: Push model + tokenizer to Hugging Face Hub
push:
	$(PYTHON) -c "from transformers import AutoModelForSequenceClassification; \
AutoModelForSequenceClassification.from_pretrained('$(MODEL_DIR)').push_to_hub('$(HUB_REPO)')"
	$(PYTHON) -c "from transformers import AutoTokenizer; \
AutoTokenizer.from_pretrained('$(MODEL_DIR)').push_to_hub('$(HUB_REPO)')"

# Step 4: Clean up
clean:
	rm -f $(SCRIPT)
	rm -rf $(MODEL_DIR)
