from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_MODEL = "cointegrated/rubert-tiny"
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
BATCH_SIZE = 100
EPOCHS = 20
output_hidden_states = True

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1, output_hidden_states=output_hidden_states)