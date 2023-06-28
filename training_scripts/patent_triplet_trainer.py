import pandas as pd
import logging
from datetime import datetime

from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses
from sentence_transformers.readers import InputExample
import torch
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import TripletEvaluator


MODEL_NAME = "patentCL"
TRAINING_CSV = "data/triplet_training.csv"
SAMPLE_SIZE = "1M"
NUM_EPOCHS = 10


logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

logging.info("Load model")
model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')

# torch check device
if torch.cuda.is_available():
    device = torch.device("cuda")
    logging.info(f"There are {torch.cuda.device_count()} GPU(s) available.")
    logging.info(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    logging.info("No GPU available, using the CPU instead.")
    device = torch.device("cpu")

print("")
print("Welcome to the Patent Triplet Trainer!")
print("")

print("Loading the triplet csv file into a pandas dataframe...")
# Load the triplet csv file into a pandas dataframe
triplet_df = pd.read_csv(TRAINING_CSV)

# Create a train and test dataframe
train_df = triplet_df.sample(frac=0.8, random_state=42)
test_df = triplet_df.drop(train_df.index)

print("Creating training and test Input Examples...")
# Create a list of InputExample objects by concatenating the a_title [SEP] a_abstract, p_title [SEP] p_abstract, n_title [SEP] n_abstract
train_examples = []
for i, row in train_df.iterrows():
    train_examples.append(
        InputExample(
            texts=[
                str(row['a_title']) + " [SEP] " + str(row['a_abstract']),
                str(row["p_title"]) + " [SEP] " + str(row["p_abstract"]),
                str(row["n_title"]) + " [SEP] " + str(row["n_abstract"])
                ], 
            label=0
            )
        )
print(f"Number of training examples: {len(train_examples)}")
    
test_examples = []
for i, row in test_df.iterrows():
    test_examples.append(
        InputExample(
            texts=[
                str(row['a_title']) + " [SEP] " + str(row['a_abstract']),
                str(row["p_title"]) + " [SEP] " + str(row["p_abstract"]),
                str(row["n_title"]) + " [SEP] " + str(row["n_abstract"])
                ], 
            label=0
            )
        )
print(f"Number of test examples: {len(test_examples)}")
    
# Create a SentencesDataset object for train and test
train_dataset = SentencesDataset(train_examples, model=model)
test_dataset = SentencesDataset(test_examples, model=model)

# Create a loss object
train_loss = losses.TripletLoss(model=model)

# Create a dataloader object
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=16)

# Create a logging object
logging = LoggingHandler()

# Create a model name
output_model_name = "own_models/" + str(MODEL_NAME) + "_epochs_" + str(NUM_EPOCHS) + "_sample_" + str(SAMPLE_SIZE) + "_" + str(datetime.now().strftime("%Y-%m-%d_%H-%M"))

warmup_steps = int(len(train_dataset) * NUM_EPOCHS / 16 * 0.1) #10% of train data for warm-up

# Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=TripletEvaluator.from_input_examples(test_examples, name='triplet-evaluation'),
    epochs=NUM_EPOCHS,
    evaluation_steps=1000,
    warmup_steps=warmup_steps,
    output_path=output_model_name,
    use_amp=True
    )

test_evaluator = TripletEvaluator.from_input_examples(test_examples, name='triplet-evaluation')
model.evaluate(test_evaluator, output_path=output_model_name)

# Save the model
model.save(output_model_name)