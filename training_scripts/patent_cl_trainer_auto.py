import pandas as pd
import logging
# Set logging to display info logs on the console
logging.getLogger().setLevel(logging.INFO)
from datetime import datetime
import os
import sys
import argparse

from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses
from sentence_transformers.readers import InputExample
import torch
import gc
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import TripletEvaluator

# Instantiate the parser
parser = argparse.ArgumentParser(description="Patentsview Triplet Trainer")

# Add the arguments
parser.add_argument(
    "-n",
    "--name",
    type=str,
    help="The name of the model to be saved.",
    required=True
)
parser.add_argument(
    "-c",
    "--csv",
    type=str,
    help="The path to the csv file containing the patents and additional information.",
    required=True,
)
parser.add_argument(
    "-col",
    "--column",
    type=str,
    help="The name of the column containing the classes.",
    default="cpc_subclasses",
)
parser.add_argument(
    "-s",
    "--sample_size",
    type=int,
    help="The number of patents to sample from the dataset.",
    default=None,
)
parser.add_argument(
    "-e",
    "--epochs",
    type=int,
    help="The number of epochs to train the model for.",
    default=10,
)
parser.add_argument(
    "-m",
    "--base_model",
    type=str,
    help="The name of the base model to use.",
    default="AI-Growth-Lab/PatentSBERTa",
)
parser.add_argument(
    "-b",
    "--batch_size",
    type=int,
    help="The batch size to use for training.",
    default=16,
)
parser.add_argument(
    "-d",
    "--duplicates",
    action="store_true",
    help="Allow duplicate classes in the dataset.",
)
parser.add_argument(
    "-fb",
    "--force_balanced",
    action="store_true",
    help="Force a balanced dataset.",
)

# Parse the arguments
args = parser.parse_args()
CUSTOM_MODEL_NAME = args.name
TRAINING_CSV = args.csv
SAMPLE_SIZE = args.sample_size
BASE_MODEL_NAME = args.base_model
ALLOW_DUPLICATE_CLASSES = args.duplicates
FORCE_BALANCED_DATASET = args.force_balanced
NUM_EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
COLUMN_NAME = args.column

output_path = f"{CUSTOM_MODEL_NAME}-{COLUMN_NAME}-S{SAMPLE_SIZE}-E{NUM_EPOCHS}-B{BATCH_SIZE}-{'exploded-' if ALLOW_DUPLICATE_CLASSES else ''}{'balanced-' if FORCE_BALANCED_DATASET else ''}{datetime.today().strftime('%Y-%m-%d')}"

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
if os.path.exists(f"{output_path}/console_output.log"):
    os.remove(f"{output_path}/console_output.log")
else:
    # Make sure the output directory exists
    os.makedirs(output_path)
file_handler = logging.FileHandler(f"{output_path}/console_output.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logging.getLogger("").addHandler(file_handler)

logging.info("Welcome to the Patent Triplet Trainer!")
logging.info("")
logging.info("You have chosen the following settings:")
logging.info(f"Model name: {CUSTOM_MODEL_NAME}")
logging.info(f"CSV file: {TRAINING_CSV}")
logging.info(f"Column name: {COLUMN_NAME}")
logging.info(f"Sample size: {SAMPLE_SIZE}")
logging.info(f"Allow duplicate classes: {ALLOW_DUPLICATE_CLASSES}")
logging.info(f"Force balanced dataset: {FORCE_BALANCED_DATASET}")
logging.info(f"Number of epochs: {NUM_EPOCHS}")
logging.info(f"Batch size: {BATCH_SIZE}")
logging.info("")
logging.info(f"Output path: {output_path}")
logging.info("")

logging.info("Load model")
model = SentenceTransformer(BASE_MODEL_NAME)

# torch check device
if torch.cuda.is_available():
    device = torch.device("cuda")
    logging.info(f"There are {torch.cuda.device_count()} GPU(s) available.")
    logging.info(f"Device name: {torch.cuda.get_device_name(0)}")
    gc.collect()
    torch.cuda.empty_cache()
    logging.info("Cache cleared.")
    torch.backends.cuda.max_split_size_mb = 70
    logging.info("Max split size set to 70MB.")
else:
    logging.info("No GPU available, using the CPU instead.")
    device = torch.device("cpu")

logging.info("Loading the patents csv file into a pandas dataframe...")
# Load the triplet csv file into a pandas dataframe
training_csv = pd.read_csv(TRAINING_CSV)

logging.info(f"Loaded {len(training_csv)} patents.")

total_papers = len(training_csv)

# Make all abstracts string
training_csv["abstract"] = training_csv["abstract"].astype(str)
# Drop all rows where the abstract is below 10 characters
training_csv = training_csv[training_csv["abstract"].str.len() > 10]

logging.info(f"Got {len(training_csv)} patents with abstracts longer than 10 characters ({len(training_csv) / total_papers * 100:.2f}%).")

# Convert the COLUMN_NAME column contents from string to list
training_csv[COLUMN_NAME] = training_csv[COLUMN_NAME].apply(lambda x: eval(x))

if ALLOW_DUPLICATE_CLASSES:
    # Explode the training_csv on the COLUMN_NAME column. The COLUMN_NAME column contains lists of areas for each paper.
    training_csv = training_csv.explode(COLUMN_NAME)
    logging.info(f"Exploded the dataframe. Got {len(training_csv)} patents ({len(training_csv) / total_papers * 100:.2f}%).")
else:
    # Only keep the rows with one entry in the COLUMN_NAME column list
    training_csv = training_csv[training_csv[COLUMN_NAME].map(len) == 1]
    # Convert the COLUMN_NAME column from a list to a string
    training_csv[COLUMN_NAME] = training_csv[COLUMN_NAME].apply(lambda x: x[0])
    logging.info(f"Kept only patents with one {COLUMN_NAME}. Got {len(training_csv)} patents ({len(training_csv) / total_papers * 100:.2f}%).")

# Get a set of all the unique COLUMN_NAME and make a new dict with a number as id and COLUMN_NAME as value
areaID_set = set(training_csv[COLUMN_NAME])
areaID_dict = {areaID: i for i, areaID in enumerate(areaID_set)}
logging.info(f"Got {len(areaID_set)} unique {COLUMN_NAME}s:")
logging.info(areaID_dict)

if SAMPLE_SIZE is None:
    SAMPLE_SIZE = len(training_csv)

# Sample the training_csv but keep the same number of patents per COLUMN_NAME so the dataset is balanced. Get a sample of SAMPLE_SIZE papers in total.
if FORCE_BALANCED_DATASET:
    training_csv = training_csv.groupby(COLUMN_NAME).apply(lambda x: x.sample(SAMPLE_SIZE // len(areaID_set), replace=True, random_state=42))
    training_csv = training_csv.reset_index(drop=True)
    logging.info(f"Got {SAMPLE_SIZE} patents after sampling with {SAMPLE_SIZE // len(areaID_set)} patents per {COLUMN_NAME} ({SAMPLE_SIZE // len(areaID_set) / len(training_csv) * 100:.2f}%).")
else:
    training_csv = training_csv.sample(SAMPLE_SIZE, random_state=42)
    logging.info(f"Got {SAMPLE_SIZE} patents after sampling randomly ({SAMPLE_SIZE / len(training_csv) * 100:.2f}%).")

# Create a train and test dataframe
train_df = training_csv.sample(frac=0.8, random_state=42)
test_df = training_csv.drop(train_df.index)

logging.info("Creating training and test input examples...")
# Create a list of InputExample objects. Take the abstract as text and the areaID_dict key as label
train_examples = [
    InputExample(texts=[str(row["abstract"])], label=areaID_dict[row[COLUMN_NAME]])
    for _, row in train_df.iterrows()
]
# print(train_examples[0])
test_examples = [
    InputExample(texts=[str(row["abstract"])], label=areaID_dict[row[COLUMN_NAME]])
    for _, row in test_df.iterrows()
]
# print(test_examples[0])

# Create a SentencesDataset object for train and test
train_dataset = SentencesDataset(train_examples, model=model)
test_dataset = SentencesDataset(test_examples, model=model)

# Create a loss object
# train_loss = losses.BatchHardSoftMarginTripletLoss(model=model)
train_loss = losses.BatchAllTripletLoss(model=model, margin=0.5)

# Create a dataloader object
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=BATCH_SIZE)

warmup_steps = int(len(train_dataset) * NUM_EPOCHS / 16 * 0.1) #10% of train data for warm-up

logging.info("Beginning training...")
# Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=NUM_EPOCHS,
    # evaluator=evaluator,
    evaluation_steps=1000,
    warmup_steps=warmup_steps,
    output_path=output_path,
    use_amp=True,
)
# test_evaluator = TripletEvaluator(test_dataloader)
# model.evaluate(test_ev aluator, output_path=output_path)

logging.info("Done training.")
logging.info("Saving the model...")

# Save the model
model.save(output_path)

logging.info("Done saving the model.")
logging.info("Done.")