import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import csv
import logging

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# File path for the variant file (update the path accordingly)
variant_file = '../data/filtered_data.csv'
# Load the CSV file
df = pd.read_csv(variant_file)

# Drop the 'is_strong' column as it's not needed right now
if 'is_strong' in df.columns:
    df = df.drop(columns=['is_strong'])

# Model and tokenizer setup
model_name = "Qwen/Qwen2.5-32B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # bf16 precision
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# System prompt
system_prompt = (
    "You are an expert in bioinformatics and genetics. "
    "Please take this variant and its additional information from a variant call format file and its corresponding header, "
    "and convert it into a human-interpretable sentence containing the same information."
)

# Open a new CSV file to write the results
output_file = '../data/variant_interpretations.csv'

# Write headers to the CSV
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['identifier', 'sentence'])  # Add CSV headers

    # Processing each line (minus the header)
    for idx, row in df.iterrows():
        # Log the progress
        logging.info(f"Processing variant {idx + 1}/{len(df)}")

        # Prepare input variant (convert the row into a formatted string for the model)
        variant_data = row.to_dict()
        variant_input = ', '.join([f"{key}: {value}" for key, value in variant_data.items()])

        # Assume identifier is in the first column of the row (modify if necessary)
        identifier = row[0]

        # Chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Variant data: {variant_input}"}
        ]

        # Tokenize the input text
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # Generate a response
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=2000,  # Adjusted to 2000 token output length
            do_sample=False  # Disable sampling for deterministic output
        )

        # Extract and decode the response
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Write the result to the CSV
        writer.writerow([identifier, response])

    logging.info("Processing complete.")

