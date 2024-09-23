# Still runs quite slow
import pandas as pd
from vllm import LLM, SamplingParams
import csv
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the GPU to be used (GPU 0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

# File path for the variant file (update the path accordingly)
variant_file = '../data/filtered_data.csv'

# Load the CSV file
logging.info(f"Loading variant file: {variant_file}")
df = pd.read_csv(variant_file)

# Drop the 'is_strong' column as it's not needed right now
if 'is_strong' in df.columns:
    logging.info("Dropping 'is_strong' column")
    df = df.drop(columns=['is_strong'])

# Model setup using vLLM with Gemma-2-27b
model_name = "google/gemma-2-27b"
logging.info(f"Setting up the model: {model_name}")
llm = LLM(model=model_name)

# Sampling parameters for text generation
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2000)

# System prompt
system_prompt = (
    "You are an expert in bioinformatics and genetics. "
    "Please take this variant and its additional information from a variant call format file and its corresponding header, "
    "and convert it into a human-interpretable sentence containing the same information."
)

# Output file for variant interpretations
output_file = '../data/variant_interpretations.csv'

# Write headers to the CSV
logging.info(f"Writing headers to the output file: {output_file}")
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['identifier', 'sentence'])  # Add CSV headers

    # Processing each line (minus the header)
    for idx, row in df.iterrows():
        # Log the start of processing for the variant
        logging.info(f"Processing variant {idx + 1}/{len(df)}")

        # Prepare input variant (convert the row into a formatted string for the model)
        variant_data = row.to_dict()
        variant_input = ', '.join([f"{key}: {value}" for key, value in variant_data.items()])

        # Assume identifier is in the first column of the row (modify if necessary)
        identifier = row[0]

        # Log the prepared input for the variant
        logging.debug(f"Variant {identifier} input: {variant_input}")

        # Full prompt for the LLM
        prompt = f"{system_prompt}\nVariant data: {variant_input}"

        # Generate a response using vLLM
        logging.info(f"Generating response for variant {identifier}")
        try:
            outputs = llm.generate([prompt], sampling_params)
            generated_text = outputs[0].outputs[0].text  # Extract the generated text (take the first output for simplicity)
            logging.info(f"Generated response for variant {identifier}: {generated_text[:50]}...")  # Log a snippet of the response
        except Exception as e:
            logging.error(f"Error generating response for variant {identifier}: {str(e)}")
            generated_text = "Error generating response"  # Handle errors gracefully

        # Write the result to the CSV
        writer.writerow([identifier, generated_text])
        logging.info(f"Variant {identifier} processed and written to output file")

logging.info("Processing complete.")

