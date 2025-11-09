import os
import json
import docx2txt
import csv
import re
from natsort import natsorted
from dotenv import load_dotenv
from pathlib import Path
import openai

# Load environment variables
dotenv_path = Path('C:/Users/LAVie/Documents/MBA8583/MyKeys/.env')
load_dotenv(dotenv_path)

# Initialize Groq client
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# Define directories
input_dir = "C:/Users/LAVie/Documents/MBA8583/InputArticles/"
output_file = "C:/Users/LAVie/Documents/MBA8583/OutputArticles/resultsGPT.csv"

# Specify the range of files to process (where to start and end in alphabetically sorted list of articles)
start = 1  # Start index (first article has index 0)
end = 10   # End index (exclusive)

# Placeholder for the function schema
function_schema = {}



def get_completion(text, model="gpt-4o-mini"):
    prompt = f"Extract insights from the following article:\n'''{text}'''"
    system_prompt = (
        "You are a helpful assistant that extracts insights from articles as JSON for a database. "
        "Use the 'extract_data' function to provide your output."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    try:
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            tools=[function_schema],  # Provide the function schema
            tool_choice="auto",       # Let the model decide when to use the function
            temperature=0
        )
    except Exception as e:
        print(f"Error during API call: {e}")
        return {}

    # Access the function call and arguments directly from the response
    message = response.choices[0].message

    if hasattr(message, 'tool_calls') and message.tool_calls:
        for tool_call in message.tool_calls:
            if tool_call.function.name == "extract_data":
                arguments = json.loads(tool_call.function.arguments)
                return arguments
    elif hasattr(message, 'content') and message.content:
        print("Assistant provided data in content instead of function call.")
        return {"summary": message.content.strip()}
    else:
        print("No function call or content provided.")
        return {}

def main():
    # Initialize data list and counter
    data = []
    counter = 0
    save_interval = 50  # Frequency of saving results to CSV

    # Get a list of all .docx files and sort them alphabetically
    docx_files = natsorted(
        [filename for filename in os.listdir(input_dir) if filename.lower().endswith(".docx")]
    )

    # Ensure the end index does not exceed the number of files
    end_index = min(end, len(docx_files))

    # Slice the list to get the subset of files
    subset_files = docx_files[start:end_index]

    # Loop through the subset of .docx files
    for filename in subset_files:
        # Skip files ending with "doclist.docx"
        if filename.endswith("doclist.docx"):
            continue
        # Skip files ending with (#).docx which indicates the article is a duplicate
        if re.search(r'\(\d+\)\.docx$', filename):
            continue

        file_path = os.path.join(input_dir, filename)

        if os.path.isfile(file_path):
            # Process the file if it exists
            text = docx2txt.process(file_path)

            # Call the LLM and perform further processing here
            try:
                gpt_result = get_completion(text)
            except Exception as e:
                print(f"Error in LLM completion: {e}")
                continue

            # Append the LLM results to the existing data
            data.append({"filename": filename, **gpt_result})
            counter += 1

            # Check if it's time to save to CSV
            if counter >= save_interval:
                save_to_csv(data)
                counter = 0  # Reset the counter
                data = []    # Clear the data list

    # Save any remaining data to CSV
    if data:
        save_to_csv(data)

def save_to_csv(data):
    # Define the headers for the CSV file
    headers = []

    # Append the data to a CSV file
    with open(output_file, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)

        # If the CSV file is empty, write the header
        if csv_file.tell() == 0:
            writer.writerow(headers)  # Write the header

        # Write the data to the CSV file
        for item in data:
            row = [item.get(header, '') for header in headers]
            writer.writerow(row)

if __name__ == "__main__":
    main()
