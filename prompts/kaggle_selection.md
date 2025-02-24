Write a Python script that enhances my existing dashboard to do the following:
Search Kaggle: Use the Kaggle API to search for datasets related to 'Dubai Property Prices' or 'UAE Real Estate' or 'Dubai Property Data'. Authenticate using a Kaggle API key (assume the user has kaggle.json set up in the default location ~/.kaggle/).
Identify and List Datasets: Retrieve a list of datasets matching the search terms. For each dataset, extract and display:
Dataset title
Dataset URL
File size
Last updated date
Number of files
Description (first 100 characters or so)
Filter by Data Quality Criteria: Apply the following filters to narrow down the datasets:
Must contain at least one CSV file
Must have been updated on or after January 1, 2020
Must have a file size between 1 MB and 1 GB (to ensure it’s substantial but manageable)
Must include keywords like 'property', 'real estate', 'price', or 'listing' in the title or description
User Interaction: Present the filtered list of datasets in a numbered menu. Allow the user to select a dataset by entering its number.
Download and Verify: Download the selected dataset as a ZIP file using the Kaggle API, extract it, and verify that it contains at least one CSV file. If successful, print the path to the extracted CSV(s). If not, notify the user and exit gracefully.
Dependencies: Use the following libraries: kaggle (for API access), datetime (for date filtering), os (for file handling), zipfile (for extraction), and re (for keyword matching). Include a comment explaining how to install the kaggle library (pip install kaggle).
Error Handling: Handle cases like no datasets found, API authentication errors, or invalid user input.
Output: Make the script verbose with print statements to guide the user through the process (e.g., 'Searching Kaggle...', 'Downloading dataset...', 'CSV found at path...').
The goal is to replicate a workflow where I can source a UAE real estate dataset (like the 'Dubai Property Data' dataset with ~10K listings from 2020-2023), download it as a CSV, and use it as a baseline for further analysis. Assume I’ll manually adjust the data later for recency (e.g., increasing prices by 20-30% for 2024 trends)."
Notes for Using This Prompt
Setup: Before running the script, ensure you’ve installed the Kaggle API (pip install kaggle) and set up your Kaggle API key. You can get the key from your Kaggle account settings and place it in ~/.kaggle/kaggle.json.
Customization: You can tweak the data quality criteria in the prompt (e.g., change the date range, file size limits, or keywords) to better fit your needs.
Output: The script will give you a list of datasets, let you pick one, and download it. You’ll then have a CSV ready for your baseline reader or further processing all visually by enhancing my existing dashboard.

Before crafting the script, take into consideration all the dependencies and ensure you’ve installed the Kaggle API (pip install kaggle) and provision for set up your Kaggle API key through the kaggle api setup dashboard. Ensure that the requirements.tx is updated dev.bat is enhanced to include the kaggle library and the script is optimized for cursor ai.