import pandas as pd
import sqlite3
from openai import OpenAI
import emoji
import re
import string
import json
import concurrent.futures
import requests
import numpy as np
import argparse  # Import argparse module


def db_to_df(db_path):
    # Path to your SQLite database
    full_data = []
    none_count = 0
    curr_pr = 1
    # Iterate through PR numbers until none are found
    while none_count < 100:
        curr_entry = get_pr_data(db_path, pr=curr_pr)
        if curr_entry is not None:
            full_data.append(curr_entry)
            none_count = 0
            print('issue number', str(curr_pr), 'processed')
        curr_pr += 1
        none_count += 1

    print('processed', str(len(full_data)), 'entries')

    df = pd.DataFrame(data=full_data, columns=get_column_names(db_path))
    return df


def get_pr_data(db_path, pr):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = f'SELECT * FROM outputTable WHERE "PR #" = {str(pr)}'
    cursor.execute(query)

    query_data = cursor.fetchall()

    if len(query_data) != 0:
        first_half = list(query_data[0][:15])
        second_half = list(query_data[0][15:])
        for i in range(len(query_data)):
            if i != 0:
                second_half = np.add(second_half, query_data[i][15:])
                second_half = second_half.tolist()

        data_entry = first_half + second_half
        return data_entry


def get_column_names(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info(outputTable)")
    column_names = [row[1] for row in cursor.fetchall()]
    conn.close()
    return column_names


def clean_text(text):
    cleaned_count = 0
    original_count = 0
    if not isinstance(text, str):
        original_count += 1
        return text

    # Remove double quotation marks
    text = text.replace('"', '')

    # Remove text starting with "DevTools" and ending with "(automated)"
    text = re.sub(r'DevTools.*?\(automated\)', '', text)

    # Lowercasing should be one of the first steps to ensure uniformity
    text = text.lower()

    # Remove emojis
    text = emoji.demojize(text)

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove special characters and punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)

    # Remove '#' characters
    text = text.replace("#", "")

    # Remove consecutive whitespaces and replace with a single space
    text = re.sub(r'\s+', ' ', text)

    # Split the text into words
    words = text.split()

    # Remove words that are over 20 characters
    words = [word for word in words if len(word) <= 20]

    # Join the remaining words back into cleaned text
    cleaned_text = ' '.join(words)

    cleaned_count += 1
    return cleaned_text


def sort_dict_by_values(d, reverse=True):
    # Sort the dictionary by its values
    sorted_dict = dict(sorted(d.items(), key=lambda item: item[1], reverse=reverse))
    return sorted_dict


def get_top_domains(n, d, df):
    # Get top n domains, drop the rest
    counter = 1
    columns_to_drop = []
    for key, value in d.items():
        if counter > n:
            columns_to_drop.append(key)
        counter += 1
    df = df.drop(columns=columns_to_drop)
    return df


def filter_domains(df):
    domains = df.columns[15:]
    columns_to_drop = []
    occurrence_dictionary = {}
    for domain in domains:
        # get occurrence rate of each domain
        column_values = df[domain].tolist()
        occurrence = column_values.count(1)
        lower_bound = int(len(df) * 0.40)
        upper_bound = int(len(df) * 0.80)

        # drop those that are outside of the bounds
        if occurrence < lower_bound or occurrence > upper_bound:
            columns_to_drop.append(domain)
        else:
            occurrence_dictionary[domain] = occurrence

    df = df.drop(columns=columns_to_drop)

    # sort occurence dictionary to determine top domains and return top 15
    occurrence_dictionary = sort_dict_by_values(occurrence_dictionary)
    num_of_domains = 15
    return get_top_domains(num_of_domains, occurrence_dictionary, df)


def generate_system_message(domain_dictionary, df):
    formatted_domains = {}
    assistant_message = {}

    # reformat domains to increase clarity for gpt model and create dictionary with only domains/subdomains (to serve as expected gpt output)
    for key, value in domain_dictionary.items():
        if key in df.columns:
            formatted_domains[key] = 'Domain'
            assistant_message[key] = 0
        # iterate through each subdomain in list and add to dictionary
        for i in range(len(value)):
            subdomain, description = list(value[i].items())[0]
            if subdomain in df.columns:
                formatted_domains[subdomain] = description
                assistant_message[subdomain] = 0

    system_message = str(formatted_domains)

    return system_message, assistant_message


def generate_gpt_messages(system_message, gpt_output, df):
    # Open the file in write mode
    with open('gpt_messages.jsonl', 'w', encoding='utf-8') as f:
        assistant_message = gpt_output
        # Iterate over the rows in the DataFrame
        for index, row in df.iterrows():
            # Create the user message by formatting the prompt with the title and body
            user_message = (
                f"Classify a GitHub issue by indicating whether each domain and subdomain is relevant to the issue based on its title: [{row['issue text']}] "
                f"and body: [{row['issue description']}]. Ensure that every domain/subdomain is accounted for, and its relevance is indicated with a 1 (relevant) or a 0 (not relevant)."
            )

            # logic to update assistant message with values in df
            for column in df.columns:
                if column in gpt_output:
                    if row[column] > 0:
                        assistant_message[column] = 1
                    else:
                        assistant_message[column] = 0

            # Construct the conversation object
            conversation_object = {
                "messages": [
                    {"role": "system",
                     "content": "Refer to these domains and subdomains when classifying " + system_message},
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": str(assistant_message)}
                ]
            }

            # Write the conversation object to one line in the file
            f.write(json.dumps(conversation_object, ensure_ascii=False) + '\n')


def fine_tune_gpt(api_key):
    client = OpenAI(api_key=api_key)
    # uploading a training file
    domain_classifier_training_file = client.files.create(
        file=open("gpt_messages.jsonl", "rb"),
        purpose="fine-tune"
    )
    print("Beginning fine tuning process")
    # creating a fine-tuned model
    ft_job_dc = client.fine_tuning.jobs.create(
        training_file=domain_classifier_training_file.id,
        model="gpt-3.5-turbo",
        suffix="issue_classifier"
    )

    while True:
        response = client.fine_tuning.jobs.retrieve(ft_job_dc.id)
        if response.status == 'succeeded':
            # Retrieving the state of a fine-tune
            issue_classifier = client.fine_tuning.jobs.retrieve(ft_job_dc.id).fine_tuned_model
            return issue_classifier
        if response.status == 'failed':
            print('process failed')
            return


def get_open_issues(owner, repo, access_token):
    data = []
    # GitHub API URL for fetching issues
    url = f'https://api.github.com/repos/{owner}/{repo}/issues'

    # Headers for authentication
    headers = {
        'Authorization': f'token {access_token}',
        'Accept': 'application/vnd.github.v3+json'
    }

    # Parameters to fetch only open issues
    params = {
        'state': 'open',
        'per_page': 100,  # Number of issues per page (maximum is 100)
        'page': 1  # Page number to start fetching from
    }

    issues = []
    while True:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            print(f'Error: {response.status_code}')
            break

        issues_page = response.json()
        if not issues_page:
            break

        issues.extend(issues_page)
        params['page'] += 1

    # Add extracted issues to dataframe
    counter = 0
    for issue in issues:
        if counter < 5:
            data.append([issue['number'], issue['title'], issue['body']])
        counter += 1
    print(f"Total issues fetched: {len(issues)}")
    df = pd.DataFrame(columns=['Issue #', 'Title', 'Body'], data=data)
    return df


def query_gpt(user_message, issue_classifier, openai_key, max_retries=5):
    client = OpenAI(api_key=openai_key)
    attempt = 0

    # attempt to query model
    while attempt < max_retries:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                client.chat.completions.create,
                model=issue_classifier,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )
            try:
                response = future.result()
                return response.choices[0].message.content
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} - An error occurred: {e}")
            finally:
                attempt += 1

    print("Failed to get a response after several retries.")
    return None


def get_gpt_responses(open_issue_df, issue_classifier, domains_string, openai_key):
    responses = {}
    for index, row in open_issue_df.iterrows():
        # create user and system messages
        user_message = (
            f"Classify a GitHub issue by indicating up to THREE domains and subdomains that are relevant to the issue based on its title: [{row['Title']}] "
            f"and body: [{row['Body']}]. Prioritize positive precision by marking an issue with a 1 only when VERY CERTAIN a domain is relevant to the issue text. Ensure that you only provide three domains and refer to ONLY THESE domains and subdomains when classifying: {domains_string}."
            f"\n\nImportant: only provide the name of the domains in list format."
        )

        # query fine tuned model
        response = query_gpt(user_message, issue_classifier, openai_key)
        responses[row['Issue #']] = response
        print("Issue #" + str(row['Issue #']) + " complete")

    with open('GPT_Responses.json', 'w') as json_file:
        json.dump(responses, json_file, indent=4)
    return responses


def responses_to_csv(gpt_responses):
    columns = ['Issue #', 'Predictions']

    prediction_data = []

    for key, value in gpt_responses.items():
        prediction_data.append([key, value])

    gpt_predictions = pd.DataFrame(columns=columns, data=prediction_data)

    return gpt_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument('--config', type=str, required=True, help='Path to the conf.json file')
    parser.add_argument('--domains', type=str, required=True, help='Path to the Domains.json file')
    parser.add_argument('--db', type=str, required=True, help='Path to the SQLite database file')
    return parser.parse_args()


def main():
    args = parse_args()  # Parse the command-line arguments

    # Load JSON from file
    with open(args.config, 'r') as f:
        repo_data = json.load(f)

    with open(args.domains, 'r') as file:
        domain_dictionary = json.load(file)

    # Get repo data
    github_key = repo_data['github_token']
    owner = repo_data['repo_owner']
    repo = repo_data['repo_name']
    openAI_key = repo_data['openAI_key']

    # Load data and preprocess
    print('Loading data from database')
    db_path = args.db  # Use the database path from the arguments
    df = db_to_df(db_path)
    columns_to_convert = df.columns[15:]
    df[columns_to_convert] = df[columns_to_convert].applymap(lambda x: 1 if x > 0 else 0)
    df['issue text'] = df['issue text'].apply(clean_text)
    df['issue description'] = df['issue description'].apply(clean_text)
    df = filter_domains(df)

    # Generate fine tuning file
    system_message, assistant_message = generate_system_message(domain_dictionary, df)
    generate_gpt_messages(system_message, assistant_message, df)

    # Fine tune GPT Model
    llm_classifier = fine_tune_gpt(openAI_key)

    # Extract open issues
    print('Extracting open issues...')
    open_issue_data = get_open_issues(owner, repo, github_key)
    open_issue_data['Title'] = open_issue_data['Title'].apply(clean_text)
    open_issue_data['Body'] = open_issue_data['Body'].apply(clean_text)

    # Classify open issues
    print('Classifying open Issues...')
    gpt_responses = get_gpt_responses(open_issue_data, llm_classifier, system_message, openAI_key)
    predictions_df = responses_to_csv(gpt_responses)
    predictions_df.to_csv('llm_prediction_data.csv', index=False)

    print('Open issue predictions written to csv')

if __name__ == "__main__":
    main()