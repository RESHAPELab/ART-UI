import pandas as pd
import sqlite3
from openai import OpenAI
import re
import string
import json
import concurrent.futures
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
import random
import argparse  # Import argparse module
import dashboard.models


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
    assistant_message = gpt_output  # Assuming gpt_output is already a dictionary that can be modified directly

    # Iterate over the rows in the DataFrame
    for index, row in df.iterrows():
        # Create the user message by formatting the prompt with the title and body
        user_message = (
            f"Classify a GitHub issue by indicating whether each domain and subdomain is relevant to the issue based on its title: [{row['issue text']}] "
            f"and body: [{row['issue description']}]. Ensure that every domain/subdomain is accounted for, and its relevance is indicated with a 1 (relevant) or a 0 (not relevant)."
        )

        # Update assistant message with values in df
        for column in df.columns:
            if column in assistant_message:
                assistant_message[column] = 1 if row[column] > 0 else 0

        # Construct the conversation object
        conversation_object = {
            "messages": [
                {"role": "system", "content": "Refer to these domains and subdomains when classifying " + system_message},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": str(assistant_message)}
            ]
        }

        # Save each conversation object to the database
        message = dashboard.models.GPTMessage(
            system_content=conversation_object["messages"][0]["content"],
            user_content=conversation_object["messages"][1]["content"],
            assistant_content=conversation_object["messages"][2]["content"]
        )
        print("message: ", message)
        message.save()


def fine_tune_gpt(api_key):
    client = OpenAI(api_key=api_key)
    # Fetch messages from the database
    messages = dashboard.models.GPTMessage.objects.all()
    if not messages.exists():
        print("No messages found in the database.")
        return
    # Create a JSONL file for training
    with open('gpt_messages.jsonl', 'w', encoding='utf-8') as f:
        for message in messages:
            conversation_object = {
                "messages": [
                    {"role": "system", "content": message.system_content},
                    {"role": "user", "content": message.user_content},
                    {"role": "assistant", "content": message.assistant_content}
                ]
            }
            f.write(json.dumps(conversation_object, ensure_ascii=False) + '\n')
        if f.tell() == 0:  # f.tell() returns the current position of the file cursor
                print("File is still empty after attempting to write.")
    # Uploading a training file
    domain_classifier_training_file = client.files.create(
        file=open("gpt_messages.jsonl", "rb"),
        purpose="fine-tune"
    )
    
    print("Beginning fine tuning process")
    # Creating a fine-tuned model
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
    for issue in issues:
        data.append([issue['number'], issue['title'], issue['body']])
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


# Random Forest Functions
def extract_text_features(data):
    X_text = data[['issue text', 'issue description']]
    X_text['combined_text'] = X_text['issue text'] + " " + X_text['issue description']
    tfidf = TfidfVectorizer(max_features=1000)
    X_text_features = tfidf.fit_transform(X_text['combined_text']).toarray()
    return X_text_features, tfidf


def transform_labels(data):
    y = data.drop(columns=['issue text', 'issue description'])
    y = y.apply(lambda x: x.index[x == 1].tolist(), axis=1)
    mlb = MultiLabelBinarizer()
    y_transformed = mlb.fit_transform(y)
    y_df = pd.DataFrame(y_transformed, columns=mlb.classes_)
    return y_df, mlb


def create_combined_features(x_text_features):
    X_combined = pd.DataFrame(x_text_features, columns=list(range(x_text_features.shape[1])))
    return X_combined


def perform_mlsmote(X, y, n_sample=500):
    def nearest_neighbour(X):
        nbs = NearestNeighbors(n_neighbors=3, metric='euclidean', algorithm='kd_tree').fit(X)
        _, indices = nbs.kneighbors(X)
        return indices

    indices2 = nearest_neighbour(X.values)
    n = len(indices2)
    new_X = np.zeros((n_sample, X.shape[1]))
    target = np.zeros((n_sample, y.shape[1]))
    for i in range(n_sample):
        reference = random.randint(0, n-1)
        neighbour = random.choice(indices2[reference, 1:])
        all_point = indices2[reference]
        nn_df = y[y.index.isin(all_point)]
        ser = nn_df.sum(axis=0, skipna=True)
        target[i] = np.array([1 if val > 2 else 0 for val in ser])
        ratio = random.random()
        gap = X.loc[reference, :] - X.loc[neighbour, :]
        new_X[i] = np.array(X.loc[reference, :] + ratio * gap)
    new_X = pd.DataFrame(new_X, columns=X.columns)
    target = pd.DataFrame(target, columns=y.columns)
    new_X = pd.concat([X, new_X], axis=0)
    target = pd.concat([y, target], axis=0)
    return new_X, target


def train_random_forest(x_train, y_train):
    clf = RandomForestClassifier(random_state=42)
    clf.fit(x_train, y_train)
    return clf


def clean_text(vectorizer, df):
    # combine text columns
    text = df[['Title', 'Body']]
    text['combined_text'] = text['Title'] + " " + text['Body']

    # vectorize text columns
    vectorized_text = vectorizer.transform(text['combined_text']).toarray()
    return vectorized_text


def predict_open_issues(open_issue_df, model, data, y_df):

    # predict for open issues
    predicted_values = model.predict(data)

    # get used domains from csv and issue numbers from df
    columns = y_df.columns
    columns = columns.insert(0, 'Issue #')
    issue_numbers = open_issue_df['Issue #'].values

    # link issue number with predictions and add to data
    prediction_data = []
    for i in range(len(predicted_values)):
        curr_prediction = predicted_values[i].tolist()
        curr_issue = [issue_numbers[i]]
        prediction_data.append(curr_issue + curr_prediction)

    # convert data to df
    prediction_df = pd.DataFrame(columns=columns, data=prediction_data)
    return prediction_df


def parse_args():
    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument('--config', type=str, required=True, help='Path to the conf.json file')
    parser.add_argument('--domains', type=str, required=True, help='Path to the Domains.json file')
    parser.add_argument('--db', type=str, required=True, help='Path to the SQLite database file')
    parser.add_argument('--method', type=str,required=True, help='Prediction method')
    return parser.parse_args()


def run_llm(args):

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


def run_rf(args):
    # Load JSON from file
    with open(args.config, 'r') as f:
        repo_data = json.load(f)

    # Get repo data
    github_key = repo_data['github_token']
    owner = repo_data['repo_owner']
    repo = repo_data['repo_name']

    # Load data and preprocess
    print('Loading data from database')
    db_path = args.db  # Use the database path from the arguments
    df = db_to_df(db_path)
    columns_to_convert = df.columns[15:]
    df[columns_to_convert] = df[columns_to_convert].applymap(lambda x: 1 if x > 0 else 0)
    df = df.drop(
        columns=['PR #', 'Pull Request', 'created_at', 'closed_at', 'userlogin', 'author_name', 'most_recent_commit',
                 'filename', 'file_commit', 'api', 'function_name', 'api_domain', 'subdomain'])
    df = df.dropna()

    print('processing data...')
    X_text_features, tfidf = extract_text_features(df)

    # Transform labels
    y_df, mlb = transform_labels(df)

    # Combine features
    X_combined = create_combined_features(X_text_features)

    # Perform MLSMOTE to augment the data
    print('balancing classes...')
    X_augmented, y_augmented = perform_mlsmote(X_combined, y_df, n_sample=500)

    print('training RF model...')
    X_combined = pd.concat([X_combined, X_augmented], axis=0)
    y_combined = pd.concat([y_df, y_augmented], axis=0)

    # Train
    clf = train_random_forest(X_combined, y_combined)

    print('collecting open issues...')
    open_issue_data = get_open_issues(owner, repo, github_key)  # get open issue number and text

    vectorized_text = clean_text(tfidf, open_issue_data)  # vectorize issue text

    print('classifying open issues...')
    predictions_df = predict_open_issues(open_issue_data, clf, vectorized_text, y_df)  # predict for open issues

    predictions_df.to_csv('open_issue_predictions.csv', index=False)  # write predictions to csv




def main():
    args = parse_args()  # Parse the command-line arguments

    if args.method == 'LLM':
        run_llm(args)

    elif args.method == 'RF':
        run_rf(args)

if __name__ == "__main__":
    main()
