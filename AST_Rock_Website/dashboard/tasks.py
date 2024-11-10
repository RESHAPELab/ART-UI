import os
import sys
from django.core.cache import cache
import tqdm


# Core Engine Classes.
from CoreEngine.src import Issue
from CoreEngine.src.classifier import (
    get_open_issues,
    get_open_issues_without_token,
)
from CoreEngine.src.database_manager import DatabaseManager
from CoreEngine.src.external import External_Model_Interface


def process_repository_issues(
    username, repo_name, openai_key, quantity, max_domains, model_select
):
    # Call to get_open_issues function
    issues = get_open_issues_without_token(username, repo_name, quantity)
    # print("open issues: ", issues)

    db = DatabaseManager(
        dbfile="CoreEngine/output/main.db",
        cachefile="CoreEngine/output/ai_result_backup.db",
        label_file="CoreEngine/data/subdomain_labels.json",
    )
    if model_select == "rf":
        external_model = External_Model_Interface(
            openai_key,
            db,
            "rf_model.pkl",
            "CoreEngine/data/domain_labels.json",
            "CoreEngine/data/subdomain_labels.json",
            "CoreEngine/data/formatted_domain_labels.json",
            repo_name,
            "CoreEngine/output/response_cache/",
        )
    if model_select == "gpt":
        external_model = External_Model_Interface(
            openai_key,
            db,
            "gpt_model.pkl",
            "CoreEngine/data/domain_labels.json",
            "CoreEngine/data/subdomain_labels.json",
            "CoreEngine/data/formatted_domain_labels.json",
            repo_name,
            "CoreEngine/output/response_cache/",
        )

    if model_select == "gpt-combined":
        external_model = External_Model_Interface(
            openai_key,
            db,
            "gpt_combined_model.pkl",
            "CoreEngine/data/domain_labels.json",
            "CoreEngine/data/subdomain_labels.json",
            "CoreEngine/data/formatted_domain_labels.json",
            repo_name,
            "CoreEngine/output/response_cache/",
        )

    # Prepare to collect responses for each issue
    responses = []

    max_issues = quantity

    for issue in tqdm.tqdm(issues[:max_issues]):
        print(f"Processing {issue.number}")
        try:
            response_in = external_model.predict_issue(issue, max_domains)
            responses.append(response_in)
        except Exception as e:
            print(f"Error processing issue: {issue}. Error: {str(e)}")
            # Optionally, you could append a None or a specific error message instead of a response
            responses.append(None)

    db.close()
    # Zip lists together for easier template use
    issues_responses = (issues, responses, model_select)

    cache_key = f"{username}_{repo_name}_issues_responses_{model_select}"
    cache.set(cache_key, issues_responses, timeout=3600)  # Cache for 1 hour

    # Retrieve from cache to test
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        print("Data was successfully cached.")
    else:
        print("Failed to cache data.")
    return None
