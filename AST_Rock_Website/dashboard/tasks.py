import os
import sys
from django.core.cache import cache

# Add the src directory to the sys.path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CoreEngine', 'src'))
sys.path.insert(0, src_dir)


from issue_class import Issue
from open_issue_classification import get_open_issues, get_open_issues_without_token
from database_manager import DatabaseManager
from external import External_Model_Interface


def process_repository_issues(username, repo_name, openai_key):
    # Call to get_open_issues function 
    issues = get_open_issues_without_token(username, repo_name)
    print("open issues: ", issues)
        

    db = DatabaseManager()
    external = External_Model_Interface(
        openai_key, db, "rf_model.pkl", "domain_labels.json", "subdomain_labels.json", None
    )

    db.close()

    db2 = DatabaseManager()
    external2 = External_Model_Interface(
        openai_key, db2, "gpt_model.pkl", "domain_labels.json", "subdomain_labels.json", None
    )

        
        # Prepare to collect responses for each issue
    responses_rf = []
    responses_gpt= []

    max_issues = 20

    for issue in issues[:max_issues]:
        try:
            response_rf = external.predict_issue(issue)
            responses_rf.append(response_rf)

            response_gpt = external2.predict_issue(issue)
            responses_gpt.append(response_gpt)
        except Exception as e:
            print(f"Error processing issue: {issue}. Error: {str(e)}")
            # Optionally, you could append a None or a specific error message instead of a response
            responses_rf.append(None)  # Indicates a failed response
            responses_gpt.append(None)  # Indicates a failed response
        
            
    print(responses_rf)
    print(responses_gpt)
        # Zip lists together for easier template use
    issues_responses = list(zip(issues, responses_rf, responses_gpt))

    cache_key = f"{username}_{repo_name}_issues_responses"
    cache.set(cache_key, issues_responses, timeout=3600)  # Cache for 1 hour

    # Retrieve from cache to test
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        print("Data was successfully cached.")
    else:
        print("Failed to cache data.")
    return None