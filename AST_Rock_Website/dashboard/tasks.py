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

        
        # Iterate over each Issue object and predict using the external model interface
    for issue in issues:
        
        response = external.predict_issue(issue)
        responses_rf.append(response)
        response = external2.predict_issue(issue)
        responses_gpt.append(response)
            
    print(responses_rf)
    print(responses_gpt)
        # Zip lists together for easier template use
    issues_responses = zip(issues, responses_rf, responses_gpt)

    cache_key = f"{username}_{repo_name}_issues_responses"
    cache.set(cache_key, issues_responses, timeout=3600)  # Cache for 1 hour
    return 