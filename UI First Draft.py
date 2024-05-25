import os
import requests
from flask import Flask, render_template, request

app = Flask(__name__)

# Replace with your actual GitHub token
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')

def get_repo_issues(repo_owner, repo_name):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch issues: {response.status_code}")
        return None

def get_repo_pull_requests(repo_owner, repo_name):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch pull requests: {response.status_code}")
        return None

def get_issue(repo_owner, repo_name, issue_number):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues/{issue_number}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch issue: {response.status_code}")
        return None

def get_predictions(issue_number):
    # Placeholder function - replace with actual prediction fetching logic
    return {
        "required_skills": ["Python", "API Integration"],
        "estimated_time": "2 hours"
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/issue', methods=['GET'])
def display_bug_info():
    repo_owner = request.args.get('repo_owner')
    repo_name = request.args.get('repo_name')
    issue_number = int(request.args.get('issue_number'))

    issue = get_issue(repo_owner, repo_name, issue_number)
    pull_requests = get_repo_pull_requests(repo_owner, repo_name)
    predictions = get_predictions(issue_number)

    issue_data = {}
    pr_data = {}
    
    if issue:
        issue_data = {
            "title": issue['title'],
            "body": issue['body'],
            "predictions": predictions
        }
    
    if pull_requests:
        for pr in pull_requests:
            if pr['issue_url'].split('/')[-1] == str(issue_number):
                pr_data = {
                    "title": pr['title'],
                    "body": pr['body']
                }
                break
    
    return render_template('issue.html', issue=issue_data, pr=pr_data)

if __name__ == '__main__':
    app.run(debug=True)


