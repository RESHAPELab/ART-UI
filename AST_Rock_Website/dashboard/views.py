import os
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
import requests
from allauth.socialaccount.models import SocialToken
# views.py in your Django app
from allauth.socialaccount.providers.oauth2.views import OAuth2Adapter
from allauth.socialaccount.providers.github.views import GitHubOAuth2Adapter
from allauth.socialaccount.views import SignupView
from allauth.socialaccount.models import SocialAccount
from django.contrib.auth.decorators import login_required
from allauth.socialaccount.models import SocialAccount, SocialToken
import requests
from external_file import External_Model_Interface
from open_issue import get_open_issues
from database_manager_file import DatabaseManager
import sys
import os
import json

# Add the src directory to the sys.path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CoreEngine', 'src'))
sys.path.insert(0, src_dir)


from issue_class import Issue




@login_required
def your_repositories(request):
    user = request.user
    print(user)
    
    try:
        # Get the social token for GitHub
        social_account = SocialAccount.objects.get(user=user, provider='github')
        token = SocialToken.objects.get(account=social_account).token
        print(f"Token: {token}")

        # GitHub API endpoint to fetch repositories
        url = 'https://api.github.com/user/repos'
        headers = {'Authorization': f'token {token}'}
        response = requests.get(url, headers=headers)
        repositories = response.json()
        
        repo_names = [repo['name'] for repo in repositories]
        print(f"Repositories: {repo_names}")

        # Store data in the session
        request.session['github_token'] = token
        request.session['username'] = user.username
        request.session['repositories'] = repo_names
        print(request.session.get('github_token'))
        print(request.session.get('username'))
        print(request.session.get('repositories'))
        # Send both username and repo_names to the template
        return render(request, 'your_repositories.html', {
            'username': request.session.get('username'),
            'repositories': request.session.get('repositories')
        })
        
    except SocialToken.DoesNotExist:
        print("GitHub token not found")
        return render(request, 'your_repositories.html', {'error': 'GitHub token not found'})

@login_required
def repo_detail(request, repo_name):

    
    token = request.session.get('github_token')
    username = request.session.get('username')
    repositories = request.session.get('repositories', [])



    # Call to get_open_issues function 
    issues = get_open_issues(username, repo_name, token)
    print("open issues: ", issues)


    if issues is None:
        return render(request, 'repo_detail.html', {
            'responses': 'No issues found.'
        })
    else:
        openai_key = os.getenv('OPENAI_API_KEY')  # Ensure you store OpenAI API key in session or settings
    # Adapt issue data for display if necessary
    
    db = DatabaseManager()
    external = External_Model_Interface(
        openai_key, db, "CoreEngine/output/rf_model.pkl", "CoreEngine/data/domain_labels.json", None
    )

    db.close()

    
    responses_rf = external.predict_issue(issues)
    
    # Zip lists together for easier template use
    issues_responses = zip(issues, responses_rf)
    
    return render(request, 'repo_detail.html', {
        'repo_name': repo_name,
        'issues_responses': issues_responses,
    })

    
def index(request):
    return render(request, 'index.html')

@login_required
def your_dashboard(request):
    return render(request, 'your_dashboard.html')


@login_required
def repositories_by_link(request):
    return render(request, 'repositories_by_link.html')

def home(request):
    return render(request, 'index.html')










