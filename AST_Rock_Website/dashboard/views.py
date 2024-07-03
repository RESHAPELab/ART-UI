import os
import pickle
import re
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.urls import reverse
import pandas as pd
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
import sys
import os
import json
from .tasks import process_repository_issues
from django.core.cache import cache

# Add the src directory to the sys.path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CoreEngine', 'src'))
sys.path.insert(0, src_dir)


from issue_class import Issue
from open_issue_classification import get_open_issues, get_open_issues_without_token
from database_manager import DatabaseManager
from external import External_Model_Interface



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


    # Call to get_open_issues function 
    issues = get_open_issues(username, repo_name, token)
    print("open issues: ", issues)



    openai_key = os.getenv('OPENAI_API_KEY')  # Ensure you store OpenAI API key in session or settings
    # Adapt issue data for display if necessary
        

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

    issues_responses_list = list(zip(issues, responses_rf, responses_gpt))
    if not issues_responses_list:
        print("There is no issues")
        # If there are no issues, render a page with a specific message
        return render(request, 'repo_detail.html', {
            'repo_name': repo_name,
            'message': 'No Open Issues Found in this Repository'
        })

    print("There is some issues")
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
    token = request.session.get('github_token')
    if request.method == 'POST':
        github_link = request.POST.get('github_link')
        match = re.search(r"github\.com/(.+?)/(.+?)(\.git|$)", github_link)
        if match:
            username, repo_name = match.groups()[:2]
            openai_key = os.getenv('OPENAI_API_KEY')
            # Trigger the Celery task
            process_repository_issues.delay(username, repo_name, openai_key)
            # Immediately redirect to a loading page
            return HttpResponseRedirect(reverse('splash_screen', kwargs={'repo_name': repo_name}))
    return render(request, 'repositories_by_link.html')

@login_required
def render_issues_results(request, username, repo_name):
    cache_key = f"{username}_{repo_name}_issues_responses"
    issues_responses = cache.get(cache_key)

    if not issues_responses:
        return render(request, 'repo_detail.html', {
            'repo_name': repo_name,
            'message': 'No Open Issues Found in this Repository'
        })

    return render(request, 'repo_detail.html', {
        'repo_name': repo_name,
        'issues_responses': issues_responses
    })

def home(request):
    return render(request, 'index.html')

def splash_screen(request):
    return render(request, 'splash_screen.html')










