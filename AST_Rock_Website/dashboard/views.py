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
from open_issue_classification import get_open_issues, get_gpt_responses, fine_tune_gpt, generate_system_message
import json


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

    # Load the domains from the file
    with open('Domains.json', 'r') as file:
        domains_data = json.load(file)
        print("Domains: ", domains_data)
        

    # Call to get_open_issues function 
    issues = get_open_issues(username, repo_name, token)
    print("open issues: ", issues)
    domains_string = generate_system_message(domains_data, issues)


    if repo_name not in repositories:
        return render(request, 'repo_detail.html', {
            'error': 'Repository not found or access not authorized.'
        })

    if issues is None:
        return render(request, 'repo_detail.html', {
            'error': 'Failed to fetch issues or no issues found.'
        })
    else:
        openai_key = os.getenv('OPENAI_API_KEY')  # Ensure you store OpenAI API key in session or settings
        issue_classifier = fine_tune_gpt(openai_key)  # You need to define or get this model id somehow
        
        responses = get_gpt_responses(issues, issue_classifier, domains_string, openai_key)
    # Adapt issue data for display if necessary
        

    return render(request, 'repo_detail.html', {
        'repo_name': repo_name,
        'issues': issues,
        'responses' : responses
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










