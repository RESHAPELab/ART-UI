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

        # Send both username and repo_names to the template
        return render(request, 'your_repositories.html', {
            'username': user.username,
            'repositories': repo_names
        })
        
    except SocialToken.DoesNotExist:
        print("GitHub token not found")
        return render(request, 'your_repositories.html', {'error': 'GitHub token not found'})

@login_required
def repo_detail(request, repo_name):
    
    return render(request, 'repo_detail.html', {'repo_name': repo_name})
    
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

