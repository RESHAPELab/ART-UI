from django.shortcuts import render
from django.contrib.auth.decorators import login_required
import requests
from allauth.socialaccount.models import SocialToken

def index(request):
    return render(request, 'index.html')

@login_required
def dashboard(request):
    token = SocialToken.objects.get(account__user=request.user, account__provider='github')
    headers = {'Authorization': f'token {token.token}'}

    repo_owner = 'your_repo_owner'
    repo_name = 'your_repo_name'

    issues_url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/issues'
    prs_url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/pulls'

    issues_response = requests.get(issues_url, headers=headers)
    prs_response = requests.get(prs_url, headers=headers)

    issues = issues_response.json() if issues_response.status_code == 200 else []
    prs = prs_response.json() if prs_response.status_code == 200 else []

    context = {
        'issues': issues,
        'pull_requests': prs,
    }
    return render(request, 'dashboard.html', context)
