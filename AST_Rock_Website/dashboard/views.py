import os
import pickle
import re
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
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
import django_rq
from . import version

# Add the src directory to the sys.path
src_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "CoreEngine", "src")
)
sys.path.insert(0, src_dir)

# Core Engine Classes.
import CoreEngine
from CoreEngine.src import Issue
from CoreEngine.src.classifier import (
    get_open_issues,
    get_open_issues_without_token,
)
from CoreEngine.src.database_manager import DatabaseManager
from CoreEngine.src.external import External_Model_Interface


@login_required
def your_repositories(request):
    user = request.user
    print(user)

    try:
        # Get the social token for GitHub
        social_account = SocialAccount.objects.get(user=user, provider="github")
        token = SocialToken.objects.get(account=social_account).token
        print(f"Token: {token}")

        # GitHub API endpoint to fetch repositories
        url = "https://api.github.com/user/repos"
        headers = {"Authorization": f"token {token}"}
        response = requests.get(url, headers=headers)
        repositories = response.json()

        repo_names = [repo["name"] for repo in repositories]
        print(f"Repositories: {repo_names}")

        # Store data in the session
        request.session["github_token"] = token
        request.session["username"] = user.username
        request.session["repositories"] = repo_names
        print(request.session.get("github_token"))
        print(request.session.get("username"))
        print(request.session.get("repositories"))
        # Send both username and repo_names to the template
        return render(
            request,
            "your_repositories.html",
            {
                "username": request.session.get("username"),
                "repositories": request.session.get("repositories"),
            },
        )

    except SocialToken.DoesNotExist:
        print("GitHub token not found")
        return render(
            request, "your_repositories.html", {"error": "GitHub token not found"}
        )


@login_required
def repo_detail(request, repo_name):

    token = request.session.get("github_token")
    username = request.session.get("username")

    # Call to get_open_issues function
    issues = get_open_issues(username, repo_name, token)
    print("open issues: ", issues)

    openai_key = os.getenv(
        "OPENAI_API_KEY"
    )  # Ensure you store OpenAI API key in session or settings
    # Adapt issue data for display if necessary

    db = DatabaseManager(
        dbfile="CoreEngine/output/main.db",
        cachefile="CoreEngine/output/ai_result_backup.db",
        label_file="CoreEngine/data/subdomain_labels.json",
    )
    external_rf = External_Model_Interface(
        openai_key,
        db,
        "CoreEngine/output/rf_model.pkl",
        "CoreEngine/data/domain_labels.json",
        "CoreEngine/data/subdomain_labels.json",
        None,
        None,
    )

    external_gpt = External_Model_Interface(
        openai_key,
        db,
        "CoreEngine/output/gpt_model.pkl",
        "CoreEngine/data/domain_labels.json",
        "CoreEngine/data/subdomain_labels.json",
        None,
        None,
    )

    # Prepare to collect responses for each issue
    responses_rf = []
    responses_gpt = []

    # Iterate over each Issue object and predict using the external model interface
    for issue in issues:
        response = external_rf.predict_issue(issue)
        responses_rf.append(response)
        response = external_gpt.predict_issue(issue)
        responses_gpt.append(response)

    print(responses_rf)
    print(responses_gpt)
    # Zip lists together for easier template use
    issues_responses = zip(issues, responses_rf, responses_gpt)
    db.close()

    issues_responses_list = list(zip(issues, responses_rf, responses_gpt))
    if not issues_responses_list:
        print("There is no issues")
        # If there are no issues, render a page with a specific message
        return render(
            request,
            "repo_detail.html",
            {
                "repo_name": repo_name,
                "message": "No Open Issues Found in this Repository",
            },
        )

    print("There is some issues")
    return render(
        request,
        "repo_detail.html",
        {
            "repo_name": repo_name,
            "issues_responses": issues_responses,
        },
    )


def index(request):
    return render(request, "index.html")


@login_required
def your_dashboard(request):
    return render(request, "your_dashboard.html")


@login_required
def repositories_by_link(request):
    token = request.session.get("github_token")
    if request.method == "POST":
        github_link = request.POST.get("github_link")
        match = re.search(r"github\.com/(.+?)/(.+?)(\.git|$)", github_link)
        if match:
            username, repo_name = match.groups()[:2]
            openai_key = os.getenv("OPENAI_API_KEY")
            # Trigger the RQ job
            queue = django_rq.get_queue("default")
            job = queue.enqueue(
                process_repository_issues, username, repo_name, openai_key
            )
            job_count = queue.count
            print(f"There are {job_count} jobs in the queue.")
            request.session["job_id_" + repo_name] = job.id
            print("Asynchronous Task is in queue")
            # Immediately redirect to a loading page
            return render(
                request,
                "splash_screen.html",
                {"repo_name": repo_name, "username": username},
            )
    return render(request, "repositories_by_link.html")


@login_required
def task_status(request, repo_name):
    queue = django_rq.get_queue("default")
    # Assuming you have stored the job ID in the session or a similar retrievable location
    job_id = request.session.get("job_id_" + repo_name, "")
    job = queue.fetch_job(job_id)
    if job is None or job.is_finished:
        return JsonResponse({"complete": True})
    else:
        return JsonResponse({"complete": False})


@login_required
def render_issues_results(request, username, repo_name):
    print("Rendering Issue Results")
    cache_key = f"{username}_{repo_name}_issues_responses"

    issues_responses = cache.get(cache_key)
    if issues_responses:
        issues_responses_list = list(issues_responses)  # Ensure it's a list
    else:
        issues_responses_list = []

    if not issues_responses_list:
        return render(
            request,
            "repo_detail.html",
            {
                "repo_name": repo_name,
                "message": "No Open Issues Found in this Repository",
            },
        )

    return render(
        request,
        "repo_detail.html",
        {
            "repo_name": repo_name,
            "issues_responses": issues_responses_list,  # Pass as list
        },
    )


def home(request):
    return render(request, "index.html")


def get_CoreEngine_version(request):
    return JsonResponse(
        {
            "CoreEngine_Version": CoreEngine.__version__,
            "Web_Version": version.WEBSITE_VERSION,
        }
    )


@login_required
def splash_screen(request, repo_name, username):
    return render(
        request, "splash_screen.html", {"repo_name": repo_name, "username": username}
    )
