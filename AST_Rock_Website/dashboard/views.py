import os
import pickle
import re
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.shortcuts import render
from django.urls import reverse
import pandas as pd
import requests

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


def index(request):
    return render(request, "index.html")


def repositories_by_link(request):
    if request.method == "POST":
        github_link = request.POST.get("github_link")
        match = re.search(r"github\.com/(.+?)/(.+?)(\.git|$)", github_link)
        if match:

            quantity = request.POST.get("quantity")
            try:
                quantity = abs(int(quantity))
            except:
                quantity = 10

            domain_quantity = request.POST.get("domain_quantity")
            try:
                domain_quantity = abs(int(domain_quantity))
            except:
                domain_quantity = 3

            if quantity > 100:
                quantity = 100
            if domain_quantity > 10:
                domain_quantity = 10

            username, repo_name = match.groups()[:2]
            openai_key = os.getenv("OPENAI_API_KEY")
            # Trigger the RQ job
            queue = django_rq.get_queue("default")
            job = queue.enqueue(
                process_repository_issues,
                username,
                repo_name,
                openai_key,
                quantity,
                domain_quantity,
            )
            job_count = queue.count
            print(f"There are {job_count} jobs in the queue.")
            request.session["job_id_" + repo_name] = job.id
            request.session["job_id_" + repo_name + "_display"] = domain_quantity
            print("Asynchronous Task is in queue")
            # Immediately redirect to a loading page
            return render(
                request,
                "splash_screen.html",
                {"repo_name": repo_name, "username": username},
            )
    return render(request, "repositories_by_link.html")


def task_status(request, repo_name):
    queue = django_rq.get_queue("default")
    # Assuming you have stored the job ID in the session or a similar retrievable location
    job_id = request.session.get("job_id_" + repo_name, "")
    job = queue.fetch_job(job_id)
    if job is None or job.is_finished:
        return JsonResponse({"complete": True})
    else:
        return JsonResponse({"complete": False})


def render_issues_results(request, username, repo_name):
    print("Rendering Issue Results")
    cache_key = f"{username}_{repo_name}_issues_responses"

    domain_q = request.session.get("job_id_" + repo_name + "_display", 3)

    issues_responses = cache.get(cache_key)

    if not issues_responses:
        return render(
            request,
            "repo_detail.html",
            {
                "repo_name": repo_name,
                "message": "No Open Issues Found in this Repository",
            },
        )

    # (issues, responses_rf, responses_gpt, responses_gpt_combo)

    issue_data, response_rf_data, response_gpt_data, response_gpt_combo_data = (
        issues_responses
    )

    new_response_rf_data = []
    for resp in response_rf_data:
        if resp is None:
            continue
        if len(resp) > domain_q:
            new_response_rf_data.append(resp[0:domain_q])
            continue
        new_response_rf_data.append(resp)

    new_response_gpt_data = []
    for resp in response_gpt_data:
        if resp is None:
            continue
        if len(resp) > domain_q:
            new_response_gpt_data.append(resp[0:domain_q])
            continue
        new_response_gpt_data.append(resp)

    new_response_gpt_combo_data = []
    for resp in response_gpt_combo_data:
        if resp is None:
            continue
        row = []
        counter = 0
        for domain, subdomain_list in resp.items():
            if counter > domain_q:
                break
            counter += 1
            row.append(subdomain_list)
        new_response_gpt_combo_data.append(row)

    issues_responses = list(
        zip(
            issue_data,
            new_response_rf_data,
            new_response_gpt_data,
            new_response_gpt_combo_data,
        )
    )

    return render(
        request,
        "repo_detail.html",
        {
            "repo_name": repo_name,
            "issues_responses": issues_responses,  # Pass as list
        },
    )


def get_CoreEngine_version(request):
    return JsonResponse(
        {
            "CoreEngine_Version": CoreEngine.__version__,
            "Web_Version": version.WEBSITE_VERSION,
            "Web_Version_Hash": version.get_version_hash(),
            "Server_Uptime": version.server_start,
        }
    )


def splash_screen(request, repo_name, username):
    return render(
        request, "splash_screen.html", {"repo_name": repo_name, "username": username}
    )
