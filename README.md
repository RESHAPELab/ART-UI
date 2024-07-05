# User Interface

Website: https://still-mesa-24591-bad3365c7700.herokuapp.com/

## About the UI

This website was built through django, a rapid web development framework. This UI is deployed through heroku and ultizies postgre and redis in the background. Open Issues are classified asynchronously through ajax logic. LLMs and Random Forest Models are ultilized to classify the open issues. 

## Core Engine

This UI ultizies the Core Engine to classify open issues. 

## File structure breakdown

AST Rock Website- Holds files at the Project level for django. 
  URls.py - Project Level URl configuration
  Settings.py - Hold key setting variables for the website. This would include configurations about the database (postgre), redis, github authentication, and much more.
  Templates- Holds Html code
DashBoard- Holds files at the app level for django.
  URls.py - App Level URl configuration
  Views.py- Logic + Rendering configuration.
  tasks.py- Used for asynchronous tasks when user is at splashscreen. 
CoreEngine- Issue Classifying Logic
  This holds the various directories and files for classifying open issues via llm models and chatgpt. 
StaticFiles- CSS + JS + Images

## Setup

Warning- Core Engine configurations are constantly changing. Added dependencies may need to be configured.
-Start a Virtual Environment
-Install required packages.
-Set up github oAuth app
-Setup heroku project
-Use redis and postgre addons
-Deploy 

  
