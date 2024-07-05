# AST Rock Website
### Overview
The AST Rock Website is a dynamic web application designed for classifying open issues on software projects. Hosted on Heroku, it leverages a robust stack including Django, PostgreSQL, and Redis to deliver a responsive and scalable user experience. This project integrates cutting-edge machine learning models, including both Large Language Models (LLMs) like GPT and traditional Random Forest classifiers, to analyze and categorize issues efficiently.

### Key Features
Asynchronous Processing: Utilizes AJAX for non-blocking data processing, ensuring a seamless user experience even during heavy computations.
Advanced Classification: Employs both LLMs and Random Forest Models to provide accurate classifications of open issues.
OAuth Integration: Features GitHub authentication allowing for secure and personalized user interactions.
Real-Time Updates: Implements Redis for efficient management of background tasks and real-time data updates.

### Core Engine
The Core Engine is the computational heart of the website, responsible for the intelligent classification of open issues using a variety of algorithms:

LLM Integration: Harnesses the power of models like GPT for natural language understanding.
Random Forest Application: Utilizes ensemble learning to classify issues based on predefined features.

### Technical Setup
Framework: Built with Django, offering a powerful, scalable back-end architecture.
Database: Uses PostgreSQL for robust data management and query capabilities.
Caching and Queues: Implements Redis for caching and managing asynchronous task queues.
Deployment: Deployed on Heroku, leveraging its dynos for flexible, container-based deployment.

### File Structure
AST Rock Website: Root directory containing Django project settings and configurations.
urls.py: Project-level URL declarations.
settings.py: Configuration settings including database, caching, and third-party integrations.
templates: HTML templates for rendering views.
Dashboard: Django app managing the user interface and asynchronous tasks.
urls.py: App-level URL configurations.
views.py: Controllers that handle requests and render responses.
tasks.py: Asynchronous task definitions for background processing.
CoreEngine: Core logic for issue classification using various machine learning models.
StaticFiles: Directory containing CSS, JavaScript, and image files for the front-end.

### Setup Instructions
Due to frequent updates and dependencies in the Core Engine, follow these steps to ensure a successful setup:

Initialize a Virtual Environment: For managing Python packages.
Install Dependencies: Ensure all required packages are installed as per requirements.txt.
Configure GitHub OAuth: Set up a GitHub OAuth app for user authentication.
Heroku Configuration: Prepare the Heroku environment with necessary add-ons like PostgreSQL and Redis.
Local Development: Run python manage.py runserver to start the development server.
Deployment: Use git push heroku master to deploy changes to Heroku.
  
