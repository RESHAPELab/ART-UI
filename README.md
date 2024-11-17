# SkillScope UI

### Overview
SkillScope UI is a dynamic web application designed for classifying open issues on software projects. This repository can be hosted locally or on a VM, anywhere Django is supported. It uses Django, SQLite, and Redis to provide a simple yet responsive demonstration of the CoreEngine of Skillscope. This project integrates models generated by the CoreEngine and features an UI for exploration. 
<br><br>
[View a running demo.](https://skillscope.codingcando.com/)
<br><br>
[View setup demo video](/AST_Rock_Website/CoreEngine/docs/demo.mp4)

### Key Features
- Asynchronous Processing: Utilizes AJAX for non-blocking data processing, ensuring a seamless user experience even during heavy computations.
- Advanced Classification: Employs both LLMs and Random Forest Models generated by CoreEngine to provide accurate classifications of open issues.

### Technical Setup
- Framework: Built with Django, offering a powerful, scalable back-end architecture.
- Database: Uses SQLite for robust data management and query capabilities.
- Caching and Queues: Implements Redis for caching and managing asynchronous task queues.
- Deployment: A VM: https://skillscope.codingcando.com/

### Deployment Instructions
1. Install Dependencies
    > ``` sh
    > sudo apt install redis-server
2. Clone Repository
    > ``` sh
    >  git clone git@github.com:RESHAPELab/ART-UI.git
    >  cd ART-UI
    >  git submodule update --init
    >  cd AST_Rock_Website
3. Create Virtual Environment
    > ``` sh
    > virtualenv venv
    > source venv/bin/activate
4. Create `.env` file
    > See the ENV section below for the required env variables.
5. Install Python Packages
    > ``` sh
    > pip install -r requirements.txt

6. Install Spacy Package
    > ``` sh
    > pip install spacy 

7. Download spacy/en_core_web_md
    > ``` sh
    > python3 -m spacy download en_core_web_md  

8.  Collect the staticfiles
    > ``` sh
    > python3 manage.py collectstatic   

9. Train the models
    > Three model files are required: `rf_model.pkl`, `gpt_model.pkl`, `gpt_combined_model.pkl`
    > Obtain these by running the instructions from the CoreEngine section.
    > **Warning:** It may take a while to train, especially the `gpt_combined_model.pkl`

10. Setup the Database
    > ``` sh
    > python3 manage.py migrate
11. Run. Use two terminals (make sure to keep the virtual environment slected)
    ``` sh
        # One terminal:
        gunicorn AST_Rock_Website.wsgi -b 127.0.0.1:1234
        # Second terminal:
        python3 manage.py rqworker default
    ```

### .ENV Requirements
- `DJANGO_SETTINGS_MODULE` : Should be the location of the settings file. `AST_Rock_Website.settings`
- `OPENAI_API_KEY` : OpenAI key that is able to access the pre-trained models trained from CoreEngine.
- `REDIS_URL` : Location of the redis-server
- `DJANGO_SECRET_KEY` : Django Secret Key for session management

### File Structure
Inside `AST_Rock_Website/AST_Rock_Website/`
- urls.py: Project-level URL declarations.
- settings.py: Configuration settings including database, caching, and third-party integrations.
- templates: HTML templates for rendering views.

Inside `dashboard/`: 
- urls.py: App-level URL configurations.
- views.py: Controllers that handle requests and render responses.
- tasks.py: Asynchronous task definitions for background processing.

Inside `StaticFiles/`:
- Directory containing CSS, JavaScript, and image files for the front-end.
