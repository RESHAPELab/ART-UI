# When making any changes, change the website version variable below:

WEBSITE_VERSION = "1.0.0"


## Put all "code-related" files here. This calculates a hash that
## works a lot like a basic version estimator.
## if the hashes are different, something changed.

import datetime
import glob
import hashlib
import os

code_paths = [
    "AST_Rock_Website/templates/**/*.html",
    "AST_Rock_Website/urls.py",
    "dashboard/*.py",
    "staticfiles/js/**",
    "staticfiles/images/**",
    "staticfiles/css/**",
    "staticfiles/webfonts/**",
    "manage.py",
]

# No need to adjust below in updating versions.

hash_version = None
server_start = datetime.datetime.now()


def calc_hash():
    hash = hashlib.sha256()

    all_files = []
    for x in code_paths:
        files = glob.glob(x, recursive=True)
        for file in files:
            if not (os.path.isfile(file)):
                continue
            all_files.append(file)

    all_files.sort()  # guarantees same order.
    for file in all_files:
        print(file)
        with open(file, "rb") as f:
            hash.update(f.read())

    return hash.hexdigest()


def get_version_hash():
    global hash_version
    if hash_version is None:
        hash_version = calc_hash()
    return hash_version
