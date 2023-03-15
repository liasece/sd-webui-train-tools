import launch
import os
import git

my_path = os.path.dirname(os.path.realpath(__file__))

repo = git.Repo(my_path)
repo.git.submodule("update", "--init", "--recursive")

python_requirements_file = os.path.join(my_path, "requirements.txt")

with open(python_requirements_file) as file:
    launch.run_pip(f"install -r {python_requirements_file}", f"sd-webui-train-tools requirement: {python_requirements_file}")
