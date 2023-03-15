function ask_for_project_name() {
    name_ = prompt("Project name:");
    return [name_, "", ""];
}
function ask_for_project_version_name(project) {
    name_ = prompt("Version name:", "v1");
    return [project, name_];
}
