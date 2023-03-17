function ask_for_project_name() {
    name_ = prompt("Project name:");
    return [name_, "", ""];
}
function ask_for_project_version_name(project) {
    name_ = prompt("Version name:", "v1");
    return [project, name_];
}

function li_tt_showSubmitButtons(id, show) {
    gradioApp().getElementById(id).style.display = show ? "block" : "none";
}

function on_ui_update_dataset_click() {
    rememberGallerySelection("gr_project_version_dataset_gallery");
    li_tt_showSubmitButtons("update_dataset_btn", false);
    var id = randomId();
    requestProgress(id, gradioApp().getElementById("gr_project_version_dataset_gallery_container"), gradioApp().getElementById("gr_project_version_dataset_gallery"), function () {
        li_tt_showSubmitButtons("update_dataset_btn", true);
    });
    var res = create_submit_args(arguments);
    res[0] = id;
    return res;
}

function on_train_begin_click() {
    li_tt_showSubmitButtons("begin_train_btn", false);
    var id = randomId();
    requestProgress(id, gradioApp().getElementById("train_begin_btn_container"), null, function () {
        li_tt_showSubmitButtons("begin_train_btn", true);
    });
    var res = create_submit_args(arguments);
    res[0] = id;
    return res;
}

function on_ui_preview_generate_all_preview_btn_click() {
    li_tt_showSubmitButtons("preview_generate_all_preview_btn", false);
    var id = randomId();
    requestProgress(id, gradioApp().getElementById("preview_generate_all_preview_btn_container"), null, function () {
        li_tt_showSubmitButtons("preview_generate_all_preview_btn", true);
    });
    var res = create_submit_args(arguments);
    res[0] = id;
    return res;
}
