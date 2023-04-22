
import PythonContextWarper as pc
import os
if __name__ == "__main__":
    from tabulate import tabulate
    with pc.PythonContextWarper(
            to_module_path= os.path.abspath(os.path.join(os.path.dirname(__file__), "sd_scripts")), 
            path_include= os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), 
            sub_module="library",
        ):
        import sd_scripts.train_network as train_network
        ss_parser = train_network.setup_parser()
        args = []
        for action in ss_parser._actions:
            args += [[str(",".join(action.option_strings)), action.default, str(action.type), action.help]]
        print(tabulate(args, headers=['arg', 'default', 'type', 'help']))

