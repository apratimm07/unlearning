import argh
import os
import pickle
from simulator import main as main_tofu
from configs import config_tofu_olmo

CONFIG_REGISTRY = {"tofu_olmo": config_tofu_olmo}

def run(exp_id='tofu_olmo', run_id=0, runpath=''):
    if exp_id not in CONFIG_REGISTRY:
        print(f"Unknown exp_id: {exp_id}")
        return

    _, runs = CONFIG_REGISTRY[exp_id]()
    config = runs[run_id]

    if runpath:
        os.makedirs(runpath, exist_ok=True)
        os.chdir(runpath)

    with open("config.pickle", "wb") as f:
        pickle.dump(config, f)

    main_tofu(config)

if __name__ == "__main__":
    parser = argh.ArghParser()
    parser.add_commands([run])
    parser.dispatch()
