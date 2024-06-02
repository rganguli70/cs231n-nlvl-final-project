import argparse
from pathlib import Path
import json

from azure.ai.ml import MLClient, command, Input
# from azure.ai.ml.entities import Environment, BuildContext 
from azure.identity import AzureCliCredential
from azure.ai.ml.constants import AssetTypes, InputOutputModes

ws_config = json.load(open("ws_config.json"))
subscription_id = ws_config["subscription_id"]
resource_group = ws_config["resource_group_name"]
workspace_name = ws_config["workspace_name"]

def get_args(raw_args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_name", default="nlvl-experiment", help="Experiment name for AML Workspace")
    parser.add_argument("--display_name", default="nvlv-detr", help="Display name for training job")
    parser.add_argument("--data_asset", default="Charades-small", choices=["Charades-small", "Charades"], help="Name of dataset to mount for training")
    parser.add_argument("--compute", default="E4s-v3", choices=["E4s-v3", "v100"], help="AML compute target")
    parser.add_argument("--ort", default=False, action="store_true", help="Enable ONNX Runtime for accelerated training")
    parser.add_argument("--verbose", default=False, action="store_true", help="verbose")

    args = parser.parse_args(raw_args)
    return args

def main(raw_args=None):
    args = get_args(raw_args)

    ml_client = MLClient(
        AzureCliCredential(), subscription_id, resource_group, workspace_name
    )

    data_asset = ml_client.data.get(args.data_asset, label="latest")

    root_dir = Path(__file__).resolve().parent
    # environment_dir = root_dir / "environment"
    code_dir = root_dir / "src"

    env_vars = {"MODE": "cloud",
                "TOKENIZERS_PARALLELISM": "true"}
    if args.ort:
        env_vars["ORTMODULE_FALLBACK_POLICY"] = "FALLBACK_DISABLE",
    if args.verbose:
        env_vars["VERBOSE"] = "true"

    job = command(
        code=code_dir,
        command="python train_model_v2.py --data_dir ${{inputs.charades}}" \
              + " --ort" if args.ort else "",
        compute=args.compute,
        display_name=args.display_name \
                   + "-ort" if args.ort else "",
        # environment=Environment(build=BuildContext(path=environment_dir)),
        environment="cs231n-env@latest",
        environment_variables=env_vars,
        experiment_name=args.experiment_name,
        inputs={
            "charades": Input(
                type=AssetTypes.URI_FOLDER,
                path=data_asset.id,
                mode=InputOutputModes.DOWNLOAD,
            ),
        }
    )

    print("submitting PyTorch job for nlvl transformer pretrain")
    job_handle = ml_client.create_or_update(job)
    print("submitted job")

    aml_url = job_handle.studio_url
    print("job link:", aml_url)

if __name__ == "__main__":
    main()