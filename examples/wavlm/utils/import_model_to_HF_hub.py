import argparse
import os
import shutil
from huggingface_hub import HfApi

def import_model_to_HF_hub(model_repo: str, user_token: str, model_folder: str):
    api = HfApi()
    # Create the repo or make sure it exists
    api.create_repo(model_repo, private=True, repo_type="model", token=user_token, exist_ok=True)

    # Push the model to the repo
    api.upload_folder(
        folder_path=model_folder,
        repo_id=model_repo,
        repo_type="model",
        token=user_token,
    )

def create_temp_folder(model_path: str):
    temp_folder = "temp_folder"
    os.makedirs(temp_folder, exist_ok=True)
    shutil.copy(model_path, temp_folder+"/pytorch_model.bin")
    shutil.copy("./examples/wavlm/utils/config/config.json", temp_folder)
    shutil.copy("./examples/wavlm/utils/config/preprocessor_config.json", temp_folder)
    return temp_folder

def delete_temp_folder(temp_folder: str):
    shutil.rmtree(temp_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import model to Hugging Face Hub")
    parser.add_argument("model_repo", type=str, help="The name of the HF model repository")
    parser.add_argument("user_token", type=str, help="The user token for authentication", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("model_path", type=str, help="The path to the model files")

    args = parser.parse_args()
    model_folder = create_temp_folder(args.model_path)
    print("Created temp folder")
    import_model_to_HF_hub(args.model_repo, args.user_token, model_folder)
    print("Model imported to HF Hub")
    delete_temp_folder(model_folder)
    print("Deleted temp folder")
