import argparse
import os
import shutil
from huggingface_hub import HfApi
from tempfile import TemporaryDirectory

def import_model_to_HF_hub(model_path: str ,model_repo: str, user_token: str):
    api = HfApi()
    # Create the repo or make sure it exists
    api.create_repo(model_repo, private=True, repo_type="model", token=user_token, exist_ok=True)

    # Push the model to the repo
    with TemporaryDirectory() as temp_folder:
        shutil.copy(model_path, temp_folder+"/pytorch_model.bin")
        shutil.copy("./examples/wavlm/utils/config/config.json", temp_folder)
        shutil.copy("./examples/wavlm/utils/config/preprocessor_config.json", temp_folder)
        api.upload_folder(
            folder_path=temp_folder,
            repo_id=model_repo,
            repo_type="model",
            token=user_token,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import model to Hugging Face Hub")
    parser.add_argument("model_repo", type=str, help="The name of the HF model repository")
    parser.add_argument("user_token", type=str, help="The user token for authentication", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("model_path", type=str, help="The path to the model files")

    args = parser.parse_args()
    import_model_to_HF_hub(args.model_path, args.model_repo, args.user_token)
    print("Model imported to HF Hub")
