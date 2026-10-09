"""
Automated GitHub Push Helper for DeepShield
Pushes local repository commits to GitHub without requiring system Xcode tools.
"""

import sys
import getpass
from dulwich import porcelain
from dulwich.repo import Repo


def push_to_remote():
    repo = Repo(".")
    print("=== 🛡️ DeepShield GitHub Push Utility ===")

    # Check for existing remotes
    config = repo.get_config()
    existing_remote = None
    try:
        existing_remote = config.get((b"remote", b"origin"), b"url").decode()
        print(f"Existing remote URL: {existing_remote}")
    except KeyError:
        pass

    if len(sys.argv) > 1:
        remote_url = sys.argv[1]
    else:
        prompt_text = f"Enter GitHub repository URL [{existing_remote or 'https://github.com/USER/REPO.git'}]: "
        remote_url = input(prompt_text).strip() or existing_remote

    if not remote_url:
        print("Error: No remote URL provided.")
        return

    # Add or update remote
    config.set((b"remote", b"origin"), b"url", remote_url.encode())
    config.write_to_path()
    print(f"Set origin -> {remote_url}")

    # Set branch to main
    try:
        repo.refs[b"refs/heads/main"] = repo.head()
    except Exception:
        pass

    print("\nAttempting push to GitHub...")
    try:
        porcelain.push(repo, remote_location=remote_url, refspecs=b"refs/heads/main:refs/heads/main")
        print("✅ Successfully pushed to GitHub main branch!")
    except Exception as e:
        err_msg = str(e)
        if "Authentication" in err_msg or "401" in err_msg or "403" in err_msg:
            print("\n🔒 Authentication Required:")
            print("GitHub requires a Personal Access Token (classic) with 'repo' scope.")
            print("You can format your URL with token like: https://<TOKEN>@github.com/<USER>/<REPO>.git")
            print("Example:")
            print("  python3 push_to_github.py https://ghp_YourTokenHere@github.com/yourname/DeepShield.git")
        else:
            print(f"Push failed: {e}")


if __name__ == "__main__":
    push_to_remote()
