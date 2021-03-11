# Using VSCode remote with Colab

1. Open `env/colab_vscode.ipynb` in a colab instance and choose appropriate runtime config. 
2. To automatically load git credentials (needed if you want to commit), create the files `colab/git_email.txt` and `colab/git_user.txt` in your GDrive, containing your email address and git username. Otherwise set `INIT_GIT_CREDS = False` at the top of `env/colab_vscode.ipynb`
3. Run the cell. The first time you do this you'll need to follow the steps in the "Client machine configuration" section of the output window. Note that "<PUT_THE_ABSOLUTE_CLOUDFLARE_PATH_HERE>" means the file path of the cloudflared executable.
4. Wait for the "Your repository was closed with success" section to pop up and click the button.
5. In VSCode press 'Enter' for the first two prompts, then enter the password `melons69`.
6. Once connected open the VSCode terminal and type `bash env/vscode_extensions.sh` to install the Python extensions

### Notes

- You can use git through the VSCode terminal.
- You can open and run notebooks in VSCode (with the extensions installed by `vscode_extensions.sh`)
- `wandb` has to be initialized first using the main colab notebook for some reason.