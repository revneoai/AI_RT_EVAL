import os
import subprocess
import inquirer
from colorama import init, Fore, Style
import time
from typing import List, Optional

init()  # Initialize colorama for colored output

class GitHubHelper:
    def __init__(self):
        self.username: Optional[str] = None
        self.repo_name: str = "ai-collaborative-platform"
    
    def print_step(self, message: str):
        """Print step with formatting"""
        print(f"\n{Fore.CYAN}=== {message} ==={Style.RESET_ALL}")
        time.sleep(1)  # Give user time to read
    
    def print_info(self, message: str):
        """Print info message"""
        print(f"{Fore.GREEN}ℹ {message}{Style.RESET_ALL}")
    
    def print_warning(self, message: str):
        """Print warning message"""
        print(f"{Fore.YELLOW}⚠ {message}{Style.RESET_ALL}")
    
    def run_command(self, command: List[str], show_output: bool = True) -> bool:
        """Run a shell command and return success status"""
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE if not show_output else None,
                stderr=subprocess.PIPE if not show_output else None,
                text=True
            )
            return result.returncode == 0
        except Exception as e:
            self.print_warning(f"Error running command: {e}")
            return False
    
    def check_git_installed(self) -> bool:
        """Check if git is installed"""
        self.print_step("Checking Git Installation")
        if self.run_command(["git", "--version"], show_output=False):
            self.print_info("Git is installed!")
            return True
        else:
            self.print_warning("Git is not installed!")
            print(f"{Fore.YELLOW}Please install Git from: https://git-scm.com/downloads{Style.RESET_ALL}")
            return False
    
    def configure_git(self):
        """Configure Git settings"""
        self.print_step("Configuring Git")
        
        questions = [
            inquirer.Text('name', message="What's your name?"),
            inquirer.Text('email', message="What's your email?"),
            inquirer.Text('username', message="What's your GitHub username?")
        ]
        
        answers = inquirer.prompt(questions)
        self.username = answers['username']
        
        self.run_command(["git", "config", "--global", "user.name", answers['name']])
        self.run_command(["git", "config", "--global", "user.email", answers['email']])
        
        self.print_info("Git configured successfully!")
    
    def initialize_repository(self):
        """Initialize git repository"""
        self.print_step("Initializing Git Repository")
        
        if os.path.exists(".git"):
            self.print_info("Git repository already initialized!")
            return
        
        if self.run_command(["git", "init"]):
            self.print_info("Repository initialized!")
        
        self.print_info("Adding files to repository...")
        self.run_command(["git", "add", "."])
        
        self.run_command(["git", "commit", "-m", "Initial commit"])
        self.print_info("Files committed!")
    
    def setup_github_remote(self):
        """Setup GitHub remote repository"""
        self.print_step("Setting up GitHub Remote")
        
        print(f"{Fore.YELLOW}Please follow these steps:{Style.RESET_ALL}")
        print("1. Go to GitHub.com")
        print("2. Click '+' in top right")
        print("3. Select 'New repository'")
        print(f"4. Name it: {self.repo_name}")
        print("5. Don't initialize with README")
        print("6. Click 'Create repository'")
        
        input("\nPress Enter when you've created the repository...")
        
        remote_url = f"https://github.com/{self.username}/{self.repo_name}.git"
        self.run_command(["git", "remote", "add", "origin", remote_url])
        self.run_command(["git", "branch", "-M", "main"])
        
        if self.run_command(["git", "push", "-u", "origin", "main"]):
            self.print_info("Code pushed to GitHub!")
    
    def explain_github_actions(self):
        """Explain GitHub Actions"""
        self.print_step("Understanding GitHub Actions")
        
        print(f"\n{Fore.GREEN}Your GitHub Actions workflow will:{Style.RESET_ALL}")
        print("1. Run automatically on push to main/develop branches")
        print("2. Run your tests and check code coverage")
        print("3. Check code style with flake8, black, and isort")
        
        print(f"\n{Fore.GREEN}To view your Actions:{Style.RESET_ALL}")
        print(f"1. Visit: https://github.com/{self.username}/{self.repo_name}/actions")
        print("2. Click on the latest workflow run")
        print("3. View the test and lint job results")
        
        input("\nPress Enter to continue...")
    
    def show_next_steps(self):
        """Show next steps"""
        self.print_step("Next Steps")
        
        print(f"\n{Fore.GREEN}Your repository is set up! Here's what you can do next:{Style.RESET_ALL}")
        print("1. Create a new branch:")
        print("   git checkout -b feature/new-feature")
        
        print("\n2. Make changes and commit:")
        print("   git add .")
        print("   git commit -m 'Add new feature'")
        
        print("\n3. Push changes:")
        print("   git push origin feature/new-feature")
        
        print("\n4. Create a Pull Request on GitHub")
        
        print(f"\nRepository URL: https://github.com/{self.username}/{self.repo_name}")
    
    def run(self):
        """Run the helper"""
        if not self.check_git_installed():
            return
        
        self.configure_git()
        self.initialize_repository()
        self.setup_github_remote()
        self.explain_github_actions()
        self.show_next_steps()

if __name__ == "__main__":
    helper = GitHubHelper()
    helper.run() 