import click
import os
import shutil
from typing import Optional

@click.group()
def cli():
    """AI Platform CLI tool"""
    pass

@cli.command()
@click.argument('name')
@click.option('--template', default='basic', help='Module template to use')
def generate_module(name: str, template: str):
    """Generate a new module with boilerplate"""
    template_dir = f'tools/templates/module/{template}'
    target_dir = f'backend/core/{name}'
    shutil.copytree(template_dir, target_dir)
    click.echo(f"Generated module: {name} using template: {template}")

@cli.command()
@click.option('--coverage/--no-coverage', default=True, help='Run with coverage')
def test(coverage: bool):
    """Run all tests"""
    cmd = 'pytest --cov=backend tests/' if coverage else 'pytest tests/'
    os.system(cmd)

@cli.command()
def run():
    """Run the application"""
    os.system('docker-compose up')

if __name__ == '__main__':
    cli()
