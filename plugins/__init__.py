"""
Copyright (c) 2026, g023 (https://github.com/g023)

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Plugin system for the Qwen-Image generation pipeline.
Auto-discovers and registers plugins from subfolders.
"""
import importlib
import pkgutil
from pathlib import Path


class PluginRegistry:
    """Central registry for all pipeline plugins."""
    _plugins = {
        'sampler': {},
        'scheduler': {},
        'prompt_processor': {},
    }

    @classmethod
    def register(cls, category, name):
        """Decorator to register a plugin class."""
        def decorator(klass):
            cls._plugins[category][name] = klass
            return klass
        return decorator

    @classmethod
    def get(cls, category, name, **kwargs):
        """Instantiate a plugin by category and name."""
        if name not in cls._plugins[category]:
            available = list(cls._plugins[category].keys())
            raise ValueError(f"Unknown {category} '{name}'. Available: {available}")
        return cls._plugins[category][name](**kwargs)

    @classmethod
    def list_plugins(cls, category):
        """List available plugin names for a category."""
        return list(cls._plugins[category].keys())

    @classmethod
    def list_all(cls):
        """List all registered plugins."""
        return {cat: list(plugins.keys()) for cat, plugins in cls._plugins.items()}


def _discover_plugins():
    """Import all plugin modules to trigger registration."""
    plugin_dir = Path(__file__).parent
    for subdir in ['samplers', 'schedulers', 'prompt_processors']:
        pkg_path = plugin_dir / subdir
        if pkg_path.exists():
            pkg_name = f"plugins.{subdir}"
            for _importer, name, _ispkg in pkgutil.iter_modules([str(pkg_path)]):
                if not name.startswith('_'):
                    importlib.import_module(f"{pkg_name}.{name}")


_discover_plugins()
