# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'crispyn'
copyright = '2023, energyinpython'
author = 'Aleksandra Bączkiewicz'

release = '0.1'
version = '0.0.5'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
	'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
	'nbsphinx',
	'autoapi.extension',
]
autoapi_type = 'python'
autoapi_dirs = ["../src"]  # location to parse for API reference

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
