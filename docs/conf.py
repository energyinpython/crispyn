# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'crispyn'
copyright = '2023, energyinpython'
author = 'Aleksandra Bączkiewicz'

release = '0.1'
version = '0.0.6'

# -- General configuration

extensions = ['autoapi.extension',
'nbsphinx',
'sphinx_rtd_theme',
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
