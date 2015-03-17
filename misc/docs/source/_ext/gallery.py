"""
A directive for including a matplotlib plot in a gallery, mostly optimized for
Bootstrap.

Options
-------

The ``gallery-plot`` directive supports the following required options:

* ``target``: text (URI or reference name)
    Makes the image into a hyperlink reference ("clickable"). The option
    argument may be a URI (relative or absolute), or a reference name with
    underscore suffix (e.g. `` `a name`_ ``).
* ``alt``: text
    Alternate text: a short description of the image, displayed by applications
    that cannot display images, or spoken by applications for visually impaired
    users.

Configuration options
---------------------

Configuration options are the same as matplotlib's plot directive, with the
addition of the following:

* ``gallery_plot_classes``: list of text
    A list of classes to apply to the gallery plot's div. The 'thumbnail' class
    is automatically added.
"""
from __future__ import print_function

from docutils import nodes
from docutils.parsers.rst import Directive, DirectiveError, directives


class Gallery(Directive):
    name = 'gallery-plot'
    has_content = True
    required_arguments = 0
    optional_arguments = 2
    final_argument_whitespace = False
    option_spec = {
        'alt': directives.unchanged_required,
        'target': directives.unchanged_required,
    }

    def run(self):
        language = self.state_machine.language
        document = self.state.document
        directive, messages = directives.directive('plot', language, document)
        self.state.parent += messages

        # Temporarily override template
        #   * Add target to figure
        #   * Add classes for Bootstrap
        target = ':target: ' + self.options.pop('target', '')
        title = self.options['alt']
        classes = ' '.join(document.settings.env.config.gallery_plot_classes +
                           ['thumbnail'])
        template = '''
{{ only_html }}

   {% for img in images %}
   .. figure:: {{ build_dir }}/{{ img.basename }}.png
      ''' + target + '''
      :figclass: ''' + classes + '''
      {% for option in options -%}
      {{ option }}
      {% endfor %}

   {% endfor %}

{{ only_latex }}

   {% for img in images %}
   {% if 'pdf' in img.formats -%}
   .. image:: {{ build_dir }}/{{ img.basename }}.pdf
      ''' + target + '''
   {% endif -%}
   {% endfor %}

{{ only_texinfo }}

   {% for img in images %}
   .. image:: {{ build_dir }}/{{ img.basename }}.png
      ''' + target + '''
      {% for option in options -%}
      {{ option }}
      {% endfor %}

   {% endfor %}

'''

        plot_template_orig = document.settings.env.config.plot_template
        document.settings.env.config.plot_template = template

        # Don't bother with the high resolution version
        plot_formats_orig = document.settings.env.config.plot_formats
        plot_formats = []
        for f in plot_formats_orig:
            if isinstance(f, str) and 'hires' in f:
                continue
            elif isinstance(f, tuple) and 'hires' in f[0]:
                continue
            plot_formats.append(f)
        document.settings.env.config.plot_formats = plot_formats

        options = {
            'alt': title,
        }

        directive_instance = directive('plot', self.arguments, options,
                                       self.content, self.lineno,
                                       self.content_offset, self.block_text,
                                       self.state, self.state_machine)
        try:
            result = directive_instance.run()
        except DirectiveError as error:
            msg_node = self.reporter.system_message(error.level, error.msg,
                                                    line=self.lineno)
            msg_node += nodes.literal_block(self.block_text, self.block_text)
            result = [msg_node]

        # Restore original settings
        document.settings.env.config.plot_template = plot_template_orig
        document.settings.env.config.plot_formats = plot_formats_orig

        return result


def fix_thumbnail_class(app, doctree):
    """
    Move thumbnail class from figure to reference.

    The :figclass: option puts 'thumbnail' on the figure node, which ends up on
    the outer <div>, but we want 'thumbnail' to be on the <a>, so move it to
    the reference node.
    """

    if not doctree.has_name('gallery'):
        return

    for fig in doctree.traverse(condition=nodes.figure):
        try:
            fig['classes'].remove('thumbnail')
        except ValueError:
            continue
        for ref in fig.traverse(condition=nodes.reference):
            ref['classes'].append('thumbnail')


def setup(app):
    app.setup_extension('matplotlib.sphinxext.plot_directive')
    app.add_directive('gallery-plot', Gallery)
    app.add_config_value('gallery_plot_classes', [], 'html')
    app.connect('doctree-read', fix_thumbnail_class)
