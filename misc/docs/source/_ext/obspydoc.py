
import re

def post_process_html(app, pagename, templatename, context, doctree):
    try:
        context['body'] = \
            context['body'].replace("&#8211;", '<span class="dash" />')
    except:
        pass

    if not doctree or not doctree.has_name('citations'):
        return

    # Fix citations list
    body = re.sub(
        r'<td><table class="first last docutils citation" '
        r'''frame="void" id="(?P<tag>[A-Za-z0-9]+)" rules="none">
<colgroup><col class="label" /><col /></colgroup>
<tbody valign="top">
<tr><td class="label">(?P<content>\[[A-Za-z0-9]+\])</td><td></td></tr>
</tbody>
</table>
</td>''',
        r'<td id="\g<tag>">'
        r'<span class="label label-default">\g<content></span>'
        r'</td>',
        context['body'])
    context['body'] = body

def setup(app):
    app.connect('html-page-context', post_process_html)
