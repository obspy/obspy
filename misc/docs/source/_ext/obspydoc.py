

def post_process_html(app, pagename, templatename, context, doctree):
    try:
        context['body'] = \
            context['body'].replace("&#8211;", '<span class="dash" />')
    except:
        pass


def setup(app):
    app.connect('html-page-context', post_process_html)
