import functools


@functools.lru_cache(maxsize=2)
def _get_html_templates_env():
    # For _html_repr_()
    import jinja2

    templates_env = jinja2.Environment(
        loader=jinja2.PackageLoader(
            package_name="obspy.html_templates", package_path="repr"
        ),
        autoescape=jinja2.select_autoescape(default=True, default_for_string=True),
    )

    return templates_env


def _get_html_template(name):
    return _get_html_templates_env().get_template(name)